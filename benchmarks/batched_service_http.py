"""Compare real local HTTP requests with and without tabctx coalescing."""
import concurrent.futures
import json
import threading
import time
import urllib.request
from http.server import HTTPServer

import numpy as np
import pandas as pd
import torch

from housing_affordability.batching import BatchedHousingService
from housing_affordability.features import ROOT
from housing_affordability.server import BatchedHTTPServer, HousingService, create_handler


class QueuedHTTPServer(HTTPServer):
    request_queue_size = 256


def load_burst(port, rows, concurrency):
    n = len(rows)
    barrier = threading.Barrier(concurrency)
    latency = [None] * n
    predictions = [None] * n

    def send(i):
        if i < concurrency:
            barrier.wait()
        payload = json.dumps({'records': [rows[i]]}).encode()
        req = urllib.request.Request(f'http://127.0.0.1:{port}/predict', data=payload,
                                     headers={'Content-Type': 'application/json'})
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=180) as response:
            item = json.load(response)['predictions'][0]
        latency[i] = time.perf_counter()-start
        predictions[i] = item['prediction']
        if item['Zip'] != rows[i]['Zip']:
            raise ValueError('Response ZIP/order mismatch')

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(send, range(n)))
    elapsed = time.perf_counter()-start
    return {'seconds': elapsed, 'requests_per_second': n/elapsed,
            'median_latency_seconds': float(np.median(latency)),
            'p95_latency_seconds': float(np.percentile(latency, 95)), 'predictions': predictions}


def main():
    torch.set_num_threads(4)
    train = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
    test = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip': str})
    unseen = test.loc[~test.Zip.isin(train.Zip)].reset_index(drop=True)
    rows = json.loads(unseen.head(256).to_json(orient='records'))
    expected = pd.read_csv(ROOT/'artifacts/external_acs/foundation_realtor_test.csv', dtype={'Zip': str})
    if not expected[['Zip','State']].equals(unseen[['Zip','State']]):
        raise ValueError('Reference ZIP mismatch')
    service = HousingService(train, cache=True)
    report = {'device': torch.cuda.get_device_name(0),
              'scope': 'localhost HTTP, cached model, includes JSON parsing and feature preparation', 'runs': {}}
    for mode in ['serial', 'batched_5ms', 'batched_20ms']:
        window = 20 if mode == 'batched_20ms' else 5
        wrapped = service if mode == 'serial' else BatchedHousingService(service, window_ms=window)
        kind = QueuedHTTPServer if mode == 'serial' else BatchedHTTPServer
        class QuietHandler(create_handler(wrapped)):
            def log_message(self, format, *args):
                pass
        server = kind(('127.0.0.1', 0), QuietHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            # Warm up before timing each mode.
            load_burst(server.server_port, rows[:1], 1)
            report['runs'][mode] = {}
            levels = [1, 8, 32] if mode == 'serial' else [1, 8, 32, 64, 128, 256]
            for clients in levels:
                count = max(32, clients)
                trials = []
                for _ in range(2):
                    result = load_burst(server.server_port, rows[:count], clients)
                    result['max_abs_difference_vs_submission'] = float(np.max(np.abs(np.asarray(result.pop('predictions'))-expected.tabpfn.to_numpy()[:count])))
                    trials.append(result)
                report['runs'][mode][str(clients)] = trials
                print('HTTP', mode, clients, json.dumps(trials), flush=True)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    out = ROOT/'artifacts/serving/batched_http.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n')
    print('HTTP_BENCHMARK', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
