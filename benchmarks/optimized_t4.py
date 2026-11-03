"""Validate shared-key attention, CPU features, and queue latency on one T4."""
import gc
import json
import threading
import time

import numpy as np
import pandas as pd
import torch

from benchmarks.batched_service_http import load_burst
from housing_affordability.aggregated_features import enc
from housing_affordability.attention import enable_shared_key_attention
from housing_affordability.batching import QueuedHousingService
from housing_affordability.features import ROOT, Y, aggregate_history
from housing_affordability.server import BatchedHTTPServer, HousingService, create_handler


def http_run(service, rows, window):
    wrapped = QueuedHousingService(service, window_ms=window)
    class Quiet(create_handler(wrapped)):
        def log_message(self, fmt, *args):
            pass
    server = BatchedHTTPServer(('127.0.0.1', 0), Quiet)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    results = {}
    try:
        load_burst(server.server_port, rows[:1], 1)
        for clients in [1, 32, 128, 256]:
            count = 16 if clients == 1 else clients*4
            data = [rows[i % len(rows)] for i in range(count)]
            trials = []
            for _ in range(3):
                trial = load_burst(server.server_port, data, clients)
                trial.pop('predictions')
                trials.append(trial)
            results[str(clients)] = trials
            print('OPTIMIZED_HTTP', window, clients, json.dumps(trials), flush=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        wrapped.close()
    return results


def main():
    torch.set_num_threads(4)
    history = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
    test = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip': str})
    unseen = test.loc[~test.Zip.isin(history.Zip)].reset_index(drop=True)
    records = json.loads(unseen.head(256).to_json(orient='records'))
    service = HousingService(history, cache=True)
    fitted_encoder = service.builder.regional_encoder.transform
    service.builder.regional_encoder.transform = lambda q: enc(service.builder.reference, q)
    q = service.builder.transform(unseen)
    baseline = service.model.predict(q)
    report = {'gpu': torch.cuda.get_device_name(),
              'scope': 'localhost HTTP including features; 3 trials, four request waves except 16 serial requests',
              'http': {'baseline_80ms': http_run(service, records, 80)}}
    service.builder.regional_encoder.transform = fitted_encoder
    report['installed_layers'] = enable_shared_key_attention(service.model.model)
    fast = service.model.predict(q)
    report['test_max_abs_difference'] = float(np.max(np.abs(fast-baseline)))
    for window in [0.1, 5, 20, 80]:
        report['http'][f'optimized_{window}ms'] = http_run(service, records, window)
    del service, fitted_encoder
    gc.collect()
    torch.cuda.empty_cache()

    zips = np.array(sorted(history.Zip.unique()))
    np.random.default_rng(42).shuffle(zips)
    held = set(zips[::5])
    fit = history.loc[~history.Zip.isin(held)].reset_index(drop=True)
    heldout = history.loc[history.Zip.isin(held) & (history.Month == '6/1/2026')].reset_index(drop=True)
    validation = HousingService(fit, cache=True)
    q = validation.builder.transform(heldout)
    before = validation.model.predict(q)
    enable_shared_key_attention(validation.model.model)
    after = validation.model.predict(q)
    actual = heldout[Y].to_numpy()
    report['heldout'] = {'rows': len(actual),
                         'baseline_rmse': float(np.sqrt(np.mean((before-actual)**2))),
                         'optimized_rmse': float(np.sqrt(np.mean((after-actual)**2))),
                         'max_abs_difference': float(np.max(np.abs(before-after)))}
    out = ROOT/'artifacts/serving/optimized_t4.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n')
    print('OPTIMIZED_T4', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
