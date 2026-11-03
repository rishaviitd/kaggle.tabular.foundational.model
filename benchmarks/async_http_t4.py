"""Compare HTTP transports and CPU XGBoost with the same asynchronous clients."""
import asyncio
import json
import socket
import threading
import time

import aiohttp
import numpy as np
import pandas as pd
import torch
import uvicorn

from benchmarks.compare_xgboost import XGBModel
from housing_affordability.async_server import create_async_app
from housing_affordability.batching import QueuedHousingService
from housing_affordability.features import ROOT
from housing_affordability.server import BatchedHTTPServer, HousingService, create_handler


async def load(port, rows, clients, count):
    semaphore = asyncio.Semaphore(clients)
    latencies = []
    connector = aiohttp.TCPConnector(limit=clients)
    async with aiohttp.ClientSession(connector=connector) as session:
        async def send(i):
            async with semaphore:
                row = rows[i % len(rows)]
                started = time.perf_counter()
                async with session.post(f'http://127.0.0.1:{port}/predict',
                                        json={'records':[row]}) as response:
                    if response.status != 200:
                        raise RuntimeError(await response.text())
                    result = (await response.json())['predictions'][0]
                if result['Zip'] != row['Zip']:
                    raise ValueError('Response ZIP mismatch')
                latencies.append(time.perf_counter()-started)
        started = time.perf_counter()
        await asyncio.gather(*(send(i) for i in range(count)))
        seconds = time.perf_counter()-started
    return {'seconds':seconds,'requests_per_second':count/seconds,
            'median_latency_seconds':float(np.median(latencies)),
            'p95_latency_seconds':float(np.percentile(latencies,95))}


def benchmark(service, rows, backend, window):
    queued = QueuedHousingService(service, window_ms=window)
    sock = None
    if backend == 'asgi':
        sock = socket.socket()
        sock.bind(('127.0.0.1',0))
        port = sock.getsockname()[1]
        config = uvicorn.Config(create_async_app(queued), log_level='error',
                                loop='asyncio', access_log=False)
        server = uvicorn.Server(config)
        thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    else:
        class Quiet(create_handler(queued, keep_alive=backend == 'stdlib_keepalive')):
            def log_message(self, fmt, *args):
                pass
        server = BatchedHTTPServer(('127.0.0.1',0), Quiet)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    results = {}
    try:
        if backend == 'asgi':
            deadline = time.monotonic()+20
            while not server.started:
                if not thread.is_alive() or time.monotonic()>deadline:
                    raise RuntimeError('ASGI server failed to start')
                time.sleep(.01)
        asyncio.run(load(port,rows,1,1))
        for clients in [1,32,128,256]:
            count = 16 if clients==1 else clients*4
            trials = [asyncio.run(load(port,rows,clients,count)) for _ in range(3)]
            results[str(clients)] = trials
            print('ASYNC_HTTP_LEVEL',backend,window,clients,json.dumps(trials),flush=True)
    finally:
        if backend == 'asgi':
            server.should_exit = True
        else:
            server.shutdown()
            server.server_close()
        thread.join(timeout=20)
        if sock:
            sock.close()
        queued.close()
    return results


def main():
    torch.set_num_threads(4)
    train = pd.read_csv(ROOT/'data/train.csv',dtype={'Zip':str})
    test = pd.read_csv(ROOT/'data/test.csv',dtype={'Zip':str})
    unseen = test.loc[~test.Zip.isin(train.Zip)].reset_index(drop=True)
    rows = json.loads(unseen.head(256).to_json(orient='records'))
    service = HousingService(train,cache=True,fast_attention=True)
    report = {'gpu':torch.cuda.get_device_name(),'cpu_threads':4,
              'client':'aiohttp, same host; three trials, four waves except 16 serial requests',
              'runs':{}}
    for backend,window in [('stdlib',5),('asgi',5),('asgi',20),('asgi',80)]:
        report['runs'][f'tabpfn_{backend}_{window}ms'] = benchmark(service,rows,backend,window)
    service.model = XGBModel().fit(service.builder.x,service.builder.y)
    for window in [.1,20,80]:
        report['runs'][f'xgboost_asgi_{window}ms'] = benchmark(service,rows,'asgi',window)
    out=ROOT/'artifacts/serving/async_http_t4.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('ASYNC_HTTP_T4',json.dumps(report),flush=True)


if __name__=='__main__':
    main()
