"""Single-worker HTTP service for the benchmarked TabPFN v2 model."""
import argparse
import json
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

import numpy as np
import pandas as pd
import torch

from housing_affordability.features import ROOT, Y, FeatureBuilder, load_external
from housing_affordability.model import HousingModel


class BatchedHTTPServer(ThreadingHTTPServer):
    request_queue_size = 256

    def get_request(self):
        connection, address = super().get_request()
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return connection, address


class HousingService:
    def __init__(self, history, cache=False, fast_attention=False, optimize=False):
        history = history.copy()
        example = history.iloc[:1].copy()
        sentinel = next(f'{i:05}' for i in range(100000) if f'{i:05}' not in set(history.Zip))
        example['Zip'] = sentinel
        self.builder = FeatureBuilder(history, load_external(), example)
        self.model = HousingModel(cache=cache, fast_attention=fast_attention, optimize=optimize).fit(self.builder.x, self.builder.y)
        latest = history.assign(_date=pd.to_datetime(history.Month)).sort_values('_date').drop_duplicates(['Zip','State'], keep='last')
        self.seen = latest.set_index(['Zip','State'])[Y]

    def predict(self, records):
        if not isinstance(records, list) or not 0 < len(records) <= 2048:
            raise ValueError('records must be a list of 1 to 2048 rows')
        frame = pd.DataFrame(records)
        if 'Zip' not in frame or 'State' not in frame:
            raise ValueError('Each row requires Zip and State')
        if not frame.Zip.map(lambda z: isinstance(z, str) and len(z) == 5 and z.isdigit()).all():
            raise ValueError('Zip must be a five-digit string')
        prior = self.seen.reindex(pd.MultiIndex.from_frame(frame[['Zip','State']]))
        seen = prior.notna().to_numpy()
        values = prior.to_numpy(copy=True)
        if (~seen).any():
            unseen = frame.loc[~seen].reset_index(drop=True)
            x = self.builder.transform(unseen)
            values[~seen] = self.model.predict(x)
        if not np.isfinite(values).all():
            raise ValueError('Prediction is not finite')
        return [{'Zip': z, 'State': state, 'prediction': float(value), 'seen': bool(flag)}
                for z, state, value, flag in zip(frame.Zip, frame.State, values, seen)]


def create_handler(service, keep_alive=False):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1' if keep_alive else 'HTTP/1.0'
        def do_POST(self):
            try:
                if self.path != '/predict':
                    self.send_error(404)
                    return
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 4_000_000:
                    raise ValueError('Invalid request size')
                response = json.dumps({'predictions': service.predict(json.loads(self.rfile.read(size))['records'])}, allow_nan=False).encode()
                self.send_response(200)
            except (ValueError, KeyError, TypeError) as error:
                response = json.dumps({'error': str(error)}).encode()
                self.send_response(400)
                self.close_connection = True
                self.send_header('Connection', 'close')
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--fast-attention', action='store_true',
                        help='Use experimental shared-key attention with the cached v2 model')
    parser.add_argument('--optimize', action='store_true',
                        help='Use shared-key attention and fitted-schema input encoding')
    parser.add_argument('--batch-window-ms', type=float, default=0,
                        help='Enable request batching with this window')
    parser.add_argument('--batch-strategy', choices=('queued', 'tabctx'), default='queued',
                        help='Choose queued batching or the optional tabctx coalescer')
    parser.add_argument('--http-backend', choices=('stdlib', 'asgi'), default='stdlib')
    args = parser.parse_args()
    if args.batch_window_ms < 0:
        parser.error('--batch-window-ms must not be negative')
    if (args.fast_attention or args.optimize) and not args.cache:
        parser.error('Inference optimizations require --cache')
    if args.http_backend == 'asgi':
        if not args.batch_window_ms or args.batch_strategy != 'queued':
            parser.error('ASGI requires queued batching and a positive --batch-window-ms')
        try:
            import uvicorn
            import fastapi
        except ImportError:
            parser.error('Install ASGI dependencies with pip install -e ".[serving]"')
    torch.set_num_threads(4)
    history = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
    service = HousingService(history, cache=args.cache, fast_attention=args.fast_attention, optimize=args.optimize)
    if args.batch_window_ms:
        from housing_affordability.batching import BatchedHousingService, QueuedHousingService
        strategy = QueuedHousingService if args.batch_strategy == 'queued' else BatchedHousingService
        service = strategy(service, window_ms=args.batch_window_ms)

    print(f'Listening on http://{args.host}:{args.port}', flush=True)
    if args.http_backend == 'asgi':
        from housing_affordability.async_server import create_async_app
        try:
            uvicorn.run(create_async_app(service), host=args.host, port=args.port,
                        access_log=False, loop='asyncio')
        finally:
            service.close()
        return
    server_type = BatchedHTTPServer if args.batch_window_ms else HTTPServer
    server = server_type((args.host, args.port), create_handler(service, keep_alive=bool(args.batch_window_ms)))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if hasattr(service, 'close'):
            service.close()


if __name__ == '__main__':
    main()
