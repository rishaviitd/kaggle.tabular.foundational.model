"""Compare XGBoost CPU and cached TabPFN GPU on the same Kaggle machine."""
import json
import gc
import os
import platform
import threading
import time

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from benchmarks.batched_service_http import load_burst
from housing_affordability.batching import QueuedHousingService
from housing_affordability.features import ROOT, Y, aggregate_history
from housing_affordability.server import BatchedHTTPServer, HousingService, create_handler


class XGBModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.03,
                                      subsample=0.8, colsample_bytree=0.8,
                                      reg_lambda=5, tree_method='hist', n_jobs=4,
                                      random_state=42)

    @staticmethod
    def encode(frame):
        out = frame.copy()
        for name in out.select_dtypes(include=['category']).columns:
            out[name] = out[name].cat.codes.replace(-1, np.nan)
        return out.replace([np.inf, -np.inf], np.nan).astype(np.float32)

    def fit(self, x, y):
        self.model.fit(self.encode(x), y)
        return self

    def predict(self, x):
        return np.clip(self.model.predict(self.encode(x)), 0, 100)


def http_trials(service, rows, window):
    wrapped = QueuedHousingService(service, window_ms=window) if window else service
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
            trials = []
            for _ in range(2):
                result = load_burst(server.server_port, rows[:max(32, clients)], clients)
                result.pop('predictions')
                trials.append(result)
            results[str(clients)] = trials
            print('HTTP', window, clients, json.dumps(trials), flush=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if window:
            wrapped.close()
    return results


def main():
    torch.set_num_threads(4)
    history = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
    test = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip': str})
    unseen = test.loc[~test.Zip.isin(history.Zip)].reset_index(drop=True)
    rows = json.loads(unseen.head(256).to_json(orient='records'))
    report = {'cpu': platform.processor(), 'logical_cpus': os.cpu_count(),
              'gpu': torch.cuda.get_device_name(0), 'xgboost_version': xgb.__version__,
              'scope': 'same machine and features; HTTP localhost, 1 unseen ZIP/request; two bursts',
              'models': {}}
    service = HousingService(history, cache=True)
    tabpfn = service.model
    xgb_model = XGBModel().fit(service.builder.x, service.builder.y)
    prepared = service.builder.transform(unseen)
    for name, model, windows in [('tabpfn_gpu', tabpfn, [80]),
                                 ('xgboost_cpu', xgb_model, [0, 20, 80])]:
        service.model = model
        item = {'model_only': {}, 'http': {}}
        for n in [1, 32, 256, min(len(prepared), 1704)]:
            x = prepared.iloc[:n]
            model.predict(x)
            durations = []
            for _ in range(3):
                started = time.perf_counter()
                model.predict(x)
                durations.append(time.perf_counter()-started)
            item['model_only'][str(n)] = {'seconds': durations,
                                           'rows_per_second': n/float(np.median(durations))}
        for window in windows:
            item['http'][str(window)] = http_trials(service, rows, window)
        report['models'][name] = item
        print('MODEL', name, json.dumps(item), flush=True)

    # A single held-out ZIP fold, evaluated identically for both models.
    del tabpfn, service, model
    gc.collect()
    torch.cuda.empty_cache()
    zips = aggregate_history(history).Zip
    fit_idx, val_idx = next(GroupKFold(5).split(zips, groups=zips))
    fit_zips, val_zips = set(zips.iloc[fit_idx]), set(zips.iloc[val_idx])
    fit_history = history[history.Zip.isin(fit_zips)]
    heldout = aggregate_history(history[history.Zip.isin(val_zips)])
    started = time.perf_counter()
    validation_service = HousingService(fit_history, cache=True)
    q = validation_service.builder.transform(heldout)
    target = heldout[Y].to_numpy()
    pred_tab = validation_service.model.predict(q)
    val_xgb = XGBModel().fit(validation_service.builder.x, validation_service.builder.y)
    pred_xgb = val_xgb.predict(q)
    report['heldout'] = {'zip_count': len(target), 'metric': 'RMSE of ZIP-average target',
                         'tabpfn_gpu': float(np.sqrt(np.mean((pred_tab-target)**2))),
                         'xgboost_cpu': float(np.sqrt(np.mean((pred_xgb-target)**2))),
                         'seconds': time.perf_counter()-started}
    path = ROOT/'artifacts/serving/xgboost_comparison.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+'\n')
    print('COMPARISON', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
