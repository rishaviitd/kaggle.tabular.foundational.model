"""Compare tabctx request coalescing with serialized cached TabPFN on a T4.

The upstream backend casts inputs to numeric arrays and chooses its default
checkpoint. The adapter below restores our training categorical schema and
selects the exact v2 checkpoint and estimator count used by this project.
"""
import concurrent.futures
import json
import threading
import time

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tabpfn import TabPFNRegressor
from tabctx.backends.tabpfn import TabPFNBackend
from tabctx.batching import CoalescingPredictor
from tabctx.cache.manager import ContextCacheManager
from tabctx.engine import TabctxEngine
from tabctx.memory.calibration_data import A100_40GB_TABICL_CALIBRATION
from tabctx.memory.estimator import PowerLawMemoryEstimator
from tabctx.types import PredictOutcome

from housing_affordability.features import ROOT, load_external, prepare


class HousingBackend(TabPFNBackend):
    """Benchmark adapter for fixed v2 checkpoint and pandas categoricals."""
    def __init__(self, template, checkpoint):
        super().__init__(device='cuda', cache_mode='kv')
        self.template = template.iloc[:0].copy()
        self.checkpoint = checkpoint

    def frame(self, rows):
        result = pd.DataFrame(np.asarray(rows, dtype=float), columns=self.template.columns)
        for col in self.template.columns:
            dtype = self.template[col].dtype
            if isinstance(dtype, pd.CategoricalDtype):
                codes = result[col].fillna(-1).to_numpy(dtype=int)
                result[col] = pd.Categorical.from_codes(codes, categories=dtype.categories)
        return result

    def fit(self, X, y, task):
        if task != 'regression':
            raise ValueError('Housing benchmark requires regression')
        model = TabPFNRegressor(n_estimators=4, model_path=self.checkpoint, device='cuda',
                                n_preprocessing_jobs=2, random_state=42, fit_mode='fit_with_cache')
        model.fit(self.frame(X), np.asarray(y))
        return model

    def predict(self, payload, X_test, return_proba=False):
        return PredictOutcome(predictions=np.clip(payload.predict(self.frame(X_test)), 0, 100).tolist())


class T4BenchmarkEstimator:
    """Single-workload test budget based on prior T4 measurement, not production admission."""
    def estimate_bytes(self, n_train, n_test, n_features):
        if n_train != 6830 or n_features != 106 or n_test > 2048:
            raise ValueError('T4 benchmark estimator is limited to the measured workload')
        return int(10.5 * 2**30)

    def admit(self, n_train, n_test, n_features):
        return self.estimate_bytes(n_train, n_test, n_features) <= self.ceiling_bytes()

    def ceiling_bytes(self):
        return 13 * 2**30

    def admission_headroom_bytes(self, used_bytes):
        return self.ceiling_bytes()

    def confidence(self):
        return 'Single T4 workload; not calibrated for other tables or concurrency.'

    def record_observation(self, n_train, n_features, real_bytes):
        pass


def encode(frame):
    out = frame.copy()
    for col in out:
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].cat.codes
    return out.to_numpy(dtype=float)


def run_load(predictor, data, concurrency, requests=32):
    barrier = threading.Barrier(concurrency)
    latencies = []
    outputs = [None] * requests
    lock = threading.Lock()

    def invoke(i):
        if i < concurrency:
            barrier.wait()
        start = time.perf_counter()
        outcome = predictor.predict('housing', data[i:i+1])
        duration = time.perf_counter() - start
        with lock:
            latencies.append(duration)
        outputs[i] = outcome.predictions[0]

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(invoke, range(requests)))
    elapsed = time.perf_counter() - start
    return {'seconds': elapsed, 'requests_per_second': requests/elapsed,
            'latency_median': float(np.median(latencies)), 'latency_p95': float(np.percentile(latencies, 95)),
            'predictions': outputs}


def main():
    torch.set_num_threads(4)
    train = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
    test = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip': str})
    new = test.loc[~test.Zip.isin(train.Zip)].reset_index(drop=True)
    x, y, q = prepare(train, new, load_external())
    expected = pd.read_csv(ROOT/'artifacts/external_acs/foundation_realtor_test.csv', dtype={'Zip': str})
    assert expected[['Zip', 'State']].equals(new[['Zip', 'State']])
    checkpoint = hf_hub_download('Prior-Labs/TabPFN-v2-reg', 'tabpfn-v2-regressor.ckpt')
    gpu = torch.cuda.get_device_properties(0)
    prior = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION,
                                    hard_ceiling_bytes=13*2**30, gpu_capacity_bytes=gpu.total_memory)
    report = {'device': torch.cuda.get_device_name(0), 'tabctx_commit': '6397d5caad52c3705c664336b77f3fe39f0cbfd0',
              'scope': 'in-process tabctx engine/coalescer; excludes HTTP, network and feature preparation',
              'default_a100_estimate_gib': prior.estimate_bytes(len(x), 0, len(x.columns))/2**30,
              'default_a100_admits_on_t4': prior.admit(len(x), 0, len(x.columns)),
              'adapter': 'v2 checkpoint, four estimators, restore original categorical schema; fixed test-only T4 memory estimate'}
    backend = HousingBackend(x, checkpoint)
    estimator = T4BenchmarkEstimator()
    engine = TabctxEngine(backend=backend, cache=ContextCacheManager(13*2**30), estimator=estimator)
    numeric_x, numeric_q = encode(x), encode(q)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    engine.fit(numeric_x, y.to_numpy(), task='regression', dataset_id='housing')
    torch.cuda.synchronize()
    report['fit_seconds'] = time.perf_counter() - start
    pred = np.array(engine.predict('housing', numeric_q).predictions)
    report['max_abs_difference_vs_submission'] = float(np.max(np.abs(pred - expected.tabpfn.to_numpy())))
    report['mean_abs_difference_vs_submission'] = float(np.mean(np.abs(pred - expected.tabpfn.to_numpy())))
    report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
    report['peak_reserved_gib'] = torch.cuda.max_memory_reserved()/2**30
    report['runs'] = {}
    for window in [0, .005]:
        predictor = CoalescingPredictor(engine, window_s=window)
        label = 'serial' if window == 0 else 'coalesced_5ms'
        report['runs'][label] = {}
        for concurrency in [1, 8, 32]:
            result = run_load(predictor, numeric_q[:32], concurrency)
            result['max_abs_difference_vs_batch'] = float(np.max(np.abs(np.array(result.pop('predictions')) - pred[:32])))
            report['runs'][label][str(concurrency)] = result
            print(label, concurrency, json.dumps(result), flush=True)
        report['runs'][label]['engine_calls'] = predictor.engine_calls
        report['runs'][label]['batched_requests'] = predictor.batched_requests
    dest = ROOT/'artifacts/serving/tabctx_benchmark.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2)+'\n')
    print('TABCTX_BENCHMARK', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
