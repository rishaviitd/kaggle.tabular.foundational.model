"""Sweep single-T4 tabctx concurrency and batching window for one context."""
import json
import time

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tabctx.batching import CoalescingPredictor
from tabctx.cache.manager import ContextCacheManager
from tabctx.engine import TabctxEngine

from benchmarks.tabctx_t4 import HousingBackend, T4BenchmarkEstimator, encode, run_load
from housing_affordability.features import ROOT, load_external, prepare


def main():
    torch.set_num_threads(4)
    train = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
    test = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip': str})
    unseen = test.loc[~test.Zip.isin(train.Zip)].reset_index(drop=True)
    x, y, q = prepare(train, unseen, load_external())
    numeric_x, numeric_q = encode(x), encode(q)
    checkpoint = hf_hub_download('Prior-Labs/TabPFN-v2-reg', 'tabpfn-v2-regressor.ckpt')
    engine = TabctxEngine(backend=HousingBackend(x, checkpoint), cache=ContextCacheManager(13*2**30), estimator=T4BenchmarkEstimator())
    engine.fit(numeric_x, y.to_numpy(), task='regression', dataset_id='housing')
    reference = np.asarray(engine.predict('housing', numeric_q).predictions)
    saved = pd.read_csv(ROOT/'artifacts/external_acs/foundation_realtor_test.csv', dtype={'Zip': str})
    if not saved[['Zip','State']].equals(unseen[['Zip','State']]):
        raise ValueError('Reference ZIP mismatch')
    out = ROOT/'artifacts/serving/tabctx_scaling.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {'device': torch.cuda.get_device_name(0), 'scope': 'preprocessed inputs, in-process tabctx; no HTTP/network',
              'tabctx_commit': '6397d5caad52c3705c664336b77f3fe39f0cbfd0',
              'cache_reference_max_abs_difference': float(np.max(np.abs(reference-saved.tabpfn.to_numpy()))),
              'direct_batches': {}, 'concurrent': {}}

    for n in [1, 32, 128, 256, 512, 1024, len(q)]:
        times = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            pred = engine.predict('housing', numeric_q[:n]).predictions
            torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
        report['direct_batches'][str(n)] = {'seconds': times, 'median_seconds': float(np.median(times)),
                                            'rows_per_second': float(n/np.median(times)),
                                            'max_abs_difference': float(np.max(np.abs(np.asarray(pred)-reference[:n]))),
                                            'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30}
        out.write_text(json.dumps(report, indent=2)+'\n')
        print('DIRECT', n, json.dumps(report['direct_batches'][str(n)]), flush=True)

    for window in [.001, .005, .020]:
        label = str(round(window*1000))+'ms'
        report['concurrent'][label] = {}
        for concurrency in [8, 32, 64, 128, 256]:
            repeats = []
            for _ in range(3):
                predictor = CoalescingPredictor(engine, window_s=window)
                requests = min(512, concurrency*2)
                torch.cuda.reset_peak_memory_stats()
                result = run_load(predictor, numeric_q[:requests], concurrency, requests=requests)
                pred = np.asarray(result.pop('predictions'))
                result['requests'] = requests
                result['engine_calls'] = predictor.engine_calls
                result['max_abs_difference'] = float(np.max(np.abs(pred-reference[:requests])))
                result['peak_reserved_gib'] = torch.cuda.max_memory_reserved()/2**30
                repeats.append(result)
            report['concurrent'][label][str(concurrency)] = repeats
            out.write_text(json.dumps(report, indent=2)+'\n')
            print('CONCURRENT', label, concurrency, json.dumps(repeats), flush=True)
    print('SCALING_BENCHMARK', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
