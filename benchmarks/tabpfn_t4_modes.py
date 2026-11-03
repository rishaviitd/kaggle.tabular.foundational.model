"""Compare T4 inference precision, memory mode and CPU thread settings."""
import gc
import json
import time

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tabpfn import TabPFNRegressor

from housing_affordability.features import ROOT, load_external, prepare


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    return value, time.perf_counter()-start


def main():
    train = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
    test = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip': str})
    query = test.loc[~test.Zip.isin(train.Zip)].reset_index(drop=True)
    x, y, q = prepare(train, query, load_external())
    saved = pd.read_csv(ROOT/'artifacts/external_acs/foundation_realtor_test.csv', dtype={'Zip': str})
    if not saved[['Zip','State']].equals(query[['Zip','State']]):
        raise ValueError('Reference ZIP mismatch')
    checkpoint = hf_hub_download('Prior-Labs/TabPFN-v2-reg', 'tabpfn-v2-regressor.ckpt')
    plans = [
        ('auto_threads4', 4, {}),
        ('auto_threads1', 1, {}),
        ('auto_threads8', 8, {}),
        ('float16_threads4', 4, {'inference_precision': torch.float16}),
        ('float32_threads4', 4, {'inference_precision': torch.float32}),
        ('memory_saving_off', 4, {'memory_saving_mode': False}),
        ('memory_saving_on', 4, {'memory_saving_mode': True}),
    ]
    out = ROOT/'artifacts/serving/tabpfn_modes.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {'device': torch.cuda.get_device_name(0), 'scope': 'in-process model inference; prepared features; excludes HTTP', 'modes': {}}
    for name, threads, kwargs in plans:
        torch.set_num_threads(threads)
        model = None
        try:
            torch.cuda.reset_peak_memory_stats()
            model = TabPFNRegressor(n_estimators=4, model_path=checkpoint, device='cuda', n_preprocessing_jobs=2,
                                    random_state=42, fit_mode='fit_with_cache', **kwargs)
            _, fit = timed(lambda: model.fit(x, y))
            _, first = timed(lambda: model.predict(q.iloc[:1]))
            sizes = {}
            for n in [1, 32, 256, len(q)]:
                timings = []
                for _ in range(3):
                    pred, elapsed = timed(lambda: model.predict(q.iloc[:n]))
                    timings.append(elapsed)
                pred = np.clip(pred, 0, 100)
                sizes[str(n)] = {'seconds': timings, 'median_seconds': float(np.median(timings)),
                                 'rows_per_second': float(n/np.median(timings)),
                                 'max_abs_diff_vs_submission': float(np.max(np.abs(pred-saved.tabpfn.to_numpy()[:n])))}
            report['modes'][name] = {'fit_seconds': fit, 'first_prediction_seconds': first, 'batches': sizes,
                                     'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                                     'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30}
        except Exception as error:
            report['modes'][name] = {'error': f'{type(error).__name__}: {error}'}
        print('MODE', name, json.dumps(report['modes'][name]), flush=True)
        out.write_text(json.dumps(report, indent=2)+'\n')
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print('MODES_BENCHMARK', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
