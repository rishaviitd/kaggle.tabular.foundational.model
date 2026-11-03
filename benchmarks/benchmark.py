"""Measure actual TabPFN T4 latency and held-out group importance."""
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from housing_affordability.features import ROOT, Y, load_external, prepare
from housing_affordability.interpret import grouped_permutation_importance
from housing_affordability.model import HousingModel


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires a CUDA GPU')
    torch.set_num_threads(4)
    out = ROOT / 'artifacts/serving'
    out.mkdir(parents=True, exist_ok=True)
    report = {'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'model': 'TabPFN v2 regressor, 4 estimators, seed 42', 'modes': {}}
    history = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
    test = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip': str})
    query = test.loc[~test.Zip.isin(history.Zip)].reset_index(drop=True)
    start = time.perf_counter()
    external = load_external()
    report['external_load_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    x, y, q = prepare(history, query, external)
    report['full_feature_preparation_seconds'] = time.perf_counter() - start
    report['training_rows'] = len(history)
    report['training_zips'] = len(x)
    report['query_rows'] = len(q)
    report['features'] = len(x.columns)
    reference = None
    for cache in [False, True]:
        name = 'kv_cache' if cache else 'preprocessors_only'
        try:
            torch.cuda.reset_peak_memory_stats()
            model, init = timed(lambda: HousingModel(cache=cache))
            _, fit = timed(lambda: model.fit(x, y))
            _, cold = timed(lambda: model.predict(q.iloc[:1]))
            sizes = {}
            for n in [1, 32, 256, len(q)]:
                times = []
                for _ in range(3):
                    _, elapsed = timed(lambda: model.predict(q.iloc[:n]))
                    times.append(elapsed)
                sizes[str(n)] = {'seconds': times, 'median_seconds': float(np.median(times)),
                                 'rows_per_second': float(n / np.median(times))}
            predictions, _ = timed(lambda: model.predict(q))
            if reference is None:
                reference = predictions
                saved = pd.read_csv(ROOT/'artifacts/external_acs/foundation_realtor_test.csv', dtype={'Zip': str})
                if not saved[['Zip','State']].equals(query[['Zip','State']]):
                    raise ValueError('Reference predictions do not match test ZIPs')
                reference_diff = float(np.max(np.abs(predictions - saved.tabpfn.to_numpy())))
            else:
                reference_diff = float(np.max(np.abs(predictions - reference)))
            report['modes'][name] = {'model_load_seconds': init, 'fit_seconds': fit,
                                     'first_prediction_seconds': cold, 'batches': sizes,
                                     'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                                     'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
                                     'max_abs_difference_vs_reference': reference_diff}
            print(name, json.dumps(report['modes'][name]), flush=True)
            del model
        except Exception as error:
            report['modes'][name] = {'error': f'{type(error).__name__}: {error}'}
            print(name, report['modes'][name], flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        (out/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
    # ZIPs in this fold were not seen when fitting the interpretation model.
    z = np.array(sorted(history.Zip.unique()))
    np.random.default_rng(42).shuffle(z)
    held = set(z[::5])
    train = history.loc[~history.Zip.isin(held)].reset_index(drop=True)
    held_june = history.loc[history.Zip.isin(held) & (history.Month == '6/1/2026')].reset_index(drop=True)
    ix, iy, iq = prepare(train, held_june, external)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(iq), size=min(256, len(iq)), replace=False)
    ix_query = iq.iloc[indices].reset_index(drop=True)
    truth = held_june[Y].iloc[indices].to_numpy()
    model = HousingModel(cache=False).fit(ix, iy)
    scores = grouped_permutation_importance(model.predict, ix_query, truth, repeats=2)
    scores.to_csv(out/'grouped_permutation.csv', index=False)
    print('HELD_OUT_IMPORTANCE', scores.to_json(orient='records'), flush=True)
    print('BENCHMARK', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
