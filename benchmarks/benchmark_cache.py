"""Measure end-to-end KV-cached requests and verify prediction parity."""
import json
import time

import numpy as np
import pandas as pd
import torch

from housing_affordability.features import ROOT
from housing_affordability.server import HousingService


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def main():
    torch.set_num_threads(4)
    history = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
    test = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip': str})
    unseen = test.loc[~test.Zip.isin(history.Zip)].reset_index(drop=True)
    rows = json.loads(unseen.to_json(orient='records'))
    expected = pd.read_csv(ROOT/'artifacts/external_acs/foundation_realtor_test.csv', dtype={'Zip': str})
    if not expected[['Zip','State']].equals(unseen[['Zip','State']]):
        raise ValueError('Reference ZIP mismatch')
    torch.cuda.reset_peak_memory_stats()
    service, startup = timed(lambda: HousingService(history, cache=True))
    first, first_time = timed(lambda: service.predict(rows[:1]))
    batches = {}
    for n in [1, 32, 256, len(rows)]:
        samples = []
        for _ in range(3):
            last, elapsed = timed(lambda: service.predict(rows[:n]))
            samples.append(elapsed)
        batches[str(n)] = {'seconds': samples, 'median_seconds': float(np.median(samples)),
                           'rows_per_second': float(n/np.median(samples))}
    result = {'device': torch.cuda.get_device_name(0), 'startup_including_data_and_cache_seconds': startup,
              'first_unseen_request_seconds': first_time, 'batches': batches,
              'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
              'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
              'max_abs_difference_vs_submission': float(np.max(np.abs(np.array([r['prediction'] for r in last]) - expected.tabpfn))),
              'scope': 'in-process HousingService.predict, excludes HTTP and network transport'}
    dest = ROOT/'artifacts/serving/cache_benchmark.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2)+'\n')
    print('CACHE_BENCHMARK', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
