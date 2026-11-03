"""Remeasure the selected XGBoost and optimized TabPFN on the same T4 host."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmarks.async_http_t4 import benchmark
from benchmarks.tune_xgboost import TunedXGB
from housing_affordability.features import ROOT
from housing_affordability.server import HousingService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', type=Path, required=True)
    args = parser.parse_args()
    tuning = json.loads(args.selection.read_text())
    assert tuning['complete']
    selected = tuning['selected']
    torch.set_num_threads(4)
    history = pd.read_csv(ROOT / 'data/train.csv', dtype={'Zip': str})
    test = pd.read_csv(ROOT / 'data/test.csv', dtype={'Zip': str})
    unseen = test.loc[~test.Zip.isin(history.Zip)].reset_index(drop=True)
    rows = json.loads(unseen.head(256).to_json(orient='records'))
    service = HousingService(history, cache=True, optimize=True)
    tabpfn = service.model
    xgboost = TunedXGB(selected['config'], selected['refit_rounds']).fit(service.builder.x, service.builder.y)
    report = {'gpu': torch.cuda.get_device_name(), 'cpu_threads': 4,
              'cpu_info': next(line for line in Path('/proc/cpuinfo').read_text().splitlines()
                               if line.startswith('model name')),
              'scope': 'same host, localhost HTTP, unseen ZIP/request, three trials',
              'selection': selected, 'runs': {}}
    for name, model, backend, windows in [
        ('tabpfn', tabpfn, 'stdlib_keepalive', [5]),
        ('xgboost_tuned', xgboost, 'asgi', [.1, 20, 80]),
    ]:
        service.model = model
        for window in windows:
            report['runs'][f'{name}_{window}ms'] = benchmark(service, rows, backend, window)
    predictions = np.array([r['prediction'] for chunk in np.array_split(test, 8)
                            for r in service.predict(json.loads(chunk.to_json(orient='records')))])
    submission = test[['Zip', 'State']].copy()
    submission['AffordabilityPercentageTrue'] = predictions
    reference = pd.read_csv(args.selection.parent / 'submission_xgboost_tuned.csv', dtype={'Zip': str})
    assert reference[['Zip', 'State']].equals(submission[['Zip', 'State']])
    difference = float(np.max(np.abs(reference.AffordabilityPercentageTrue.to_numpy() - predictions)))
    report['max_abs_difference_vs_tuned_submission'] = difference
    if difference > 1e-5:
        raise ValueError(f'Refit predictions changed: {difference}')
    submission.to_csv(ROOT / 'submission_xgboost_tuned.csv', index=False)
    report['submission_rows'] = len(submission)
    path = ROOT / 'tuned_xgboost_comparison.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('TUNED_XGBOOST_COMPARISON', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
