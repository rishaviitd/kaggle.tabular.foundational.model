"""ZIP-cross-validated v2 prediction of May-to-June changes for seen ZIPs."""
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tabpfn import TabPFNRegressor

from housing_affordability.features import ROOT, Y


def transition_features(previous, current, prices, old_date, new_date):
    current = current.set_index('Zip').reindex(previous.Zip).reset_index()
    previous = previous.reset_index(drop=True)
    x = current.drop(columns=[Y, 'Zip', 'Month', 'YearMonth'], errors='ignore').copy()
    x['PreviousTarget'] = previous[Y].to_numpy()
    numeric = [c for c in previous.select_dtypes(include=[np.number]) if c not in [Y, 'YearMonth']]
    for col in numeric:
        x[f'Previous_{col}'] = previous[col].to_numpy()
        delta = current[col].to_numpy() - previous[col].to_numpy()
        x[f'Delta_{col}'] = delta
        x[f'RelativeDelta_{col}'] = delta / previous[col].replace(0, np.nan).to_numpy()
    zips = previous.Zip.str.zfill(5)
    x['ZipNumeric'] = pd.to_numeric(zips, errors='coerce')
    for n in [1, 2, 3]:
        x[f'ZipPrefix{n}'] = zips.str[:n]
    before = prices.reindex(previous.Zip)[old_date].to_numpy()
    after = prices.reindex(previous.Zip)[new_date].to_numpy()
    x['PreviousHomeValue'] = before
    x['CurrentHomeValue'] = after
    x['HomeValueGrowth'] = after / np.where(before == 0, np.nan, before) - 1
    x['PreviousAffordableCountProxy'] = previous[Y].to_numpy() * previous.SampleSize.to_numpy() / 100
    return x.replace([np.inf, -np.inf], np.nan)


def encode(x, q):
    x, q = x.copy(), q.copy()
    for c in x.select_dtypes(include=['object', 'string']):
        levels = pd.Index(x[c].fillna('__MISSING__').astype(str).unique())
        x[c] = pd.Categorical(x[c].fillna('__MISSING__').astype(str), categories=levels)
        q[c] = pd.Categorical(q[c].fillna('__MISSING__').astype(str), categories=levels)
    return x, q


def main():
    torch.set_num_threads(4)
    h = pd.read_csv(ROOT / 'data/train.csv', dtype={'Zip': str})
    test = pd.read_csv(ROOT / 'data/test.csv', dtype={'Zip': str})
    old = h.loc[h.Month.eq('5/1/2026')].sort_values('Zip').reset_index(drop=True)
    now = h.loc[h.Month.eq('6/1/2026')].sort_values('Zip').reset_index(drop=True)
    assert old.Zip.equals(now.Zip)
    prices = pd.read_csv(ROOT / 'artifacts/external_acs/zillow_zhvi.csv', dtype={'RegionName': str},
                         usecols=['RegionName', '2026-05-31', '2026-06-30', '2026-07-31'])
    prices.RegionName = prices.RegionName.str.zfill(5)
    prices = prices.set_index('RegionName')
    x = transition_features(old, now, prices, '2026-05-31', '2026-06-30')
    target = now[Y].to_numpy() - old[Y].to_numpy()
    z = old.Zip.to_numpy(copy=True)
    np.random.default_rng(42).shuffle(z)
    assignment = {zip_code: i % 5 for i, zip_code in enumerate(z)}
    folds = old.Zip.map(assignment).to_numpy()
    prediction = np.empty(len(old))
    constant = np.empty(len(old))
    ckpt = hf_hub_download('Prior-Labs/TabPFN-v2-reg', 'tabpfn-v2-regressor.ckpt')
    report = {'model': 'TabPFN v2', 'estimators': 4, 'fold_seed': 42,
              'scope': 'May-to-June ZIP cross-validation; temporal transfer to July is unverified', 'folds': {}}
    for fold in range(5):
        fit, val = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        xx, q = encode(x.iloc[fit], x.iloc[val])
        model = TabPFNRegressor(n_estimators=4, model_path=ckpt, device='cuda:0',
                                random_state=42, n_preprocessing_jobs=2, fit_mode='fit_with_cache')
        model.fit(xx, target[fit])
        with torch.inference_mode():
            prediction[val] = np.concatenate([model.predict(q.iloc[i:i+128]) for i in range(0, len(q), 128)])
        constant[val] = target[fit].mean()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print('SEEN_FOLD_COMPLETE', fold, flush=True)
    truth = now[Y].to_numpy()
    prior = old[Y].to_numpy()
    score = lambda delta: float(np.sqrt(np.mean((np.clip(prior + delta, 0, 100) - truth) ** 2)))
    report['persistence_rmse'] = score(0)
    report['constant_drift_rmse'] = score(constant)
    report['choices'] = [{'weight': weight, 'rmse': score((1 - weight) * constant + weight * prediction)}
                          for weight in [0., .25, .5, .75, 1.]]
    report['selected'] = min(report['choices'], key=lambda c: c['rmse'])
    pd.DataFrame({'Zip': old.Zip, 'fold': folds, 'previous': prior, 'truth': truth,
                  'model_delta': prediction, 'constant_delta': constant}).to_csv(ROOT / 'v2_seen_drift_oof.csv', index=False)
    selected = report['selected']['weight']
    seen = test.loc[test.Zip.isin(now.Zip)].reset_index(drop=True)
    prev = now.set_index('Zip').reindex(seen.Zip).reset_index()
    q = transition_features(prev, seen, prices, '2026-06-30', '2026-07-31')
    xx, q = encode(x, q)
    if selected > 0:
        model = TabPFNRegressor(n_estimators=4, model_path=ckpt, device='cuda:0',
                                random_state=42, n_preprocessing_jobs=2, fit_mode='fit_with_cache')
        model.fit(xx, target)
        with torch.inference_mode():
            delta = np.concatenate([model.predict(q.iloc[i:i+128]) for i in range(0, len(q), 128)])
        adjustment = selected * delta + (1 - selected) * target.mean()
    else:
        adjustment = np.full(len(seen), target.mean())
    seen[['Zip', 'State']].assign(prediction=np.clip(prev[Y].to_numpy() + adjustment, 0, 100)).to_csv(
        ROOT / 'v2_seen_drift_test.csv', index=False)
    report['complete'] = True
    (ROOT / 'v2_seen_drift.json').write_text(json.dumps(report, indent=2) + '\n')
    print('V2_SEEN_DRIFT_COMPLETE', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
