"""Paired five-fold accuracy comparison: TabPFN v2 and v3, four estimators."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tabpfn import TabPFNRegressor

from housing_affordability.features import ROOT, Y, FeatureBuilder, load_external


def make_model(checkpoint):
    return TabPFNRegressor(n_estimators=4, model_path=checkpoint, device='cuda:0',
                           n_preprocessing_jobs=2, random_state=42, fit_mode='fit_with_cache')


def predict_test():
    history = pd.read_csv(ROOT / 'data/train.csv', dtype={'Zip': str})
    test = pd.read_csv(ROOT / 'data/test.csv', dtype={'Zip': str})
    sample = pd.read_csv(ROOT / 'data/sample_submission.csv', dtype={'Zip': str})
    keys = ['Zip', 'State']
    assert sample[keys].equals(test[keys]) and not test.duplicated(keys).any()
    latest = history.assign(_date=pd.to_datetime(history.Month)).sort_values('_date').drop_duplicates(keys, keep='last')
    values = latest.set_index(keys)[Y].reindex(pd.MultiIndex.from_frame(test[keys])).to_numpy(copy=True)
    unseen = np.isnan(values)
    query = test.loc[unseen].reset_index(drop=True)
    builder = FeatureBuilder(history, load_external(), query.iloc[:1])
    checkpoint = hf_hub_download('Prior-Labs/tabpfn_3', 'tabpfn-v3-regressor-v3_default.ckpt')
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    evaluated = json.loads((ROOT / 'docs/benchmarks/tabpfn_accuracy_cv.json').read_text())
    assert digest == evaluated['checkpoint_sha256']['v3']
    model = make_model(checkpoint).fit(builder.x, builder.y)
    q = builder.transform(query)
    with torch.inference_mode():
        prediction = np.concatenate([model.predict(q.iloc[i:i+128].copy(deep=True))
                                     for i in range(0, len(q), 128)])
    values[unseen] = np.clip(prediction, 0, 100)
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 100)).all()
    sample[Y] = values
    path = ROOT / 'submission_tabpfn_v3.csv'
    sample.to_csv(path, index=False)
    report = {'training_rows': len(history), 'training_zips': len(builder.y),
              'submission_rows': len(sample), 'seen_zips': int((~unseen).sum()),
              'new_zips': int(unseen.sum()), 'estimators': 4, 'random_state': 42,
              'package_version': importlib.metadata.version('tabpfn'),
              'checkpoint_sha256': digest, 'submission_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'pooled_local_rmse': evaluated['pooled_oof_rmse']['v3'],
              'external_data': 'Census ACS, Zillow and Realtor.com Economic Research'}
    path.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print('V3_SUBMISSION_READY', json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict-test', action='store_true')
    args = parser.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    if args.predict_test:
        predict_test()
        return
    history = pd.read_csv(ROOT / 'data/train.csv', dtype={'Zip': str})
    external = load_external()
    zips = np.array(sorted(history.Zip.unique()))
    np.random.default_rng(42).shuffle(zips)
    assignment = {z: i % 5 for i, z in enumerate(zips)}
    checkpoints = {
        'v2': hf_hub_download('Prior-Labs/TabPFN-v2-reg', 'tabpfn-v2-regressor.ckpt'),
        'v3': hf_hub_download('Prior-Labs/tabpfn_3', 'tabpfn-v3-regressor-v3_default.ckpt'),
    }
    report = {'package_version': importlib.metadata.version('tabpfn'), 'estimators': 4,
              'fold_seed': 42, 'model_seed': 42, 'folds': {},
              'fit_mode': 'fit_with_cache', 'device': 'cuda:0',
              'scope': 'paired unseen-ZIP June OOF; same external feature pipeline; native version-specific internal preprocessing',
              'checkpoint_sha256': {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                    for name, path in checkpoints.items()}}
    output = ROOT / 'tabpfn_accuracy_cv.json'
    rows = []
    for fold in range(5):
        held = {z for z, f in assignment.items() if f == fold}
        fit = history.loc[~history.Zip.isin(held)].reset_index(drop=True)
        query = history.loc[history.Zip.isin(held) & history.Month.eq('6/1/2026')].reset_index(drop=True)
        assert not set(fit.Zip) & set(query.Zip)
        builder = FeatureBuilder(fit, external, query.iloc[:1])
        q = builder.transform(query)
        result = query[['Zip', Y]].rename(columns={Y: 'truth'}).copy()
        result['fold'] = fold
        report['folds'][str(fold)] = {}
        for name, checkpoint in checkpoints.items():
            print('START_FOLD', fold, name, 'estimators=4', flush=True)
            started = time.perf_counter()
            model = make_model(checkpoint)
            model.fit(builder.x.copy(deep=True), builder.y.copy())
            with torch.inference_mode():
                prediction = np.concatenate([model.predict(q.iloc[i:i+128].copy(deep=True))
                                             for i in range(0, len(q), 128)])
            prediction = np.clip(prediction, 0, 100)
            assert np.isfinite(prediction).all()
            result[name] = prediction
            metric = {'rmse': float(np.sqrt(np.mean((query[Y].to_numpy() - prediction) ** 2))),
                      'rows': len(query), 'training_zips': len(builder.y),
                      'fit_predict_seconds': time.perf_counter() - started}
            report['folds'][str(fold)][name] = metric
            output.write_text(json.dumps(report, indent=2) + '\n')
            print('FOLD_RESULT', fold, name, json.dumps(metric), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(result)
        pd.concat(rows, ignore_index=True).to_csv(ROOT / 'tabpfn_accuracy_cv_oof.csv', index=False)
    oof = pd.concat(rows, ignore_index=True)
    assert oof.Zip.is_unique and set(oof.Zip) == set(history.Zip)
    report['oof_rows'] = len(oof)
    report['pooled_oof_rmse'] = {name: float(np.sqrt(np.mean((oof.truth - oof[name]) ** 2)))
                                 for name in checkpoints}
    report['mean_fold_rmse'] = {name: float(np.mean([report['folds'][str(f)][name]['rmse']
                                                    for f in range(5)])) for name in checkpoints}
    report['complete'] = True
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('PAIRED_ACCURACY_COMPLETE', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
