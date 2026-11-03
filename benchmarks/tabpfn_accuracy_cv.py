"""Paired five-fold accuracy comparison: TabPFN v2 and v3, four estimators."""
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


def main():
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
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
            model = TabPFNRegressor(n_estimators=4, model_path=checkpoint, device='cuda:0',
                                    n_preprocessing_jobs=2, random_state=42, fit_mode='fit_with_cache')
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
