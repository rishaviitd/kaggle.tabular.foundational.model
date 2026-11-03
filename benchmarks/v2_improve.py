"""Test focused v2-only feature/target improvements against the saved paired OOF."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tabpfn import TabPFNRegressor

from housing_affordability.features import ROOT, Y, FeatureBuilder, load_external


def recent_history(history):
    """Use both months, with 75% June and 25% May numeric aggregation."""
    out = history.copy()
    may = history.loc[history.Month.eq('5/1/2026')].set_index('Zip')
    june = history.loc[history.Month.eq('6/1/2026')].set_index('Zip')
    for col in history.select_dtypes(include=[np.number]):
        values = .25 * may[col] + .75 * june[col]
        out[col] = out.Zip.map(values)
    return out


def property_values():
    tables = []
    for kind in ['sfr', 'condo']:
        frame = pd.read_csv(ROOT / f'artifacts/external_acs/zillow_zhvi_{kind}.csv',
                            dtype={'RegionName': str}, low_memory=False)
        frame['Zip'] = frame.RegionName.str.zfill(5)
        table = frame.set_index('Zip')[['2025-06-30', '2026-06-30']].add_prefix(f'PROPERTY_{kind}_')
        table[f'PROPERTY_{kind}_growth'] = table.iloc[:, 1] / table.iloc[:, 0].replace(0, np.nan)
        tables.append(table)
    return pd.concat(tables, axis=1)


def representation(builder, query, name, properties):
    x = builder.x.copy(deep=True)
    q = builder.transform(query)
    if name == 'property_types':
        for frame, zips in [(x, builder.reference.Zip), (q, query.Zip)]:
            extra = properties.reindex(zips).reset_index(drop=True)
            for col in extra:
                frame[col] = extra[col].to_numpy()
            for kind in ['sfr', 'condo']:
                frame[f'PROPERTY_{kind}_income_ratio'] = frame[f'PROPERTY_{kind}_2026-06-30'] / frame.AnnualMedianHouseholdIncome.replace(0, np.nan)
    elif name == 'density_bins':
        for frame in [x, q]:
            for prefix in ['ACS_B19001_E', 'ACS_VALUE_DIST_B25075_E']:
                cols = sorted(c for c in frame if c.startswith(prefix) and not c.endswith('001'))
                bins = np.diff(np.concatenate([np.zeros((len(frame), 1)), frame[cols].to_numpy()], axis=1), axis=1)
                for j in range(len(cols)):
                    frame[f'DENSITY_{prefix}{j}'] = bins[:, j]
    y = builder.y.to_numpy()
    shift_x = np.zeros(len(x))
    shift_q = np.zeros(len(q))
    if name == 'mechanism_residual':
        proxy = 100 * x['INCOME_TAIL_0.25'].to_numpy()
        valid = np.isfinite(proxy) & np.isfinite(y)
        slope, intercept = np.polyfit(proxy[valid], y[valid], 1)
        shift_x = np.nan_to_num(slope * proxy + intercept, nan=float(y.mean()))
        shift_q = np.nan_to_num(slope * 100 * q['INCOME_TAIL_0.25'].to_numpy() + intercept, nan=float(y.mean()))
    return x, y - shift_x, q, shift_q


def fit_predict(history, query, name, external, properties, checkpoint, cache=True):
    fit = recent_history(history) if name == 'recent_weighted' else history
    builder = FeatureBuilder(fit, external, query.iloc[:1])
    x, y, q, shift = representation(builder, query, name, properties)
    model = TabPFNRegressor(n_estimators=4, model_path=checkpoint, device='cuda:0',
                            n_preprocessing_jobs=2, random_state=42,
                            fit_mode='fit_with_cache' if cache else 'fit_preprocessors')
    model.fit(x, y)
    with torch.inference_mode():
        p = np.concatenate([model.predict(q.iloc[i:i+128].copy(deep=True)) for i in range(0, len(q), 128)])
    result = np.clip(p + shift, 0, 100)
    del model, builder
    gc.collect()
    torch.cuda.empty_cache()
    return result


def predict_selected(args):
    selection = json.loads(args.selection.read_text())
    assert selection['complete']
    history = pd.read_csv(ROOT / 'data/train.csv', dtype={'Zip': str})
    test = pd.read_csv(ROOT / 'data/test.csv', dtype={'Zip': str})
    baseline = pd.read_csv(args.baseline, dtype={'Zip': str})
    keys = ['Zip', 'State']
    assert baseline[keys].equals(test[keys]) and not baseline.duplicated(keys).any()
    original_hash = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    assert original_hash == '4240f71287234337409daa8c027804048424e3eb3644853534d8713b18456ed8'
    seen = test.Zip.isin(history.Zip).to_numpy()
    selected = selection['selected'].copy()
    if selected['rmse'] >= selection['baseline_rmse']:
        selected = {'variant': 'baseline', 'weight': 0., 'rmse': selection['baseline_rmse']}
    values = baseline[Y].to_numpy(copy=True)
    if selected['weight'] > 0:
        checkpoint = hf_hub_download('Prior-Labs/TabPFN-v2-reg', 'tabpfn-v2-regressor.ckpt')
        assert hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() == selection['checkpoint_sha256']
        properties = property_values() if selected['variant'] == 'property_types' else None
        prediction = fit_predict(history, test.loc[~seen].reset_index(drop=True), selected['variant'],
                                  load_external(), properties, checkpoint, cache=False)
        values[~seen] = (1 - selected['weight']) * values[~seen] + selected['weight'] * prediction
    seen_prediction = pd.read_csv(args.seen, dtype={'Zip': str})
    assert not seen_prediction.duplicated(keys).any()
    values[seen] = seen_prediction.set_index(keys).prediction.reindex(
        pd.MultiIndex.from_frame(test.loc[seen, keys])).to_numpy()
    values = np.clip(values, 0, 100)
    assert np.isfinite(values).all()
    result = baseline[keys].copy()
    result[Y] = values
    path = ROOT / 'submission_v2_improved.csv'
    result.to_csv(path, index=False)
    report = {'training_rows': len(history), 'training_zips': int(history.Zip.nunique()),
              'rows': len(result), 'seen': int(seen.sum()), 'new': int((~seen).sum()),
              'selected': selected, 'model': 'TabPFN v2 only',
              'baseline_submission_sha256': original_hash,
              'submission_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'seen_strategy': 'v2-predicted monthly change, selected on May-to-June ZIP CV',
              'external_data': 'ACS, Zillow and Realtor.com Economic Research'}
    path.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print('V2_IMPROVED_SUBMISSION_READY', json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict-test', action='store_true')
    parser.add_argument('--selection', type=Path, default=ROOT / 'docs/benchmarks/v2_improve.json')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'artifacts/external_acs/submission_realtor.csv')
    parser.add_argument('--seen', type=Path, default=ROOT / 'docs/benchmarks/v2_seen_drift_test.csv')
    args = parser.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    if args.predict_test:
        predict_selected(args)
        return
    history = pd.read_csv(ROOT / 'data/train.csv', dtype={'Zip': str})
    baseline = pd.read_csv(ROOT / 'docs/benchmarks/tabpfn_accuracy_cv_oof.csv', dtype={'Zip': str}).set_index('Zip')
    checkpoint = hf_hub_download('Prior-Labs/TabPFN-v2-reg', 'tabpfn-v2-regressor.ckpt')
    external, properties = load_external(), property_values()
    candidates = ['recent_weighted', 'property_types', 'density_bins', 'mechanism_residual']
    report = {'checkpoint': 'TabPFN-v2-reg', 'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              'estimators': 4, 'model_seed': 42, 'fold_seed': 42, 'folds': {}, 'scope': 'reused five-fold tuning; June unseen-ZIP RMSE'}
    rows = []
    path = ROOT / 'v2_improve.json'
    for fold in range(5):
        held = set(baseline.index[baseline.fold.eq(fold)])
        fit = history.loc[~history.Zip.isin(held)].reset_index(drop=True)
        query = history.loc[history.Zip.isin(held) & history.Month.eq('6/1/2026')].reset_index(drop=True)
        result = query[['Zip', Y]].rename(columns={Y: 'truth'}).copy()
        result['fold'] = fold
        result['baseline'] = baseline.reindex(query.Zip).v2.to_numpy()
        np.testing.assert_array_equal(result.truth, baseline.reindex(query.Zip).truth)
        report['folds'][str(fold)] = {}
        for name in candidates:
            started = time.perf_counter()
            print('START_VARIANT', fold, name, flush=True)
            result[name] = fit_predict(fit, query, name, external, properties, checkpoint)
            score = float(np.sqrt(np.mean((result[name] - result.truth) ** 2)))
            report['folds'][str(fold)][name] = {'rmse': score, 'seconds': time.perf_counter() - started}
            path.write_text(json.dumps(report, indent=2) + '\n')
            print('VARIANT_SCORE', fold, name, score, flush=True)
        rows.append(result)
        pd.concat(rows, ignore_index=True).to_csv(ROOT / 'v2_improve_oof.csv', index=False)
    oof = pd.concat(rows, ignore_index=True)
    score = lambda p: float(np.sqrt(np.mean((p - oof.truth) ** 2)))
    report['baseline_rmse'] = score(oof.baseline)
    choices = []
    for name in candidates:
        for weight in [.25, .5, .75, 1.]:
            p = (1 - weight) * oof.baseline + weight * oof[name]
            choices.append({'variant': name, 'weight': weight, 'rmse': score(p),
                            'fold_wins': sum(np.mean((p[oof.fold.eq(f)] - oof.truth[oof.fold.eq(f)]) ** 2) <
                                             np.mean((oof.baseline[oof.fold.eq(f)] - oof.truth[oof.fold.eq(f)]) ** 2)
                                             for f in range(5))})
    report['choices'] = [{**c, 'fold_wins': int(c['fold_wins'])} for c in choices]
    report['selected'] = min(report['choices'], key=lambda c: c['rmse'])
    # Independent ZIP groups for the May-to-June drift estimate, then transfer
    # that empirical temporal correction to the next month only as a hypothesis.
    may = history.loc[history.Month.eq('5/1/2026')].set_index('Zip')[Y]
    june = history.loc[history.Month.eq('6/1/2026')].set_index('Zip')[Y]
    delta = june - may
    drift_pred = pd.Series(index=delta.index, dtype=float)
    for fold in range(5):
        held = set(baseline.index[baseline.fold.eq(fold)])
        drift_pred.loc[drift_pred.index.isin(held)] = delta.loc[~delta.index.isin(held)].mean()
    report['seen_drift'] = {'constant': float(delta.mean()),
                            'persistence_rmse': float(np.sqrt(np.mean(delta ** 2))),
                            'crossfit_drift_rmse': float(np.sqrt(np.mean((delta - drift_pred) ** 2)))}
    selected = report['selected']
    zips = np.array(sorted(history.Zip.unique()))
    np.random.default_rng(137).shuffle(zips)
    held = set(zips[::5])
    fit = history.loc[~history.Zip.isin(held)].reset_index(drop=True)
    query = history.loc[history.Zip.isin(held) & history.Month.eq('6/1/2026')].reset_index(drop=True)
    control = fit_predict(fit, query, 'baseline', external, properties, checkpoint)
    alternative = fit_predict(fit, query, selected['variant'], external, properties, checkpoint)
    prediction = (1 - selected['weight']) * control + selected['weight'] * alternative
    report['confirmation_seed137'] = {'baseline_rmse': float(np.sqrt(np.mean((control - query[Y]) ** 2))),
                                       'candidate_rmse': float(np.sqrt(np.mean((prediction - query[Y]) ** 2)))}
    pd.DataFrame({'Zip': query.Zip, 'truth': query[Y], 'baseline': control, 'candidate': prediction}).to_csv(
        ROOT / 'v2_improve_confirmation.csv', index=False)
    report['complete'] = True
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('V2_IMPROVE_COMPLETE', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
