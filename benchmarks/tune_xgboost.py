"""Tune CPU XGBoost without touching either published comparison holdout."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from benchmarks.async_http_t4 import benchmark
from benchmarks.compare_xgboost import XGBModel
from housing_affordability.features import ROOT, Y, FeatureBuilder, load_external
from housing_affordability.server import HousingService


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - p) ** 2)))


class TunedXGB:
    def __init__(self, config, rounds=2500, early_stop=False):
        self.config = config.copy()
        params = {k: v for k, v in config.items() if k not in ('encoding', 'target')}
        params.update(n_estimators=rounds, tree_method='hist', device='cpu',
                      n_jobs=4, random_state=42, enable_categorical=True)
        if early_stop:
            params.update(early_stopping_rounds=120, eval_metric=self.metric)
        self.model = xgb.XGBRegressor(**params)

    def encode(self, frame):
        if self.config['encoding'] == 'ordinal':
            return XGBModel.encode(frame)
        # XGBoost handles numeric dtypes internally; keep categorical metadata
        # without converting every numeric column separately for each request.
        return frame.replace([np.inf, -np.inf], np.nan)

    def forward(self, y):
        p = np.clip(np.asarray(y) / 100, .0001, .9999)
        if self.config['target'] == 'logit':
            return np.log(p / (1 - p))
        if self.config['target'] == 'arcsine':
            return np.arcsin(np.sqrt(p))
        return np.asarray(y)

    def inverse(self, y):
        if self.config['target'] == 'logit':
            return 100 / (1 + np.exp(-np.clip(y, -30, 30)))
        if self.config['target'] == 'arcsine':
            return 100 * np.sin(np.clip(y, 0, np.pi / 2)) ** 2
        return np.clip(y, 0, 100)

    def metric(self, y, p):
        return rmse(self.inverse(y), self.inverse(p))

    def fit(self, x, y, validation=None):
        kwargs = {}
        if validation is not None:
            q, target = validation
            kwargs = {'eval_set': [(self.encode(q), self.forward(target))], 'verbose': False}
        self.model.fit(self.encode(x), self.forward(y), **kwargs)
        return self

    def predict(self, x):
        return self.inverse(self.model.predict(self.encode(x)))


def split_zips(history, seed):
    zips = np.array(sorted(history.Zip.unique()))
    np.random.default_rng(seed).shuffle(zips)
    return set(zips[::5])


def prepare_split(history, held, external):
    fit = history.loc[~history.Zip.isin(held)].reset_index(drop=True)
    query = history.loc[history.Zip.isin(held) & (history.Month == '6/1/2026')].reset_index(drop=True)
    assert not set(fit.Zip) & set(query.Zip)
    builder = FeatureBuilder(fit, external, query.iloc[:1])
    return builder, builder.transform(query), query


def main():
    torch.set_num_threads(4)
    history = pd.read_csv(ROOT / 'data/train.csv', dtype={'Zip': str})
    test = pd.read_csv(ROOT / 'data/test.csv', dtype={'Zip': str})
    external = load_external()
    outer = {seed: split_zips(history, seed) for seed in [42, 137]}
    reserved = outer[42] | outer[137]
    tuning_history = history.loc[~history.Zip.isin(reserved)].reset_index(drop=True)
    tuning = [prepare_split(tuning_history, split_zips(tuning_history, seed), external)
              for seed in [2026, 2027]]
    base = dict(max_depth=5, learning_rate=.03, subsample=.8,
                colsample_bytree=.8, reg_lambda=5, min_child_weight=1, reg_alpha=0)
    configs = [dict(base, encoding=e, target=t) for e in ['ordinal', 'native']
               for t in ['raw', 'arcsine', 'logit']]
    rng = np.random.default_rng(2026)
    for _ in range(24):
        configs.append(dict(max_depth=int(rng.choice([2, 3, 4, 5, 6, 8])),
                            learning_rate=float(rng.choice([.03, .06, .1])),
                            subsample=float(rng.choice([.8, 1.])),
                            colsample_bytree=float(rng.choice([.7, .9, 1.])),
                            reg_lambda=float(rng.choice([1, 5, 20])),
                            min_child_weight=int(rng.choice([1, 5, 15])),
                            reg_alpha=float(rng.choice([0, .5, 2])),
                            encoding=str(rng.choice(['ordinal', 'native'])),
                            target=str(rng.choice(['raw', 'raw', 'arcsine', 'logit']))))
    out = ROOT / 'artifacts/serving'
    out.mkdir(parents=True, exist_ok=True)
    report = {'selection': 'two inner ZIP splits, excluding union of seed42/137 outer ZIPs',
              'tuning_pool_zips': int(tuning_history.Zip.nunique()),
              'xgboost_version': xgb.__version__, 'trials': []}
    for i, config in enumerate(configs):
        scores, rounds = [], []
        started = time.perf_counter()
        for builder, q, query in tuning:
            model = TunedXGB(config, early_stop=True).fit(builder.x, builder.y, (q, query[Y]))
            scores.append(rmse(query[Y], model.predict(q)))
            rounds.append(model.model.best_iteration + 1)
        trial = {'id': i, 'config': config, 'rmse': scores, 'mean_rmse': float(np.mean(scores)),
                 'rounds': rounds, 'seconds': time.perf_counter() - started}
        report['trials'].append(trial)
        (out / 'xgboost_tuning.json').write_text(json.dumps(report, indent=2) + '\n')
        print('TUNING_TRIAL', json.dumps(trial), flush=True)
    best = min(report['trials'], key=lambda t: t['mean_rmse'])
    rounds = int(np.median(best['rounds']))
    report['selected'] = dict(best, refit_rounds=rounds)
    report['heldout'] = {}
    for seed, held in outer.items():
        builder, q, query = prepare_split(history, held, external)
        model = TunedXGB(best['config'], rounds).fit(builder.x, builder.y)
        prediction = model.predict(q)
        baseline = XGBModel().fit(builder.x, builder.y).predict(q)
        report['heldout'][str(seed)] = {'rows': len(query), 'tuned_rmse': rmse(query[Y], prediction),
                                      'baseline_rmse': rmse(query[Y], baseline)}
        pd.DataFrame({'Zip': query.Zip, 'truth': query[Y], 'xgboost': prediction,
                      'baseline': baseline}).to_csv(out / f'xgboost_seed{seed}_heldout.csv', index=False)
        print('HELDOUT', seed, report['heldout'][str(seed)], flush=True)
    unseen = test.loc[~test.Zip.isin(history.Zip)].reset_index(drop=True)
    builder = FeatureBuilder(history, external, unseen.iloc[:1])
    model = TunedXGB(best['config'], rounds).fit(builder.x, builder.y)
    model.model.save_model(out / 'xgboost_tuned.ubj')
    service = HousingService.__new__(HousingService)
    service.builder, service.model = builder, model
    latest = history.assign(_date=pd.to_datetime(history.Month)).sort_values('_date').drop_duplicates(['Zip', 'State'], keep='last')
    service.seen = latest.set_index(['Zip', 'State'])[Y]
    values = service.seen.reindex(pd.MultiIndex.from_frame(test[['Zip', 'State']])).to_numpy(copy=True)
    mask = np.isnan(values)
    values[mask] = model.predict(builder.transform(test.loc[mask].reset_index(drop=True)))
    sample = pd.read_csv(ROOT / 'data/sample_submission.csv', dtype={'Zip': str})
    assert sample[['Zip', 'State']].equals(test[['Zip', 'State']])
    sample[Y] = values
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 100)).all()
    path = ROOT / 'submission_xgboost_tuned.csv'
    sample.to_csv(path, index=False)
    report['submission'] = {'file': path.name, 'rows': len(sample), 'seen': int((~mask).sum()),
                            'new': int(mask.sum()), 'training_rows': len(history),
                            'training_zips': int(history.Zip.nunique()),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    rows = json.loads(unseen.head(256).to_json(orient='records'))
    report['cpu_info'] = next(line for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
    report['gpu'] = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    report['http'] = {}
    for window in [.1, 20, 80]:
        report['http'][str(window)] = benchmark(service, rows, 'asgi', window)
    report['complete'] = True
    (out / 'xgboost_tuning.json').write_text(json.dumps(report, indent=2) + '\n')
    (ROOT / 'xgboost_tuning.json').write_text(json.dumps(report, indent=2) + '\n')
    print('XGBOOST_TUNING_COMPLETE', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
