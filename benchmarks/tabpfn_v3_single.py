"""Compare single-estimator v3 and v2 on the existing held-out ZIP splits."""
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tabpfn import TabPFNRegressor

from benchmarks.async_http_t4 import benchmark
from housing_affordability.features import ROOT, Y, FeatureBuilder, load_external
from housing_affordability.server import HousingService


class SingleModel:
    def __init__(self, checkpoint):
        self.model = TabPFNRegressor(n_estimators=1, model_path=checkpoint,
                                    device='cuda:0', n_preprocessing_jobs=2,
                                    random_state=42, fit_mode='fit_with_cache')

    def fit(self, x, y):
        self.model.fit(x, y)
        return self

    def predict(self, x):
        with torch.inference_mode():
            parts = [self.model.predict(x.iloc[i:i+128]) for i in range(0, len(x), 128)]
        return np.clip(np.concatenate(parts), 0, 100)


def main():
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    history = pd.read_csv(ROOT / 'data/train.csv', dtype={'Zip': str})
    test = pd.read_csv(ROOT / 'data/test.csv', dtype={'Zip': str})
    external = load_external()
    props = torch.cuda.get_device_properties(0)
    import psutil
    report = {'gpu': props.name, 'gpu_vram_bytes': props.total_memory,
              'cpu': next(line for line in Path('/proc/cpuinfo').read_text().splitlines()
                          if line.startswith('model name')),
              'logical_cpus': psutil.cpu_count(), 'cpu_threads': 4,
              'system_ram_bytes': psutil.virtual_memory().total,
              'estimators': 1, 'fit_mode': 'fit_with_cache', 'heldout': {},
              'scope': 'same features and ZIP-held-out June targets; no v2-specific attention patches'}
    checkpoints = {
        'v3_single': hf_hub_download('Prior-Labs/tabpfn_3', 'tabpfn-v3-regressor-v3_default.ckpt'),
        'v2_single': hf_hub_download('Prior-Labs/TabPFN-v2-reg', 'tabpfn-v2-regressor.ckpt'),
    }
    report['checkpoint_bytes'] = {name: Path(path).stat().st_size for name, path in checkpoints.items()}
    dest = ROOT / 'tabpfn_v3_single.json'
    for seed in [42, 137]:
        zips = np.array(sorted(history.Zip.unique()))
        np.random.default_rng(seed).shuffle(zips)
        held = set(zips[::5])
        fit = history.loc[~history.Zip.isin(held)].reset_index(drop=True)
        query = history.loc[history.Zip.isin(held) & history.Month.eq('6/1/2026')].reset_index(drop=True)
        builder = FeatureBuilder(fit, external, query.iloc[:1])
        q = builder.transform(query)
        report['heldout'][str(seed)] = {}
        for name, checkpoint in checkpoints.items():
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            print('START_MODEL', seed, name, len(builder.y), len(query), flush=True)
            model = SingleModel(checkpoint).fit(builder.x, builder.y)
            pred = model.predict(q)
            assert np.isfinite(pred).all()
            item = {'rows': len(query), 'training_zips': len(builder.y),
                    'rmse': float(np.sqrt(np.mean((query[Y].to_numpy() - pred) ** 2))),
                    'fit_predict_seconds': time.perf_counter() - started,
                    'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved()}
            report['heldout'][str(seed)][name] = item
            pd.DataFrame({'Zip': query.Zip, 'truth': query[Y], 'prediction': pred}).to_csv(
                ROOT / f'{name}_seed{seed}_heldout.csv', index=False)
            dest.write_text(json.dumps(report, indent=2) + '\n')
            print('HELDOUT_RESULT', seed, name, json.dumps(item), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    report['mean_local_rmse'] = {name: float(np.mean([report['heldout'][str(s)][name]['rmse']
                                                    for s in [42, 137]])) for name in checkpoints}
    unseen = test.loc[~test.Zip.isin(history.Zip)].reset_index(drop=True)
    builder = FeatureBuilder(history, external, unseen.iloc[:1])
    service = HousingService.__new__(HousingService)
    service.builder = builder
    latest = history.assign(_date=pd.to_datetime(history.Month)).sort_values('_date').drop_duplicates(['Zip', 'State'], keep='last')
    service.seen = latest.set_index(['Zip', 'State'])[Y]
    torch.cuda.reset_peak_memory_stats()
    print('FULL_FIT_V3', len(builder.y), flush=True)
    service.model = SingleModel(checkpoints['v3_single']).fit(builder.x, builder.y)
    report['full_fit_training_rows'] = len(history)
    report['full_fit_training_zips'] = len(builder.y)
    prediction = service.model.predict(builder.transform(unseen))
    unseen[['Zip', 'State']].assign(prediction=prediction).to_csv(ROOT / 'v3_single_test_predictions.csv', index=False)
    rows = json.loads(unseen.head(256).to_json(orient='records'))
    report['http'] = benchmark(service, rows, 'stdlib_keepalive', 5)
    report['full_fit_peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['full_fit_peak_gpu_reserved_bytes'] = torch.cuda.max_memory_reserved()
    report['complete'] = True
    dest.write_text(json.dumps(report, indent=2) + '\n')
    print('TABPFN_V3_SINGLE_COMPLETE', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
