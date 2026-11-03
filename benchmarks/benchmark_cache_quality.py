"""Check cached and uncached accuracy on a ZIP-held-out fold."""
import gc
import json

import numpy as np
import pandas as pd
import torch

from housing_affordability.features import ROOT, Y, load_external, prepare
from housing_affordability.model import HousingModel


def main():
    torch.set_num_threads(4)
    history = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
    zips = np.array(sorted(history.Zip.unique()))
    np.random.default_rng(42).shuffle(zips)
    held = set(zips[::5])
    training = history.loc[~history.Zip.isin(held)].reset_index(drop=True)
    query = history.loc[history.Zip.isin(held) & (history.Month == '6/1/2026')].reset_index(drop=True)
    x, y, q = prepare(training, query, load_external())
    predictions = {}
    for cache in [False, True]:
        model = HousingModel(cache=cache).fit(x, y)
        predictions['cache' if cache else 'baseline'] = model.predict(q)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    actual = query[Y].to_numpy()
    diff = predictions['cache'] - predictions['baseline']
    result = {'heldout_zip_rows': len(query), 'baseline_new_zip_rmse': float(np.mean((actual-predictions['baseline'])**2)**.5),
              'cached_new_zip_rmse': float(np.mean((actual-predictions['cache'])**2)**.5),
              'mean_absolute_prediction_difference': float(np.mean(np.abs(diff))),
              'max_absolute_prediction_difference': float(np.max(np.abs(diff)))}
    dest = ROOT/'artifacts/serving/cache_quality.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2)+'\n')
    print('CACHE_QUALITY', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
