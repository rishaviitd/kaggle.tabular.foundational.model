"""Create a weighted average of validated full-data model predictions."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from evaluate import DATA, ROOT, TARGET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='JSON mapping from model names to weights')
    args = parser.parse_args()
    weights = json.loads(args.weights)
    allowed = {'base', 'pfn16', 'pfn_arcsine', 'tabicl'}
    if not weights or set(weights) - allowed or any(v < 0 for v in weights.values()) or not np.isclose(sum(weights.values()), 1):
        raise ValueError('Invalid model weights')
    train = pd.read_csv(ROOT / 'data/train.csv', dtype={'Zip': str})
    test = pd.read_csv(ROOT / 'data/test.csv', dtype={'Zip': str})
    sample = pd.read_csv(ROOT / 'data/sample_submission.csv', dtype={'Zip': str})
    keys = ['Zip', 'State']
    if not sample[keys].equals(test[keys]) or test.duplicated(keys).any():
        raise ValueError('Test/sample key mismatch')
    history = train.assign(_date=pd.to_datetime(train.Month)).sort_values('_date').drop_duplicates(keys, keep='last')
    prior = history.set_index(keys)[TARGET].reindex(pd.MultiIndex.from_frame(test[keys]))
    seen = prior.notna().to_numpy()
    expected = test.loc[~seen, keys].reset_index(drop=True)
    components = {}
    for filename, names in [('foundation_realtor_test.csv', ['base']),
                            ('foundation_realtor_tuning_test.csv', ['pfn16', 'pfn_arcsine', 'tabicl'])]:
        required = [name for name in names if weights.get(name, 0) > 0]
        if not required:
            continue
        path = DATA / filename
        meta = json.loads(path.with_suffix('.json').read_text())
        if meta['training_rows'] != 13660 or meta['training_zips'] != 6830:
            raise ValueError('Predictions do not use all labeled rows')
        pred = pd.read_csv(path, dtype={'Zip': str})
        if not pred[keys].equals(expected):
            raise ValueError('Prediction keys do not match test ZIPs')
        for name in required:
            components[name] = pred['tabpfn' if name == 'base' else name].to_numpy()
    values = prior.to_numpy(copy=True)
    values[~seen] = sum(weights[n]*components[n] for n in components)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 100)).all():
        raise ValueError('Invalid predictions')
    output = sample[keys].copy()
    output[TARGET] = values
    dest = DATA / 'submission_realtor_tuned.csv'
    output.to_csv(dest, index=False)
    saved = pd.read_csv(dest, dtype={'Zip': str})
    if list(saved.columns) != list(sample.columns) or not saved[keys].equals(sample[keys]):
        raise ValueError('Saved file format mismatch')
    record = {'rows': len(saved), 'seen': int(seen.sum()), 'new': int((~seen).sum()),
              'weights': weights, 'submission_sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
    dest.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
