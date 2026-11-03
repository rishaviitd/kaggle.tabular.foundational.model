"""Assemble and validate the Zillow price-tier TabPFN submission."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from evaluate import DATA, ROOT, TARGET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--realtor', action='store_true')
    args = parser.parse_args()
    train = pd.read_csv(ROOT / 'data/train.csv', dtype={'Zip': str})
    test = pd.read_csv(ROOT / 'data/test.csv', dtype={'Zip': str})
    sample = pd.read_csv(ROOT / 'data/sample_submission.csv', dtype={'Zip': str})
    path = DATA / ('foundation_realtor_test.csv' if args.realtor else 'foundation_zillow_tiers_test.csv')
    pred = pd.read_csv(path, dtype={'Zip': str})
    metadata = json.loads(path.with_suffix('.json').read_text())
    if metadata['training_rows'] != 13660 or metadata['training_zips'] != 6830:
        raise ValueError('Predictions were not fitted on the full training data')
    keys = ['Zip', 'State']
    if not sample[keys].equals(test[keys]) or test.duplicated(keys).any():
        raise ValueError('Test/sample key mismatch')
    history = train.assign(_date=pd.to_datetime(train.Month)).sort_values('_date').drop_duplicates(keys, keep='last')
    prior = history.set_index(keys)[TARGET].reindex(pd.MultiIndex.from_frame(test[keys]))
    seen = prior.notna().to_numpy()
    if not pred[keys].equals(test.loc[~seen, keys].reset_index(drop=True)):
        raise ValueError('Unseen ZIP prediction key mismatch')
    values = prior.to_numpy(copy=True)
    values[~seen] = pred.tabpfn.to_numpy()
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 100)).all():
        raise ValueError('Invalid predictions')
    output = sample[keys].copy()
    output[TARGET] = values
    dest = DATA / ('submission_realtor.csv' if args.realtor else 'submission_zillow_tiers.csv')
    output.to_csv(dest, index=False)
    saved = pd.read_csv(dest, dtype={'Zip': str})
    if list(saved.columns) != list(sample.columns) or not saved[keys].equals(sample[keys]):
        raise ValueError('Saved file format mismatch')
    record = {'rows': len(saved), 'seen': int(seen.sum()), 'new': int((~seen).sum()),
              'training': metadata, 'submission_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
              'predictions_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    dest.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
