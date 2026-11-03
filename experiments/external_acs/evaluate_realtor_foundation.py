"""Evaluate Realtor.com Economic Research listing data on held-out ZIPs."""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sklearn.model_selection import GroupKFold
from tabpfn import TabPFNRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from housing_affordability.aggregated_features import feats, enc, Y
from evaluate import DATA, read_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    train = pd.read_csv(ROOT / 'data/train.csv', dtype={'Zip': str})
    income = read_table('b19001')
    income = income.div(income.B19001_E001.replace(0, np.nan), axis=0).add_prefix('ACS_')
    home = read_table('b25077').add_prefix('ACS_')
    values = read_table('b25075')
    values = values.div(values.B25075_E001.replace(0, np.nan), axis=0).add_prefix('ACS_VALUE_DIST_')
    census = pd.concat([income, home, values], axis=1)
    extras = {}
    for name in ['zhvi', 'zhvi_low', 'zhvi_high']:
        data = pd.read_csv(DATA / f'zillow_{name}.csv', dtype={'RegionName': str}, low_memory=False)
        data['Zip'] = data.RegionName.str.zfill(5)
        assert data.Zip.is_unique
        dates = ['2024-06-30', '2025-06-30', '2026-05-31', '2026-06-30']
        extra = data.set_index('Zip')[dates].add_prefix(f'ZILLOW_{name}_')
        extra[f'ZILLOW_{name}_annual_growth'] = extra[f'ZILLOW_{name}_2026-06-30'] / extra[f'ZILLOW_{name}_2025-06-30'].replace(0, np.nan)
        extras[name] = extra
    zillow = pd.concat([extras['zhvi'], extras['zhvi_low'], extras['zhvi_high']], axis=1)
    listings = pd.read_csv(DATA / 'realtor_zip_2026may_june.csv', dtype={'postal_code': str}, low_memory=False)
    listings = listings.loc[listings.month_date_yyyymm == 202606].copy()
    listings.postal_code = listings.postal_code.str.zfill(5)
    assert listings.postal_code.is_unique
    listings = listings.set_index('postal_code').drop(columns=['month_date_yyyymm', 'zip_name', 'quality_flag'])
    listings = listings.apply(pd.to_numeric, errors='coerce').add_prefix('REALTOR_')
    prices = listings[['REALTOR_median_listing_price', 'REALTOR_average_listing_price',
                       'REALTOR_median_listing_price_per_square_foot']]
    extras = {'listing_prices': pd.concat([zillow, prices], axis=1),
              'listing_all': pd.concat([zillow, listings], axis=1)}
    agg = {c: 'mean' for c in train.select_dtypes(include=[np.number]).columns if c != Y}
    agg.update({c: 'first' for c in ['State', 'City', 'Metro']})
    table = train.groupby('Zip', as_index=False).agg(agg)
    table[Y] = train.groupby('Zip')[Y].mean().to_numpy()
    z = np.array(sorted(table.Zip.unique()))
    np.random.default_rng(args.seed).shuffle(z)
    table['fold'] = table.Zip.map({zip_code: i % 5 for i, zip_code in enumerate(z)})
    checkpoint = hf_hub_download('Prior-Labs/TabPFN-v2-reg', 'tabpfn-v2-regressor.ckpt')
    outputs = []
    for fold in range(5):
        a = table.loc[table.fold != fold].drop(columns='fold').reset_index(drop=True)
        held = set(table.loc[table.fold == fold, 'Zip'])
        v = train.loc[train.Zip.isin(held) & (train.Month == '6/1/2026')].reset_index(drop=True)
        x, q = feats(a), feats(v)
        regional = np.zeros((len(a), 12), np.float32)
        for ia, ib in GroupKFold(5).split(a, groups=a.Zip):
            regional[ib] = enc(a.iloc[ia].reset_index(drop=True), a.iloc[ib].reset_index(drop=True))
        cols = [f'Regional{i}' for i in range(12)]
        x[cols], q[cols] = regional, enc(a, v)
        x = pd.concat([x, census.reindex(a.Zip).reset_index(drop=True)], axis=1)
        q = pd.concat([q, census.reindex(v.Zip).reset_index(drop=True)], axis=1).reindex(columns=x.columns)
        for frame in (x, q):
            for prefix in ['ACS_B19001_E', 'ACS_VALUE_DIST_B25075_E']:
                cols = sorted(c for c in frame if c.startswith(prefix) and not c.endswith('001'))
                frame[cols] = frame[cols].cumsum(axis=1, skipna=False)
        for col in x.select_dtypes(include=['object', 'string']).columns:
            x[col] = x[col].fillna('__MISSING__').astype(str)
            q[col] = q[col].fillna('__MISSING__').astype(str)
            levels = pd.Index(x[col].unique())
            x[col] = pd.Categorical(x[col], categories=levels)
            q[col] = pd.Categorical(q[col], categories=levels)
        row = pd.DataFrame({'Zip': v.Zip, 'fold': fold, 'truth': v[Y]})
        for name, extra in extras.items():
            xx = pd.concat([x, extra.reindex(a.Zip).reset_index(drop=True)], axis=1)
            qq = pd.concat([q, extra.reindex(v.Zip).reset_index(drop=True)], axis=1)
            for frame in (xx, qq):
                price = frame['ZILLOW_zhvi_2026-06-30']
                frame['PRICE_TO_MEDIAN_INCOME'] = price / frame.AnnualMedianHouseholdIncome.replace(0, np.nan)
                if name in ('listing_prices', 'listing_all'):
                    # ACS income bins; top bin interpolation caps at $300k.
                    edges = np.array([0, 10000, 15000, 20000, 25000, 30000, 35000, 40000,
                                      45000, 50000, 60000, 75000, 100000, 125000, 150000, 200000, 300000])
                    # x already has cumulative shares; recover individual shares.
                    cdf = frame[[f'ACS_B19001_E{i:03d}' for i in range(2, 18)]].to_numpy(float)
                    shares = np.diff(np.concatenate([np.zeros((len(frame), 1)), cdf], axis=1), axis=1)
                    for factor in [.10, .15, .20, .25, .30, .35, .40]:
                        threshold = price.to_numpy() * factor
                        below = np.clip((threshold[:, None] - edges[:-1]) / np.diff(edges), 0, 1)
                        frame[f'INCOME_TAIL_{factor:.2f}'] = 1 - (shares * below).sum(axis=1)
            model = TabPFNRegressor(n_estimators=4, model_path=checkpoint, device='cuda', n_preprocessing_jobs=2, random_state=42+fold)
            model.fit(xx, a[Y])
            p = model.predict(qq)
            row[name] = np.clip(p, 0, 100)
            assert np.isfinite(row[name]).all()
            print(f'fold={fold} {name} RMSE={np.mean((v[Y]-row[name])**2)**.5:.4f}', flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        outputs.append(row)
        pd.concat(outputs, ignore_index=True).to_csv(DATA / f'foundation_realtor_seed{args.seed}_oof.csv', index=False)
    for name in extras:
        out = pd.concat(outputs, ignore_index=True)
        print(f'{name} weighted proxy={.2*1.252406176+.8*np.mean((out.truth-out[name])**2)**.5:.4f}', flush=True)


if __name__ == '__main__':
    main()
