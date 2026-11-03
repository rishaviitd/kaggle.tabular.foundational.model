"""Reproduce the external-feature TabPFN training and query representation."""
from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(os.environ.get('HOUSING_AFFORDABILITY_ROOT', Path(__file__).resolve().parents[1])).resolve()
from housing_affordability.aggregated_features import feats, enc, Y

DATA = ROOT / 'artifacts/external_acs'


def read_table(name):
    frame = pd.read_csv(DATA / f'{name}_zcta.dat', sep='|', dtype={'GEO_ID': str}, low_memory=False)
    frame['Zip'] = frame.GEO_ID.str.removeprefix('860Z200US')
    if not frame.Zip.str.fullmatch(r'\d{5}').all() or frame.Zip.duplicated().any():
        raise ValueError(f'Invalid or duplicate Census ZCTA: {name}')
    cols = [c for c in frame if '_E' in c]
    for col in cols:
        frame[col] = pd.to_numeric(frame[col], errors='coerce').where(lambda x: x >= 0)
    return frame.set_index('Zip')[cols]


def aggregate_history(history):
    agg = {c: 'mean' for c in history.select_dtypes(include=[np.number]).columns if c != Y}
    agg.update({c: 'first' for c in ['State', 'City', 'Metro']})
    out = history.groupby('Zip', as_index=False).agg(agg)
    out[Y] = history.groupby('Zip')[Y].mean().to_numpy()
    return out


def load_external():
    income = read_table('b19001')
    income = income.div(income.B19001_E001.replace(0, np.nan), axis=0).add_prefix('ACS_')
    home = read_table('b25077').add_prefix('ACS_')
    values = read_table('b25075')
    values = values.div(values.B25075_E001.replace(0, np.nan), axis=0).add_prefix('ACS_VALUE_DIST_')
    census = pd.concat([income, home, values], axis=1)
    tables = []
    for name in ['zhvi', 'zhvi_low', 'zhvi_high']:
        data = pd.read_csv(DATA / f'zillow_{name}.csv', dtype={'RegionName': str}, low_memory=False)
        data['Zip'] = data.RegionName.str.zfill(5)
        if not data.Zip.is_unique:
            raise ValueError(f'Duplicate Zillow ZIP in {name}')
        dates = ['2024-06-30', '2025-06-30', '2026-05-31', '2026-06-30']
        extra = data.set_index('Zip')[dates].add_prefix(f'ZILLOW_{name}_')
        extra[f'ZILLOW_{name}_annual_growth'] = extra[f'ZILLOW_{name}_2026-06-30'] / extra[f'ZILLOW_{name}_2025-06-30'].replace(0, np.nan)
        tables.append(extra)
    listings = pd.read_csv(DATA / 'realtor_zip_2026may_june.csv', dtype={'postal_code': str}, low_memory=False)
    listings = listings.loc[listings.month_date_yyyymm == 202606].copy()
    listings.postal_code = listings.postal_code.str.zfill(5)
    if not listings.postal_code.is_unique:
        raise ValueError('Duplicate Realtor ZIP')
    listings = listings.set_index('postal_code').drop(columns=['month_date_yyyymm', 'zip_name', 'quality_flag'])
    listings = listings.apply(pd.to_numeric, errors='coerce').add_prefix('REALTOR_')
    prices = listings[['REALTOR_median_listing_price', 'REALTOR_average_listing_price', 'REALTOR_median_listing_price_per_square_foot']]
    return census, pd.concat(tables + [prices], axis=1)


def prepare(history, query, external):
    """Return full-fit features and unseen-ZIP features; never use query targets."""
    a = aggregate_history(history)
    query = query.reset_index(drop=True)
    if set(a.Zip) & set(query.Zip):
        raise ValueError('Model features are intended for unseen ZIPs only')
    census, extra = external
    regional = np.zeros((len(a), 12), np.float32)
    for ia, ib in GroupKFold(5).split(a, groups=a.Zip):
        regional[ib] = enc(a.iloc[ia].reset_index(drop=True), a.iloc[ib].reset_index(drop=True))
    x, q = feats(a), feats(query)
    cols = [f'Regional{i}' for i in range(12)]
    x[cols], q[cols] = regional, enc(a, query)
    x = pd.concat([x, census.reindex(a.Zip).reset_index(drop=True)], axis=1)
    q = pd.concat([q, census.reindex(query.Zip).reset_index(drop=True)], axis=1).reindex(columns=x.columns)
    for col in x.select_dtypes(include=['object', 'string']).columns:
        x[col] = x[col].fillna('__MISSING__').astype(str)
        q[col] = q[col].fillna('__MISSING__').astype(str)
        levels = pd.Index(x[col].unique())
        x[col] = pd.Categorical(x[col], categories=levels)
        q[col] = pd.Categorical(q[col], categories=levels)
    for frame in (x, q):
        for prefix in ['ACS_B19001_E', 'ACS_VALUE_DIST_B25075_E']:
            cols = sorted(c for c in frame if c.startswith(prefix) and not c.endswith('001'))
            frame[cols] = frame[cols].cumsum(axis=1, skipna=False)
    x = pd.concat([x, extra.reindex(a.Zip).reset_index(drop=True)], axis=1)
    q = pd.concat([q, extra.reindex(query.Zip).reset_index(drop=True)], axis=1)
    for frame in (x, q):
        price = frame['ZILLOW_zhvi_2026-06-30']
        frame['PRICE_TO_MEDIAN_INCOME'] = price / frame.AnnualMedianHouseholdIncome.replace(0, np.nan)
        edges = np.array([0,10000,15000,20000,25000,30000,35000,40000,45000,50000,60000,75000,100000,125000,150000,200000,300000])
        cdf = frame[[f'ACS_B19001_E{i:03d}' for i in range(2, 18)]].to_numpy(float)
        shares = np.diff(np.concatenate([np.zeros((len(frame), 1)), cdf], axis=1), axis=1)
        for factor in [.10, .15, .20, .25, .30, .35, .40]:
            below = np.clip((price.to_numpy()[:, None]*factor - edges[:-1]) / np.diff(edges), 0, 1)
            frame[f'INCOME_TAIL_{factor:.2f}'] = 1 - (shares * below).sum(axis=1)
    return x, a[Y], q


class FeatureBuilder:
    """Cache fit-side features and regional summaries for repeated requests."""
    def __init__(self, history, external, example_query):
        self.external = external
        self.reference = aggregate_history(history)
        self.x, self.y, _ = prepare(history, example_query, external)
        first_extra = self.x.columns.get_loc('ZILLOW_zhvi_2024-06-30')
        self.base_columns = self.x.columns[:first_extra]
        self.training_zips = set(self.reference.Zip)

    def transform(self, query):
        query = query.reset_index(drop=True)
        if set(query.Zip) & self.training_zips:
            raise ValueError('transform expects previously unseen ZIPs')
        census, extra = self.external
        q = feats(query)
        q[[f'Regional{i}' for i in range(12)]] = enc(self.reference, query)
        q = pd.concat([q, census.reindex(query.Zip).reset_index(drop=True)], axis=1).reindex(columns=self.base_columns)
        for col in q.columns:
            if isinstance(self.x[col].dtype, pd.CategoricalDtype):
                q[col] = pd.Categorical(q[col].fillna('__MISSING__').astype(str), categories=self.x[col].cat.categories)
        for prefix in ['ACS_B19001_E', 'ACS_VALUE_DIST_B25075_E']:
            cols = sorted(c for c in q if c.startswith(prefix) and not c.endswith('001'))
            q[cols] = q[cols].cumsum(axis=1, skipna=False)
        q = pd.concat([q, extra.reindex(query.Zip).reset_index(drop=True)], axis=1)
        price = q['ZILLOW_zhvi_2026-06-30']
        q['PRICE_TO_MEDIAN_INCOME'] = price / q.AnnualMedianHouseholdIncome.replace(0, np.nan)
        edges = np.array([0,10000,15000,20000,25000,30000,35000,40000,45000,50000,60000,75000,100000,125000,150000,200000,300000])
        cdf = q[[f'ACS_B19001_E{i:03d}' for i in range(2,18)]].to_numpy(float)
        shares = np.diff(np.concatenate([np.zeros((len(q),1)), cdf], axis=1), axis=1)
        for factor in [.10,.15,.20,.25,.30,.35,.40]:
            below = np.clip((price.to_numpy()[:,None]*factor-edges[:-1])/np.diff(edges),0,1)
            q[f'INCOME_TAIL_{factor:.2f}'] = 1-(shares*below).sum(axis=1)
        return q.reindex(columns=self.x.columns)
