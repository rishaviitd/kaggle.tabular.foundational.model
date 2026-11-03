"""Fit the evaluated Zillow and Realtor listing-price model on all labeled ZIPs."""
import argparse,gc,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sklearn.model_selection import GroupKFold
from tabpfn import TabPFNRegressor

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from housing_affordability.aggregated_features import feats,enc,Y
from evaluate import DATA,read_table


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--home-value-distribution',action='store_true');args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Expected an available CUDA GPU')
    torch.set_num_threads(4)
    train=pd.read_csv(ROOT/'data/train.csv',dtype={'Zip':str});test=pd.read_csv(ROOT/'data/test.csv',dtype={'Zip':str})
    assert not train.duplicated(['Zip','Month']).any() and train.groupby('Zip').size().eq(2).all()
    assert not test.duplicated(['Zip','State']).any()
    new=test.loc[~test.Zip.isin(train.Zip)].reset_index(drop=True)
    assert not set(train.Zip)&set(new.Zip)
    income=read_table('b19001');home=read_table('b25077')
    income=income.div(income.B19001_E001.replace(0,np.nan),axis=0).add_prefix('ACS_')
    census=pd.concat([income,home.add_prefix('ACS_')],axis=1)
    if args.home_value_distribution:
        values=read_table('b25075');values=values.div(values.B25075_E001.replace(0,np.nan),axis=0).add_prefix('ACS_VALUE_DIST_')
        census=pd.concat([census,values],axis=1)
    assert census.index.is_unique and train.Zip.isin(census.index).all() and new.Zip.isin(census.index).all()
    agg={c:'mean' for c in train.select_dtypes(include=[np.number]).columns if c!=Y}
    agg.update({c:'first' for c in ['State','City','Metro']})
    a=train.groupby('Zip',as_index=False).agg(agg);a[Y]=train.groupby('Zip')[Y].mean().to_numpy()
    ra=np.zeros((len(a),12),np.float32)
    for ia,ib in GroupKFold(5).split(a,groups=a.Zip):ra[ib]=enc(a.iloc[ia].reset_index(drop=True),a.iloc[ib].reset_index(drop=True))
    x,q=feats(a),feats(new);cols=[f'Regional{i}' for i in range(12)];x[cols]=ra;q[cols]=enc(a,new)
    x=pd.concat([x,census.reindex(a.Zip).reset_index(drop=True)],axis=1)
    q=pd.concat([q,census.reindex(new.Zip).reset_index(drop=True)],axis=1).reindex(columns=x.columns)
    for col in x.select_dtypes(include=['object','string']).columns:
        x[col]=x[col].fillna('__MISSING__').astype(str);q[col]=q[col].fillna('__MISSING__').astype(str)
        levels=pd.Index(x[col].unique());x[col]=pd.Categorical(x[col],categories=levels);q[col]=pd.Categorical(q[col],categories=levels)
    ckpt=hf_hub_download('Prior-Labs/TabPFN-v2-reg','tabpfn-v2-regressor.ckpt')
    if not args.home_value_distribution:raise ValueError('The selected model needs home value distributions')
    for frame in (x,q):
        for prefix in ['ACS_B19001_E','ACS_VALUE_DIST_B25075_E']:
            cols=sorted(c for c in frame if c.startswith(prefix) and not c.endswith('001'))
            frame[cols]=frame[cols].cumsum(axis=1,skipna=False)
    tables=[]
    for name in ['zhvi','zhvi_low','zhvi_high']:
        data=pd.read_csv(DATA/f'zillow_{name}.csv',dtype={'RegionName':str},low_memory=False)
        data['Zip']=data.RegionName.str.zfill(5)
        assert data.Zip.is_unique
        dates=['2024-06-30','2025-06-30','2026-05-31','2026-06-30']
        extra=data.set_index('Zip')[dates].add_prefix(f'ZILLOW_{name}_')
        extra[f'ZILLOW_{name}_annual_growth']=extra[f'ZILLOW_{name}_2026-06-30']/extra[f'ZILLOW_{name}_2025-06-30'].replace(0,np.nan)
        tables.append(extra)
    listings=pd.read_csv(DATA/'realtor_zip_2026may_june.csv',dtype={'postal_code':str},low_memory=False)
    listings=listings.loc[listings.month_date_yyyymm==202606].copy()
    listings.postal_code=listings.postal_code.str.zfill(5)
    assert listings.postal_code.is_unique
    listings=listings.set_index('postal_code').drop(columns=['month_date_yyyymm','zip_name','quality_flag'])
    listings=listings.apply(pd.to_numeric,errors='coerce').add_prefix('REALTOR_')
    prices=listings[['REALTOR_median_listing_price','REALTOR_average_listing_price','REALTOR_median_listing_price_per_square_foot']]
    extra=pd.concat(tables+[prices],axis=1)
    x=pd.concat([x,extra.reindex(a.Zip).reset_index(drop=True)],axis=1)
    q=pd.concat([q,extra.reindex(new.Zip).reset_index(drop=True)],axis=1)
    for frame in (x,q):
        price=frame['ZILLOW_zhvi_2026-06-30']
        frame['PRICE_TO_MEDIAN_INCOME']=price/frame.AnnualMedianHouseholdIncome.replace(0,np.nan)
        edges=np.array([0,10000,15000,20000,25000,30000,35000,40000,45000,50000,60000,75000,100000,125000,150000,200000,300000])
        cdf=frame[[f'ACS_B19001_E{i:03d}' for i in range(2,18)]].to_numpy(float)
        shares=np.diff(np.concatenate([np.zeros((len(frame),1)),cdf],axis=1),axis=1)
        for factor in [.10,.15,.20,.25,.30,.35,.40]:
            threshold=price.to_numpy()*factor
            below=np.clip((threshold[:,None]-edges[:-1])/np.diff(edges),0,1)
            frame[f'INCOME_TAIL_{factor:.2f}']=1-(shares*below).sum(axis=1)
    model=TabPFNRegressor(n_estimators=4,model_path=ckpt,device='cuda',n_preprocessing_jobs=2,random_state=42)
    model.fit(x,a[Y]);p=np.clip(model.predict(q),0,100)
    if not np.isfinite(p).all():raise ValueError('Non-finite predictions')
    result=new[['Zip','State']].copy();result['tabpfn']=p
    print(f'Fitted {len(train)} rows / {len(a)} ZIPs, predicted {len(new)} new ZIPs',flush=True)
    output=DATA/'foundation_realtor_test.csv';output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(output,index=False)
    info={'training_rows':len(train),'training_zips':len(a),'new_test_zips':len(new),'home_value_distribution':True,'external_data':'2020-2024 ACS, June 2026 Zillow home-value tiers and Realtor.com Economic Research listing prices','seed':42}
    output.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n');print(output,flush=True)

if __name__=='__main__':main()
