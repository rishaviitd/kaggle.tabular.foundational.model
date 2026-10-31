"""Score the dev-selected regional-summary and LightGBM ensemble."""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from tabicl import TabICLRegressor
from features import features, group_columns, encode_groups, TARGET, ROOT

parser=argparse.ArgumentParser()
parser.add_argument('--base',required=True)
parser.add_argument('--save',help='Optional CSV for validation predictions')
args=parser.parse_args()
tr=pd.read_csv(ROOT/'data/splits/train_local.csv',dtype={'Zip':str})
dev=pd.read_csv(ROOT/'data/splits/dev_local.csv',dtype={'Zip':str})
val=pd.read_csv(ROOT/'data/splits/val_local.csv',dtype={'Zip':str})
if set(dev.Zip)&set(val.Zip):
    raise ValueError('Development and validation ZIPs overlap.')
saved=np.load(args.base)
if not np.array_equal(saved['zip'].astype(str),val.Zip.to_numpy()):
    raise ValueError('Saved predictions do not align with validation rows.')
pred=saved['prediction'].copy()
seen=val.Zip.isin(tr.Zip).to_numpy()
new=val.loc[~seen].reset_index(drop=True)
fit=pd.concat([tr,dev],ignore_index=True)
if set(fit.Zip)&set(new.Zip):
    raise ValueError('A new validation ZIP appears in fitting data.')
keys=group_columns(fit);newkeys=group_columns(new);y=fit[TARGET].to_numpy()
oof=pd.DataFrame(index=range(len(fit)))
for a,b in GroupKFold(n_splits=5,shuffle=True,random_state=42).split(fit,groups=fit.Zip):
    temp=encode_groups(keys.iloc[a],y[a],keys.iloc[b]);temp.index=b
    for c in temp:oof.loc[b,c]=temp[c]
new_encoded=encode_groups(keys,y,newkeys)
X=features(fit).reset_index(drop=True).drop(columns='City')
U=features(new).reset_index(drop=True).drop(columns='City')
for data in (X,U):
    data['income_to_required_income']=data.AnnualMedianHouseholdIncome/(40*data.RentIndex)
    data['rent_to_income_x_owner']=data.rent_to_income*data.OwnerPercent/100
    data['children_per_adult']=data.AvgNumberofChildren/data.AvgNumberofAdults
X=pd.concat([X,oof],axis=1);U=pd.concat([U,new_encoded],axis=1)
model=TabICLRegressor(n_estimators=4,batch_size=1,device=None,n_jobs=4,random_state=42)
model.fit(X,y)
pred[~seen]=.25*pred[~seen]+.75*model.predict(U)
truth=val[TARGET].to_numpy()
seen_rmse=mean_squared_error(truth[seen],pred[seen])**.5
new_rmse=mean_squared_error(truth[~seen],pred[~seen])**.5
print(f'validation seen RMSE: {seen_rmse:.4f}')
print(f'validation new RMSE: {new_rmse:.4f}')
print(f'validation weighted RMSE: {.2*seen_rmse+.8*new_rmse:.4f}')
if args.save:
    from pathlib import Path
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'Zip':val.Zip,'State':val.State,'seen':seen,'prediction':pred}).to_csv(args.save,index=False)
