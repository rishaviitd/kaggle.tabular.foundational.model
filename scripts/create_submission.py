"""Create the dev-selected ensemble's Kaggle submission from official data."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from tabicl import TabICLRegressor
from features import features, group_columns, encode_groups, TARGET, ROOT

parser=argparse.ArgumentParser()
parser.add_argument('--base',required=True)
parser.add_argument('--save',required=True)
args=parser.parse_args()
train=pd.read_csv(ROOT/'data/train.csv',dtype={'Zip':str})
test=pd.read_csv(ROOT/'data/test.csv',dtype={'Zip':str})
sample=pd.read_csv(ROOT/'data/sample_submission.csv',dtype={'Zip':str})
saved=np.load(args.base)
if not np.array_equal(saved['zip'].astype(str),test.Zip.to_numpy()) or not np.array_equal(saved['state'].astype(str),test.State.to_numpy()):
    raise ValueError('Base predictions do not match test rows.')
if list(sample.columns)!=['Zip','State',TARGET] or not test[['Zip','State']].equals(sample[['Zip','State']]):
    raise ValueError('Test rows do not match the sample submission.')
if test.duplicated(['Zip','State']).any():
    raise ValueError('Duplicate test ZIP and state.')
pred=saved['prediction'].copy();seen=saved['seen']
new=test.loc[~seen].reset_index(drop=True)
if set(train.Zip)&set(new.Zip):
    raise ValueError('A new test ZIP appears in the training data.')
keys=group_columns(train);newkeys=group_columns(new);y=train[TARGET].to_numpy()
oof=pd.DataFrame(index=range(len(train)))
for a,b in GroupKFold(n_splits=5,shuffle=True,random_state=42).split(train,groups=train.Zip):
    temp=encode_groups(keys.iloc[a],y[a],keys.iloc[b]);temp.index=b
    for c in temp:oof.loc[b,c]=temp[c]
new_encoded=encode_groups(keys,y,newkeys)
X=features(train).reset_index(drop=True).drop(columns='City')
U=features(new).reset_index(drop=True).drop(columns='City')
for data in (X,U):
    data['income_to_required_income']=data.AnnualMedianHouseholdIncome/(40*data.RentIndex)
    data['rent_to_income_x_owner']=data.rent_to_income*data.OwnerPercent/100
    data['children_per_adult']=data.AvgNumberofChildren/data.AvgNumberofAdults
X=pd.concat([X,oof],axis=1);U=pd.concat([U,new_encoded],axis=1)
model=TabICLRegressor(n_estimators=4,batch_size=1,device=None,n_jobs=4,random_state=42)
model.fit(X,y)
pred[~seen]=.25*pred[~seen]+.75*model.predict(U)
pred=np.clip(pred,0,100)
if not np.isfinite(pred).all():
    raise ValueError('Submission contains missing or infinite predictions.')
out=sample[['Zip','State']].copy();out[TARGET]=pred
path=Path(args.save);path.parent.mkdir(parents=True,exist_ok=True)
temp=path.with_suffix('.csv.tmp')
out.to_csv(temp,index=False)
temp.replace(path)
print('submission rows:',len(out),'seen:',int(seen.sum()),'new:',int((~seen).sum()))
print('prediction min, median, max:',*[round(x,2) for x in (pred.min(),np.median(pred),pred.max())])
print('first predictions:')
print(out.head(5).to_string(index=False))
print('saved:',path)
