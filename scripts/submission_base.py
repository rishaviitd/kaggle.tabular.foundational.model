"""Fit the validated LightGBM baseline on both labeled months."""
import argparse
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from features import ROOT,TARGET,CAT,ZIP_CAT,features

parser=argparse.ArgumentParser()
parser.add_argument('--save',required=True)
args=parser.parse_args()
train=pd.read_csv(ROOT/'data/train.csv',dtype={'Zip':str})
test=pd.read_csv(ROOT/'data/test.csv',dtype={'Zip':str})
X=features(train);T=features(test)
cats=CAT+ZIP_CAT
for col in cats:
    X[col]=X[col].astype('category')
    T[col]=T[col].where(T[col].isin(X[col].cat.categories),None).astype(X[col].dtype)
model=LGBMRegressor(n_estimators=1000,learning_rate=.03,num_leaves=31,min_child_samples=30,reg_lambda=5,verbosity=-1,n_jobs=4,random_state=42)
model.fit(X,train[TARGET],categorical_feature=cats)
pred=model.predict(T)
history=train.assign(_date=pd.to_datetime(train.Month)).sort_values('_date').drop_duplicates(['Zip','State'],keep='last')
prior=history.set_index(['Zip','State'])[TARGET]
keys=pd.MultiIndex.from_frame(test[['Zip','State']])
matched=prior.reindex(keys)
seen=matched.notna().to_numpy()
pred[seen]=matched[seen].to_numpy()
np.savez(args.save,zip=test.Zip.to_numpy(dtype=str),state=test.State.to_numpy(dtype=str),prediction=pred,seen=seen)
print('test rows',len(test),'seen',seen.sum(),'new',(~seen).sum())
