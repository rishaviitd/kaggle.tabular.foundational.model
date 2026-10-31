"""Final validation check using train_local and dev_local only for fitting."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from features import ROOT, TARGET, CAT, ZIP_CAT, features

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--save', type=Path, help='Save predictions for an ensemble')
    args=parser.parse_args()
    tr=pd.read_csv(ROOT/'data/splits/train_local.csv',dtype={'Zip':str})
    dev=pd.read_csv(ROOT/'data/splits/dev_local.csv',dtype={'Zip':str})
    val=pd.read_csv(ROOT/'data/splits/val_local.csv',dtype={'Zip':str})
    fit=pd.concat([tr,dev],ignore_index=True)
    X=features(fit);V=features(val)
    cats=CAT+ZIP_CAT
    for c in cats:
        X[c]=X[c].astype('category')
        V[c]=pd.Categorical(V[c],categories=X[c].cat.categories)
    model=LGBMRegressor(n_estimators=1000,learning_rate=.03,num_leaves=31,min_child_samples=30,reg_lambda=5,verbosity=-1,n_jobs=4,random_state=42)
    model.fit(X,fit[TARGET],categorical_feature=cats)
    pred=model.predict(V)
    seen=set(tr.Zip)
    is_seen=val.Zip.isin(seen).to_numpy()
    prior=dict(zip(tr.Zip,tr[TARGET]))
    pred[is_seen]=np.array([prior[z] for z in val.loc[is_seen,'Zip']])
    if args.save:
        np.savez(args.save, zip=val.Zip.to_numpy(dtype=str), prediction=pred)
    truth=val[TARGET].to_numpy()
    seen_rmse=mean_squared_error(truth[is_seen],pred[is_seen])**.5
    new_rmse=mean_squared_error(truth[~is_seen],pred[~is_seen])**.5
    print(f'val seen rows={is_seen.sum()} rmse={seen_rmse:.4f}')
    print(f'val new rows={(~is_seen).sum()} rmse={new_rmse:.4f}')
    print(f'val weighted_rmse={.2*seen_rmse+.8*new_rmse:.4f}')
if __name__=='__main__':main()
