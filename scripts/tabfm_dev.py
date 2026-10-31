"""Evaluate Google's TabFM on unseen development ZIPs with bounded memory."""
import argparse
import json
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sklearn.metrics import mean_squared_error
from tabfm import TabFMRegressor
from tabfm.src.pytorch.tabfm_v1_0_0 import TabFM_HF
from features import features, ROOT, TARGET

parser=argparse.ArgumentParser()
parser.add_argument('--context',type=int,default=256)
parser.add_argument('--estimators',type=int,default=2)
parser.add_argument('--query-batch',type=int,default=64)
args=parser.parse_args()
train=pd.read_csv(ROOT/'data/splits/train_local.csv',dtype={'Zip':str})
dev=pd.read_csv(ROOT/'data/splits/dev_local.csv',dtype={'Zip':str})
seen=dev.Zip.isin(train.Zip).to_numpy()
new=dev.loc[~seen].reset_index(drop=True)
X=features(train).drop(columns='City')
U=features(new).drop(columns='City')
for frame in (X,U):
    frame['income_to_required_income']=frame.AnnualMedianHouseholdIncome/(40*frame.RentIndex)
    frame['rent_to_income_x_owner']=frame.rent_to_income*frame.OwnerPercent/100
    frame['children_per_adult']=frame.AvgNumberofChildren/frame.AvgNumberofAdults

repo='google/tabfm-1.0.0-pytorch'
cfg_path=hf_hub_download(repo,'regression/config.json')
weight_path=hf_hub_download(repo,'regression/model.safetensors')
with open(cfg_path) as f: config=json.load(f)
torch.set_num_threads(4)
device='mps' if torch.backends.mps.is_available() else 'cpu'
if device=='mps':
    torch.mps.set_per_process_memory_fraction(.65)
print('Loading checkpoint on',device,flush=True)
with torch.device('meta'):
    model=TabFM_HF(**config)
weights=load_file(weight_path,device='cpu')
model.load_state_dict(weights,strict=True,assign=True)
model=model.to(device=device,dtype=torch.bfloat16).eval()
del weights
reg=TabFMRegressor(model=model,n_estimators=args.estimators,
                   max_num_rows=args.context,batch_size=1,
                   cache_context=True,maybe_quantize_kv_cache=False,
                   random_state=42)
reg.fit(X,train[TARGET])
pred=[]
for start in range(0,len(U),args.query_batch):
    pred.extend(reg.predict(U.iloc[start:start+args.query_batch]))
    print('predicted',min(start+args.query_batch,len(U)),'of',len(U),flush=True)
pred=np.asarray(pred)
prior=dict(zip(train.Zip,train[TARGET]))
seen_y=dev.loc[seen,TARGET].to_numpy()
seen_pred=np.array([prior[z] for z in dev.loc[seen,'Zip']])
seen_rmse=mean_squared_error(seen_y,seen_pred)**.5
new_rmse=mean_squared_error(new[TARGET],pred)**.5
print('dev seen RMSE',seen_rmse)
print('dev new RMSE',new_rmse)
print('dev weighted RMSE',.2*seen_rmse+.8*new_rmse)
