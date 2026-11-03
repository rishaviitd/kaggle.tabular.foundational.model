"""Measure experimental attention against the existing cached T4 model."""
import json
import numpy as np
import pandas as pd
import torch

from benchmarks.profile_t4 import ARCH, ORIGINAL, shared_key_attention, timed
from benchmarks.triton_attention import shared_attention
from housing_affordability.features import ROOT
from housing_affordability.server import HousingService


def kernel(q,k,v,*args,**kwargs):
    if k is not None and k.shape[2] == 1 and q.shape[2]>1:
        return shared_attention(q,k,v)
    return ORIGINAL(q,k,v,*args,**kwargs)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    errors = []
    for rows in [1,7,32]:
        q = torch.randn(2,rows,6,32,device='cuda',dtype=torch.float16)
        k = torch.randn(2,97,1,32,device='cuda',dtype=torch.float16)
        v = torch.randn_like(k)
        reference = shared_key_attention(q,k,v)
        actual = shared_attention(q,k,v)
        errors.append(float((reference-actual).abs().max()))
    print('TRITON_UNIT', errors, flush=True)
    train = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip':str})
    test = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip':str})
    query = test.loc[~test.Zip.isin(train.Zip)].reset_index(drop=True)
    service = HousingService(train,cache=True)
    x = service.builder.transform(query)
    baseline = {n:service.model.predict(x.iloc[:n]) for n in [1,32,128,256,len(x)]}
    report = {'gpu':torch.cuda.get_device_name(), 'unit_max_errors':errors, 'modes':{}}
    for name,fn in [('shared_key',shared_key_attention),('triton',kernel)]:
        ARCH.scaled_dot_product_attention=fn
        results = {}
        for n in [1,32,128,256,len(x)]:
            q = x.iloc[:n]
            _, first = timed(lambda:service.model.predict(q))
            runs = [timed(lambda:service.model.predict(q)) for _ in range(3)]
            results[str(n)] = {'first_seconds':first, 'seconds':[s for _,s in runs],
                               'max_abs_difference':float(np.max(np.abs(runs[-1][0]-baseline[n])))}
            print('TRITON_SIZE',name,n,json.dumps(results[str(n)]),flush=True)
        report['modes'][name]=results
    ARCH.scaled_dot_product_attention=ORIGINAL
    out=ROOT/'artifacts/serving/triton_t4.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('TRITON_T4',json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
