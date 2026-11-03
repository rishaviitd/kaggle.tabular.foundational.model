"""Tune cache reuse on representative attention shapes, without model downloads."""
import json
import time
import numpy as np
import torch

from benchmarks.triton_attention import shared_attention
from housing_affordability.attention import shared_key_sdpa


def measure(fn):
    torch.cuda.synchronize()
    start=time.perf_counter()
    out=fn()
    torch.cuda.synchronize()
    return out,time.perf_counter()-start


def main():
    torch.manual_seed(42)
    report={'gpu':torch.cuda.get_device_name(),'shapes':{}}
    for rows in [32,128,256]:
        q=torch.randn(155,rows,6,32,device='cuda',dtype=torch.float16)
        k=torch.randn(155,6830,1,32,device='cuda',dtype=torch.float16)
        v=torch.randn_like(k)
        reference=shared_key_sdpa(q,k,v)
        base=[measure(lambda:shared_key_sdpa(q,k,v))[1] for _ in range(5)]
        results={'pytorch_seconds':base,'candidates':{}}
        for bm,bn,warps in [(64,64,4),(128,64,4),(128,64,8),(256,64,8),
                            (128,32,4),(256,32,8),(128,128,8),(256,128,8)]:
            name=f'q{bm}_k{bn}_w{warps}'
            try:
                fn=lambda:shared_attention(q,k,v,query_block=bm,key_block=bn,warps=warps)
                fn()
                runs=[measure(fn) for _ in range(5)]
                result={'seconds':[s for _,s in runs],
                         'max_abs_error':float((runs[-1][0]-reference).abs().max())}
            except Exception as error:
                result={'error':repr(error)}
            results['candidates'][name]=result
            print('KERNEL_CANDIDATE',rows,name,json.dumps(result),flush=True)
        report['shapes'][str(rows)]=results
    print('KERNEL_TUNING',json.dumps(report),flush=True)


if __name__=='__main__':
    main()
