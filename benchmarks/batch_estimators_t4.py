"""Test batching compatible fitted v2 ensemble members without refitting."""
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from benchmarks.profile_t4 import timed
from housing_affordability.features import ROOT
from housing_affordability.server import HousingService
from benchmarks.execution import disable_diagnostic_barriers
from tabpfn.architectures.tabpfn_v2 import TabPFNV2Cache
from tabpfn.architectures.kv_cache import KVCache


def merge_caches(executor):
    groups=defaultdict(list)
    for i,(cache,members) in enumerate(zip(executor.kv_caches,executor.cache_groups)):
        model_index=executor.ensemble_members[members[0]].config._model_index
        width=cache.feature_cache['column_selection_mask'].shape[0]//cache.train_shape[0]
        key=(model_index,cache.train_shape[1],width,tuple(cache.kv[0].key.shape[1:]),
             tuple((k,tuple(v.shape[1:])) for k,v in cache.feature_cache.items()))
        groups[key].append(i)
    new_caches,new_groups=[],[]
    for positions in groups.values():
        caches=[executor.kv_caches[i] for i in positions]
        members=[m for i in positions for m in executor.cache_groups[i]]
        if len(caches)==1:
            merged=caches[0]
        else:
            features={name:torch.cat([c.feature_cache[name] for c in caches],dim=0)
                      for name in caches[0].feature_cache}
            embedding=torch.cat([c.test_y_embedding for c in caches],dim=0)
            shape=(sum(c.train_shape[0] for c in caches),caches[0].train_shape[1])
            merged=TabPFNV2Cache(kv=KVCache._consume_and_concatenate_layers(caches),
                                 feature_cache=features,test_y_embedding=embedding,train_shape=shape)
        new_caches.append(merged)
        new_groups.append(members)
    executor.kv_caches,executor.cache_groups=new_caches,new_groups


def main():
    torch.set_num_threads(4)
    train=pd.read_csv(ROOT/'data/train.csv',dtype={'Zip':str})
    test=pd.read_csv(ROOT/'data/test.csv',dtype={'Zip':str})
    service=HousingService(train,cache=True,fast_attention=True)
    x=service.builder.transform(test.loc[~test.Zip.isin(train.Zip)].reset_index(drop=True))
    model=service.model
    disable_diagnostic_barriers(model.model)
    executor=model.model.executor_
    report={'gpu':torch.cuda.get_device_name(),'groups_before':executor.cache_groups,
            'cache_shapes':[list(c.kv[0].key.shape) for c in executor.kv_caches],'modes':{}}
    reference={}
    for name in ['sequential','batched']:
        try:
            if name=='batched':
                merge_caches(executor)
                report['groups_after']=executor.cache_groups
            sizes={}
            for n in [1,32,128,256,len(x)]:
                q=x.iloc[:n]
                model.predict(q)
                runs=[timed(lambda:model.predict(q)) for _ in range(3)]
                pred=runs[-1][0]
                if name=='sequential':
                    reference[n]=pred
                sizes[str(n)]={'seconds':[s for _,s in runs],
                               'max_abs_difference':float(np.max(np.abs(pred-reference[n])))}
            report['modes'][name]=sizes
        except Exception as error:
            report['modes'][name]={'error':repr(error)}
    print('BATCH_ESTIMATORS',json.dumps(report),flush=True)


if __name__=='__main__':
    main()
