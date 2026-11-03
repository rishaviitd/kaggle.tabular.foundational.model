"""Validate the shipped optimizations and compare both models on one T4 host."""
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmarks.async_http_t4 import benchmark
from benchmarks.compare_xgboost import XGBModel
from housing_affordability.aggregated_features import enc
from housing_affordability.features import ROOT, Y
from housing_affordability.server import HousingService


def main():
    torch.set_num_threads(4)
    history=pd.read_csv(ROOT/'data/train.csv',dtype={'Zip':str})
    test=pd.read_csv(ROOT/'data/test.csv',dtype={'Zip':str})
    unseen=test.loc[~test.Zip.isin(history.Zip)].reset_index(drop=True)
    rows=json.loads(unseen.head(256).to_json(orient='records'))
    cpu=Path('/proc/cpuinfo').read_text()
    report={'gpu':torch.cuda.get_device_name(), 'cpu_model':next(
        line.split(':',1)[1].strip() for line in cpu.splitlines() if line.startswith('model name')),
        'torch_cpu_threads':torch.get_num_threads(),
        'scope':'localhost HTTP, aiohttp clients; 3 trials, four waves except 16 serial requests',
        'runs':{}}
    service=HousingService(history,cache=True)
    fitted=service.builder.regional_encoder.transform
    service.builder.regional_encoder.transform=lambda q:enc(service.builder.reference,q)
    q=service.builder.transform(unseen)
    before=service.model.predict(q)
    report['runs']['previous_tabpfn_stdlib_80ms']=benchmark(service,rows,'stdlib',80)
    service.builder.regional_encoder.transform=fitted
    pd.testing.assert_frame_equal(q,service.builder.transform(unseen))
    service.model.optimize_for_inference()
    after=service.model.predict(q)
    report['test_max_abs_difference']=float(np.max(np.abs(before-after)))
    report['runs']['optimized_tabpfn_keepalive_5ms']=benchmark(service,rows,'stdlib_keepalive',5)
    report['runs']['optimized_tabpfn_asgi_5ms']=benchmark(service,rows,'asgi',5)
    report['runs']['optimized_tabpfn_asgi_20ms']=benchmark(service,rows,'asgi',20)
    report['runs']['optimized_tabpfn_asgi_80ms']=benchmark(service,rows,'asgi',80)
    service.model=XGBModel().fit(service.builder.x,service.builder.y)
    for window in [.1,20,80]:
        report['runs'][f'xgboost_asgi_{window}ms']=benchmark(service,rows,'asgi',window)
    del service,fitted
    gc.collect()
    torch.cuda.empty_cache()
    report['heldout']={}
    for seed in [42,137]:
        zips=np.array(sorted(history.Zip.unique()))
        np.random.default_rng(seed).shuffle(zips)
        held=set(zips[::5])
        fit=history.loc[~history.Zip.isin(held)].reset_index(drop=True)
        query=history.loc[history.Zip.isin(held)&(history.Month=='6/1/2026')].reset_index(drop=True)
        validation=HousingService(fit,cache=True)
        q=validation.builder.transform(query)
        baseline=validation.model.predict(q)
        sample_before=np.array([validation.model.predict(q.iloc[[i]])[0] for i in range(32)])
        validation.model.optimize_for_inference()
        optimized=validation.model.predict(q)
        sample_after=np.array([validation.model.predict(q.iloc[[i]])[0] for i in range(32)])
        student=XGBModel().fit(validation.builder.x,validation.builder.y).predict(q)
        target=query[Y].to_numpy()
        score=lambda pred:float(np.sqrt(np.mean((pred-target)**2)))
        report['heldout'][str(seed)]={'rows':len(target),'baseline_rmse':score(baseline),
            'optimized_rmse':score(optimized),'xgboost_rmse':score(student),
            'max_abs_difference':float(np.max(np.abs(optimized-baseline))),
            'single_row_sample_max_abs_difference':float(np.max(np.abs(sample_after-sample_before)))}
        del validation
        gc.collect()
        torch.cuda.empty_cache()
    out=ROOT/'artifacts/serving/final_inference_comparison.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('FINAL_INFERENCE_COMPARISON',json.dumps(report),flush=True)


if __name__=='__main__':
    main()
