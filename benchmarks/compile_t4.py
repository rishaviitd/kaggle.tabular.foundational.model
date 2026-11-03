"""Test compiled transformer blocks without compiling data-dependent encoders."""
import json
import time

import numpy as np
import pandas as pd
import torch

from benchmarks.profile_t4 import ARCH, ORIGINAL, shared_key_attention, timed
from housing_affordability.features import ROOT
from housing_affordability.server import HousingService


def main():
    torch.set_num_threads(4)
    train = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
    test = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip': str})
    query = test.loc[~test.Zip.isin(train.Zip)].reset_index(drop=True)
    service = HousingService(train, cache=True)
    x = service.builder.transform(query)
    models = [cache.get(device) for cache in service.model.model.executor_.model_caches
              for device in cache.get_devices()]
    baseline = {n: service.model.predict(x.iloc[:n]) for n in [1,32,128,256]}
    report = {'gpu': torch.cuda.get_device_name(), 'modes': {}}
    out = ROOT/'artifacts/serving/compiled_blocks.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    for name in ['shared_key_eager', 'shared_key_compiled_safe']:
        ARCH.scaled_dot_product_attention = shared_key_attention
        if 'compiled' in name:
            for model in models:
                for block in model.blocks:
                    compiled = torch.compile(block.forward, dynamic=True, mode='reduce-overhead')
                    def forward(*args, _compiled=compiled, **kwargs):
                        output, cache = _compiled(*args, **kwargs)
                        return output.clone(), cache
                    block.forward = forward
        try:
            results = {}
            for n in [1,32,128,256]:
                q = x.iloc[:n]
                def predict():
                    if 'compiled' in name:
                        torch.compiler.cudagraph_mark_step_begin()
                    return service.model.predict(q)
                _, first = timed(predict)
                runs = [timed(predict) for _ in range(3)]
                results[str(n)] = {'first_seconds': first,
                                   'seconds': [s for _,s in runs],
                                   'max_abs_difference': float(np.max(np.abs(runs[-1][0]-baseline[n])))}
                print('COMPILE_SIZE', name, n, json.dumps(results[str(n)]), flush=True)
            report['modes'][name] = results
        except Exception as error:
            report['modes'][name] = {'error': repr(error)}
        out.write_text(json.dumps(report,indent=2)+'\n')
    ARCH.scaled_dot_product_attention = ORIGINAL
    print('COMPILED_BLOCKS', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
