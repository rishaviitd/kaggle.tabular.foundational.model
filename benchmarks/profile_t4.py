"""Profile the existing T4 model and test shared-key attention layouts."""
import cProfile
import importlib
import io
import json
import pstats
import time

import numpy as np
import pandas as pd
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from housing_affordability.features import ROOT
from housing_affordability.server import HousingService


def shared_key_attention(q, k, v, *args, **kwargs):
    """Treat query heads as queries when all heads share one K/V head."""
    if (k is not None and v is not None and k.shape[2] == 1
            and q.shape[2] > 1 and not kwargs.get('quantized_kv')):
        b, s, h, d = q.shape
        query = q.reshape(b, s*h, d).unsqueeze(1)
        key, value = k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        return out.squeeze(1).reshape(b, s, h, d)
    return ORIGINAL(q, k, v, *args, **kwargs)


ARCH = importlib.import_module('tabpfn.architectures.tabpfn_v2')
ORIGINAL = ARCH.scaled_dot_product_attention


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, time.perf_counter()-start


def main():
    torch.set_num_threads(4)
    train = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
    test = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip': str})
    unseen = test.loc[~test.Zip.isin(train.Zip)].reset_index(drop=True)
    service = HousingService(train, cache=True)
    q = service.builder.transform(unseen)
    model = service.model
    out = ROOT/'artifacts/serving/attention_profile.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {'gpu': torch.cuda.get_device_name(), 'modes': {}}
    baseline = {}
    model.predict(q.iloc[:1])
    cpu = cProfile.Profile()
    cpu.enable()
    model.predict(q.iloc[:1])
    cpu.disable()
    stream = io.StringIO()
    pstats.Stats(cpu, stream=stream).sort_stats('cumulative').print_stats(35)
    report['cpu_profile'] = stream.getvalue()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as prof:
        model.predict(q.iloc[:1])
    report['gpu_profile'] = prof.key_averages().table(sort_by='self_device_time_total', row_limit=25)
    for name in ['baseline', 'shared_key']:
        ARCH.scaled_dot_product_attention = ORIGINAL if name == 'baseline' else shared_key_attention
        sizes = {}
        try:
            for n in [1, 32, 128, 256, len(q)]:
                x = q.iloc[:n]
                model.predict(x)
                runs = [timed(lambda: model.predict(x)) for _ in range(3)]
                pred = runs[-1][0]
                if name == 'baseline':
                    baseline[n] = pred
                sizes[str(n)] = {'seconds': [s for _, s in runs],
                                  'median_seconds': float(np.median([s for _, s in runs])),
                                  'max_abs_difference': float(np.max(np.abs(pred-baseline[n])))}
            report['modes'][name] = sizes
        except Exception as error:
            report['modes'][name] = {'error': repr(error)}
        finally:
            ARCH.scaled_dot_product_attention = ORIGINAL
        print('ATTENTION_MODE', name, json.dumps(report['modes'][name]), flush=True)
        out.write_text(json.dumps(report, indent=2)+'\n')
    print('ATTENTION_PROFILE', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
