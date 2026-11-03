"""Test typed housing features against TabPFN's generic dtype inference."""
import json
import warnings

import numpy as np
import pandas as pd
import torch

from benchmarks.profile_t4 import timed
from housing_affordability.features import ROOT
from housing_affordability.server import HousingService
from benchmarks.execution import disable_diagnostic_barriers
from tabpfn.preprocessing.clean import clean_data_transform, process_text_na_dataframe
from tabpfn.preprocessing.datamodel import FeatureModality


def main():
    torch.set_num_threads(4)
    train=pd.read_csv(ROOT/'data/train.csv',dtype={'Zip':str})
    test=pd.read_csv(ROOT/'data/test.csv',dtype={'Zip':str})
    service=HousingService(train,cache=True,fast_attention=True)
    x=service.builder.transform(test.loc[~test.Zip.isin(train.Zip)].reset_index(drop=True))
    model=service.model.model
    encoder=model.ordinal_encoder_
    cat_indices=model.inferred_feature_schema_.indices_for(FeatureModality.CATEGORICAL)
    def fast_clean(frame):
        frame=frame.copy(deep=False)
        frame.columns=range(frame.shape[1])
        for col,value in getattr(encoder,'constant_values_',{}).items():
            frame[col]=value
        for col in cat_indices:
            frame[col]=frame[col].astype('category')
        for col in frame:
            if isinstance(frame[col].dtype,pd.CategoricalDtype) and col not in cat_indices:
                frame[col]=frame[col].astype('string')
        return process_text_na_dataframe(frame,ord_encoder=encoder,
            passthrough_inf=model.get_inference_config().PASSTHROUGH_INF)
    clean=fast_clean(x)
    expected=clean_data_transform(x.to_numpy(),cat_indices=cat_indices,
        ord_encoder=encoder,passthrough_inf=model.get_inference_config().PASSTHROUGH_INF)
    mismatch=~np.isclose(clean,expected,rtol=0,atol=0,equal_nan=True)
    columns=[str(x.columns[i]) for i in np.flatnonzero(mismatch.any(axis=0))]
    reference={n:service.model.predict(x.iloc[:n]) for n in [1,32,128,256,len(x)]}
    disable_diagnostic_barriers(model)
    report={'feature_rows_equal':len(x) if not mismatch.any() else 0,
            'mismatched_columns':columns,'modes':{}}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for mode in ['async'] + ([] if mismatch.any() else ['async_fast_clean']):
            if mode.endswith('clean'):
                model.ordinal_encoder_=None
            sizes={}
            for n in [1,32,128,256,len(x)]:
                q=x.iloc[:n]
                def predict():
                    data=fast_clean(q) if mode.endswith('clean') else q
                    return np.clip(model.predict(data),0,100)
                predict()
                runs=[timed(predict) for _ in range(3)]
                sizes[str(n)]={'seconds':[s for _,s in runs],
                               'max_abs_difference':float(np.max(np.abs(runs[-1][0]-reference[n])))}
            report['modes'][mode]=sizes
            print('FAST_CLEAN_MODE',mode,json.dumps(sizes),flush=True)
    print('FAST_CLEAN',json.dumps(report),flush=True)


if __name__=='__main__':
    main()
