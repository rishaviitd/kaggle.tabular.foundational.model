"""Persistent TabPFN model with optional training-context KV cache."""
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tabpfn import TabPFNRegressor


class HousingModel:
    def __init__(self, cache=False, device='cuda'):
        checkpoint = hf_hub_download('Prior-Labs/TabPFN-v2-reg', 'tabpfn-v2-regressor.ckpt')
        self.model = TabPFNRegressor(n_estimators=4, model_path=checkpoint, device=device,
                                    n_preprocessing_jobs=2, random_state=42,
                                    fit_mode='fit_with_cache' if cache else 'fit_preprocessors')

    def fit(self, x, y):
        self.model.fit(x, y)
        return self

    def predict(self, x, batch_size=None):
        if len(x) == 0:
            return np.array([], dtype=float)
        batch_size = batch_size or len(x)
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        with torch.inference_mode():
            parts = [self.model.predict(x.iloc[i:i+batch_size]) for i in range(0, len(x), batch_size)]
        return np.clip(np.concatenate(parts), 0, 100)
