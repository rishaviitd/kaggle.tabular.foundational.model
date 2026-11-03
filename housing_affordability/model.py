"""Persistent TabPFN model with optional training-context KV cache."""
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tabpfn import TabPFNRegressor


class HousingModel:
    def __init__(self, cache=False, device='cuda', fast_attention=False, optimize=False):
        if (fast_attention or optimize) and not cache:
            raise ValueError('Inference optimizations require cache=True')
        self.fast_attention = fast_attention or optimize
        self.optimize = optimize
        self.input_encoder = None
        checkpoint = hf_hub_download('Prior-Labs/TabPFN-v2-reg', 'tabpfn-v2-regressor.ckpt')
        self.model = TabPFNRegressor(n_estimators=4, model_path=checkpoint, device=device,
                                    n_preprocessing_jobs=2, random_state=42,
                                    fit_mode='fit_with_cache' if cache else 'fit_preprocessors')

    def fit(self, x, y):
        self.model.fit(x, y)
        self.feature_columns_ = x.columns.copy()
        self.input_encoder = None
        if self.optimize:
            self.optimize_for_inference()
        elif self.fast_attention:
            from housing_affordability.attention import enable_shared_key_attention
            enable_shared_key_attention(self.model)
        return self

    def optimize_for_inference(self):
        """Optimize this fitted cached v2 model without changing its ensemble."""
        from tabpfn.preprocessing.datamodel import FeatureModality
        from housing_affordability.attention import enable_shared_key_attention
        from housing_affordability.input_encoder import HousingInputEncoder
        enable_shared_key_attention(self.model)
        self.optimize = True
        self.fast_attention = True
        if self.input_encoder is None:
            self.input_encoder = HousingInputEncoder(self.model.ordinal_encoder_,
                self.model.inferred_feature_schema_.indices_for(FeatureModality.CATEGORICAL), self.feature_columns_,
                passthrough_inf=self.model.get_inference_config().PASSTHROUGH_INF)
            self.model.ordinal_encoder_ = None
        return self

    def predict(self, x, batch_size=None):
        if len(x) == 0:
            return np.array([], dtype=float)
        batch_size = batch_size or len(x)
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        with torch.inference_mode():
            parts = []
            for i in range(0, len(x), batch_size):
                query = x.iloc[i:i+batch_size]
                if self.input_encoder is not None:
                    query = self.input_encoder.transform(query)
                parts.append(self.model.predict(query))
        return np.clip(np.concatenate(parts), 0, 100)
