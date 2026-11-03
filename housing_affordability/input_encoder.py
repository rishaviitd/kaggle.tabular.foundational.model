"""Reuse the fitted housing schema instead of re-inferring pandas dtypes."""
import numpy as np
import pandas as pd


class HousingInputEncoder:
    def __init__(self, ordinal_encoder, categorical_indices, columns, passthrough_inf=False):
        self.encoder = ordinal_encoder
        self.categorical_indices = set(categorical_indices)
        self.columns = columns.copy()
        self.passthrough_inf = passthrough_inf

    def transform(self, frame):
        from tabpfn.preprocessing.clean import process_text_na_dataframe

        if not isinstance(frame, pd.DataFrame) or not frame.columns.equals(self.columns):
            raise ValueError('Optimized input must preserve the fitted housing columns and order')
        frame = frame.copy(deep=False)
        frame.columns = range(frame.shape[1])
        for col, value in (getattr(self.encoder, 'constant_values_', None) or {}).items():
            frame[col] = value
        for col in self.categorical_indices:
            frame[col] = frame[col].astype('category')
        for col in frame:
            if isinstance(frame[col].dtype, pd.CategoricalDtype) and col not in self.categorical_indices:
                # TabPFN's generic path sees these as strings after ndarray
                # conversion. Preserve its missing/unknown-value distinction.
                frame[col] = frame[col].astype('string')
        encoded = process_text_na_dataframe(frame, ord_encoder=self.encoder,
                                            passthrough_inf=self.passthrough_inf)
        return pd.DataFrame(encoded.astype(np.float64, copy=False), columns=self.columns)
