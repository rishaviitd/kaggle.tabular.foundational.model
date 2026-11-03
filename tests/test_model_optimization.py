"""Check the model wrapper keeps raw-input predictions after optimization."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from tabpfn.preprocessing.clean import clean_data, clean_data_transform
from tabpfn.preprocessing.datamodel import FeatureSchema

from housing_affordability.model import HousingModel


class ModelOptimizationTests(unittest.TestCase):
    def test_raw_inputs_remain_equivalent_after_optimization_and_refit(self):
        class Inner:
            def fit(self, x, y):
                self.inferred_feature_schema_ = FeatureSchema.from_only_categorical_indices([0],3)
                _, self.ordinal_encoder_, _ = clean_data(x.to_numpy(), self.inferred_feature_schema_)

            def get_inference_config(self):
                return SimpleNamespace(PASSTHROUGH_INF=False)

            def predict(self, x):
                matrix = clean_data_transform(x.to_numpy(), cat_indices=[0], ord_encoder=self.ordinal_encoder_)
                return np.nan_to_num(matrix).sum(axis=1)

        train = pd.DataFrame({'State':pd.Categorical(['MA','CA','MA']),
                               'ZipPrefix3':pd.Categorical(['021','902','021']),
                               'amount':[10.,20.,30.]})
        query = pd.DataFrame({'State':pd.Categorical(['MA',None,'TX']),
                               'ZipPrefix3':pd.Categorical(['021',None,'999']),
                               'amount':[np.nan,15.,17.]})
        wrapper = HousingModel.__new__(HousingModel)
        wrapper.model = Inner()
        wrapper.optimize = wrapper.fast_attention = False
        wrapper.input_encoder = None
        wrapper.fit(train, [1,2,3])
        expected = wrapper.predict(query)
        with patch('housing_affordability.attention.enable_shared_key_attention'):
            wrapper.optimize_for_inference()
            np.testing.assert_array_equal(wrapper.predict(query, batch_size=2), expected)
            wrapper.optimize_for_inference()
            np.testing.assert_array_equal(wrapper.predict(query), expected)
            wrapper.fit(train, [1,2,3])
            np.testing.assert_array_equal(wrapper.predict(query), expected)
