"""Match TabPFN cleaning for known, missing, and unknown housing categories."""
import unittest

import numpy as np
import pandas as pd

from housing_affordability.input_encoder import HousingInputEncoder
from tabpfn.preprocessing.clean import clean_data, clean_data_transform
from tabpfn.preprocessing.datamodel import FeatureSchema


class InputEncoderTests(unittest.TestCase):
    def test_preserves_missing_unknown_and_infinite_values(self):
        train = pd.DataFrame({'State':pd.Categorical(['MA','CA','MA']),
                               'ZipPrefix3':pd.Categorical(['021','902','021']),
                               'amount':[10.,20.,30.]})
        schema = FeatureSchema.from_only_categorical_indices([0], 3)
        _, encoder, _ = clean_data(train.to_numpy(), schema)
        query = pd.DataFrame({'State':pd.Categorical(['MA',None,'TX']),
                              'ZipPrefix3':pd.Categorical(['021',None,'999']),
                              'amount':[np.nan,np.inf,17.]})
        expected = clean_data_transform(query.to_numpy(), cat_indices=[0],
                                        ord_encoder=encoder, passthrough_inf=True)
        codec = HousingInputEncoder(encoder, [0], train.columns, passthrough_inf=True)
        actual = codec.transform(query)
        np.testing.assert_allclose(actual.to_numpy(), expected, rtol=0, atol=0, equal_nan=True)
        self.assertEqual(actual.columns.tolist(), train.columns.tolist())
        self.assertTrue(pd.isna(query.loc[1,'State']))

    def test_rejects_reordered_columns(self):
        codec = HousingInputEncoder(None, [], pd.Index(['a','b']))
        with self.assertRaises(ValueError):
            codec.transform(pd.DataFrame({'b':[1.], 'a':[2.]}))
