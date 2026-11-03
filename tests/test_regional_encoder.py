"""Ensure cached regional features match the reference implementation."""
import unittest

import numpy as np
import pandas as pd

from housing_affordability.aggregated_features import RegionalEncoder, enc, Y


class RegionalEncoderTests(unittest.TestCase):
    def test_matches_reference_for_known_missing_and_new_regions(self):
        reference = pd.DataFrame({'Zip': ['02134', '02135', '90210'],
                                  'State': ['MA', 'MA', 'CA'],
                                  'City': ['Boston', 'Boston', None],
                                  'Metro': ['Boston', 'Boston', None],
                                  Y: [30., 50., 90.]})
        query = pd.DataFrame({'Zip': ['02136', '90211', '99999'],
                              'State': ['MA', 'CA', 'AK'],
                              'City': ['Boston', None, 'New'],
                              'Metro': ['Boston', None, 'New']})
        fitted = RegionalEncoder(reference)
        np.testing.assert_array_equal(fitted.transform(query), enc(reference, query))
        np.testing.assert_array_equal(fitted.transform(query.iloc[[2, 0]]),
                                      enc(reference, query.iloc[[2, 0]]))
