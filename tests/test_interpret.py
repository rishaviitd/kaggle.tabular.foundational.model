"""Small behavioral tests for explanation and feature-group handling."""
import unittest

import numpy as np
import pandas as pd

from housing_affordability.interpret import feature_groups, grouped_permutation_importance, local_group_sensitivity


class InterpretationTests(unittest.TestCase):
    def setUp(self):
        self.x = pd.DataFrame({'ZILLOW_zhvi_2026-06-30': np.arange(20.),
                               'REALTOR_median_listing_price': np.ones(20),
                               'State': pd.Categorical(['A', 'B'] * 10)})
        self.predict = lambda frame: 3*frame['ZILLOW_zhvi_2026-06-30'].to_numpy()

    def test_groups_cover_columns_once(self):
        grouped = [col for group in feature_groups(self.x.columns).values() for col in group]
        self.assertEqual(sorted(grouped), sorted(self.x.columns))

    def test_permutation_ranks_used_feature_above_unused(self):
        result = grouped_permutation_importance(self.predict, self.x, self.predict(self.x), repeats=2)
        score = result.set_index('group').rmse_increase_mean
        self.assertGreater(score['Zillow home values'], 0)
        self.assertEqual(score['Realtor listing prices'], 0)

    def test_local_sensitivity(self):
        result = local_group_sensitivity(self.predict, self.x.iloc[[0]], self.x, samples=10)
        score = result.set_index('group').mean_absolute_change
        self.assertGreater(score['Zillow home values'], 0)
        self.assertEqual(score['Realtor listing prices'], 0)


if __name__ == '__main__':
    unittest.main()
