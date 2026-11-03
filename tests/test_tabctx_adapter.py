"""Optional tabctx adapter tests; no GPU or checkpoint required."""
import importlib.util
import unittest

import numpy as np
import pandas as pd


@unittest.skipUnless(importlib.util.find_spec('tabctx'), 'tabctx is optional')
class TabctxAdapterTests(unittest.TestCase):
    def test_categorical_round_trip(self):
        from benchmarks.tabctx_t4 import HousingBackend, encode

        x = pd.DataFrame({'numeric': [1., np.nan], 'location': pd.Categorical(['NY', 'CA'])})
        backend = HousingBackend.__new__(HousingBackend)
        backend.template = x.iloc[:0].copy()
        pd.testing.assert_frame_equal(x, backend.frame(encode(x)))

    def test_benchmark_estimator_rejects_unmeasured_shapes(self):
        from benchmarks.tabctx_t4 import T4BenchmarkEstimator

        estimator = T4BenchmarkEstimator()
        self.assertTrue(estimator.admit(6830, 32, 106))
        with self.assertRaises(ValueError):
            estimator.admit(6830, 32, 107)


if __name__ == '__main__':
    unittest.main()
