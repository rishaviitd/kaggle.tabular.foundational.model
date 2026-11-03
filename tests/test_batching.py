"""Concurrent batching behavior without a GPU."""
import concurrent.futures
import importlib.util
import threading
import unittest


@unittest.skipUnless(importlib.util.find_spec('tabctx'), 'tabctx is optional')
class BatchingTests(unittest.TestCase):
    def test_coalesces_and_preserves_order(self):
        from housing_affordability.batching import BatchedHousingService

        class Fake:
            calls = 0

            def predict(self, records):
                self.calls += 1
                return [{'Zip': r['Zip'], 'value': r['value']} for r in records]

        base = Fake()
        service = BatchedHousingService(base, window_ms=50)
        barrier = threading.Barrier(8)

        def call(i):
            barrier.wait()
            return service.predict([{'Zip': f'{i:05d}', 'State': 'CA', 'value': i}])[0]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(call, range(8)))
        self.assertEqual([row['value'] for row in results], list(range(8)))
        self.assertLess(base.calls, 8)

    def test_invalid_inputs_do_not_poison_valid_request(self):
        from housing_affordability.batching import BatchedHousingService

        class Fake:
            def predict(self, records):
                if any(row['value'] < 0 for row in records):
                    raise ValueError('invalid value')
                return records

        service = BatchedHousingService(Fake(), window_ms=50)
        barrier = threading.Barrier(2)

        def call(value):
            barrier.wait()
            try:
                return service.predict([{'Zip': '90210', 'State': 'CA', 'value': value}])
            except ValueError:
                return 'invalid'

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            good, bad = list(pool.map(call, [1, -1]))
        self.assertEqual(good[0]['value'], 1)
        self.assertEqual(bad, 'invalid')


if __name__ == '__main__':
    unittest.main()
