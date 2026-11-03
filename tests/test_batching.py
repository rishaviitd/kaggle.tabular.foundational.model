"""Concurrent batching behavior without a GPU."""
import concurrent.futures
import importlib.util
import threading
import time
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


class QueueTests(unittest.TestCase):
    def test_batches_while_worker_is_busy_and_isolates_errors(self):
        from housing_affordability.batching import QueuedHousingService

        class Fake:
            def __init__(self):
                self.calls = []
                self.started = threading.Event()

            def predict(self, rows):
                self.calls.append(len(rows))
                self.started.set()
                time.sleep(.04)
                if any(row['value'] < 0 for row in rows):
                    raise ValueError('invalid value')
                return rows

        base = Fake()
        service = QueuedHousingService(base, window_ms=5)
        def call(i):
            return service.predict([{'Zip': f'{i:05}', 'State': 'CA', 'value': i}])
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            first = pool.submit(call, 0)
            self.assertTrue(base.started.wait(1))
            rest = [pool.submit(call, i) for i in range(1, 6)]
            self.assertEqual(first.result()[0]['value'], 0)
            self.assertEqual([f.result()[0]['value'] for f in rest], list(range(1, 6)))
        self.assertEqual(base.calls, [1, 5])

        barrier = threading.Barrier(2)
        def checked(i):
            barrier.wait()
            return service.predict([{'Zip': '90210', 'State': 'CA', 'value': i}])
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            good = pool.submit(checked, 1)
            bad = pool.submit(checked, -1)
            self.assertEqual(good.result()[0]['value'], 1)
            with self.assertRaises(ValueError):
                bad.result()


if __name__ == '__main__':
    unittest.main()
