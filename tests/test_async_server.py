"""Check async response routing and input-error isolation without a GPU."""
import asyncio
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('fastapi') and importlib.util.find_spec('httpx'),
                     'Optional serving/test dependencies are not installed')
class AsyncServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        from housing_affordability.async_server import create_async_app
        from housing_affordability.batching import QueuedHousingService

        class Fake:
            calls = 0
            def predict(self, rows):
                self.calls += 1
                if any(row['value'] < 0 for row in rows):
                    raise ValueError('Invalid value')
                return rows
        self.base = Fake()
        self.service = QueuedHousingService(self.base, window_ms=5)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_async_app(self.service)),
                                        base_url='http://housing.test')

    async def asyncTearDown(self):
        await self.client.aclose()
        self.service.close()

    async def test_concurrent_requests_keep_their_own_results(self):
        replies = await asyncio.gather(*(self.client.post('/predict', json={
            'records':[{'Zip':f'{i:05}', 'State':'CA', 'value':i}]}) for i in range(12)))
        self.assertEqual([r.status_code for r in replies], [200]*12)
        self.assertEqual([r.json()['predictions'][0]['value'] for r in replies], list(range(12)))
        self.assertLess(self.base.calls, 12)

    async def test_bad_request_does_not_poison_valid_request(self):
        good, bad = await asyncio.gather(*(self.client.post('/predict', json={
            'records':[{'Zip':'90210', 'State':'CA', 'value':value}]}) for value in [1,-1]))
        self.assertEqual(good.status_code, 200)
        self.assertEqual(bad.status_code, 400)
        malformed = await self.client.post('/predict', json={'records':[{'Zip':90210}]})
        self.assertEqual(malformed.status_code, 400)
