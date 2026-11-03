"""Contract tests for request routing with a lightweight model stub."""
import unittest
import json
import threading
import urllib.request
import http.client
from http.server import HTTPServer

import numpy as np
import pandas as pd

from housing_affordability.server import HousingService, BatchedHTTPServer, create_handler


class Builder:
    def transform(self, frame):
        return frame


class Model:
    def predict(self, frame):
        return np.full(len(frame), 23.5)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = HousingService.__new__(HousingService)
        self.service.builder = Builder()
        self.service.model = Model()
        self.service.seen = pd.Series([42.], index=pd.MultiIndex.from_tuples([('02134', 'MA')]))

    def test_seen_and_unseen_keep_request_order(self):
        rows = [{'Zip': '02134', 'State': 'MA'}, {'Zip': '90210', 'State': 'CA'}]
        result = self.service.predict(rows)
        self.assertEqual([r['prediction'] for r in result], [42., 23.5])
        self.assertEqual([r['seen'] for r in result], [True, False])

    def test_zip_must_remain_a_five_digit_string(self):
        with self.assertRaises(ValueError):
            self.service.predict([{'Zip': 2134, 'State': 'MA'}])

    def test_http_prediction(self):
        server = HTTPServer(('127.0.0.1', 0), create_handler(self.service))
        thread = threading.Thread(target=server.handle_request)
        thread.start()
        try:
            payload = json.dumps({'records': [{'Zip': '02134', 'State': 'MA'}]}).encode()
            req = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/predict', data=payload,
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=3) as response:
                result = json.load(response)
            self.assertEqual(result['predictions'][0]['prediction'], 42.)
        finally:
            thread.join(timeout=3)
            server.server_close()

    def test_batched_transport_reuses_connection(self):
        server = BatchedHTTPServer(('127.0.0.1', 0), create_handler(self.service, keep_alive=True))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
        payload = json.dumps({'records': [{'Zip': '02134', 'State': 'MA'}]})
        try:
            first_socket = None
            for _ in range(2):
                connection.request('POST', '/predict', body=payload,
                                   headers={'Content-Type': 'application/json'})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read())['predictions'][0]['prediction'], 42.)
                if first_socket is None:
                    first_socket = connection.sock
                self.assertIs(connection.sock, first_socket)
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
