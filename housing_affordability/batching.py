"""Optional same-context request coalescing for the housing HTTP service."""
import json
import threading
import time
from collections import deque
from concurrent.futures import Future


class BatchedHousingService:
    """Coalesce concurrent requests, while keeping one in-flight GPU call.

    The tabctx coalescer is used directly; this is not tabctx's full HTTP
    server. Invalid inputs are retried individually by tabctx so one malformed
    request does not poison valid requests in the same batch.
    """
    def __init__(self, service, window_ms=5):
        if window_ms <= 0:
            raise ValueError('window_ms must be positive')
        try:
            from tabctx.batching import CoalescingPredictor
            from tabctx.errors import InvalidInputError
            from tabctx.types import PredictOutcome
        except ImportError as error:
            raise RuntimeError('Install the optional batching dependency: pip install -e ".[batching]"') from error

        class Engine:
            def __init__(self):
                self.lock = threading.Lock()

            def predict(self, dataset_id, records, return_proba=False):
                if not dataset_id.startswith('housing:') or return_proba:
                    raise InvalidInputError('Unsupported prediction request')
                try:
                    with self.lock:
                        return PredictOutcome(predictions=service.predict(records))
                except (ValueError, KeyError, TypeError) as error:
                    raise InvalidInputError(str(error)) from error

        self._predictor = CoalescingPredictor(Engine(), window_s=window_ms/1000)
        self._invalid = InvalidInputError

    def predict(self, records):
        if not isinstance(records, list) or not 0 < len(records) <= 2048:
            raise ValueError('records must be a list of 1 to 2048 rows')
        if any(not isinstance(row, dict) or not isinstance(row.get('Zip'), str)
               or len(row['Zip']) != 5 or not row['Zip'].isdigit()
               or not isinstance(row.get('State'), str) for row in records):
            raise ValueError('Each row requires a five-digit string Zip and a string State')
        # Do not combine requests with different schemas: pandas can otherwise
        # fill a missing field from another request and change validation.
        signature = sorted({tuple(sorted(row)) for row in records})
        dataset_id = 'housing:' + json.dumps(signature, separators=(',', ':'))
        try:
            return self._predictor.predict(dataset_id, records).predictions
        except self._invalid as error:
            raise ValueError(str(error)) from error


class QueuedHousingService:
    """Accumulate requests while a single worker is running inference."""
    def __init__(self, service, window_ms=20):
        if window_ms <= 0:
            raise ValueError('window_ms must be positive')
        self.service = service
        self.window = window_ms / 1000
        self.condition = threading.Condition()
        self.pending = deque()
        self.closed = False
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def close(self):
        """Drain accepted requests and release the worker's model reference."""
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.worker.join()

    def predict(self, records):
        return self.submit(records).result()

    def submit(self, records):
        """Submit without blocking an HTTP event loop."""
        if not isinstance(records, list) or not 0 < len(records) <= 2048:
            raise ValueError('records must be a list of 1 to 2048 rows')
        if any(not isinstance(row, dict) or not isinstance(row.get('Zip'), str)
               or len(row['Zip']) != 5 or not row['Zip'].isdigit()
               or not isinstance(row.get('State'), str) for row in records):
            raise ValueError('Each row requires a five-digit string Zip and a string State')
        signature = tuple(sorted({tuple(sorted(row)) for row in records}))
        future = Future()
        with self.condition:
            if self.closed:
                raise RuntimeError('Prediction queue is closed')
            self.pending.append((signature, records, future, time.monotonic()))
            self.condition.notify()
        return future

    def _run(self):
        while True:
            with self.condition:
                while not self.pending:
                    if self.closed:
                        return
                    self.condition.wait()
                signature, records, future, arrived = self.pending.popleft()
                deadline = arrived + self.window
                while time.monotonic() < deadline:
                    self.condition.wait(timeout=max(0, deadline - time.monotonic()))
                members = [(records, future)]
                remaining = deque()
                count = len(records)
                while self.pending:
                    key, rows, result, timestamp = self.pending.popleft()
                    if key == signature and count + len(rows) <= 2048:
                        members.append((rows, result))
                        count += len(rows)
                    else:
                        remaining.append((key, rows, result, timestamp))
                self.pending = remaining
            members = [(rows, future) for rows, future in members
                       if future.set_running_or_notify_cancel()]
            if members:
                self._execute(members)

    def _execute(self, members):
        try:
            predictions = self.service.predict([row for rows, _ in members for row in rows])
            if len(predictions) != sum(len(rows) for rows, _ in members):
                raise RuntimeError('Model returned an unexpected prediction count')
        except (ValueError, KeyError, TypeError):
            for rows, future in members:
                try:
                    future.set_result(self.service.predict(rows))
                except Exception as error:
                    future.set_exception(error)
            return
        except Exception as error:
            for _, future in members:
                future.set_exception(error)
            return
        offset = 0
        for rows, future in members:
            future.set_result(predictions[offset:offset + len(rows)])
            offset += len(rows)
