"""Optional same-context request coalescing for the housing HTTP service."""
import json
import threading


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
