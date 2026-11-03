"""Experiment removing diagnostic barriers; no clear gain was measured."""
from types import FunctionType, MethodType


def disable_diagnostic_barriers(regressor):
    """Keep GPU work on its stream; prediction's final CPU copy still waits.

    This is scoped to one fitted executor, without modifying TabPFN globals.
    The executor's internal forward-time metric becomes enqueue time; external
    latency measurements must still synchronize or consume CPU predictions.
    """
    from tabpfn.inference import InferenceEngineExplicitKVCache, _TimedIterator

    executor = regressor.executor_
    devices = executor.get_devices()
    if not isinstance(executor, InferenceEngineExplicitKVCache):
        raise ValueError('Barrier removal requires the pinned cached executor')
    if len(devices) != 1 or devices[0].type != 'cuda':
        raise ValueError('Barrier removal requires exactly one CUDA device')

    class EnqueueTimer(_TimedIterator):
        def __init__(self, inner, devices):
            super().__init__(inner, devices)
            self._synchronize = False

    original = executor.iter_outputs.__func__
    namespace = dict(original.__globals__, _TimedIterator=EnqueueTimer)
    function = FunctionType(original.__code__, namespace, original.__name__,
                            original.__defaults__, original.__closure__)
    function.__kwdefaults__ = original.__kwdefaults__
    executor.iter_outputs = MethodType(function, executor)
