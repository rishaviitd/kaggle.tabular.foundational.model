"""Check that diagnostic barrier removal stays local to one executor."""
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from benchmarks.execution import disable_diagnostic_barriers


class ExecutionTests(unittest.TestCase):
    def test_single_device_iterator_keeps_values_without_global_barrier(self):
        from tabpfn.inference import InferenceEngineExplicitKVCache
        executor = InferenceEngineExplicitKVCache.__new__(InferenceEngineExplicitKVCache)
        executor.get_devices = lambda: [torch.device('cuda:0')]
        def outputs(self):
            yield from _TimedIterator(iter([1, 2]), self.get_devices())
        executor.iter_outputs = MethodType(outputs, executor)
        regressor = SimpleNamespace(executor_=executor)
        disable_diagnostic_barriers(regressor)
        with patch('torch.cuda.synchronize') as barrier:
            self.assertEqual(list(executor.iter_outputs()), [1, 2])
            barrier.assert_not_called()
        self.assertNotIn('_TimedIterator', outputs.__globals__)

    def test_rejects_cpu_executor(self):
        from tabpfn.inference import InferenceEngineExplicitKVCache
        executor = InferenceEngineExplicitKVCache.__new__(InferenceEngineExplicitKVCache)
        executor.get_devices = lambda: [torch.device('cpu')]
        with self.assertRaises(ValueError):
            disable_diagnostic_barriers(SimpleNamespace(executor_=executor))
