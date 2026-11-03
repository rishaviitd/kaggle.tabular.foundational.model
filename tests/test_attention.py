"""Numerical equivalence of shared-key attention for rows and heads."""
import unittest

import torch

from housing_affordability.attention import shared_key_sdpa, enable_shared_key_attention


class SharedKeyAttentionTests(unittest.TestCase):
    def test_matches_explicit_repeated_heads(self):
        torch.manual_seed(42)
        for rows in [1, 7]:
            with self.subTest(rows=rows):
                q = torch.randn(2, rows, 6, 8, dtype=torch.float64)
                k = torch.randn(2, 13, 1, 8, dtype=torch.float64)
                v = torch.randn_like(k)
                expected = torch.nn.functional.scaled_dot_product_attention(
                    q.permute(0,2,1,3),
                    k.permute(0,2,1,3).repeat_interleave(6, dim=1),
                    v.permute(0,2,1,3).repeat_interleave(6, dim=1)).permute(0,2,1,3)
                torch.testing.assert_close(shared_key_sdpa(q, k, v), expected)

    def test_rejects_multiple_key_heads(self):
        with self.assertRaises(ValueError):
            shared_key_sdpa(torch.randn(1,1,6,8), torch.randn(1,3,2,8),
                            torch.randn(1,3,2,8))

    def test_installation_is_local_and_preserves_cached_projection(self):
        from types import SimpleNamespace
        from tabpfn.architectures.tabpfn_v2 import TabPFNV2, TabPFNV2Config
        from tabpfn.architectures.kv_cache import KVCacheEntry

        torch.manual_seed(7)
        config = TabPFNV2Config(emsize=24, nhead=3, nlayers=1, max_num_classes=0)
        model = TabPFNV2(config=config, n_out=8)
        layer = model.blocks[0].per_column_attention_between_cells
        other = TabPFNV2(config=config, n_out=8)
        untouched = other.blocks[0].per_column_attention_between_cells.forward
        cache = KVCacheEntry(key=torch.randn(2,11,1,8), value=torch.randn(2,11,1,8))
        query = torch.randn(2,5,24)
        expected = layer(query, cached_kv=cache)[0]
        models = SimpleNamespace(get_devices=lambda: ['cpu'], get=lambda _: model)
        regressor = SimpleNamespace(executor_=SimpleNamespace(kv_caches=[], model_caches=[models]))
        self.assertEqual(enable_shared_key_attention(regressor), 1)
        torch.testing.assert_close(layer(query, cached_kv=cache)[0], expected)
        self.assertEqual(enable_shared_key_attention(regressor), 0)
        self.assertEqual(other.blocks[0].per_column_attention_between_cells.forward, untouched)
