"""Shared-key attention for the pinned TabPFN v2 cached inference path."""
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def shared_key_sdpa(query, key, value):
    """Avoid copying one shared key/value head for every query head.

    Inputs and output use (batch, sequence, heads, head_dim). Each query/head
    pair attends independently to the same keys, so folding heads into the
    query sequence leaves the attention operation unchanged.
    """
    if key.shape[2] != 1 or value.shape[2] != 1:
        raise ValueError('Shared-key attention requires exactly one K/V head')
    b, s, h, d = query.shape
    q = query.reshape(b, s*h, d).unsqueeze(1)
    k, v = key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3)
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        result = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return result.squeeze(1).reshape(b, s, h, d)


def enable_shared_key_attention(regressor):
    """Install on this fitted model only; training and other models are untouched."""
    from tabpfn.architectures.tabpfn_v2 import TabPFNV2

    executor = regressor.executor_
    if not hasattr(executor, 'kv_caches'):
        raise ValueError('Shared-key attention requires fit_with_cache')
    installed = 0
    for model_cache in executor.model_caches:
        for device in model_cache.get_devices():
            model = model_cache.get(device)
            if not isinstance(model, TabPFNV2):
                raise ValueError('Shared-key attention supports the pinned TabPFN v2 only')
            for block in model.blocks:
                layer = block.per_column_attention_between_cells
                if getattr(layer, '_housing_shared_key', False):
                    continue
                original = layer.forward

                def forward(x, single_eval_pos=None, *, cached_kv=None, return_kv=False,
                            _layer=layer, _original=original):
                    if cached_kv is None or return_kv:
                        return _original(x, single_eval_pos, cached_kv=cached_kv,
                                         return_kv=return_kv)
                    b, s, _ = x.shape
                    q = _layer.q_projection(x).view(b, s, -1, _layer.head_dim)
                    k, v = cached_kv.key, cached_kv.value
                    if k.dtype != q.dtype:
                        k, v = k.to(q.dtype), v.to(q.dtype)
                    out = shared_key_sdpa(q, k, v).reshape(b, s, -1)
                    return _layer.out_projection(out), None

                layer.forward = forward
                layer._housing_shared_key = True
                installed += 1
    return installed
