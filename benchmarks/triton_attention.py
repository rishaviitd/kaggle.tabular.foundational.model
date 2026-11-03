"""Experimental T4 shared-key kernel; online softmax as in Triton's tutorial.

Reference: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
This supports inference only, FP16, one shared key/value head, no mask/dropout.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _attention(Q, K, V, O, P, MAX, SUM, M: tl.constexpr, N: tl.constexpr,
               D: tl.constexpr, SPLITS: tl.constexpr, BM: tl.constexpr,
               BN: tl.constexpr, SCALE: tl.constexpr):
    tile, batch, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = tile*BM + tl.arange(0, BM)
    d = tl.arange(0, D)
    q = tl.load(Q+batch*M*D+m[:,None]*D+d[None,:], m[:,None]<M, 0)
    maximum = tl.full((BM,), -float('inf'), tl.float32)
    total = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM,D), tl.float32)
    chunk: tl.constexpr = triton.cdiv(N, SPLITS)
    for start in range(0, chunk, BN):
        n = part*chunk + start + tl.arange(0, BN)
        valid = (n<N) & (n<(part+1)*chunk)
        key = tl.load(K+batch*N*D+n[None,:]*D+d[:,None], valid[None,:], 0)
        value = tl.load(V+batch*N*D+n[:,None]*D+d[None,:], valid[:,None], 0)
        logits = tl.dot(q, key)*SCALE
        logits = tl.where(valid[None,:], logits, -float('inf'))
        next_max = tl.maximum(maximum, tl.max(logits, 1))
        alpha = tl.exp2(maximum-next_max)
        probability = tl.exp2(logits-next_max[:,None])
        total = total*alpha + tl.sum(probability, 1)
        acc = acc*alpha[:,None]
        acc = tl.dot(probability.to(tl.float16), value, acc)
        maximum = next_max
    if SPLITS == 1:
        result = acc/total[:,None]
        tl.store(O+batch*M*D+m[:,None]*D+d[None,:], result, m[:,None]<M)
    else:
        tl.store(P+(batch*SPLITS+part)*M*D+m[:,None]*D+d[None,:], acc, m[:,None]<M)
        tl.store(MAX+(batch*SPLITS+part)*M+m, maximum, m<M)
        tl.store(SUM+(batch*SPLITS+part)*M+m, total, m<M)


@triton.jit
def _reduce(P, MAX, SUM, O, M: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr):
    row, batch = tl.program_id(0), tl.program_id(1)
    part, d = tl.arange(0,SPLITS), tl.arange(0,D)
    maximum = tl.load(MAX+(batch*SPLITS+part)*M+row)
    total = tl.load(SUM+(batch*SPLITS+part)*M+row)
    alpha = tl.exp2(maximum-tl.max(maximum,0))
    acc = tl.load(P+(batch*SPLITS+part[:,None])*M*D+row*D+d[None,:])
    result = tl.sum(acc*alpha[:,None],0)/tl.sum(total*alpha,0)
    tl.store(O+batch*M*D+row*D+d, result)


def shared_attention(query, key, value, query_block=32, key_block=64, warps=4):
    b,s,h,d = query.shape
    if query.dtype != torch.float16 or d != 32 or key.shape[2] != 1:
        raise ValueError('Experimental kernel requires FP16, head_dim=32, one K/V head')
    q = query.reshape(b,s*h,d).contiguous()
    k, v = key.squeeze(2).contiguous(), value.squeeze(2).contiguous()
    m, n = s*h, k.shape[1]
    splits = 8 if m <= 16 else 1
    bm = 16 if m <= 16 else query_block
    out = torch.empty_like(q)
    partial = torch.empty((b,splits,m,d), device=q.device, dtype=torch.float32) if splits>1 else out
    maxima = torch.empty((b,splits,m), device=q.device, dtype=torch.float32) if splits>1 else out
    totals = torch.empty_like(maxima) if splits>1 else out
    _attention[(triton.cdiv(m,bm),b,splits)](q,k,v,out,partial,maxima,totals,m,n,d,splits,bm,key_block,
                                          d**-.5*1.4426950408889634,num_warps=warps,num_stages=2)
    if splits>1:
        _reduce[(m,b)](partial,maxima,totals,out,m,d,splits,num_warps=4)
    return out.reshape(b,s,h,d)
