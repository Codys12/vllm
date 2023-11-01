from typing import Optional

import torch
import triton
import triton.language as tl


# Grid: (num_seqs, NUM_KV_HEADS, max_num_partitions)
@triton.jit
def _paged_attn_kernel(
    out_ptr,                                 # [num_seqs, QUERY_GROUP_SIZE * NUM_KV_HEADS, HEAD_SIZE]
    q_ptr,                                   # [num_seqs, QUERY_GROUP_SIZE * NUM_KV_HEADS, HEAD_SIZE]
    k_cache_ptr,                             # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    v_cache_ptr,                             # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    context_lens_ptr,                        # [num_seqs]
    block_tables_ptr,                        # [num_seqs, max_num_blocks_per_seq]
    alibi_slopes_ptr,                        # [QUERY_GROUP_SIZE * NUM_KV_HEADS]
    max_num_blocks_per_seq,
    attn_scale,
    USE_ALIBI: tl.constexpr,
    Q_STRIDE: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    KV_HEAD_STRIDE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    partition_idx = tl.program_id(2)

    context_len = tl.load(context_lens_ptr + seq_idx)
    context_start_idx = partition_idx * PARTITION_SIZE
    if context_start_idx >= context_len:
        # Early exit.
        return
    # This block processes [context_start_idx, context_end_idx).
    context_end_idx = context_start_idx + PARTITION_SIZE
    context_end_idx = tl.minimum(context_end_idx, context_len)

    # Load queries.
    query_offset = seq_idx * Q_STRIDE + kv_head_idx * QUERY_GROUP_SIZE * HEAD_SIZE
    # NOTE(woosuk): Here we assume that QUERY_GROUP_SIZE and HEAD_SIZE are both powers of 2.
    query_offset += tl.arange(0, QUERY_GROUP_SIZE)[:, None] * HEAD_SIZE + tl.arange(0, HEAD_SIZE)[None, :]
    query = tl.load(q_ptr + query_offset)

    # Initialize accumulators.
    m_i = tl.zeros([QUERY_GROUP_SIZE], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([QUERY_GROUP_SIZE], dtype=tl.float32)
    acc = tl.zeros([QUERY_GROUP_SIZE, HEAD_SIZE], dtype=tl.float32)

    # NOTE: KV_BLOCK_SIZE must be >= 16.
    for start_idx in range(context_start_idx, context_end_idx, KV_BLOCK_SIZE):
        start_idx = tl.multiple_of(start_idx, KV_BLOCK_SIZE)
        block_idx = start_idx // KV_BLOCK_SIZE
        block_number = tl.load(block_tables_ptr + seq_idx * max_num_blocks_per_seq + block_idx)

        # Load a key block.
        kv_offset = block_number * KV_BLOCK_STRIDE + kv_head_idx * KV_HEAD_STRIDE
        kv_offset += tl.arange(0, KV_BLOCK_SIZE)[:, None] * HEAD_SIZE + tl.arange(0, HEAD_SIZE)[None, :]
        kv_mask = (start_idx + tl.arange(0, KV_BLOCK_SIZE)[:, None]) < context_len
        key = tl.load(k_cache_ptr + kv_offset, mask=kv_mask, other=0.0)

        # Compute attention.
        if QUERY_GROUP_SIZE == 1:
            # MHA.
            qk = query * key
            qk = tl.sum(qk.to(tl.float32), axis=1)[None, :]
        else:
            # MQA/GQA.
            # NOTE(woosuk): QUERY_GROUP_SIZE must be >= 16.
            # Initialize qk to indicate that it uses FP32.
            qk = tl.zeros([QUERY_GROUP_SIZE, KV_BLOCK_SIZE], dtype=tl.float32)
            qk = tl.dot(query, key.T)
        qk *= attn_scale
        qk = tl.where(start_idx + tl.arange(0, KV_BLOCK_SIZE) < context_len, qk, float("-inf"))

        # Compute m, l, and p.
        m_ij = tl.max(qk, axis=1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij

        # Update accumulators.
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]

        # Load a value block.
        value = tl.load(v_cache_ptr + kv_offset, mask=kv_mask, other=0.0)

        p = p.to(value.dtype)
        if QUERY_GROUP_SIZE == 1:
            # MHA.
            acc += tl.sum((p.T * value).to(tl.float32), axis=0)[None, :]
        else:
            # MQA/GQA.
            # NOTE(woosuk): QUERY_GROUP_SIZE must be >= 16.
            acc += tl.dot(p, value)

        # Update m and l.
        m_i = m_i_new
        l_i = l_i_new

    # NOTE: out_offset can be different from query_offset.
    # NOTE: We assume the out tensor is contiguous.
    out_offset = (seq_idx * NUM_KV_HEADS + kv_head_idx) * QUERY_GROUP_SIZE * HEAD_SIZE
    tl.store(out_ptr + query_offset, acc)


def paged_attention(
    out: torch.Tensor,                  # [num_seqs, NUM_QUERY_GROUPS * QUERY_GROUP_SIZE, HEAD_SIZE]
    query: torch.Tensor,                # [num_seqs, NUM_QUERY_GROUPS * QUERY_GROUP_SIZE, HEAD_SIZE]
    key_cache: torch.Tensor,            # [num_blocks, NUM_QUERY_GROUPS, KV_BLOCK_SIZE, HEAD_SIZE]
    value_cache: torch.Tensor,          # [num_blocks, NUM_QUERY_GROUPS, KV_BLOCK_SIZE, HEAD_SIZE]
    context_lens: torch.Tensor,         # [num_seqs]
    block_tables: torch.Tensor,         # [num_seqs, max_num_blocks_per_seq]
    alibi_slopes: Optional[torch.Tensor], # [NUM_QUERY_GROUPS * QUERY_GROUP_SIZE]
    attn_scale: float,
    max_context_len: int,
    partition_size: int = 512,
) -> None:
    num_seqs = query.shape[0]
    num_kv_heads = key_cache.shape[1]
    kv_block_size = key_cache.shape[2]
    head_size = key_cache.shape[3]
    query_group_size = query.shape[1] // num_kv_heads
    max_num_blocks_per_seq = block_tables.shape[1]
    query_stride = query.stride(0)
    kv_block_stride = key_cache.stride(0)
    kv_head_stride = key_cache.stride(1)
    use_alibi = alibi_slopes is not None

    # TEMP
    max_num_partitions = triton.cdiv(max_context_len, partition_size)
    assert max_num_partitions == 1
    assert not use_alibi

    grid = (num_seqs, num_kv_heads, max_num_partitions)
    _paged_attn_kernel[grid](
        out,
        query,
        key_cache,
        value_cache,
        context_lens,
        block_tables,
        alibi_slopes,
        max_num_blocks_per_seq,
        attn_scale,
        use_alibi,
        query_stride,
        kv_block_stride,
        kv_head_stride,
        head_size,
        query_group_size,
        num_kv_heads,
        kv_block_size,
        partition_size,
    )
    return out



def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


def ref_single_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    block_size = value_cache.shape[2]
    head_size = value_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables = block_tables.cpu().tolist()
    context_lens = context_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size
            k = key_cache[block_number, :, block_offset, :]
            keys.append(k)
            v = value_cache[block_number, :, block_offset, :]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(context_len, device="cuda").int()
            alibi_bias = (position_ids - context_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(
                1, 1, -1)

        out = ref_masked_attention(q, keys, values, scale, alibi_bias)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)


if __name__ == '__main__':
    torch.set_default_dtype(torch.half)
    import random
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    NUM_SEQS = 32
    NUM_QUERY_GROUPS = 12
    QUERY_GROUP_SIZE = 1
    NUM_BLOCKS = 7000
    HEAD_SIZE = 64
    KV_BLOCK_SIZE = 16
    CONTEXT_LENS = [random.randint(1, 512) for _ in range(NUM_SEQS)]
    MAX_NUM_BLOCKS_PER_SEQ = (max(CONTEXT_LENS) + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE

    attn_scale = HEAD_SIZE ** -0.5
    q = torch.empty(NUM_SEQS, NUM_QUERY_GROUPS * QUERY_GROUP_SIZE, HEAD_SIZE).cuda()
    q.uniform_(-attn_scale, attn_scale)
    out = torch.empty_like(q)

    k_cache = torch.empty(NUM_BLOCKS, NUM_QUERY_GROUPS, KV_BLOCK_SIZE, HEAD_SIZE).cuda()
    k_cache.uniform_(-attn_scale, attn_scale)
    v_cache = torch.empty_like(k_cache)
    v_cache.uniform_(-attn_scale, attn_scale)

    block_tables = torch.randint(0, NUM_BLOCKS, (NUM_SEQS, MAX_NUM_BLOCKS_PER_SEQ)).cuda()
    alibi_slopes = None
    context_lens = torch.tensor(CONTEXT_LENS, dtype=torch.int32).cuda()
    max_context_len = max(CONTEXT_LENS)

    paged_attention(
        out,
        q,
        k_cache,
        v_cache,
        context_lens,
        block_tables,
        alibi_slopes,
        attn_scale,
        max_context_len,
    )

    ref_out = torch.empty_like(out)
    ref_single_query_cached_kv_attention(
        ref_out,
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        attn_scale,
        alibi_slopes,
    )
    print(torch.allclose(out, ref_out, atol=1e-3))
    print(torch.max(torch.abs(out - ref_out)))
