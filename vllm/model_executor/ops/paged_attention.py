from typing import Optional

import torch
import triton
import triton.language as tl


# Grid: (num_seqs, NUM_KV_HEADS, max_num_partitions)
@triton.jit
def _paged_attn_mha_kernel(
    m_i_ptr,                                 # [num_seqs, NUM_KV_HEADS, max_num_partitions]
    l_i_ptr,                                 # [num_seqs, NUM_KV_HEADS, max_num_partitions]
    out_ptr,                                 # [num_seqs, NUM_KV_HEADS, max_num_partitions, HEAD_SIZE]
    q_ptr,                                   # [num_seqs, NUM_KV_HEADS, HEAD_SIZE]
    k_cache_ptr,                             # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    v_cache_ptr,                             # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    context_lens_ptr,                        # [num_seqs]
    block_tables_ptr,                        # [num_seqs, max_num_blocks_per_seq]
    alibi_slopes_ptr,                        # [NUM_KV_HEADS]
    max_num_blocks_per_seq,
    attn_scale,
    USE_ALIBI: tl.constexpr,
    Q_STRIDE: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    KV_HEAD_STRIDE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    partition_idx = tl.program_id(2)
    max_num_partitions = tl.num_programs(2)

    USE_PARTITIONING = PARTITION_SIZE > 0
    context_len = tl.load(context_lens_ptr + seq_idx)
    # This block processes [context_start_idx, context_end_idx).
    if USE_PARTITIONING:
        context_start_idx = partition_idx * PARTITION_SIZE
        if context_start_idx >= context_len:
            return
        context_end_idx = tl.minimum(context_start_idx + PARTITION_SIZE, context_len)
    else:
        context_start_idx = 0
        context_end_idx = context_len

    # Define offsets.
    block_offset = tl.arange(0, KV_BLOCK_SIZE)
    # NOTE(woosuk): Here we assume HEAD_SIZE is a power of 2.
    head_offset = tl.arange(0, HEAD_SIZE)
    kv_offset = kv_head_idx * KV_HEAD_STRIDE
    kv_offset += block_offset[:, None] * HEAD_SIZE + head_offset[None, :]

    # Load queries.
    query_offset = seq_idx * Q_STRIDE + kv_head_idx * HEAD_SIZE
    query_offset += head_offset
    # query: [1, HEAD_SIZE]
    query = tl.load(q_ptr + query_offset)[None, :]

    # Initialize accumulators.
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_SIZE], dtype=tl.float32)

    for start_idx in range(context_start_idx, context_end_idx, KV_BLOCK_SIZE):
        start_idx = tl.multiple_of(start_idx, KV_BLOCK_SIZE)
        block_idx = start_idx // KV_BLOCK_SIZE
        block_number = tl.load(block_tables_ptr + seq_idx * max_num_blocks_per_seq + block_idx)

        # Load a key block.
        kv_block_offset = block_number * KV_BLOCK_STRIDE + kv_offset
        mask_offset = start_idx + block_offset
        kv_mask = mask_offset[:, None] < context_len
        # key: [KV_BLOCK_SIZE, HEAD_SIZE]
        key = tl.load(k_cache_ptr + kv_block_offset, mask=kv_mask, other=0.0)

        # Compute attention.
        # qk: [KV_BLOCK_SIZE]
        qk = tl.sum(query * key, axis=1)
        qk *= attn_scale
        qk = tl.where(mask_offset < context_len, qk, float("-inf"))

        # Compute m, l, and p.
        # m_ij: [1]
        m_ij = tl.max(qk, axis=0)
        # m_i_new: [1]
        m_i_new = tl.maximum(m_i, m_ij)

        # p: [KV_BLOCK_SIZE]
        p = tl.exp(qk - m_i_new)
        # alpha: [1]
        alpha = tl.exp(m_i - m_i_new)
        acc *= alpha

        # Load a value block.
        # value: [KV_BLOCK_SIZE, HEAD_SIZE]
        value = tl.load(v_cache_ptr + kv_block_offset, mask=kv_mask, other=0.0)
        acc += tl.sum(p[:, None] * value, axis=0)

        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_i_new
    acc = acc / l_i

    # Store the current partition's m and l for later reduction.
    if USE_PARTITIONING:
        partition_offset = (seq_idx * NUM_KV_HEADS + kv_head_idx) * max_num_partitions
        partition_offset += partition_idx
        tl.store(m_i_ptr + partition_offset, m_i)
        tl.store(l_i_ptr + partition_offset, l_i)

    # NOTE: Unlike the query tensor, we assume the out tensor is contiguous.
    out_offset = (seq_idx * NUM_KV_HEADS + kv_head_idx) * max_num_partitions * HEAD_SIZE
    out_offset += partition_idx * HEAD_SIZE
    out_offset += head_offset
    tl.store(out_ptr + out_offset, acc)


# Grid: (num_seqs, NUM_KV_HEADS, max_num_partitions)
@triton.jit
def _paged_attn_kernel(
    m_i_ptr,                                 # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    l_i_ptr,                                 # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    out_ptr,                                 # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE, HEAD_SIZE]
    q_ptr,                                   # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    k_cache_ptr,                             # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    v_cache_ptr,                             # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    context_lens_ptr,                        # [num_seqs]
    block_tables_ptr,                        # [num_seqs, max_num_blocks_per_seq]
    alibi_slopes_ptr,                        # [NUM_KV_HEADS * QUERY_GROUP_SIZE]
    max_num_blocks_per_seq,
    attn_scale,
    USE_ALIBI: tl.constexpr,
    Q_STRIDE: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    KV_HEAD_STRIDE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    PADDED_QUERY_GROUP_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    partition_idx = tl.program_id(2)
    max_num_partitions = tl.num_programs(2)

    USE_PARTITIONING = PARTITION_SIZE > 0
    context_len = tl.load(context_lens_ptr + seq_idx)
    # This block processes [context_start_idx, context_end_idx).
    if USE_PARTITIONING:
        context_start_idx = partition_idx * PARTITION_SIZE
        if context_start_idx >= context_len:
            # Early exit.
            return
        context_end_idx = tl.minimum(context_start_idx + PARTITION_SIZE, context_len)
    else:
        context_start_idx = 0
        context_end_idx = context_len

    # Load queries.
    query_offset = seq_idx * Q_STRIDE + kv_head_idx * QUERY_GROUP_SIZE * HEAD_SIZE
    # NOTE(woosuk): Here we assume HEAD_SIZE is a power of 2.
    query_offset += tl.arange(0, PADDED_QUERY_GROUP_SIZE)[:, None] * HEAD_SIZE + tl.arange(0, HEAD_SIZE)[None, :]
    group_mask = tl.arange(0, PADDED_QUERY_GROUP_SIZE)[:, None] < QUERY_GROUP_SIZE
    # query: [PADDED_QUERY_GROUP_SIZE, HEAD_SIZE]
    query = tl.load(q_ptr + query_offset, mask=group_mask, other=0.0)

    # Initialize accumulators.
    m_i = tl.zeros([PADDED_QUERY_GROUP_SIZE], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([PADDED_QUERY_GROUP_SIZE], dtype=tl.float32)
    acc = tl.zeros([PADDED_QUERY_GROUP_SIZE, HEAD_SIZE], dtype=tl.float32)

    # NOTE: KV_BLOCK_SIZE must be >= 16.
    for start_idx in range(context_start_idx, context_end_idx, KV_BLOCK_SIZE):
        start_idx = tl.multiple_of(start_idx, KV_BLOCK_SIZE)
        block_idx = start_idx // KV_BLOCK_SIZE
        block_number = tl.load(block_tables_ptr + seq_idx * max_num_blocks_per_seq + block_idx)

        # Load a key block.
        kv_offset = block_number * KV_BLOCK_STRIDE + kv_head_idx * KV_HEAD_STRIDE
        kv_offset += tl.arange(0, KV_BLOCK_SIZE)[:, None] * HEAD_SIZE + tl.arange(0, HEAD_SIZE)[None, :]
        kv_mask = (start_idx + tl.arange(0, KV_BLOCK_SIZE)[:, None]) < context_len
        # key: [KV_BLOCK_SIZE, HEAD_SIZE]
        key = tl.load(k_cache_ptr + kv_offset, mask=kv_mask, other=0.0)

        # Compute attention.
        if PADDED_QUERY_GROUP_SIZE == 1:
            # MHA.
            # query: [1, HEAD_SIZE]
            # qk: [KV_BLOCK_SIZE, HEAD_SIZE]
            qk = query * key
            # qk: [1, KV_BLOCK_SIZE]
            qk = tl.sum(qk.to(tl.float32), axis=1)[None, :]
        else:
            # MQA/GQA.
            # qk: [PADDED_QUERY_GROUP_SIZE, KV_BLOCK_SIZE]
            qk = tl.dot(query, key.T, out_dtype=tl.float32)
        # qk: [PADDED_QUERY_GROUP_SIZE, KV_BLOCK_SIZE]
        qk *= attn_scale
        qk = tl.where(start_idx + tl.arange(0, KV_BLOCK_SIZE) < context_len, qk, float("-inf"))

        # Compute m, l, and p.
        # m_ij: [PADDED_QUERY_GROUP_SIZE]
        m_ij = tl.max(qk, axis=1)
        # m_i_new: [PADDED_QUERY_GROUP_SIZE]
        m_i_new = tl.maximum(m_i, m_ij)

        # p: [PADDED_QUERY_GROUP_SIZE, KV_BLOCK_SIZE]
        p = tl.exp(qk - m_i_new[:, None])
        # alpha: [PADDED_QUERY_GROUP_SIZE]
        alpha = tl.exp(m_i - m_i_new)
        acc *= alpha[:, None]

        # Load a value block.
        # value: [KV_BLOCK_SIZE, HEAD_SIZE]
        value = tl.load(v_cache_ptr + kv_offset, mask=kv_mask, other=0.0)

        p = p.to(value.dtype)
        if PADDED_QUERY_GROUP_SIZE == 1:
            # MHA.
            acc += tl.sum((p.T * value).to(tl.float32), axis=0)[None, :]
        else:
            # MQA/GQA.
            acc += tl.dot(p, value, out_dtype=tl.float32)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    acc = acc / l_i[:, None]

    # Store the current partition's m and l for later reduction.
    if USE_PARTITIONING:
        partition_offset = (seq_idx * NUM_KV_HEADS + kv_head_idx) * max_num_partitions * QUERY_GROUP_SIZE
        partition_offset += partition_idx * QUERY_GROUP_SIZE
        partition_offset += tl.arange(0, PADDED_QUERY_GROUP_SIZE)
        mask = tl.arange(0, PADDED_QUERY_GROUP_SIZE) < QUERY_GROUP_SIZE
        tl.store(m_i_ptr + partition_offset, m_i, mask=mask)
        tl.store(l_i_ptr + partition_offset, l_i, mask=mask)

    # NOTE: Unlike the query tensor, we assume the out tensor is contiguous.
    out_offset = (seq_idx * NUM_KV_HEADS + kv_head_idx) * max_num_partitions * QUERY_GROUP_SIZE * HEAD_SIZE
    out_offset += partition_idx * QUERY_GROUP_SIZE * HEAD_SIZE
    out_offset += tl.arange(0, PADDED_QUERY_GROUP_SIZE)[:, None] * HEAD_SIZE + tl.arange(0, HEAD_SIZE)[None, :]
    group_mask = tl.arange(0, PADDED_QUERY_GROUP_SIZE)[:, None] < QUERY_GROUP_SIZE
    tl.store(out_ptr + out_offset, acc, mask=group_mask) 


# Grid: (num_seqs, NUM_KV_HEADS, 1)
@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=1),
        triton.Config({}, num_stages=2, num_warps=1),
        triton.Config({}, num_stages=3, num_warps=1),
        triton.Config({}, num_stages=1, num_warps=2),
        triton.Config({}, num_stages=2, num_warps=2),
        triton.Config({}, num_stages=3, num_warps=2),
        triton.Config({}, num_stages=1, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=3, num_warps=4),
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=3, num_warps=8),
    ],
    key=[],
)
@triton.jit
def _paged_attn_v1_kernel(
    out_ptr,
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    context_lens_ptr,
    block_tables_ptr,
    alibi_slopes_ptr,
    max_num_blocks_per_seq,
    attn_scale,
    USE_ALIBI: tl.constexpr,
    Q_STRIDE: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    KV_HEAD_STRIDE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    PADDED_QUERY_GROUP_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
):
    # NOTE: The first two inputs are unused.
    if PADDED_QUERY_GROUP_SIZE == 1:
        _paged_attn_mha_kernel(
            out_ptr, out_ptr, out_ptr, q_ptr, k_cache_ptr, v_cache_ptr, context_lens_ptr, block_tables_ptr,
            alibi_slopes_ptr, max_num_blocks_per_seq, attn_scale, USE_ALIBI, Q_STRIDE, KV_BLOCK_STRIDE,
            KV_HEAD_STRIDE, HEAD_SIZE, NUM_KV_HEADS, KV_BLOCK_SIZE, PARTITION_SIZE=0,
        )
    else:
        _paged_attn_kernel(
            out_ptr, out_ptr, out_ptr, q_ptr, k_cache_ptr, v_cache_ptr, context_lens_ptr, block_tables_ptr,
            alibi_slopes_ptr, max_num_blocks_per_seq, attn_scale, USE_ALIBI, Q_STRIDE, KV_BLOCK_STRIDE,
            KV_HEAD_STRIDE, HEAD_SIZE, QUERY_GROUP_SIZE, PADDED_QUERY_GROUP_SIZE, NUM_KV_HEADS, KV_BLOCK_SIZE, PARTITION_SIZE=0,
        )


# Grid: (num_seqs, NUM_KV_HEADS, max_num_partitions)
@triton.jit
def _paged_attn_v2_kernel(
    m_i_ptr,
    l_i_ptr,
    tmp_out_ptr,
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    context_lens_ptr,
    block_tables_ptr,
    alibi_slopes_ptr,
    max_num_blocks_per_seq,
    attn_scale,
    USE_ALIBI: tl.constexpr,
    Q_STRIDE: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    KV_HEAD_STRIDE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    PADDED_QUERY_GROUP_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
):
    _paged_attn_kernel(
        m_i_ptr, l_i_ptr, tmp_out_ptr, q_ptr, k_cache_ptr, v_cache_ptr, context_lens_ptr, block_tables_ptr,
        alibi_slopes_ptr, max_num_blocks_per_seq, attn_scale, USE_ALIBI, Q_STRIDE, KV_BLOCK_STRIDE,
        KV_HEAD_STRIDE, HEAD_SIZE, QUERY_GROUP_SIZE, PADDED_QUERY_GROUP_SIZE, NUM_KV_HEADS, KV_BLOCK_SIZE, PARTITION_SIZE,
    )


# Grid: (num_seqs, NUM_KV_HEADS)
@triton.jit
def _paged_attn_v2_reduce_kernel(
    out_ptr,                                 # [num_seqs, NUM_KV_HEADS, QUERY_GROUP_SIZE, HEAD_SIZE]
    m_i_ptr,                                 # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    l_i_ptr,                                 # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE]
    tmp_out_ptr,                             # [num_seqs, NUM_KV_HEADS, max_num_partitions, QUERY_GROUP_SIZE, HEAD_SIZE]
    context_lens_ptr,                        # [num_seqs]
    max_num_partitions,
    HEAD_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    NUM_PARTITIONS: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    context_len = tl.load(context_lens_ptr + seq_idx)
    num_partitions = tl.cdiv(context_len, PARTITION_SIZE)
    group_head_offset = tl.arange(0, QUERY_GROUP_SIZE)[:, None] * HEAD_SIZE + tl.arange(0, HEAD_SIZE)[None, :]
    if num_partitions == 1:
        # No reduction needed. Only copy tmp_out to out.
        tmp_out_offset = (seq_idx * NUM_KV_HEADS + kv_head_idx) * max_num_partitions * QUERY_GROUP_SIZE * HEAD_SIZE
        tmp_out_offset += group_head_offset
        tmp_out = tl.load(tmp_out_ptr + tmp_out_offset)
        out_offset = (seq_idx * NUM_KV_HEADS + kv_head_idx) * QUERY_GROUP_SIZE * HEAD_SIZE
        out_offset += group_head_offset
        tl.store(out_ptr + out_offset, tmp_out)
        return

    # Get the global max logit.
    offset = (seq_idx * NUM_KV_HEADS + kv_head_idx) * max_num_partitions * QUERY_GROUP_SIZE
    offset += tl.arange(0, NUM_PARTITIONS)[:, None] * QUERY_GROUP_SIZE + tl.arange(0, QUERY_GROUP_SIZE)[None, :]
    mask = tl.arange(0, NUM_PARTITIONS)[:, None] < num_partitions
    # m_i: [NUM_PARTITIONS, QUERY_GROUP_SIZE]
    m_i = tl.load(m_i_ptr + offset, mask=mask, other=float("-inf"))
    # m: [QUERY_GROUP_SIZE]
    m = tl.max(m_i, axis=0)

    # Rescale the exp sums and compute the global sum.
    # l_i: [NUM_PARTITIONS, QUERY_GROUP_SIZE]
    l_i = tl.load(l_i_ptr + offset, mask=mask, other=0.0)
    l_i *= tl.exp(m_i - m[None, :])
    # l: [QUERY_GROUP_SIZE]
    l = tl.sum(l_i, axis=0)
    # r: [NUM_PARTITIONS, QUERY_GROUP_SIZE]
    r = l_i / l[None, :]

    # Aggregate tmp_out to out.
    tmp_out_offset = (seq_idx * NUM_KV_HEADS + kv_head_idx) * max_num_partitions * QUERY_GROUP_SIZE * HEAD_SIZE
    tmp_out_offset += tl.arange(0, NUM_PARTITIONS)[:, None, None] * QUERY_GROUP_SIZE * HEAD_SIZE
    tmp_out_offset += tl.arange(0, QUERY_GROUP_SIZE)[None, :, None] * HEAD_SIZE
    tmp_out_offset += tl.arange(0, HEAD_SIZE)[None, None, :]
    # tmp_out: [NUM_PARTITIONS, QUERY_GROUP_SIZE, HEAD_SIZE]
    tmp_out = tl.load(tmp_out_ptr + tmp_out_offset, mask=mask[:, :, None], other=0.0)
    # out: [QUERY_GROUP_SIZE, HEAD_SIZE]
    out = tl.sum((tmp_out * r[:, :, None]).to(tl.float32), axis=0)

    # Store output.
    out_offset = (seq_idx * NUM_KV_HEADS + kv_head_idx) * QUERY_GROUP_SIZE * HEAD_SIZE
    out_offset += group_head_offset
    tl.store(out_ptr + out_offset, out)


def paged_attention(
    out: torch.Tensor,                      # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    query: torch.Tensor,                    # [num_seqs, NUM_KV_HEADS * QUERY_GROUP_SIZE, HEAD_SIZE]
    key_cache: torch.Tensor,                # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    value_cache: torch.Tensor,              # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    context_lens: torch.Tensor,             # [num_seqs]
    block_tables: torch.Tensor,             # [num_seqs, max_num_blocks_per_seq]
    alibi_slopes: Optional[torch.Tensor],   # [NUM_KV_HEADS * QUERY_GROUP_SIZE]
    attn_scale: float,
    max_context_len: int,
    v2_partition_size: int = 512,
    version: Optional[int] = None,
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

    if query_group_size == 1:
        padded_group_size = 1
    elif query_group_size < 16:
        padded_group_size = 16
    else:
        padded_group_size = triton.next_power_of_2(query_group_size)
    # FIXME: Remove these constraints.
    assert head_size in [64, 128, 256]
    assert kv_block_size >= 16
    assert query_group_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]

    # TODO: Support ALiBi.
    assert not use_alibi
    # TODO: Tune num_warps and num_stages.

    max_num_partitions = triton.cdiv(max_context_len, v2_partition_size)
    grid = (num_seqs, num_kv_heads, max_num_partitions)
    use_v1 = version == 1 or max_num_partitions == 1
    if use_v1:
        _paged_attn_v1_kernel[grid](
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
            padded_group_size,
            num_kv_heads,
            kv_block_size,
        )
    else:
        m_i = torch.empty(
            size=(num_seqs, num_kv_heads, max_num_partitions, query_group_size),
            dtype=torch.float32,
            device=out.device,
        )
        l_i = torch.empty_like(m_i)
        tmp_out = torch.empty(
            size=(num_seqs, num_kv_heads, max_num_partitions, query_group_size, head_size),
            dtype=out.dtype,
            device=out.device,
        )
        _paged_attn_v2_kernel[grid](
            m_i,
            l_i,
            tmp_out,
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
            padded_group_size,
            num_kv_heads,
            kv_block_size,
            v2_partition_size,
        )
        reduce_grid = (num_seqs, num_kv_heads)
        _paged_attn_v2_reduce_kernel[reduce_grid](
            out,
            m_i,
            l_i,
            tmp_out,
            context_lens,
            max_num_partitions,
            head_size,
            query_group_size,
            num_kv_heads,
            v2_partition_size,
            triton.next_power_of_2(max_num_partitions),
        )


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
    NUM_KV_HEADS = 8
    QUERY_GROUP_SIZE = 1
    NUM_BLOCKS = 7000
    HEAD_SIZE = 128
    KV_BLOCK_SIZE = 16
    MAX_SEQ_LEN = 512
    CONTEXT_LENS = [random.randint(1, MAX_SEQ_LEN) for _ in range(NUM_SEQS)]
    MAX_NUM_BLOCKS_PER_SEQ = (max(CONTEXT_LENS) + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE

    attn_scale = HEAD_SIZE ** -0.5
    qkv = torch.empty(NUM_SEQS, (QUERY_GROUP_SIZE + 2) * NUM_KV_HEADS, HEAD_SIZE).cuda()
    qkv.uniform_(-attn_scale, attn_scale)
    q = qkv[:, :-2 * NUM_KV_HEADS, :]
    out = torch.empty_like(q)

    k_cache = torch.empty(NUM_BLOCKS, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE).cuda()
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
