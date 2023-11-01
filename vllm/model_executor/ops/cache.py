import torch
import triton
import triton.language as tl


# Grid: (num_tokens)
@triton.jit
def _reshape_and_cache_kernel(
    k_ptr,                              # [num_tokens, NUM_KV_HEADS, HEAD_SIZE]
    v_ptr,                              # [num_tokens, NUM_KV_HEADS, HEAD_SIZE]
    k_cache_ptr,                        # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    v_cache_ptr,                        # [num_blocks, NUM_KV_HEADS, KV_BLOCK_SIZE, HEAD_SIZE]
    slot_mapping_ptr,                   # [num_tokens]
    K_STRIDE: tl.constexpr,
    V_STRIDE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx < 0:
        # Padding token. Ignore.
        return

    block_idx = slot_idx // KV_BLOCK_SIZE
    block_offset = slot_idx % KV_BLOCK_SIZE
    head_offset = tl.arange(0, HEAD_SIZE)[None, :]
    kv_cache_offset = (block_idx * NUM_KV_HEADS * KV_BLOCK_SIZE + block_offset) * HEAD_SIZE + head_offset

    for head_idx in range(NUM_KV_HEADS):
        key_offset = token_idx * K_STRIDE + head_idx * HEAD_SIZE + head_offset
        key = tl.load(k_ptr + key_offset)
        key_cache_offset = kv_cache_offset + head_idx * KV_BLOCK_SIZE * HEAD_SIZE
        tl.store(k_cache_ptr + key_cache_offset, key)

    for head_idx in range(NUM_KV_HEADS):
        value_offset = token_idx * V_STRIDE + head_idx * HEAD_SIZE + head_offset
        value = tl.load(v_ptr + value_offset)
        value_cache_offset = kv_cache_offset + head_idx * KV_BLOCK_SIZE * HEAD_SIZE
        tl.store(v_cache_ptr + value_cache_offset, value)


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    num_tokens = key.shape[0]
    num_kv_heads = key.shape[1]
    head_size = key.shape[2]
    block_size = key_cache.shape[2]
    key_stride = key.stride(0)
    value_stride = value.stride(0)

    grid = (num_tokens,)
    _reshape_and_cache_kernel[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key_stride,
        value_stride,
        num_kv_heads,
        head_size,
        block_size,
    )


if __name__ == "__main__":
    key = torch.randn(10, 12, 64).cuda()
    value = torch.randn_like(key)
    key_cache = torch.randn(1000, 12, 16, 64).cuda()
    value_cache = torch.randn_like(key_cache)

    slot_mapping = torch.randint(0, 1000, (10,)).cuda()

    cloned_key_cache = key_cache.clone()
    cloned_value_cache = value_cache.clone()
    reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
    )

    num_tokens = key.shape[0]
    block_size = key_cache.shape[2]
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indicies[i]
        block_offset = block_offsets[i]
        cloned_key_cache[block_idx, :, block_offset, :] = key[i]
        cloned_value_cache[block_idx, :, block_offset, :] = value[i]

    print(torch.allclose(cloned_key_cache, key_cache))
    print(torch.allclose(cloned_value_cache, value_cache))
