# SPDX-License-Identifier: MIT
# Test: Verify that quantize→dispatch and dispatch→quantize produce identical results.
# This proves FP8 + scale_block_size=32 dispatch correctly transports tokens and scale factors.
#
# Path A: BF16 → fused_dispatch_permute → blockwise_quantize (at expert)
# Path B: BF16 → blockwise_quantize → fused_dispatch_permute (FP8 + float32 scale)
# Comparison: dequantized results should be bitwise identical.
#
# Usage:
#   python test_mxfp8_dispatch.py --num-processes 4
#   HIDDEN_DIM=7168 NUM_TOKENS_PER_RANK=512 python test_mxfp8_dispatch.py --num-processes 8
#
# Optional: TE MXFP8 variant (requires TransformerEngine):
#   TE_PATH=/path/to/TransformerEngine python test_mxfp8_dispatch.py --num-processes 4

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
te_path = os.environ.get("TE_PATH", "")
if te_path and os.path.isdir(te_path):
    sys.path.insert(0, te_path)

import argparse
import torch
import torch.distributed as dist
import deep_ep
from utils import init_dist

# TE import (optional)
try:
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    HAS_TE = True
except ImportError:
    HAS_TE = False

# Config (overridable via env vars)
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 7168))
MAX_NUM_OF_TOKENS_PER_RANK = int(os.environ.get("MAX_NUM_OF_TOKENS_PER_RANK", 4096))
NUM_TOKENS_PER_RANK = int(os.environ.get("NUM_TOKENS_PER_RANK", 256))
NUM_LOCAL_EXPERTS = int(os.environ.get("NUM_LOCAL_EXPERTS", 8))
TOPK = int(os.environ.get("TOPK", 8))
PAD_MULTIPLE = 32
SCALE_BLOCK_SIZE = 32
SEED = int(os.environ.get("SEED", 1025))

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


# ============================================================
# Simple blockwise FP8 quantizer (float32 scales)
# ============================================================

def blockwise_quantize_fp8(tensor: torch.Tensor, block_size: int = 32):
    """Per-block FP8 E4M3 quantization with float32 scales.

    Args:
        tensor: [num_tokens, hidden_dim] BF16
        block_size: elements per scale factor

    Returns:
        fp8_data: [num_tokens, hidden_dim] uint8 (FP8 E4M3 bit pattern)
        scale:    [num_tokens, hidden_dim // block_size] float32
    """
    num_tokens, hidden_dim = tensor.shape
    assert hidden_dim % block_size == 0
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0

    blocks = tensor.float().reshape(num_tokens, -1, block_size)
    absmax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = absmax / FP8_MAX

    fp8 = (blocks / scale).reshape(num_tokens, hidden_dim).to(torch.float8_e4m3fn)
    return fp8.view(torch.uint8), scale.squeeze(-1)


def blockwise_dequantize_fp8(fp8_data: torch.Tensor, scale: torch.Tensor, block_size: int = 32):
    """Dequantize per-block FP8 back to BF16."""
    num_tokens, hidden_dim = fp8_data.shape
    fp8_float = fp8_data.view(torch.float8_e4m3fn).float()
    blocks = fp8_float.reshape(num_tokens, -1, block_size)
    return (blocks * scale.unsqueeze(-1)).reshape(num_tokens, hidden_dim).bfloat16()


# ============================================================
# Helpers
# ============================================================

def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape:
        return False
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def make_routing(num_tokens, num_experts, topk):
    routing_map = torch.zeros(num_tokens, num_experts, device="cuda", dtype=torch.bool)
    probs = torch.zeros(num_tokens, num_experts, device="cuda", dtype=torch.float32)
    for i in range(num_tokens):
        selected = torch.randperm(num_experts, device="cuda")[:topk]
        routing_map[i, selected] = True
        probs[i, selected] = 1.0
    return routing_map, probs


def status(ok):
    return "PASS" if ok else "FAIL"


# ============================================================
# Test 1: Simple blockwise quantizer (float32 scales)
# ============================================================

def test_blockwise_commutativity(group, rank, world_size):
    """quantize(dispatch(x)) == dispatch(quantize(x)) using simple blockwise FP8."""
    num_experts = NUM_LOCAL_EXPERTS * world_size

    if rank == 0:
        print(f"\n{'='*60}")
        print(f" Test 1: Blockwise FP8 Commutativity (scale_block_size={SCALE_BLOCK_SIZE})")
        print(f" hidden_dim={HIDDEN_DIM}, tokens={NUM_TOKENS_PER_RANK}, "
              f"experts={num_experts}, topk={TOPK}, pad_multiple={PAD_MULTIPLE}")
        print(f"{'='*60}")

    # Buffers
    buf_bf16 = deep_ep.HybridEPBuffer(
        group=group, hidden_dim=HIDDEN_DIM,
        max_num_of_tokens_per_rank=MAX_NUM_OF_TOKENS_PER_RANK,
        num_local_experts=NUM_LOCAL_EXPERTS, use_fp8=False,
    )
    buf_fp8 = deep_ep.HybridEPBuffer(
        group=group, hidden_dim=HIDDEN_DIM,
        max_num_of_tokens_per_rank=MAX_NUM_OF_TOKENS_PER_RANK,
        num_local_experts=NUM_LOCAL_EXPERTS, use_fp8=True,
        scale_block_size=SCALE_BLOCK_SIZE,
    )

    # Test data
    hidden_bf16 = torch.randn(NUM_TOKENS_PER_RANK, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
    routing_map, probs = make_routing(NUM_TOKENS_PER_RANK, num_experts, TOPK)
    dist.barrier()

    # --- Path A: dispatch BF16 → quantize at expert ---
    disp_bf16, disp_probs_a, _, tpe_a, handle_a = buf_bf16.dispatch_with_permute(
        hidden=hidden_bf16, routing_map=routing_map, probs=probs,
        pad_multiple=PAD_MULTIPLE, fuse_permute_dispatch=True,
    )
    fp8_a, scale_a = blockwise_quantize_fp8(disp_bf16, SCALE_BLOCK_SIZE)
    deq_a = blockwise_dequantize_fp8(fp8_a, scale_a, SCALE_BLOCK_SIZE)

    # --- Path B: quantize → dispatch FP8 ---
    fp8_pre, scale_pre = blockwise_quantize_fp8(hidden_bf16, SCALE_BLOCK_SIZE)
    disp_fp8, disp_probs_b, disp_scale_b, tpe_b, handle_b = buf_fp8.dispatch_with_permute(
        hidden=fp8_pre, scaling_factor=scale_pre, routing_map=routing_map, probs=probs,
        pad_multiple=PAD_MULTIPLE, fuse_permute_dispatch=True,
    )
    deq_b = blockwise_dequantize_fp8(disp_fp8, disp_scale_b, SCALE_BLOCK_SIZE)

    # --- Compare ---
    # FP8 data: bitwise identical for all tokens (padding = 0 in both paths)
    fp8_ok = bitwise_equal(fp8_a, disp_fp8)

    # Scale: only compare non-padding tokens (padding has scale mismatch: 1e-12 vs 0)
    real_mask = (fp8_a.view(torch.float8_e4m3fn).float().abs().sum(dim=-1) > 0)
    if real_mask.any():
        scale_real_a = scale_a[real_mask]
        scale_real_b = disp_scale_b[real_mask]
        scale_ok = bitwise_equal(scale_real_a, scale_real_b)
    else:
        scale_ok = True

    # Dequantized: should match for all tokens (padding: 0*any = 0)
    deq_ok = torch.allclose(deq_a, deq_b, atol=0, rtol=0)

    # tokens_per_expert
    tpe_ok = bitwise_equal(tpe_a, tpe_b)

    # Compute diffs for reporting
    fp8_mismatch = (fp8_a != disp_fp8).sum().item()
    fp8_total = fp8_a.numel()

    if real_mask.any():
        scale_abs = (scale_real_a.float() - scale_real_b.float()).abs()
        scale_abs_max = scale_abs.max().item()
        scale_abs_mean = scale_abs.mean().item()
        scale_denom = scale_real_a.float().abs().clamp(min=1e-12)
        scale_rel = (scale_abs / scale_denom)
        scale_rel_max = scale_rel.max().item()
        scale_rel_mean = scale_rel.mean().item()
    else:
        scale_abs_max = scale_abs_mean = scale_rel_max = scale_rel_mean = 0.0

    deq_abs = (deq_a.float() - deq_b.float()).abs()
    deq_abs_max = deq_abs.max().item()
    deq_abs_mean = deq_abs.mean().item()
    deq_denom = deq_a.float().abs().clamp(min=1e-12)
    deq_rel = (deq_abs / deq_denom)
    deq_rel_max = deq_rel.max().item()
    deq_rel_mean = deq_rel.mean().item()

    dist.barrier()
    for i in range(world_size):
        if i == rank:
            print(f"  [Rank {rank}] tpe={status(tpe_ok)}  fp8={status(fp8_ok)}  "
                  f"scale(real)={status(scale_ok)}  deq={status(deq_ok)}")
            print(f"    fp8 mismatches: {fp8_mismatch}/{fp8_total}")
            print(f"    scale  abs_diff: max={scale_abs_max:.6e}, mean={scale_abs_mean:.6e}  "
                  f"rel_diff: max={scale_rel_max:.6e}, mean={scale_rel_mean:.6e}")
            print(f"    deq    abs_diff: max={deq_abs_max:.6e}, mean={deq_abs_mean:.6e}  "
                  f"rel_diff: max={deq_rel_max:.6e}, mean={deq_rel_mean:.6e}")
        dist.barrier()

    all_pass = fp8_ok and scale_ok and deq_ok and tpe_ok
    if rank == 0:
        print(f"\n  Test 1 Result: {status(all_pass)}")
    return all_pass


# ============================================================
# Test 2: TE MXFP8 quantizer (E8M0 scales via float32 transport)
# ============================================================

def test_te_mxfp8_commutativity(group, rank, world_size):
    """Same as Test 1 but using TE's MXFP8Quantizer with E8M0 ↔ float32 scale conversion."""
    if not HAS_TE:
        if rank == 0:
            print(f"\n  Test 2: SKIPPED (TransformerEngine not found)")
        return True

    num_experts = NUM_LOCAL_EXPERTS * world_size

    if rank == 0:
        print(f"\n{'='*60}")
        print(f" Test 2: TE MXFP8 Commutativity (E8M0 scale via float32 transport)")
        print(f"{'='*60}")

    # Buffers
    buf_bf16 = deep_ep.HybridEPBuffer(
        group=group, hidden_dim=HIDDEN_DIM,
        max_num_of_tokens_per_rank=MAX_NUM_OF_TOKENS_PER_RANK,
        num_local_experts=NUM_LOCAL_EXPERTS, use_fp8=False,
    )
    buf_fp8 = deep_ep.HybridEPBuffer(
        group=group, hidden_dim=HIDDEN_DIM,
        max_num_of_tokens_per_rank=MAX_NUM_OF_TOKENS_PER_RANK,
        num_local_experts=NUM_LOCAL_EXPERTS, use_fp8=True,
        scale_block_size=SCALE_BLOCK_SIZE,
    )

    # TE MXFP8 quantizer (rowwise only, block_size=32)
    quantizer = te.MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False,
    )

    # Test data
    hidden_bf16 = torch.randn(NUM_TOKENS_PER_RANK, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
    routing_map, probs = make_routing(NUM_TOKENS_PER_RANK, num_experts, TOPK)
    dist.barrier()

    # --- Path A: dispatch BF16 → TE MXFP8 quantize at expert ---
    disp_bf16, _, _, tpe_a, _ = buf_bf16.dispatch_with_permute(
        hidden=hidden_bf16, routing_map=routing_map, probs=probs,
        pad_multiple=PAD_MULTIPLE, fuse_permute_dispatch=True,
    )
    mxfp8_a = quantizer(disp_bf16)
    fp8_a = mxfp8_a._rowwise_data                # uint8, FP8 E4M3
    e8m0_a = mxfp8_a._rowwise_scale_inv           # uint8, E8M0

    # --- Path B: TE MXFP8 quantize → dispatch FP8 ---
    mxfp8_pre = quantizer(hidden_bf16)
    fp8_pre = mxfp8_pre._rowwise_data              # uint8
    e8m0_pre = mxfp8_pre._rowwise_scale_inv         # uint8

    # Convert E8M0 (uint8) → float32 for HybridEP transport
    # E8M0: value represents 2^(val - 127)
    e8m0_float32 = torch.pow(2.0, e8m0_pre.to(torch.float32) - 127.0)

    # Handle E8M0 scale_inv shape: may be padded to [round_up(T,128), round_up(H//32,4)]
    # HybridEP expects [T, H//32]. Trim padding if present.
    scales_per_token = HIDDEN_DIM // SCALE_BLOCK_SIZE
    e8m0_float32_trimmed = e8m0_float32[:NUM_TOKENS_PER_RANK, :scales_per_token]

    disp_fp8, _, disp_scale_b, tpe_b, _ = buf_fp8.dispatch_with_permute(
        hidden=fp8_pre, scaling_factor=e8m0_float32_trimmed,
        routing_map=routing_map, probs=probs,
        pad_multiple=PAD_MULTIPLE, fuse_permute_dispatch=True,
    )

    # Convert transported float32 back to E8M0 (lossless for valid E8M0 values)
    # scale_float32 = 2^(e8m0 - 127), so e8m0 = log2(scale_float32) + 127
    disp_e8m0_b = (torch.log2(disp_scale_b.clamp(min=2.0**-127)) + 127.0).round().to(torch.uint8)

    # --- Compare (real tokens only) ---
    # Trim e8m0_a to match shape (TE may pad differently than HybridEP output)
    num_permuted = disp_fp8.shape[0]
    e8m0_a_trimmed = e8m0_a[:num_permuted, :scales_per_token]

    real_mask = (disp_fp8.float().abs().sum(dim=-1) > 0)

    fp8_ok = bitwise_equal(fp8_a[:num_permuted], disp_fp8)

    if real_mask.any():
        e8m0_ok = bitwise_equal(e8m0_a_trimmed[real_mask], disp_e8m0_b[real_mask])
    else:
        e8m0_ok = True

    # Dequantize comparison
    deq_a = mxfp8_a.dequantize(dtype=torch.bfloat16)[:num_permuted]
    # Manual dequantize for Path B: fp8 * scale_inv
    disp_fp8_float = disp_fp8.view(torch.float8_e4m3fn).float()
    blocks_b = disp_fp8_float.reshape(num_permuted, -1, SCALE_BLOCK_SIZE)
    deq_b = (blocks_b * disp_scale_b.unsqueeze(-1)).reshape(num_permuted, HIDDEN_DIM).bfloat16()
    deq_close = torch.allclose(deq_a, deq_b, atol=1e-3, rtol=1e-2)

    tpe_ok = bitwise_equal(tpe_a, tpe_b)

    # Compute diffs for reporting
    fp8_mismatch2 = (fp8_a[:num_permuted] != disp_fp8).sum().item()
    fp8_total2 = disp_fp8.numel()

    if real_mask.any():
        e8m0_mismatch = (e8m0_a_trimmed[real_mask] != disp_e8m0_b[real_mask]).sum().item()
        e8m0_total = e8m0_a_trimmed[real_mask].numel()
    else:
        e8m0_mismatch = 0
        e8m0_total = 0

    deq_abs2 = (deq_a.float() - deq_b.float()).abs()
    deq_abs_max2 = deq_abs2.max().item()
    deq_abs_mean2 = deq_abs2.mean().item()
    deq_denom2 = deq_a.float().abs().clamp(min=1e-12)
    deq_rel2 = (deq_abs2 / deq_denom2)
    deq_rel_max2 = deq_rel2.max().item()
    deq_rel_mean2 = deq_rel2.mean().item()

    dist.barrier()
    for i in range(world_size):
        if i == rank:
            print(f"  [Rank {rank}] tpe={status(tpe_ok)}  fp8={status(fp8_ok)}  "
                  f"e8m0(real)={status(e8m0_ok)}  deq≈={status(deq_close)}")
            print(f"    fp8 mismatches: {fp8_mismatch2}/{fp8_total2}")
            print(f"    e8m0 mismatches: {e8m0_mismatch}/{e8m0_total}")
            print(f"    deq    abs_diff: max={deq_abs_max2:.6e}, mean={deq_abs_mean2:.6e}  "
                  f"rel_diff: max={deq_rel_max2:.6e}, mean={deq_rel_mean2:.6e}")
        dist.barrier()

    all_pass = fp8_ok and e8m0_ok and deq_close and tpe_ok
    if rank == 0:
        print(f"\n  Test 2 Result: {status(all_pass)}")
    return all_pass


# ============================================================
# Main
# ============================================================

def test_main(local_rank, num_local_ranks, args):
    _, _, group = init_dist(local_rank, num_local_ranks)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        pass1 = test_blockwise_commutativity(group, rank, world_size)
        pass2 = test_te_mxfp8_commutativity(group, rank, world_size)

    dist.barrier()
    if rank == 0:
        print(f"\n{'='*60}")
        overall = pass1 and pass2
        print(f" ALL TESTS: {status(overall)}")
        print(f"{'='*60}")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FP8 dispatch with scale_block_size=32")
    parser.add_argument("--num-processes", type=int, default=4)
    args = parser.parse_args()
    torch.multiprocessing.spawn(test_main, args=(args.num_processes, args), nprocs=args.num_processes)
