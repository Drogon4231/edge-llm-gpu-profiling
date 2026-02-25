import argparse
import numpy as np
import torch
import torch.nn.functional as F
import nvtx

def has_sdpa():
    return hasattr(F, "scaled_dot_product_attention")

def manual_attention(q,k,v, causal=False):
    d = q.shape[-1]
    scale = 1.0 / np.sqrt(d)
    scores = torch.matmul(q, k.transpose(-2,-1)) * scale   # [B,H,S,S]
    if causal:
        S = q.shape[-2]
        mask = torch.triu(torch.ones((S,S), device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    o = torch.matmul(p, v)
    return o

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--S", type=int, default=512)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--dtype", choices=["fp16","fp32"], default="fp16")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--use_sdpa", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    device="cuda"
    dtype = torch.float16 if args.dtype=="fp16" else torch.float32

    q = torch.randn((args.B,args.H,args.S,args.D), device=device, dtype=dtype)
    k = torch.randn((args.B,args.H,args.S,args.D), device=device, dtype=dtype)
    v = torch.randn((args.B,args.H,args.S,args.D), device=device, dtype=dtype)

    use_sdpa = args.use_sdpa and has_sdpa()

    def fn():
        if use_sdpa:
            return F.scaled_dot_product_attention(q,k,v, is_causal=args.causal)
        else:
            return manual_attention(q,k,v, causal=args.causal)

    # Warmup (not timed)
    for _ in range(args.warmup):
        _ = fn()
    torch.cuda.synchronize()

    # Timed with CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    nvtx.push_range(f"ATTN_EVENTS_B{args.B}_H{args.H}_S{args.S}_D{args.D}_{args.dtype}_{'sdpa' if use_sdpa else 'manual'}")
    start.record()
    for _ in range(args.iters):
        _ = fn()
    end.record()
    torch.cuda.synchronize()
    nvtx.pop_range()

    total_ms = start.elapsed_time(end)
    per_iter_ms = total_ms / args.iters

    print({
        "B":args.B,"H":args.H,"S":args.S,"D":args.D,"dtype":args.dtype,
        "causal":args.causal,"impl":"sdpa" if use_sdpa else "manual",
        "iters":args.iters,
        "total_ms":float(total_ms),
        "per_iter_ms":float(per_iter_ms),
    })

if __name__=="__main__":
    main()
