import argparse
import numpy as np
import torch
import torch.nn.functional as F
import nvtx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--S", type=int, default=4096)   # KV length
    ap.add_argument("--T", type=int, default=1)      # query length (microbatch tokens)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--dtype", choices=["fp16","fp32"], default="fp16")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=200)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    device="cuda"
    dtype = torch.float16 if args.dtype=="fp16" else torch.float32

    q = torch.randn((args.B,args.H,args.T,args.D), device=device, dtype=dtype)
    k = torch.randn((args.B,args.H,args.S,args.D), device=device, dtype=dtype)
    v = torch.randn((args.B,args.H,args.S,args.D), device=device, dtype=dtype)

    assert hasattr(F, "scaled_dot_product_attention")

    def fn():
        return F.scaled_dot_product_attention(q,k,v, is_causal=False)

    for _ in range(args.warmup):
        _ = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    nvtx.push_range(f"DECODE_MICROBATCH_S{args.S}_T{args.T}")
    start.record()
    for _ in range(args.iters):
        _ = fn()
    end.record()
    torch.cuda.synchronize()
    nvtx.pop_range()

    total_ms = start.elapsed_time(end)
    # normalize per token
    per_token_us = (total_ms * 1000.0) / (args.iters * args.T)
    print({"S":args.S,"T":args.T,"per_token_us":float(per_token_us)})

if __name__=="__main__":
    main()
