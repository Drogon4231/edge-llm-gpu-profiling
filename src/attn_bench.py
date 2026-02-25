import argparse, time
import numpy as np
import torch
import torch.nn.functional as F
import nvtx

def now_ms(): return time.perf_counter() * 1e3

def sync(): torch.cuda.synchronize()

def quantiles(samples):
    a = np.array(samples, dtype=np.float64)
    return dict(
        mean_ms=float(a.mean()),
        p50_ms=float(np.percentile(a,50)),
        p95_ms=float(np.percentile(a,95)),
        p99_ms=float(np.percentile(a,99)),
        min_ms=float(a.min()),
        max_ms=float(a.max()),
    )

def manual_attention(q,k,v, causal=False):
    # q,k,v: [B, H, S, D]
    d = q.shape[-1]
    scale = 1.0 / np.sqrt(d)
    scores = torch.matmul(q, k.transpose(-2,-1)) * scale   # [B,H,S,S]
    if causal:
        S = q.shape[-2]
        mask = torch.triu(torch.ones((S,S), device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    o = torch.matmul(p, v)                                 # [B,H,S,D]
    return o

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--dtype", choices=["fp16","fp32"], default="fp16")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--use_sdpa", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    device="cuda"
    dtype = torch.float16 if args.dtype=="fp16" else torch.float32

    q = torch.randn((args.B,args.H,args.S,args.D), device=device, dtype=dtype)
    k = torch.randn((args.B,args.H,args.S,args.D), device=device, dtype=dtype)
    v = torch.randn((args.B,args.H,args.S,args.D), device=device, dtype=dtype)

    has_sdpa = hasattr(F, "scaled_dot_product_attention")
    use_sdpa = args.use_sdpa and has_sdpa

    def fn():
        if use_sdpa:
            # SDPA expects [B,H,S,D]
            return F.scaled_dot_product_attention(q,k,v, is_causal=args.causal)
        else:
            return manual_attention(q,k,v, causal=args.causal)

    # Warmup
    for _ in range(args.warmup):
        _ = fn()
    sync()

    # Timed
    samples=[]
    nvtx.push_range(f"ATTN_TIMED_B{args.B}_H{args.H}_S{args.S}_D{args.D}_{args.dtype}_{'sdpa' if use_sdpa else 'manual'}")
    for _ in range(args.iters):
        t0=now_ms()
        _ = fn()
        sync()
        samples.append(now_ms()-t0)
    nvtx.pop_range()

    stats = quantiles(samples)
    print({
        "B":args.B,"H":args.H,"S":args.S,"D":args.D,"dtype":args.dtype,
        "causal":args.causal,"impl":"sdpa" if use_sdpa else "manual",
        **stats
    })

if __name__=="__main__":
    main()
