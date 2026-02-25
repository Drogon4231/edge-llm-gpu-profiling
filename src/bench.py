import argparse, os, time, json, subprocess, statistics
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import psutil
import cpuinfo

# Optional torch import (so file exists even before torch is installed)
try:
    import torch
except Exception as e:
    torch = None

def now_ms():
    return time.perf_counter() * 1e3

def quantiles_ms(samples_ms: List[float]) -> dict:
    a = np.array(samples_ms, dtype=np.float64)
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "std_ms": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
    }

def run_cmd(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return ""

def env_snapshot() -> dict:
    info = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "uname": run_cmd(["uname", "-a"]),
        "tegra_release": run_cmd(["bash", "-lc", "cat /etc/nv_tegra_release 2>/dev/null || true"]),
        "nvpmodel_q": run_cmd(["bash", "-lc", "sudo nvpmodel -q 2>/dev/null || true"]),
        "jetson_clocks_show": run_cmd(["bash", "-lc", "sudo jetson_clocks --show 2>/dev/null || true"]),
        "cpu": cpuinfo.get_cpu_info().get("brand_raw", ""),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }
    if torch is not None:
        info.update({
            "torch": getattr(torch, "__version__", ""),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda": getattr(getattr(torch, "version", None), "cuda", None),
            "cudnn": (torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None),
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        })
    return info

def sync_if_cuda(device: str):
    if torch is not None and device == "cuda":
        torch.cuda.synchronize()

def torch_dtype(dtype_str: str):
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype {dtype_str}")

def bench_gemm(device: str, dtype: str, m: int, n: int, k: int, batch: int, warmup: int, iters: int) -> Tuple[List[float], dict]:
    assert torch is not None, "torch not installed"
    assert device in ("cpu", "cuda")
    tdtype = torch_dtype(dtype)

    # Create batched matrices: [B, M, K] x [B, K, N] -> [B, M, N]
    # Using torch.bmm for clearer DL-like behavior
    A = torch.randn((batch, m, k), device=device, dtype=tdtype)
    B = torch.randn((batch, k, n), device=device, dtype=tdtype)

    def fn():
        return torch.bmm(A, B)

    # Warmup
    for _ in range(warmup):
        _ = fn()
    sync_if_cuda(device)

    # Timed
    samples = []
    for _ in range(iters):
        t0 = now_ms()
        _ = fn()
        sync_if_cuda(device)
        t1 = now_ms()
        samples.append(t1 - t0)

    # Approx FLOPs: 2 * B * M * N * K
    flops = 2.0 * batch * m * n * k
    stats = quantiles_ms(samples)
    avg_s = stats["mean_ms"] / 1e3
    stats["tflops"] = float((flops / avg_s) / 1e12) if avg_s > 0 else 0.0
    return samples, stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/raw_csv/bench.csv")
    ap.add_argument("--bench", choices=["gemm"], default="gemm")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    run = {
        "bench": args.bench,
        "device": args.device,
        "dtype": args.dtype,
        "m": args.m, "n": args.n, "k": args.k,
        "batch": args.batch,
        "warmup": args.warmup,
        "iters": args.iters,
        "tag": args.tag,
        **env_snapshot(),
    }

    if torch is None:
        raise RuntimeError("torch not installed. Install NVIDIA Jetson PyTorch, then rerun.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("torch CUDA not available. Check JetPack PyTorch install.")

    # Run benchmark
    if args.bench == "gemm":
        _, stats = bench_gemm(args.device, args.dtype, args.m, args.n, args.k, args.batch, args.warmup, args.iters)
    else:
        raise ValueError("Unknown bench")

    run.update(stats)

    # Append to CSV
    df = pd.DataFrame([run])
    if os.path.exists(args.out):
        df.to_csv(args.out, mode="a", header=False, index=False)
    else:
        df.to_csv(args.out, index=False)

    print(json.dumps(run, indent=2))

if __name__ == "__main__":
    main()
