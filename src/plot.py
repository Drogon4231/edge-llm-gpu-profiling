import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/raw_csv/bench.csv")

# Batch sweep plot (if present)
bdf = df[df["tag"]=="b_sweep"].sort_values("batch")
if len(bdf):
    plt.figure()
    plt.plot(bdf["batch"], bdf["tflops"], marker="o")
    plt.xlabel("Batch")
    plt.ylabel("TFLOPs (approx)")
    plt.title("GEMM Throughput vs Batch")
    plt.grid(True)
    plt.savefig("results/plots/gemm_tflops_vs_batch.png", dpi=200)

    plt.figure()
    plt.plot(bdf["batch"], bdf["p50_ms"], marker="o", label="p50")
    plt.plot(bdf["batch"], bdf["p95_ms"], marker="o", label="p95")
    plt.xlabel("Batch")
    plt.ylabel("Latency (ms)")
    plt.title("GEMM Latency vs Batch")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/gemm_latency_vs_batch.png", dpi=200)

# Size sweep plot (if present)
sdf = df[df["tag"]=="size_sweep"].sort_values("m")
if len(sdf):
    plt.figure()
    plt.plot(sdf["m"], sdf["tflops"], marker="o")
    plt.xlabel("Matrix size (M=N=K)")
    plt.ylabel("TFLOPs (approx)")
    plt.title("GEMM Throughput vs Size")
    plt.grid(True)
    plt.savefig("results/plots/gemm_tflops_vs_size.png", dpi=200)

print("Saved plots to results/plots/")
