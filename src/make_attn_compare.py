import csv, os, subprocess, ast

os.makedirs("results/raw_csv", exist_ok=True)
path="results/raw_csv/attn_compare.csv"

with open(path,"w",newline="") as f:
    csv.writer(f).writerow(["impl","B","H","S","D","dtype","mean_ms","p50_ms","p95_ms","p99_ms","min_ms","max_ms"])

def run(cmd):
    out = subprocess.check_output(cmd, text=True).strip()
    return ast.literal_eval(out)

for S in [64,128,256,512,1024]:
    d = run(["python","attn_bench.py","--B","1","--H","8","--S",str(S),"--D","64","--dtype","fp16","--iters","50","--warmup","30","--use_sdpa"])
    m = run(["python","attn_bench.py","--B","1","--H","8","--S",str(S),"--D","64","--dtype","fp16","--iters","50","--warmup","30"])
    with open(path,"a",newline="") as f:
        w=csv.writer(f)
        w.writerow([d["impl"],d["B"],d["H"],d["S"],d["D"],d["dtype"],d["mean_ms"],d["p50_ms"],d["p95_ms"],d["p99_ms"],d["min_ms"],d["max_ms"]])
        w.writerow([m["impl"],m["B"],m["H"],m["S"],m["D"],m["dtype"],m["mean_ms"],m["p50_ms"],m["p95_ms"],m["p99_ms"],m["min_ms"],m["max_ms"]])
    print(f"S={S}: sdpa={d['mean_ms']:.3f} ms, manual={m['mean_ms']:.3f} ms")
print("Wrote", path)
