import csv, os, subprocess, ast

os.makedirs("results/raw_csv", exist_ok=True)
path="results/raw_csv/decode_kv_sweep.csv"

with open(path,"w",newline="") as f:
    csv.writer(f).writerow(["impl","B","H","S","D","dtype","iters","per_iter_us"])

def run(cmd):
    out = subprocess.check_output(cmd, text=True).strip()
    return ast.literal_eval(out)

for S in [256,512,1024,2048,4096,8192]:
    d = run(["python","kv_decode_bench.py","--B","1","--H","8","--S",str(S),"--D","64","--dtype","fp16","--iters","4000","--warmup","300","--use_sdpa"])
    with open(path,"a",newline="") as f:
        csv.writer(f).writerow([d["impl"],d["B"],d["H"],d["S"],d["D"],d["dtype"],d["iters"],d["per_iter_us"]])
    print(f"S={S} -> {d['per_iter_us']:.2f} us/token")
print("Wrote", path)
