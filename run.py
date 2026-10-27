"""Run the paper's experiments with fixed hyperparameters (no grid search).

    python run.py synthetic  --reps 5
    python run.py multiclass --reps 3
    python run.py stock      --reps 10
    python run.py student    --reps 10
    python run.py all --reps 5 --workers 8

Results are appended to results/<experiment>.csv (one row per model x setting x repetition),
so an interrupted run can be resumed: finished rows are skipped.
"""
import argparse, csv, os, time
from multiprocessing import Pool
import numpy as np
import torch

from compgcn import data as dt
from compgcn.train import fit, DEFAULT_HP, NEURAL

MODELS = list(NEURAL) + ["linear-ilr", "constant"]


def jobs_for(exp, reps, models):
    J = []
    if exp == "synthetic":
        for n in (1000, 2000):
            for p in (10, 100):
                for s, (a, b) in dt.SNR_SETTINGS.items():
                    for r in range(reps):
                        J += [dict(exp=exp, setting=f"n={n},p={p},snr={s}", n=n, p=p, C=2, a=a, b=b, rep=r, model=m) for m in models]
    elif exp == "multiclass":
        for p in (10, 100):
            for C, (a, b) in dt.MULTICLASS.items():
                for r in range(reps):
                    J += [dict(exp=exp, setting=f"p={p},C={C}", n=2000, p=p, C=C, a=a, b=b, rep=r, model=m) for m in models]
    elif exp == "stock":
        for y in (2019, 2020, 2021):
            for r in range(reps):
                J += [dict(exp=exp, setting=str(y), year=y, rep=r, model=m) for m in models]
    elif exp == "student":
        for t in (1, 2, 3, 4):
            for r in range(reps):
                J += [dict(exp=exp, setting=f"term{t}", term=t, rep=r, model=m) for m in models]
    return J


_cache = {}


def load(j, adj_mode):
    key = (j["exp"], j["setting"], j["rep"])
    if key in _cache:
        return _cache[key]
    rng = np.random.default_rng(1000 + j["rep"])
    if j["exp"] in ("synthetic", "multiclass"):
        d = dt.synthetic(n=j["n"], p=j["p"], D=3, C=j["C"], a=j["a"], b=j["b"], seed=j["rep"])
        split = dt.split_per_class(d.y, 30, 40, rng)          # Sec. 4.3
        A = dt.norm_adj(d.edges, d.n, "sym")
    elif j["exp"] == "stock":
        d = dt.stock(j["year"])
        split = dt.split_random(d.n, 80, 80, rng)             # Sec. 5.1.2
        A = dt.norm_adj(d.edges, d.n, adj_mode)
    else:
        d = dt.student(j["term"])
        split = dt.split_per_class(d.y, 40, 40, rng)          # 80 train / 80 val, class-balanced (assumed)
        A = dt.norm_adj(d.edges, d.n, "sym")
    _cache.clear(); _cache[key] = (d, A, split)
    return d, A, split


def work(args):
    j, adj_mode = args
    torch.set_num_threads(1)
    d, A, split = load(j, adj_mode)
    t = time.time()
    m = fit(j["model"], d, A, split, DEFAULT_HP, seed=j["rep"])
    return {**{k: j[k] for k in ("exp", "setting", "rep", "model")},
            **m, "seconds": round(time.time() - t, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["synthetic", "multiclass", "stock", "student", "all"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--stock-adj", default="dgl", choices=["dgl", "sym"],
                    help="directed (DGL GraphConv-style) or symmetrized stock graph")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    exps = ["synthetic", "multiclass", "stock", "student"] if a.exp == "all" else [a.exp]
    fields = ["exp", "setting", "rep", "model", "acc", "bal_acc", "mse", "epochs", "seconds"]
    for exp in exps:
        path = f"results/{exp}.csv"
        done = set()
        if os.path.exists(path):
            with open(path) as f:
                done = {(r["setting"], int(r["rep"]), r["model"]) for r in csv.DictReader(f)}
        J = [j for j in jobs_for(exp, a.reps, a.models) if (j["setting"], j["rep"], j["model"]) not in done]
        print(f"{exp}: {len(J)} runs to do ({len(done)} already done)", flush=True)
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f, Pool(a.workers) as pool:
            w = csv.DictWriter(f, fields)
            if new:
                w.writeheader()
            for i, row in enumerate(pool.imap(work, [(j, a.stock_adj) for j in J], chunksize=1)):
                w.writerow(row); f.flush()
                if i % 20 == 0:
                    print(f"  {i+1}/{len(J)} {row}", flush=True)


if __name__ == "__main__":
    main()
