"""E3b: county vote shares (2016 from 2012 + covariates). Results appended to results/compmp_e3.csv
(resumable). Each row is one (model, loss, config, split kind, seed); selection on validation happens in
summarize_e3.py.

    python run_e3.py --workers 4
"""
import argparse, csv, itertools, os, time
from multiprocessing import Pool
import numpy as np
import torch

from compgcn import elections as E, e3
from compgcn.compmp import adjacency

GRID = [dict(lr=lr, hidden=h, dropout=dp) for lr, h, dp in itertools.product((0.005, 0.01), (8, 16), (0.2, 0.5))]
NEURAL = [("CompMP-full", "aitchison"), ("CompMP-full", "kl"), ("CompMP-full", "multinomial"),
          ("CompMP-FR", "hellinger"), ("CompMP-FR", "kl"),
          ("GCN-ilr", "aitchison"), ("GCN-ilr", "kl"), ("GCN-ilr", "multinomial"),
          ("GCN-softmax", "kl")]
# non-neural baselines with their own 4-value grids (selected on validation like the rest)
OTHER = [("constant", None), ("persistence", None)] + [("linear-ilr", l) for l in (0.01, 0.1, 1.0, 10.0)] \
    + [("LP", a) for a in (0.5, 0.8, 0.9, 0.99)] + [("LP-delta", a) for a in (0.5, 0.8, 0.9, 0.99)] \
    + [("Lap-MNL", l) for l in (0.1, 1.0, 10.0, 100.0)]
SPLITS, SEEDS = ("random", "state"), range(10)
FIELDS = ["split", "seed", "model", "loss", "lr", "hidden", "dropout", "param", "val_loss", "epochs",
          "val_aitchison", "val_kl", "val_hellinger", "val_mae", "aitchison", "kl", "hellinger", "mae", "seconds"]

_el = {}


def data():
    if not _el:
        el = E.load()
        _el["el"], _el["A"] = el, adjacency(el.edges, el.n, "sym")
    return _el["el"], _el["A"]


def jobs():
    J = []
    for kind, seed in itertools.product(SPLITS, SEEDS):
        J += [dict(split=kind, seed=seed, model=m, param=p) for m, p in OTHER]
        J += [dict(split=kind, seed=seed, model=m, loss=l, **hp) for m, l in NEURAL for hp in GRID]
    return J


def key(j):
    return tuple(str(j.get(k, "")) for k in ("split", "seed", "model", "loss", "lr", "hidden", "dropout", "param"))


def work(j):
    torch.set_num_threads(1)
    el, A = data()
    sp = E.split(el, j["split"], j["seed"])
    t = time.time()
    m, p = j["model"], j.get("param")
    if m == "constant":
        r = e3.fit_constant(el, sp)
    elif m == "persistence":
        r = e3.fit_persistence(el, sp)
    elif m == "linear-ilr":
        r = e3.fit_linear(el, sp, p)
    elif m in ("LP", "LP-delta"):
        r = e3.fit_lp(el, A.numpy().astype(float), sp, p, delta=m == "LP-delta")
    elif m == "Lap-MNL":
        r = e3.fit_lap_mnl(el, el.edges, sp, p, j["seed"])
    else:
        r = e3.fit_neural(m, j["loss"], el, A, sp, j, j["seed"])
    return {**{k: j.get(k, "") for k in FIELDS[:8]}, **r, "seconds": round(time.time() - t, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/compmp_e3.csv")
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {key(r) for r in csv.DictReader(f)}
    J = [j for j in jobs() if key(j) not in done]
    print(f"e3: {len(J)} runs to do ({len(done)} done)", flush=True)
    new = not os.path.exists(a.out)
    t0 = time.time()
    with open(a.out, "a", newline="") as f, Pool(a.workers) as pool:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for i, row in enumerate(pool.imap_unordered(work, J, chunksize=1)):
            w.writerow(row); f.flush()
            if (i + 1) % 50 == 0 or i + 1 == len(J):
                print(f"  {i+1}/{len(J)}  {(time.time()-t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
