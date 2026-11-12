"""CompMP experiments. Results are appended to results/compmp_<exp>.csv (resumable).

    python run_compmp.py e1 --workers 4    # basis dependence: fixed hyperparameters, 10 random bases x 5 seeds
    python run_compmp.py e2 --workers 4    # scalar targets: small grid, every row is one (model, hp, seed)

Selection on validation happens in summarize_compmp.py; nothing here looks at test data for a decision.
"""
import argparse, copy, csv, itertools, os, time
from multiprocessing import Pool
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from compgcn import data as dt
from compgcn.ilr import block_ilr, random_basis
from compgcn.models import COMPGCN, GCN, MLP
from compgcn.compmp import CompMP, adjacency, clr_blocks
from compgcn.train import DEFAULT_HP

D = 3
# name -> (nonlinearity, linear map, adjacency normalization, scalar head)
COMPMP = {"CompMP-a": ("naive", "off", "row", "stack"), "CompMP-a-M": ("naive", "per", "row", "stack"),
          "CompMP-b": ("norm", "off", "row", "stack"), "CompMP-c": ("part", "off", "row", "stack"),
          "CompMP-c-M": ("part", "per", "row", "stack"),
          # full zero-sum linear maps: linear part == a GCN on ilr (tests/check_compmp.py), so with "sym" the only
          # difference to GCN-ilr is the basis-free nonlinearity
          # "prop" head: inner products then propagation, like GCN's second layer (the "stack" head overfits);
          # hidden / (D-1) compositions, i.e. the same hidden width in ilr coordinates as GCN-ilr
          "CompMP-full": ("part", "full", "sym", "prop"), "CompMP-full-row": ("part", "full", "row", "prop")}
BASELINES = {"GCN-ilr": (GCN, "ilr"), "COMP-GCN": (COMPGCN, "ilr"), "GCN-scalar": (GCN, "raw"),
             "MLP-ilr": (MLP, "ilr")}
GRID = [dict(lr=lr, hidden=h, dropout=dp) for lr, h, dp in itertools.product((0.005, 0.01), (8, 16), (0.2, 0.5))]
LOGREG_C = (0.01, 0.1, 1.0, 10.0)      # linear-ilr gets the same budget size (4 configs; it has no lr/hidden/dropout)

SYN = {}
for p in (10, 100):
    for s, (a, b) in dt.SNR_SETTINGS.items():
        SYN[f"p={p},snr={s}"] = dict(p=p, C=2, a=a, b=b)
    for C, (a, b) in dt.MULTICLASS.items():
        SYN[f"p={p},C={C}"] = dict(p=p, C=C, a=a, b=b)
E1_SETTINGS = ["p=10,snr=1.78", "p=100,snr=1.78", "p=10,C=5", "p=100,C=5"]


def bases_for(basis_id, p):
    """basis_id 0 = Helmert (the default everywhere); otherwise one random basis per block."""
    if basis_id == 0:
        return None
    rng = np.random.default_rng(10_000 + basis_id)
    return [random_basis(D, rng) for _ in range(p)]


_cache = {}


def load(setting, seed):
    key = (setting, seed)
    if key not in _cache:
        s = SYN[setting]
        d = dt.synthetic(n=2000, p=s["p"], D=D, C=s["C"], a=s["a"], b=s["b"], seed=seed)
        split = dt.split_per_class(d.y, 30, 40, np.random.default_rng(1000 + seed))   # as in run.py
        _cache.clear()
        _cache[key] = (d, split, adjacency(d.edges, d.n, "row"), dt.norm_adj(d.edges, d.n, "sym").to_dense())
    return _cache[key]


def train(model, x, A, y, split, hp):
    """Training loop of compgcn/train.py (Adam, early stopping on val loss, patience 50, value clipping)."""
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=DEFAULT_HP["weight_decay"])
    tr, va, te = (torch.tensor(s) for s in split)
    best, best_state, wait = np.inf, None, 0
    for ep in range(DEFAULT_HP["max_epochs"]):
        model.train(); opt.zero_grad()
        F.cross_entropy(model(x, A)[tr], y[tr]).backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.cross_entropy(model(x, A)[va], y[va]).item()
        if vl < best:
            best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= DEFAULT_HP["patience"]:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred = model(x, A).argmax(1)
    acc = lambda idx: float((pred[idx] == y[idx]).float().mean())
    return dict(val_loss=best, val_acc=acc(va), acc=acc(te), epochs=ep + 1)


def fit(j):
    d, split, A_row, A_sym = load(j["setting"], j["seed"])
    bases = bases_for(j["basis"], len(d.D))
    m, hp = j["model"], j
    torch.manual_seed(j["seed"])
    y = torch.tensor(d.y, dtype=torch.long)
    if m in COMPMP:
        nl, within, adj, head = COMPMP[m]
        x = torch.tensor(clr_blocks(d.X, d.D), dtype=torch.float32)
        width = hp["hidden"] // (D - 1) if head == "prop" else hp["hidden"]
        model = CompMP(len(d.D), D, width, d.n_classes, hp["dropout"], nl, within,
                       None if bases is None else bases[0], head=head)
        return train(model, x, A_row if adj == "row" else A_sym, y, split, hp)
    if m in BASELINES:
        cls, kind = BASELINES[m]
        x = torch.tensor(block_ilr(d.X, d.D, bases) if kind == "ilr" else d.X, dtype=torch.float32)
        model = COMPGCN(d.D, hp["hidden"], d.n_classes, hp["dropout"]) if cls is COMPGCN \
            else cls(x.shape[1], hp["hidden"], d.n_classes, hp["dropout"])
        return train(model, x, A_sym, y, split, hp)
    tr, va, te = split
    if m == "linear-ilr":
        Z = block_ilr(d.X, d.D, bases)
        clf = LogisticRegression(C=j["C"], max_iter=5000).fit(Z[tr], d.y[tr])
        P = clf.predict_proba(Z)
        return dict(val_loss=float(-np.mean(np.log(P[va, d.y[va]] + 1e-12))),
                    val_acc=float(np.mean(P[va].argmax(1) == d.y[va])), acc=float(np.mean(P[te].argmax(1) == d.y[te])))
    if m == "constant":
        c = np.bincount(d.y[tr]).argmax()
        return dict(val_acc=float(np.mean(d.y[va] == c)), acc=float(np.mean(d.y[te] == c)))
    raise ValueError(m)


FIELDS = ["exp", "setting", "seed", "basis", "model", "lr", "hidden", "dropout", "C",
          "val_loss", "val_acc", "acc", "epochs", "seconds"]


def jobs(exp):
    J = []
    if exp == "e1":
        hp = dict(lr=DEFAULT_HP["lr"], hidden=DEFAULT_HP["hidden"], dropout=DEFAULT_HP["dropout"])
        for s, seed, b in itertools.product(E1_SETTINGS, range(5), range(10)):
            for m in ["CompMP-a", "CompMP-a-M", "GCN-ilr"] + (["CompMP-b", "CompMP-c"] if b < 3 else []):
                J.append(dict(exp=exp, setting=s, seed=seed, basis=b, model=m, **hp))
    elif exp == "e2":
        for s, seed in itertools.product(SYN, range(10)):
            # the old CompMP-a/b/c-M variants were dropped from E2 after E1 (rows already run are kept)
            for m in ["CompMP-c", "CompMP-full", "CompMP-full-row"] + list(BASELINES):
                J += [dict(exp=exp, setting=s, seed=seed, basis=0, model=m, **hp) for hp in GRID]
            J += [dict(exp=exp, setting=s, seed=seed, basis=0, model="linear-ilr", C=c) for c in LOGREG_C]
            J.append(dict(exp=exp, setting=s, seed=seed, basis=0, model="constant"))
    return J


def key(j):
    return tuple(str(j.get(k, "")) for k in ("setting", "seed", "basis", "model", "lr", "hidden", "dropout", "C"))


def work(j):
    torch.set_num_threads(1)
    t = time.time()
    r = fit(j)
    return {**{k: j.get(k, "") for k in FIELDS[:9]}, **r, "seconds": round(time.time() - t, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["e1", "e2"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="run only the first N pending jobs (timing)")
    ap.add_argument("--out", help="results CSV (default results/compmp_<exp>.csv)")
    a = ap.parse_args()
    path = a.out or f"results/compmp_{a.exp}.csv"
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {key(r) for r in csv.DictReader(f)}
    J = [j for j in jobs(a.exp) if key(j) not in done]
    if a.limit:
        J = J[:a.limit]
    print(f"{a.exp}: {len(J)} runs to do ({len(done)} done)", flush=True)
    new = not os.path.exists(path)
    t0 = time.time()
    with open(path, "a", newline="") as f, Pool(a.workers) as pool:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for i, row in enumerate(pool.imap(work, J, chunksize=1)):
            w.writerow(row); f.flush()
            if (i + 1) % 100 == 0 or i + 1 == len(J):
                print(f"  {i+1}/{len(J)}  {(time.time()-t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
