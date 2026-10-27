"""US county presidential elections (Jia & Benson, KDD 2020 data) for compositional node regression.

Source: github.com/000Justin000/gnn-residual-correlation, datasets/election: 3,112 counties (no Alaska),
county adjacency, Dem / GOP / other vote counts for 2008, 2012, 2016, and county covariates.

Task: predict each county's 2016 vote shares (D = 3) from its 2012 shares (one input composition)
and six covariates, transductively on the county adjacency graph.
"""
import os
import subprocess
from dataclasses import dataclass
import numpy as np
import pandas as pd

REPO = "https://github.com/000Justin000/gnn-residual-correlation"
SUB = "datasets/election"
PARTIES = ("dem", "gop", "oth")


def fetch(root="data/raw/gnn-residual-correlation"):
    """Sparse-clone the election folder once; returns its path."""
    path = os.path.join(root, SUB)
    if not os.path.exists(os.path.join(path, "election.csv")):
        os.makedirs(os.path.dirname(root), exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", REPO, root], check=True)
        subprocess.run(["git", "-C", root, "sparse-checkout", "set", SUB], check=True)
    return path


@dataclass
class Election:
    fips: np.ndarray        # (n,) 5-digit codes
    state: np.ndarray       # (n,) 2-digit state codes
    counts: np.ndarray      # (n, 3) 2016 votes (target)
    prev: np.ndarray        # (n, 3) 2012 shares, zeros replaced (input for log-ratio models, persistence)
    prev_raw: np.ndarray    # (n, 3) 2012 shares with zeros (input for the Fisher-Rao model)
    n_zero_prev: int        # zero 2012 counts replaced
    covars: np.ndarray      # (n, k) standardized covariates
    covar_names: list
    edges: np.ndarray       # (E, 2) undirected county adjacency, indices into fips
    n_imputed: int          # covariate cells filled with the column mean

    @property
    def n(self):
        return len(self.fips)

    @property
    def shares(self):
        return self.counts / self.counts.sum(1, keepdims=True)


def _read(path, name, cols, fips_col="FIPS"):
    d = pd.read_csv(os.path.join(path, name), dtype={fips_col: str}, thousands=",")
    d.columns = [c.strip() for c in d.columns]
    d[fips_col] = d[fips_col].str.zfill(5)
    return d.set_index(fips_col)[cols]


def load(root="data/raw/gnn-residual-correlation") -> Election:
    path = fetch(root)
    e = pd.read_csv(os.path.join(path, "election.csv"), dtype={"fips_code": str})
    e["fips_code"] = e["fips_code"].str.zfill(5)
    e = e.set_index("fips_code").sort_index()
    counts = e[[f"{p}_2016" for p in PARTIES]].to_numpy(float)
    prev_c = e[[f"{p}_2012" for p in PARTIES]].to_numpy(float)
    prev_raw = prev_c / prev_c.sum(1, keepdims=True)
    # multiplicative replacement of zero counts by half a vote: zeros get 0.5 / total, the other parts
    # are scaled down so their ratios are unchanged (Martin-Fernandez et al. 2003)
    delta = np.where(prev_c == 0, 0.5 / prev_c.sum(1, keepdims=True), 0.0)
    prev = np.where(prev_c == 0, delta, prev_raw * (1 - delta.sum(1, keepdims=True)))
    n_zero = int((prev_c == 0).sum())

    # covariates measured up to 2016 (none is a vote count)
    cov = pd.concat([
        _read(path, "education.csv", ["BachelorRate2016"]),
        _read(path, "income.csv", ["MedianIncome2016"]),
        _read(path, "population.csv", ["POP_ESTIMATE_2016", "R_birth_2016", "R_death_2016"]),
        _read(path, "unemployment.csv", ["Unemployment_rate_2016"]),
    ], axis=1).reindex(e.index).apply(pd.to_numeric, errors="coerce")
    cov["MedianIncome2016"] = np.log(cov["MedianIncome2016"])
    cov["POP_ESTIMATE_2016"] = np.log(cov["POP_ESTIMATE_2016"])
    n_imp = int(cov.isna().sum().sum())
    cov = cov.fillna(cov.mean())
    X = ((cov - cov.mean()) / cov.std()).to_numpy(float)

    # adjacency.txt: "name<TAB>fips<TAB>neighbour name<TAB>neighbour fips", the county columns left
    # blank on continuation lines
    idx = {f: i for i, f in enumerate(e.index)}
    edges, cur = [], None
    with open(os.path.join(path, "adjacency.txt"), encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            if parts[1].strip():
                cur = parts[1].strip().zfill(5)
            nb = parts[3].strip().zfill(5)
            if cur in idx and nb in idx and cur != nb:
                edges.append((idx[cur], idx[nb]))
    edges = np.unique(np.sort(np.array(edges), 1), axis=0)

    assert (counts > 0).all(), "2016 targets contain zeros"
    return Election(np.array(e.index), np.array([f[:2] for f in e.index]), counts, prev, prev_raw, n_zero, X,
                    list(cov.columns), edges, n_imp)


def split(el: Election, kind: str, seed: int, frac=(0.6, 0.2)):
    """Transductive node split. kind="random": counties at random; kind="state": whole states held out
    (spatially blocked), states assigned at random until each part reaches its share of counties."""
    rng = np.random.default_rng(seed)
    if kind == "random":
        idx = rng.permutation(el.n)
        a, b = int(frac[0] * el.n), int((frac[0] + frac[1]) * el.n)
        return idx[:a], idx[a:b], idx[b:]
    states = rng.permutation(np.unique(el.state))
    parts, target, cum = [[], [], []], np.cumsum([frac[0], frac[1], 1.0]) * el.n, 0
    for s in states:
        members = np.where(el.state == s)[0]
        k = int(np.searchsorted(target, cum, side="right"))
        parts[min(k, 2)] += list(members)
        cum += len(members)
    return tuple(np.array(p) for p in parts)
