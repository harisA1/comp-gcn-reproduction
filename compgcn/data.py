"""Datasets: synthetic SBM + compositional features, stock comovement, student failure."""
from dataclasses import dataclass
import numpy as np
import torch
from sklearn.datasets import make_classification

from .ilr import block_ilr_inv


@dataclass
class GraphData:
    X: np.ndarray          # (n, M) raw compositions, p blocks
    D: list                # parts per block
    edges: np.ndarray      # (E, 2) source -> target
    y: np.ndarray          # labels (int) or targets (float)
    task: str              # "clf" or "reg"
    n_classes: int = 1

    @property
    def n(self):
        return self.X.shape[0]


def norm_adj(edges: np.ndarray, n: int, mode: str = "sym") -> torch.Tensor:
    """Normalized adjacency with self-loops, as a sparse (n, n) tensor.

    Rows are receivers. mode="sym": symmetrize, D^-1/2 (A+I) D^-1/2 (Kipf & Welling).
    mode="dgl": keep direction; h_dst = sum_src h_src / sqrt(outdeg(src) * indeg(dst)),
    which is what DGL GraphConv(norm='both') does on a directed graph.
    """
    src, dst = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    if mode == "sym":
        src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])
    pairs = np.unique(np.stack([src, dst], 1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    loops = np.arange(n)
    src = np.concatenate([pairs[:, 0], loops])
    dst = np.concatenate([pairs[:, 1], loops])
    outdeg = np.bincount(src, minlength=n).astype(float)
    indeg = np.bincount(dst, minlength=n).astype(float)
    w = 1.0 / np.sqrt(outdeg[src] * indeg[dst])
    idx = torch.tensor(np.stack([dst, src]))
    return torch.sparse_coo_tensor(idx, torch.tensor(w, dtype=torch.float32), (n, n), check_invariants=False).coalesce()


# ---------------------------------------------------------------- synthetic (Sec. 4.1)
def sbm(labels: np.ndarray, a: float, b: float, rng) -> np.ndarray:
    n = len(labels)
    s, t = a / n, b / n
    iu = np.triu_indices(n, 1)
    same = labels[iu[0]] == labels[iu[1]]
    keep = rng.random(len(iu[0])) < np.where(same, s, t)
    return np.stack([iu[0][keep], iu[1][keep]], 1)


def synthetic(n=1000, p=10, D=3, C=2, a=245, b=205, noise_var=0.8, seed=0) -> GraphData:
    rng = np.random.default_rng(seed)
    # sklearn's default n_informative=2 supports at most 4 classes; the paper's C=5..7 needs more.
    # We keep the default for C <= 4 and use the minimum that works (3) for C = 5..8 (assumption).
    n_inf = 2 if C <= 4 else 3
    Z, y = make_classification(n_samples=n, n_features=p * (D - 1), n_classes=C, n_informative=n_inf,
                               n_clusters_per_class=1, random_state=seed)
    Z = Z + rng.normal(0, np.sqrt(noise_var), Z.shape)
    X = block_ilr_inv(Z, [D] * p)
    return GraphData(X, [D] * p, sbm(y, a, b, rng), y, "clf", C)


def snr(a, b, C=2):
    return (a - b) ** 2 / (C * (a + (C - 1) * b))


SNR_SETTINGS = {1: (240, 210), 1.78: (245, 205), 2.78: (250, 200), 4: (120, 80)}
MULTICLASS = {3: (220, 170), 4: (220, 156), 5: (220, 143), 6: (220, 131), 7: (220, 121)}


# ---------------------------------------------------------------- real data (Sec. 5)
def load_npz(path, task) -> GraphData:
    z = np.load(path)
    y = z["y"]
    C = int(y.max()) + 1 if task == "clf" else 1
    return GraphData(z["X"].astype(float), [int(d) for d in z["D"]], z["E"], y, task, C)


def stock(year, root="data"):
    return load_npz(f"{root}/stock_{year}.npz", "reg")


def student(term, root="data"):
    return load_npz(f"{root}/student_term{term}.npz", "clf")


# ---------------------------------------------------------------- splits
def split_per_class(y, n_train, n_val, rng):
    """n_train / n_val labelled nodes per class, rest test (synthetic protocol)."""
    tr, va = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        tr += list(idx[:n_train]); va += list(idx[n_train:n_train + n_val])
    te = np.setdiff1d(np.arange(len(y)), tr + va)
    return np.array(tr), np.array(va), te


def split_random(n, n_train, n_val, rng):
    idx = rng.permutation(n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
