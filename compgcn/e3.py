"""E3: compositional node regression (county vote shares). Models, losses, metrics and baselines.

Every model maps (prev shares, covariates, graph) to log-probabilities lp (n, D) of the predicted
composition, so all losses and metrics are computed the same way for every model.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ilr import clr, helmert_basis, ilr
from .compmp import CompMP, center
from .models import GCN
from .train import DEFAULT_HP

EPS = 1e-6


# ------------------------------------------------------------------ Fisher-Rao (square-root sphere) model
class SphereLayer(nn.Module):
    """States are unit vectors u = sqrt(p) in the positive orthant of the sphere (zeros allowed).
    Aggregation: weighted chordal mean A u (normalized by the nonlinearity below). Then a linear map per
    channel pair, scalar covariates, bias, ReLU per part and projection back onto the sphere. The
    projection makes the layer nonlinear in every coordinate system, so it is not a GCN on any fixed
    transform of the shares. ReLU can produce exact zero parts."""
    def __init__(self, c_in, c_out, D, n_scalar=0):
        super().__init__()
        bound = np.sqrt(6.0 / ((c_in + c_out) * D))
        self.W = nn.Parameter((torch.rand(c_out, c_in, D, D) * 2 - 1) * bound)
        self.Ws = nn.Parameter((torch.rand(c_out, c_in, D, D) * 2 - 1) * bound)      # skip: own state
        self.V = nn.Linear(n_scalar, c_out * D, bias=False) if n_scalar else None
        self.Vs = nn.Linear(n_scalar, c_out * D, bias=False) if n_scalar else None
        self.b = nn.Parameter(torch.full((c_out, D), 1 / np.sqrt(D)))   # start inside the orthant
        self._au, self._as = None, None

    def forward(self, u, A, s=None, cache=False):
        n = u.shape[0]
        if cache and self._au is not None:
            au = self._au
        else:
            au = (A @ u.reshape(n, -1)).reshape(u.shape)
            if cache:
                self._au = au
        z = torch.einsum("kjde,nje->nkd", self.W, au) + torch.einsum("kjde,nje->nkd", self.Ws, u) + self.b
        if self.V is not None:
            if self._as is None:
                self._as = A @ s
            z = z + (self.V(self._as) + self.Vs(s)).reshape(z.shape)
        z = F.relu(z)
        return z / (z.norm(dim=-1, keepdim=True) + EPS)


class SphereGNN(nn.Module):
    def __init__(self, D, hidden, dropout, n_scalar):
        super().__init__()
        self.l1 = SphereLayer(1, hidden, D, n_scalar)
        self.l2 = SphereLayer(hidden, 1, D)
        self.dp, self.D = dropout, D

    def forward(self, u, A, s):
        h = self.l1(u, A, s, cache=True)
        if self.training and self.dp > 0:     # drop whole channels: replace by the barycenter direction
            keep = (torch.rand(h.shape[0], h.shape[1], 1) > self.dp).to(h.dtype)
            h = keep * h + (1 - keep) / np.sqrt(self.D)
        p = self.l2(h, A)[:, 0] ** 2
        return torch.log((p + EPS) / (1 + self.D * EPS))


# ------------------------------------------------------------------ wrappers giving log-probabilities
class CompMPComp(nn.Module):
    def __init__(self, D, hidden, dropout, n_scalar):
        super().__init__()
        self.m = CompMP(1, D, max(1, hidden // (D - 1)), 1, dropout, "part", "full", out="comp",
                        n_scalar=n_scalar, skip=True)

    def forward(self, h, A, s):
        return F.log_softmax(self.m(h, A, s)[:, 0], -1)


class SkipGCN(nn.Module):
    """2-layer GCN with a self-connection in each layer: H' = A H W + H W_self + b (as in GraphSAGE), so a
    node's own features are not diluted by its neighbours. ReLU -> dropout between the layers."""
    def __init__(self, i, hidden, out, dropout):
        super().__init__()
        self.a1, self.s1 = nn.Linear(i, hidden), nn.Linear(i, hidden, bias=False)
        self.a2, self.s2 = nn.Linear(hidden, out), nn.Linear(hidden, out, bias=False)
        for m in (self.a1, self.s1, self.a2, self.s2):
            nn.init.xavier_uniform_(m.weight)
        self.dp, self._ax = dropout, None

    def forward(self, x, A):
        if self._ax is None:
            self._ax = A @ x
        h = F.dropout(F.relu(self.a1(self._ax) + self.s1(x)), self.dp, self.training)
        return self.a2(A @ h) + self.s2(h)


class GCNilr(nn.Module):
    """GCN (with self-connections) on [ilr(prev), covariates]; outputs ilr coordinates -> composition."""
    def __init__(self, i, D, hidden, dropout):
        super().__init__()
        self.g = SkipGCN(i, hidden, D - 1, dropout)
        self.register_buffer("Psi", torch.tensor(helmert_basis(D), dtype=torch.float32))

    def forward(self, x, A, s=None):
        return F.log_softmax(self.g(x, A) @ self.Psi, -1)


class GCNsoftmax(nn.Module):
    """STdGCN-style: GCN on [raw prev shares, covariates], softmax output (trained with KL)."""
    def __init__(self, i, D, hidden, dropout):
        super().__init__()
        self.g = SkipGCN(i, hidden, D, dropout)

    def forward(self, x, A, s=None):
        return F.log_softmax(self.g(x, A), -1)


# ------------------------------------------------------------------ losses and metrics
def loss_fn(name, lp, shares, counts):
    if name == "aitchison":
        return ((center(lp) - center(torch.log(shares))) ** 2).sum(1).mean()
    if name == "kl":
        return (shares * (torch.log(shares) - lp)).sum(1).mean()
    if name == "multinomial":          # count-weighted: per-vote negative log-likelihood
        return -(counts * lp).sum() / counts.sum()
    if name == "hellinger":
        return ((torch.exp(lp / 2) - torch.sqrt(shares)) ** 2).sum(1).mean()
    raise ValueError(name)


def metrics(p, s, prefix=""):
    """p, s: (m, D) predicted and true shares (numpy). Aitchison distance uses p clipped at EPS."""
    pc = np.clip(p, EPS, None); pc = pc / pc.sum(1, keepdims=True)
    return {f"{prefix}aitchison": float(np.mean(np.linalg.norm(clr(pc) - clr(s), axis=1))),
            f"{prefix}kl": float(np.mean((s * (np.log(s) - np.log(pc))).sum(1))),
            f"{prefix}hellinger": float(np.mean(np.sqrt(0.5 * ((np.sqrt(p) - np.sqrt(s)) ** 2).sum(1)))),
            f"{prefix}mae": float(np.mean(np.abs(p - s)))}


def evaluate(p, el, split):
    s = el.shares
    return {**metrics(p[split[1]], s[split[1]], "val_"), **metrics(p[split[2]], s[split[2]])}


# ------------------------------------------------------------------ neural training (protocol of train.py)
NEURAL = {"CompMP-full": ("aitchison", "kl", "multinomial"),
          "CompMP-FR": ("hellinger", "kl", "multinomial"),
          "GCN-ilr": ("aitchison", "kl", "multinomial"),
          "GCN-softmax": ("kl",)}


def inputs(name, el):
    s = torch.tensor(el.covars, dtype=torch.float32)
    if name == "CompMP-full":
        return torch.tensor(clr(el.prev)[:, None, :], dtype=torch.float32), s
    if name == "CompMP-FR":
        return torch.tensor(np.sqrt(el.prev_raw)[:, None, :], dtype=torch.float32), s
    if name == "GCN-ilr":
        return torch.tensor(np.concatenate([ilr(el.prev), el.covars], 1), dtype=torch.float32), s
    if name == "GCN-softmax":
        return torch.tensor(np.concatenate([el.prev, el.covars], 1), dtype=torch.float32), s
    raise ValueError(name)


def build(name, x, hp, D=3, n_scalar=6):
    if name == "CompMP-full":
        return CompMPComp(D, hp["hidden"], hp["dropout"], n_scalar)
    if name == "CompMP-FR":
        return SphereGNN(D, max(1, hp["hidden"] // (D - 1)), hp["dropout"], n_scalar)
    if name == "GCN-ilr":
        return GCNilr(x.shape[1], D, hp["hidden"], hp["dropout"])
    return GCNsoftmax(x.shape[1], D, hp["hidden"], hp["dropout"])


def fit_neural(name, loss, el, A, split, hp, seed):
    torch.manual_seed(seed)
    x, s = inputs(name, el)
    model = build(name, x, hp)
    shares = torch.tensor(el.shares, dtype=torch.float32)
    counts = torch.tensor(el.counts, dtype=torch.float32)
    tr, va, _ = (torch.tensor(i) for i in split)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=DEFAULT_HP["weight_decay"])
    best, best_state, wait = np.inf, None, 0
    for ep in range(DEFAULT_HP["max_epochs"]):
        model.train(); opt.zero_grad()
        loss_fn(loss, model(x, A, s)[tr], shares[tr], counts[tr]).backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        opt.step(); model.eval()
        with torch.no_grad():
            vl = loss_fn(loss, model(x, A, s)[va], shares[va], counts[va]).item()
        if vl < best:
            best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= DEFAULT_HP["patience"]:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        p = torch.exp(model(x, A, s)).numpy().astype(float)
    p = p / p.sum(1, keepdims=True)
    return {"val_loss": best, "epochs": ep + 1, **evaluate(p, el, split)}


# ------------------------------------------------------------------ non-neural baselines
def _from_clr(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def fit_constant(el, split):
    p = np.tile(_from_clr(clr(el.shares[split[0]]).mean(0, keepdims=True)), (el.n, 1))   # closed geometric mean
    return evaluate(p, el, split)


def fit_persistence(el, split):
    return evaluate(el.prev, el, split)


def fit_linear(el, split, lam):
    """Ridge regression of ilr(2016 shares) on [ilr(2012 shares), covariates, 1]; no graph."""
    X = np.concatenate([ilr(el.prev), el.covars, np.ones((el.n, 1))], 1)
    Y = ilr(el.shares)
    tr = split[0]
    B = np.linalg.solve(X[tr].T @ X[tr] + lam * np.eye(X.shape[1]), X[tr].T @ Y[tr])
    return evaluate(_from_clr((X @ B) @ helmert_basis(3)), el, split)


def _propagate(A, Y, mask, alpha, iters=100):
    """Label propagation (Zhou et al.): F <- alpha A F + (1 - alpha) Y, for the labels and the mask."""
    F_, M = Y.copy(), mask.astype(float)[:, None]
    m = M.copy()
    for _ in range(iters):
        F_ = alpha * (A @ F_) + (1 - alpha) * Y
        m = alpha * (A @ m) + (1 - alpha) * M
    return F_, m


def fit_lp(el, A, split, alpha, delta=False):
    """Label propagation in clr. delta=True propagates the 2012 -> 2016 change on top of persistence."""
    tr = split[0]
    target = clr(el.shares) - (clr(el.prev) if delta else 0)
    Y = np.zeros_like(target); Y[tr] = target[tr]
    mask = np.zeros(el.n, bool); mask[tr] = True
    F_, m = _propagate(A, Y, mask, alpha)
    fallback = target[tr].mean(0, keepdims=True)
    Z = np.where(m > 1e-8, F_ / np.maximum(m, 1e-8), fallback)
    Z[tr] = target[tr]
    return evaluate(_from_clr(Z + (clr(el.prev) if delta else 0)), el, split)


def fit_lap_mnl(el, edges, split, lam, seed=0):
    """Laplacian-regularized multinomial logit (Tomlinson & Benson style): logits = X B + theta_i, with
    theta smoothed over the graph by lam * sum_edges ||theta_i - theta_j||^2. KL loss on training counties."""
    torch.manual_seed(seed)
    X = torch.tensor(np.concatenate([ilr(el.prev), el.covars, np.ones((el.n, 1))], 1), dtype=torch.float32)
    S = torch.tensor(el.shares, dtype=torch.float32)
    tr = torch.tensor(split[0])
    e = torch.tensor(edges)
    B = torch.zeros(X.shape[1], 3, requires_grad=True)
    th = torch.zeros(el.n, 3, requires_grad=True)
    opt = torch.optim.LBFGS([B, th], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        lp = F.log_softmax(X @ B + th, -1)
        l = loss_fn("kl", lp[tr], S[tr], None) + lam * ((th[e[:, 0]] - th[e[:, 1]]) ** 2).sum() / el.n \
            + 1e-4 * (th ** 2).sum() / el.n
        l.backward()
        return l
    opt.step(closure)
    with torch.no_grad():
        p = torch.softmax(X @ B + th, -1).numpy().astype(float)
    return evaluate(p, el, split)
