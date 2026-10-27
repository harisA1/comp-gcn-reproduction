"""Compositional message passing (CompMP): node states stay compositions in every layer.

A node state is p compositions of D parts, stored as clr vectors: a tensor (n, p, D) whose
last axis sums to zero. Storing clr avoids choosing an ilr basis; only the naive
nonlinearity uses one.

One layer (Aitchison-linear part, then a nonlinearity):
  aggregation   A_hat @ clr        (row-normalized A_hat: Aitchison barycenter of the neighbours)
  mixing        out[k] = sum_j alpha_kj * in[j]          (powering + perturbation)
  within (opt.) z -> P M z, M (D x D) learnable           (shared or one per output composition)
  bias          z -> z + P b                              (perturbation by a learnable composition)
  nonlinearity  "naive": ilr with a fixed basis, ReLU per coordinate, back to clr (basis-dependent)
                "norm":  z -> g(|z|) z                     (rotation-invariant, basis-free)
                "part":  z -> P f(z), f = LeakyReLU per part (basis-free, permutation-equivariant)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ilr import clr, helmert_basis

NONLINS = ("naive", "norm", "part")


def center(z):
    """Apply P = I - 11'/D on the last axis."""
    return z - z.mean(-1, keepdim=True)


def clr_blocks(X, D):
    """(n, p*D) raw compositions with p equal blocks of D parts -> (n, p, D) clr."""
    assert len(set(D)) == 1, "CompMP needs blocks of equal size"
    n, d = X.shape[0], D[0]
    return clr(X.reshape(n, len(D), d))


def adjacency(edges, n, mode="row"):
    """Dense normalized adjacency with self-loops (receivers are rows), undirected.

    mode="row": D^-1 (A+I), rows sum to 1, so A_hat @ clr is a true Aitchison barycenter.
    mode="sym": D^-1/2 (A+I) D^-1/2, as in GCN.
    """
    A = np.zeros((n, n), dtype=np.float32)
    A[edges[:, 0], edges[:, 1]] = 1
    A[edges[:, 1], edges[:, 0]] = 1
    np.fill_diagonal(A, 1)
    deg = A.sum(1)
    if mode == "row":
        A = A / deg[:, None]
    elif mode == "sym":
        s = 1 / np.sqrt(deg)
        A = A * s[:, None] * s[None, :]
    else:
        raise ValueError(mode)
    return torch.tensor(A)


def comp_dropout(h, p, training):
    """Drop whole compositions (set to the neutral element, clr = 0), rescale the rest."""
    if not training or p == 0:
        return h
    keep = (torch.rand(h.shape[0], h.shape[1], 1) > p).to(h.dtype)
    return h * keep / (1 - p)


class CompLayer(nn.Module):
    def __init__(self, p_in, p_out, D, nonlin=None, within="off", basis=None):
        super().__init__()
        self.within = within
        if within == "full":
            # every input -> output channel pair gets its own D x D map, output projected back to the
            # zero-sum hyperplane: the general zero-sum-preserving linear map (as expressive as a dense
            # linear layer on ilr coordinates)
            bound = np.sqrt(6.0 / ((p_in + p_out) * D))
            self.W = nn.Parameter((torch.rand(p_out, p_in, D, D) * 2 - 1) * bound)
        else:
            self.alpha = nn.Parameter(torch.empty(p_out, p_in))
            nn.init.xavier_uniform_(self.alpha)
        if within == "shared":
            self.M = nn.Parameter(torch.eye(D))
        elif within == "per":
            self.M = nn.Parameter(torch.eye(D).repeat(p_out, 1, 1))
        elif within not in ("off", "full"):
            raise ValueError(within)
        self.bias = nn.Parameter(torch.zeros(p_out, D))
        self.nonlin = nonlin
        if nonlin == "naive":
            Psi = helmert_basis(D) if basis is None else basis
            self.register_buffer("Psi", torch.tensor(Psi, dtype=torch.float32))   # (D-1, D)
        elif nonlin == "norm":
            # output norm rho(r) = softplus(a(r - c)) - softplus(-a c): rho(0) = 0, smooth, ~ a(r - c) for large r
            self.a_raw = nn.Parameter(torch.full((p_out,), 0.5413))   # softplus^-1(1)
            self.c = nn.Parameter(torch.zeros(p_out))
        elif nonlin not in (None, "part"):
            raise ValueError(nonlin)
        self._ah = None

    def forward(self, h, A, cache=False):
        n = h.shape[0]
        if cache:                       # first layer: the input is fixed, compute A @ clr once
            if self._ah is None:
                self._ah = (A @ h.reshape(n, -1)).reshape(h.shape)
            ah = self._ah
        else:
            ah = (A @ h.reshape(n, -1)).reshape(h.shape)
        if self.within == "full":
            z = center(torch.einsum("kjde,nje->nkd", self.W, ah))
        else:
            z = torch.einsum("kj,njd->nkd", self.alpha, ah)
        if self.within == "shared":
            z = center(z @ self.M.T)
        elif self.within == "per":
            z = center(torch.einsum("kde,nke->nkd", self.M, z))
        z = z + center(self.bias)
        return self.act(z)

    def act(self, z):
        if self.nonlin == "naive":
            return F.relu(z @ self.Psi.T) @ self.Psi
        if self.nonlin == "norm":
            r = z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            a, c = F.softplus(self.a_raw)[:, None], self.c[:, None]
            rho = F.softplus(a * (r - c)) - F.softplus(-a * c)
            return z * rho / r
        if self.nonlin == "part":
            return center(F.leaky_relu(z, 0.1))
        return z


class CompMP(nn.Module):
    """CompLayer (p -> hidden, nonlinearity) -> composition dropout -> CompLayer (hidden -> q, linear) -> readout.

    out="scalar": q = hidden; the readout is a sum of Aitchison inner products of each output
      composition with learned compositions, i.e. a Linear layer on the flattened clr state.
    out="comp":   q = number of target compositions; the output is the clr of the prediction
      (prediction = softmax of it).
    layers > 2 inserts hidden -> hidden CompLayers with the nonlinearity, each with a residual
    connection in clr (h + layer(h), i.e. perturbation), against over-smoothing towards the
    global Aitchison barycenter.
    head (scalar targets only): "stack" = CompLayer (hidden -> hidden) then Linear readout (used in E1);
      "prop" = Aitchison inner products with learned compositions, then propagation, like GCN's second
      layer (A H W). "stack" stacks two linear maps and overfits small training sets much faster.
    """
    def __init__(self, p, D, hidden, n_out, dropout, nonlin="part", within="off", basis=None, out="scalar",
                 layers=2, head="stack"):
        super().__init__()
        self.l1 = CompLayer(p, hidden, D, nonlin, within, basis)
        self.mid = nn.ModuleList([CompLayer(hidden, hidden, D, nonlin, within, basis) for _ in range(layers - 2)])
        q = hidden if out == "scalar" else n_out
        self.head = head if out == "scalar" else "stack"
        self.l2 = CompLayer(hidden, q, D, None, within) if self.head == "stack" else None
        self.readout = nn.Linear(q * D, n_out) if out == "scalar" else None
        if self.readout is not None:
            nn.init.xavier_uniform_(self.readout.weight); nn.init.zeros_(self.readout.bias)
        self.dp = dropout

    def forward(self, h, A):
        h = self.l1(h, A, cache=True)
        for layer in self.mid:
            h = h + layer(comp_dropout(h, self.dp, self.training), A)
        h = comp_dropout(h, self.dp, self.training)
        if self.head == "prop":
            return A @ self.readout(h.reshape(h.shape[0], -1))
        h = self.l2(h, A)
        if self.readout is None:
            return h                                  # (n, q, D) clr of the predicted compositions
        return self.readout(h.reshape(h.shape[0], -1))


def aitchison_loss(pred_clr, target_clr):
    """Mean squared Aitchison distance; inputs (n, q, D) clr."""
    return ((pred_clr - target_clr) ** 2).sum(-1).mean()
