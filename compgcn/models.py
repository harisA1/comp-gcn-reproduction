"""COMP-GCN (Lu, Wang & Zhao, Information Fusion 2025) and the competing models.

Every model takes (x, A) where x is the node-feature matrix it was built for
(raw compositions, ILR coordinates, or a constant) and A the normalized adjacency.
Dropout sits before the output layer in all models (Sec. 3.4).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def prop(A, x):
    """A @ x for dense or sparse A."""
    return torch.sparse.mm(A, x) if A.is_sparse else A @ x


def xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class GCNLayer(nn.Module):
    """H' = A H W + b. With cache=True (first layer, fixed input) A H is computed once."""
    def __init__(self, i, o, bias=True, cache=False):
        super().__init__()
        self.lin = nn.Linear(i, o, bias=False)
        self.bias = nn.Parameter(torch.zeros(o)) if bias else None
        self.cache, self._ax = cache, None

    def forward(self, x, A):
        if self.cache:
            if self._ax is None:
                self._ax = prop(A, x)
            h = self.lin(self._ax)
        else:
            h = prop(A, self.lin(x))
        return h if self.bias is None else h + self.bias


class CGCNLayer(nn.Module):
    """Compositional graph convolution, Eq. (7): H = A ILR(X) ILR(W)'.

    Input is ILR(X), shape (n, M - p). Each block j of D_j - 1 ilr coordinates is
    propagated with A and projected onto its own weight vector ilr(w_j), giving one
    scalar per composition: output shape (n, p). Equivalent to a GCN layer whose
    weight matrix is block-diagonal with one output column per composition.
    """
    def __init__(self, D):
        super().__init__()
        sizes = [d - 1 for d in D]
        self.p = len(D)
        self.register_buffer("block", torch.repeat_interleave(torch.arange(self.p), torch.tensor(sizes)))
        # Glorot-uniform per block (fan_in = D_j - 1, fan_out = 1)
        bound = torch.tensor([np.sqrt(6.0 / (s + 1)) for s in sizes], dtype=torch.float32).repeat_interleave(torch.tensor(sizes))
        self.w = nn.Parameter((torch.rand(sum(sizes)) * 2 - 1) * bound)
        self._az = None

    def forward(self, z, A):
        if self._az is None:           # input is fixed, so A ILR(X) is computed once
            self._az = prop(A, z)
        h = self._az * self.w                       # (n, M-p)
        out = torch.zeros(h.shape[0], self.p, dtype=h.dtype, device=h.device)
        return out.index_add_(1, self.block, h)                   # block-wise sums -> (n, p)


class COMPGCN(nn.Module):
    """Fig. 3(a): CGCN -> GCN -> ReLU -> dropout -> Linear."""
    def __init__(self, D, hidden, out, dropout):
        super().__init__()
        self.cgcn = CGCNLayer(D)
        self.gcn = GCNLayer(len(D), hidden)
        self.out = nn.Linear(hidden, out)
        self.dp = dropout
        self.apply(xavier)

    def forward(self, z, A):
        h = F.relu(self.gcn(self.cgcn(z, A), A))
        return self.out(F.dropout(h, self.dp, self.training))


class GCN(nn.Module):
    """Fig. 3(b): GCN -> ReLU -> dropout -> GCN. Used for GCN-scalar / GCN-ilr / GCN-graph."""
    def __init__(self, i, hidden, out, dropout):
        super().__init__()
        self.g1, self.g2, self.dp = GCNLayer(i, hidden, cache=True), GCNLayer(hidden, out), dropout
        self.apply(xavier)

    def forward(self, x, A):
        h = F.relu(self.g1(x, A))
        return self.g2(F.dropout(h, self.dp, self.training), A)


class MLP(nn.Module):
    """Fig. 3(c): Linear -> ReLU -> dropout -> Linear."""
    def __init__(self, i, hidden, out, dropout):
        super().__init__()
        self.l1, self.l2, self.dp = nn.Linear(i, hidden), nn.Linear(hidden, out), dropout
        self.apply(xavier)

    def forward(self, x, A=None):
        return self.l2(F.dropout(F.relu(self.l1(x)), self.dp, self.training))


class SGC(nn.Module):
    """Wu et al. 2019: A^K X W, K = 2 (propagation precomputed and cached)."""
    def __init__(self, i, hidden, out, dropout, K=2):
        super().__init__()
        self.lin, self.K, self.dp, self._cache = nn.Linear(i, out), K, dropout, None
        self.apply(xavier)

    def forward(self, x, A):
        if self._cache is None:
            h = x
            for _ in range(self.K):
                h = prop(A, h)
            self._cache = h
        return self.lin(F.dropout(self._cache, self.dp, self.training))


class GCNIIConv(nn.Module):
    def __init__(self, h, alpha, beta):
        super().__init__()
        self.W, self.alpha, self.beta = nn.Linear(h, h, bias=False), alpha, beta

    def forward(self, x, x0, A):
        s = (1 - self.alpha) * prop(A, x) + self.alpha * x0
        return (1 - self.beta) * s + self.beta * self.W(s)


class GCNII(nn.Module):
    """Chen et al. 2020: Linear -> 2 x GCNII layers -> ReLU -> dropout -> Linear."""
    def __init__(self, i, hidden, out, dropout, alpha=0.1, lam=0.5):
        super().__init__()
        self.inp = nn.Linear(i, hidden)
        self.convs = nn.ModuleList([GCNIIConv(hidden, alpha, float(np.log(lam / l + 1))) for l in (1, 2)])
        self.out, self.dp = nn.Linear(hidden, out), dropout
        self.apply(xavier)

    def forward(self, x, A):
        h0 = h = F.relu(self.inp(x))
        for c in self.convs:
            h = F.relu(c(h, h0, A))
        return self.out(F.dropout(h, self.dp, self.training))
