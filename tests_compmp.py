"""Checks for CompMP: basis (in)variance of the nonlinearities and the barycenter aggregation."""
import numpy as np, torch
from compgcn.ilr import clr, random_basis
from compgcn.data import synthetic
from compgcn.compmp import CompMP, adjacency, clr_blocks

torch.manual_seed(0)
rng = np.random.default_rng(0)
d = synthetic(n=200, p=4, D=4, seed=1)
A = adjacency(d.edges, d.n, "row")
H = torch.tensor(clr_blocks(d.X, d.D), dtype=torch.float32)
B1, B2 = random_basis(4, rng), random_basis(4, rng)

# 1) aggregation in clr == closed weighted geometric mean of the neighbours (rows of A sum to 1)
Ad = A.numpy().astype(float)
x = d.X[:, :4]
geo = np.exp(Ad @ np.log(x)); geo /= geo.sum(1, keepdims=True)
assert np.allclose(clr(geo), Ad @ clr(x), atol=1e-8)
assert np.allclose(A.sum(1), 1, atol=1e-6)


def outputs(nonlin, within, basis):
    torch.manual_seed(1)
    m = CompMP(4, 4, 8, 2, 0.0, nonlin, within, basis).eval()
    with torch.no_grad():
        return m(H, A).numpy()


# 2) "norm" and "part" are exactly basis-invariant; "naive" is not
for within in ("off", "per"):
    for nl in ("norm", "part"):
        assert np.allclose(outputs(nl, within, B1), outputs(nl, within, B2)), (nl, within)
    diff = np.abs(outputs("naive", within, B1) - outputs("naive", within, B2)).max()
    assert diff > 1e-3, diff
    print(f"within={within}: naive output differs between two bases by up to {diff:.3f}; norm/part identical")

# 3) states stay compositions: every intermediate clr vector sums to zero
m = CompMP(4, 4, 8, 1, 0.2, "norm", "per", out="comp").train()
assert torch.allclose(m(H, A).sum(-1), torch.zeros(1), atol=1e-5)
print("CompMP checks passed")
