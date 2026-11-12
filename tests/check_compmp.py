"""Checks for CompMP: basis (in)variance of the nonlinearities and the barycenter aggregation."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # run from anywhere
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

# 4) equivalence: a linear CompLayer with the full zero-sum map == a dense linear GCN layer on ilr coordinates.
#    Any ilr weight G (blocks (D-1) x (D-1)) and bias c is realized by W = Psi' G Psi and b = Psi' c.
from compgcn.compmp import CompLayer
from compgcn.ilr import helmert_basis
Psi = helmert_basis(4)
Pt = torch.tensor(Psi, dtype=torch.float32)
G, c = torch.randn(6, 4, 3, 3), torch.randn(6, 3)
layer = CompLayer(4, 6, 4, None, "full")
with torch.no_grad():
    layer.W.copy_(torch.einsum("da,kjab,be->kjde", Pt.T, G, Pt))
    layer.bias.copy_(c @ Pt)
    out_ilr = layer(H, A) @ Pt.T                                        # CompMP output in ilr coordinates
    Z = H @ Pt.T                                                        # ilr(X), (n, p, D-1)
    AZ = (A @ Z.reshape(d.n, -1)).reshape(Z.shape)
    ref = torch.einsum("kjab,njb->nka", G, AZ) + c                      # GCN layer on ilr: A Z G' + c
assert torch.allclose(out_ilr, ref, atol=1e-4), (out_ilr - ref).abs().max()
print("linear full CompLayer == dense linear GCN layer on ilr coordinates")

# 5) the basis-free nonlinearities commute with permutations of the parts (S_D-equivariance)
z = H[:50]
perm = torch.tensor([2, 0, 3, 1])
for nl in ("part", "norm"):
    act = CompLayer(4, 4, 4, nl).act
    with torch.no_grad():
        assert torch.allclose(act(z[..., perm]), act(z)[..., perm], atol=1e-6), nl
# the full model is still exactly basis-invariant (it never uses a basis)
assert np.allclose(outputs("part", "full", B1), outputs("part", "full", B2))
print("CompMP checks passed")
