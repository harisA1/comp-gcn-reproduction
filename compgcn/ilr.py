"""Isometric log-ratio (ilr) transform for blocks of compositions."""
import numpy as np


def helmert_basis(D: int) -> np.ndarray:
    """Orthonormal basis of the zero-sum hyperplane in R^D, shape (D-1, D).

    Row i is clr(xi_i) for the i-th element of an orthonormal basis of the
    simplex (the contrast matrix Psi in the paper, Definition 3).
    """
    Psi = np.zeros((D - 1, D))
    for i in range(1, D):
        Psi[i - 1, :i] = 1.0 / i
        Psi[i - 1, i] = -1.0
        Psi[i - 1] *= np.sqrt(i / (i + 1.0))
    return Psi


def random_basis(D: int, rng: np.random.Generator) -> np.ndarray:
    """A random orthonormal basis of the zero-sum hyperplane (Helmert rotated)."""
    Q, _ = np.linalg.qr(rng.standard_normal((D - 1, D - 1)))
    return Q @ helmert_basis(D)


def clr(x: np.ndarray) -> np.ndarray:
    lx = np.log(x)
    return lx - lx.mean(-1, keepdims=True)


def ilr(x: np.ndarray, Psi: np.ndarray | None = None) -> np.ndarray:
    """x: (..., D) compositions -> (..., D-1) ilr coordinates."""
    if Psi is None:
        Psi = helmert_basis(x.shape[-1])
    return clr(x) @ Psi.T


def ilr_inv(z: np.ndarray, Psi: np.ndarray | None = None) -> np.ndarray:
    if Psi is None:
        Psi = helmert_basis(z.shape[-1] + 1)
    e = np.exp(z @ Psi)
    return e / e.sum(-1, keepdims=True)


def block_ilr(X: np.ndarray, D: list[int], bases=None) -> np.ndarray:
    """ILR(X) of the paper: X is (n, M) with p blocks of sizes D -> (n, M - p)."""
    out, s = [], 0
    for j, d in enumerate(D):
        out.append(ilr(X[:, s:s + d], None if bases is None else bases[j]))
        s += d
    return np.concatenate(out, 1)


def block_ilr_inv(Z: np.ndarray, D: list[int]) -> np.ndarray:
    out, s = [], 0
    for d in D:
        out.append(ilr_inv(Z[:, s:s + d - 1]))
        s += d - 1
    return np.concatenate(out, 1)
