"""Training protocol of Sec. 4.3 and a registry of all compared methods."""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from .ilr import block_ilr
from .models import COMPGCN, GCN, MLP, SGC, GCNII

# Hyperparameters from the Fig. 6 caption (the only fixed setting the paper reports)
DEFAULT_HP = dict(lr=0.005, weight_decay=5e-4, hidden=8, dropout=0.2,
                  max_epochs=10000, patience=50)

# name -> (model class, input representation)
NEURAL = {
    "COMP-GCN":   (COMPGCN, "ilr"),
    "GCN-ilr":    (GCN, "ilr"),      # missing baseline: ilr inputs + ordinary GCN
    "GCN-scalar": (GCN, "raw"),
    "GCN-graph":  (GCN, "ones"),
    "MLP":        (MLP, "raw"),
    "MLP-ilr":    (MLP, "ilr"),
    "SGC":        (SGC, "raw"),
    "GCNII":      (GCNII, "raw"),
}
LINEAR = ["linear-ilr"]   # logistic-ilr (classification) / OLS-ilr (regression)
CONSTANT = ["constant"]   # majority class / training mean


def features(data, kind, bases=None):
    if kind == "raw":
        return data.X
    if kind == "ilr":
        return block_ilr(data.X, data.D, bases)
    if kind == "ones":
        return np.ones((data.n, 1))
    raise ValueError(kind)


def metrics(data, pred, idx):
    y = data.y[idx]
    if data.task == "reg":
        return {"mse": float(np.mean((pred[idx] - y) ** 2))}
    return {"acc": float(np.mean(pred[idx] == y)),
            "bal_acc": float(balanced_accuracy_score(y, pred[idx]))}


def fit_neural(name, data, A, split, hp=DEFAULT_HP, seed=0, bases=None, verbose=False):
    torch.manual_seed(seed)
    if A.is_sparse and data.n <= 6000:
        A = A.to_dense()               # these graphs are small and dense-ish: dense matmul is faster
    cls, kind = NEURAL[name]
    x = torch.tensor(features(data, kind, bases), dtype=torch.float32)
    out = data.n_classes if data.task == "clf" else 1
    model = COMPGCN(data.D, hp["hidden"], out, hp["dropout"]) if cls is COMPGCN \
        else cls(x.shape[1], hp["hidden"], out, hp["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    tr, va, te = (torch.tensor(s) for s in split)
    if data.task == "clf":
        y = torch.tensor(data.y, dtype=torch.long)
        lossf = F.cross_entropy
    else:
        y = torch.tensor(data.y, dtype=torch.float32)
        lossf = lambda o, t: F.mse_loss(o.squeeze(-1), t)

    best, best_state, wait = np.inf, None, 0
    for ep in range(hp["max_epochs"]):
        model.train(); opt.zero_grad()
        loss = lossf(model(x, A)[tr], y[tr])
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(x, A)[va], y[va]).item()
        if vl < best:
            best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= hp["patience"]:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        o = model(x, A)
    pred = o.argmax(1).numpy() if data.task == "clf" else o.squeeze(-1).numpy()
    m = metrics(data, pred, split[2]); m["epochs"] = ep + 1
    return m


def fit_linear(data, split, bases=None):
    Z = features(data, "ilr", bases)
    tr = split[0]
    if data.task == "clf":
        model = LogisticRegression(max_iter=5000).fit(Z[tr], data.y[tr])
    else:
        model = LinearRegression().fit(Z[tr], data.y[tr])
    return metrics(data, model.predict(Z), split[2])


def fit_constant(data, split):
    tr = split[0]
    if data.task == "clf":
        pred = np.full(data.n, np.bincount(data.y[tr]).argmax())
    else:
        pred = np.full(data.n, data.y[tr].mean())
    return metrics(data, pred, split[2])


def fit(name, data, A, split, hp=DEFAULT_HP, seed=0, bases=None):
    if name in NEURAL:
        return fit_neural(name, data, A, split, hp, seed, bases)
    if name == "linear-ilr":
        return fit_linear(data, split, bases)
    if name == "constant":
        return fit_constant(data, split)
    raise ValueError(name)
