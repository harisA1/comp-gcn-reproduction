"""Summarize the CompMP experiments into results/compmp_summary.md.

E1: for each setting and model, the spread of test accuracy across ilr bases (seed and split fixed)
    against the spread across seeds.
E2: per model and setting, the grid config with the lowest mean validation loss over seeds is
    selected (validation only); its test accuracy is reported, plus paired differences to GCN-ilr.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

OUT = "results/compmp_summary.md"


def e1_table(df):
    rows = []
    for (s, m), g in df.groupby(["setting", "model"], sort=False):
        per_seed = g.groupby("seed")["acc"]
        rows.append(dict(setting=s, model=m, bases=g["basis"].nunique(),
                         mean=g["acc"].mean(),
                         sd_bases=per_seed.std(ddof=1).mean(),        # within a seed, across bases
                         range_bases=(per_seed.max() - per_seed.min()).mean(),
                         sd_seeds=per_seed.mean().std(ddof=1)))       # across seeds, basis-averaged
    return pd.DataFrame(rows)


def select(df):
    """Rows of the validation-selected config, one per (setting, model, seed)."""
    hp = ["lr", "hidden", "dropout", "C"]
    d = df.copy()
    d[hp] = d[hp].fillna("-").astype(str)
    d["val_loss"] = d["val_loss"].fillna(0)            # constant: a single config
    mean_vl = d.groupby(["setting", "model"] + hp)["val_loss"].mean().reset_index()
    best = mean_vl.loc[mean_vl.groupby(["setting", "model"])["val_loss"].idxmin()]
    return d.merge(best[["setting", "model"] + hp], on=["setting", "model"] + hp), best


def paired(sel, ref="GCN-ilr"):
    rows = []
    for (s, m), g in sel.groupby(["setting", "model"]):
        if m == ref:
            continue
        r = sel[(sel.setting == s) & (sel.model == ref)].set_index("seed")["acc"]
        diff = (g.set_index("seed")["acc"] - r).dropna()
        h = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
        rows.append(dict(setting=s, model=m, diff=diff.mean(), lo=diff.mean() - h, hi=diff.mean() + h))
    return pd.DataFrame(rows)


def main():
    out = ["# CompMP results\n"]
    if os.path.exists("results/compmp_e1.csv"):
        t = e1_table(pd.read_csv("results/compmp_e1.csv"))
        out += ["## E1: basis dependence (test accuracy, fixed hyperparameters lr 0.005, hidden 8, dropout 0.2)\n",
                "`sd_bases`: std across ilr bases with seed, split and initialization fixed, averaged over 5 seeds. "
                "`range_bases`: max - min across bases, averaged over seeds. `sd_seeds`: std across seeds of the "
                "basis-averaged accuracy. Basis 0 is Helmert, the others random. CompMP-b/c were run on 3 bases "
                "(they cannot depend on the basis).\n",
                t.to_markdown(index=False, floatfmt=".4f"), "\n"]
    if os.path.exists("results/compmp_e2.csv"):
        df = pd.read_csv("results/compmp_e2.csv")
        sel, best = select(df)
        order = ["CompMP-a", "CompMP-b", "CompMP-c", "CompMP-c-M", "GCN-ilr", "COMP-GCN", "GCN-scalar",
                 "MLP-ilr", "linear-ilr", "constant"]
        settings = list(dict.fromkeys(df["setting"]))
        g = sel.groupby(["model", "setting"])["acc"]
        cell = (g.mean().map("{:.3f}".format) + " ± " + g.std(ddof=1).map("{:.3f}".format)).unstack()
        cell = cell.reindex(index=[m for m in order if m in cell.index], columns=settings)
        n = sel.groupby(["model", "setting"]).size()
        out += ["## E2: scalar targets, n = 2000 (test accuracy, mean ± std over seeds)\n",
                f"Validation-selected config per model and setting (8-config grid: lr {{0.005, 0.01}} × hidden "
                f"{{8, 16}} × dropout {{0.2, 0.5}}; linear-ilr: 4 values of C). Seeds per cell: {n.min()}–{n.max()}.\n"]
        for half in ("p=10", "p=100"):
            cols = [c for c in settings if c.startswith(half + ",")]
            out += [f"### {half}\n", cell[cols].to_markdown(), "\n"]
        p = paired(sel)
        p["cell"] = p.apply(lambda r: f"{r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]", axis=1)
        pt = p.pivot(index="model", columns="setting", values="cell")
        pt = pt.reindex(index=[m for m in order if m in pt.index], columns=settings)
        out += ["### Paired difference to GCN-ilr (same seeds and splits), mean [95% CI]\n"]
        for half in ("p=10", "p=100"):
            cols = [c for c in settings if c.startswith(half + ",")]
            out += [f"#### {half}\n", pt[cols].to_markdown(), "\n"]
        wins = p.assign(w=p.lo > 0, l=p.hi < 0).groupby("model")[["w", "l"]].sum().astype(int)
        wins["ties"] = p.groupby("model").size() - wins.w - wins.l
        out += ["Settings where the CI excludes 0 (win / loss vs GCN-ilr):\n",
                wins.rename(columns={"w": "wins", "l": "losses"}).reindex([m for m in order if m in wins.index]).to_markdown(), "\n"]
        best = best.drop(columns="val_loss")
        out += ["### Selected configs\n", best.to_markdown(index=False), "\n"]
    open(OUT, "w").write("\n".join(out))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
