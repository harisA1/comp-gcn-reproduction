"""Summarize E3b (county vote shares) into results/compmp_e3_summary.md.

Per (split, model, loss), the config with the lowest mean validation criterion over seeds is selected:
neural models use their own validation loss, ridge / LP / constant / persistence the validation Aitchison
distance, Lap-MNL the validation KL (its training loss). Test metrics are reported for that config only.
Paired differences use the seeds both models have.
"""
import numpy as np
import pandas as pd
from scipy import stats

KEY = ["split", "seed", "model", "loss", "lr", "hidden", "dropout", "param"]
HP = ["lr", "hidden", "dropout", "param"]
METRICS = ["aitchison", "kl", "hellinger", "mae"]


def load(path="results/compmp_e3.csv"):
    d = pd.read_csv(path, keep_default_na=False)
    num = ["val_loss", "val_aitchison", "val_kl"] + METRICS
    d[num] = d[num].apply(pd.to_numeric, errors="coerce")
    d = d.drop_duplicates(subset=KEY)
    d["sel"] = d.val_loss.where(d.val_loss.notna(), d.val_aitchison)
    d.loc[d.model == "Lap-MNL", "sel"] = d.loc[d.model == "Lap-MNL", "val_kl"]
    d["name"] = np.where(d.loss != "", d.model + " (" + d.loss + ")", d.model)
    return d


def select(d):
    g = d.groupby(["split", "name"] + HP)["sel"].mean().reset_index()
    best = g.loc[g.groupby(["split", "name"])["sel"].idxmin()]
    return d.merge(best[["split", "name"] + HP], on=["split", "name"] + HP)


def paired(sel, a, b, metric):
    x = sel[sel.name == a].set_index("seed")[metric]
    y = sel[sel.name == b].set_index("seed")[metric]
    diff = (x - y).dropna()
    if len(diff) < 2:
        return f"n={len(diff)}"
    h = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
    return f"{diff.mean():+.4f} [{diff.mean() - h:+.4f}, {diff.mean() + h:+.4f}] (n={len(diff)})"


def main():
    d = load()
    sel = select(d)
    out = ["# E3b: county vote shares (2016 from 2012 + covariates)\n",
           "Incomplete run, stopped by decision: random split with 3–4 of 10 seeds, the state-blocked split "
           "not run. Configs selected on validation only (see summarize_e3.py). Lower is better for all "
           "metrics; mean ± std over seeds.\n"]
    for split in sorted(sel.split.unique()):
        s = sel[sel.split == split]
        g = s.groupby("name")
        t = pd.DataFrame({"seeds": g.seed.nunique()})
        for m in METRICS:
            t[m] = g[m].mean().map("{:.4f}".format) + " ± " + g[m].std(ddof=1).map("{:.4f}".format)
        t = t.loc[g["mae"].mean().sort_values().index]
        out += [f"## Split: {split}\n", t.to_markdown(), "\n"]
        pairs = [("CompMP-full (aitchison)", "Lap-MNL"), ("CompMP-full (aitchison)", "GCN-ilr (aitchison)"),
                 ("CompMP-FR (hellinger)", "CompMP-full (aitchison)")]
        rows = [{"comparison": f"{a} − {b}", **{m: paired(s, a, b, m) for m in METRICS}} for a, b in pairs]
        out += ["Paired differences (negative = first model better), mean [95% CI]:\n",
                pd.DataFrame(rows).to_markdown(index=False), "\n"]
    open("results/compmp_e3_summary.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
