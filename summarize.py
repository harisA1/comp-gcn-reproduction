"""Compare our runs with the paper's tables. Writes results/summary.md."""
import os
import pandas as pd

# ---- Paper numbers (Lu, Wang & Zhao 2025), transcribed from Tables 1, 3, 5.
# linear-ilr = logistic-ilr (classification) or OLS-ilr (regression).
SNR = [1, 1.78, 2.78, 4]
T1 = {  # (n, p): {model: [acc at SNR 1, 1.78, 2.78, 4]}
    (1000, 10): {"COMP-GCN": [.794, .87, .922, .949], "GCN-scalar": [.729, .832, .91, .953],
                 "GCN-graph": [.505, .506, .506, .505], "MLP": [.732, .732, .73, .731],
                 "MLP-ilr": [.763, .765, .765, .764], "SGC": [.719, .79, .845, .876],
                 "GCNII": [.543, .582, .609, .832], "linear-ilr": [.732] * 4},
    (1000, 100): {"COMP-GCN": [.713, .814, .893, .927], "GCN-scalar": [.542, .598, .622, .827],
                  "GCN-graph": [.507, .504, .514, .505], "MLP": [.581, .581, .582, .58],
                  "MLP-ilr": [.607, .609, .61, .608], "SGC": [.644, .713, .782, .822],
                  "GCNII": [.749, .781, .827, .923], "linear-ilr": [.595] * 4},
    (2000, 10): {"COMP-GCN": [.795, .863, .916, .948], "GCN-scalar": [.752, .841, .908, .954],
                 "GCN-graph": [.505, .503, .505, .504], "MLP": [.736, .735, .737, .735],
                 "MLP-ilr": [.77, .77, .771, .77], "SGC": [.714, .782, .841, .881],
                 "GCNII": [.55, .581, .608, .788], "linear-ilr": [.75] * 4},
    (2000, 100): {"COMP-GCN": [.702, .79, .867, .919], "GCN-scalar": [.538, .58, .618, .85],
                  "GCN-graph": [.505, .502, .504, .504], "MLP": [.586, .586, .585, .586],
                  "MLP-ilr": [.603, .604, .604, .602], "SGC": [.616, .677, .739, .786],
                  "GCNII": [.768, .806, .844, .93], "linear-ilr": [.608] * 4},
}
T3 = {"COMP-GCN": [.01798, .01656, .00934], "GCN-graph": [.01812, .01667, .00955],
      "GCN-scalar": [.01818, .01663, .00951], "MLP": [.02329, .01917, .01044],
      "MLP-ilr": [.02787, .02055, .01256], "SGC": [.0258, .02081, .01106],
      "GCNII": [.03447, .0635, .22885], "linear-ilr": [.01978, .01473, .01007]}
T5 = {"COMP-GCN": [.52643, .52963, .5445, .53456], "GCN-graph": [.5015, .50191, .50454, .5],
      "GCN-scalar": [.50953, .50642, .5405, .51694], "MLP": [.52734, .52094, .53815, .53243],
      "MLP-ilr": [.51813, .52198, .53117, .52491], "SGC": [.50042, .50226, .51417, .50738],
      "GCNII": [.51526, .5186, .54954, .54509], "linear-ilr": [.53155, .52756, .54314, .53136]}

ORDER = ["COMP-GCN", "GCN-ilr", "GCN-scalar", "GCN-graph", "MLP", "MLP-ilr", "SGC", "GCNII",
         "linear-ilr", "constant"]


def ours(path, metric):
    df = pd.read_csv(path, dtype={"setting": str})
    g = df.groupby(["setting", "model"])[metric].agg(["mean", "std", "count"])
    return g


def cell(g, setting, model, paper, fmt):
    try:
        r = g.loc[(setting, model)]
        s = f"{r['mean']:{fmt}} ± {r['std']:{fmt}}"
    except KeyError:
        s = "–"
    return f"{s} ({paper:{fmt}})" if paper is not None else f"{s} (–)"


def table(g, settings, paper_cols, metric_fmt, header):
    lines = ["| model | " + " | ".join(header) + " |", "|---" * (len(header) + 1) + "|"]
    for m in ORDER:
        row = [cell(g, s, m, paper_cols.get(m, [None] * len(settings))[i], metric_fmt)
               for i, s in enumerate(settings)]
        lines.append(f"| {m} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    out = ["# Reproduction vs paper\n",
           "Cells: our mean ± std over repetitions (paper value in brackets). Fixed hyperparameters "
           "from the Fig. 6 caption (lr 0.005, L2 5e-4, hidden 8, dropout 0.2, patience 50), no grid search.\n"]
    if os.path.exists("results/student.csv") and os.path.getsize("results/student.csv") > 100:
        g = ours("results/student.csv", "acc")
        out += ["\n## Student failure (Table 5), test accuracy\n",
                table(g, [f"term{t}" for t in (1, 2, 3, 4)], T5, ".3f", ["Sem 1", "Sem 2", "Sem 3", "Sem 4"])]
        g = ours("results/student.csv", "bal_acc")
        out += ["\n\nBalanced accuracy (not in paper):\n",
                table(g, [f"term{t}" for t in (1, 2, 3, 4)], {}, ".3f", ["Sem 1", "Sem 2", "Sem 3", "Sem 4"])]
    if os.path.exists("results/stock.csv") and os.path.getsize("results/stock.csv") > 100:
        g = ours("results/stock.csv", "mse")
        out += ["\n\n## Stock comovement (Table 3), test MSE\n",
                table(g, ["2019", "2020", "2021"], T3, ".5f", ["2019", "2020", "2021"])]
    if os.path.exists("results/synthetic.csv") and os.path.getsize("results/synthetic.csv") > 100:
        g = ours("results/synthetic.csv", "acc")
        for (n, p), cols in T1.items():
            out += [f"\n\n## Synthetic binary (Table 1), n={n}, p={p}, test accuracy\n",
                    table(g, [f"n={n},p={p},snr={s}" for s in SNR], cols, ".3f", [f"SNR={s}" for s in SNR])]
    if os.path.exists("results/multiclass.csv") and os.path.getsize("results/multiclass.csv") > 100:
        df = pd.read_csv("results/multiclass.csv")
        piv = df.pivot_table(index="model", columns="setting", values="acc", aggfunc="mean")
        cols = [f"p={p},C={C}" for p in (10, 100) for C in range(3, 8)]
        piv = piv.reindex(index=[m for m in ORDER if m in piv.index], columns=[c for c in cols if c in piv.columns])
        out += ["\n\n## Multiclass (Fig. 4), mean test accuracy\n", piv.round(3).to_markdown()]
    open("results/summary.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
