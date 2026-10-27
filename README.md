# COMP-GCN reimplementation

Reimplementation of Lu, Wang & Zhao, "Graph convolutional network for compositional data",
*Information Fusion* 117 (2025) 102798, in plain PyTorch (no DGL needed). The authors shared the data, not the code.

## Layout

```
compgcn/ilr.py      ilr / inverse ilr, Helmert and random orthonormal bases, block-wise ILR(X)
compgcn/data.py     synthetic SBM data (Sec. 4.1), stock + student loaders, normalized adjacency, splits
compgcn/models.py   CGCN layer (Eq. 7), COMP-GCN, GCN, MLP, SGC, GCNII
compgcn/train.py    training protocol (Sec. 4.3), model registry, linear-ilr and constant baselines
run.py              runs the experiments, appends to results/<exp>.csv (resumable)
summarize.py        our numbers next to the paper's Tables 1, 3, 5 -> results/summary.md
tests_check.py      ilr isometry and CGCN == paper's simplex formulation (Eq. 4+5) checks
prepare_data.py     converts the authors' .xlsx files into data/*.npz
results/            raw per-run results (CSV) and summary.md for the runs reported below
```

The datasets are not included. They were shared by the authors on request; ask them for a copy, then run
`python prepare_data.py /path/to/COMP-GNN_data`.

## Run

```
pip install torch numpy pandas scikit-learn openpyxl tabulate
python prepare_data.py /path/to/COMP-GNN_data
python tests_check.py
python run.py student --reps 10
python run.py stock --reps 10
python run.py synthetic --reps 20 --workers 16
python run.py multiclass --reps 10 --workers 16
python summarize.py
```

## Models

| name | input | architecture |
|---|---|---|
| COMP-GCN | ILR(X) | CGCN -> GCN -> ReLU -> dropout -> Linear (Fig. 3a) |
| GCN-ilr | ILR(X) | ordinary 2-layer GCN. **Not in the paper**: the ablation that isolates the CGCN layer |
| GCN-scalar | raw proportions | GCN -> ReLU -> dropout -> GCN (Fig. 3b) |
| GCN-graph | constant 1 | same as GCN-scalar |
| MLP / MLP-ilr | raw / ILR(X) | Linear -> ReLU -> dropout -> Linear (Fig. 3c) |
| SGC | raw | A^2 X W, K=2 |
| GCNII | raw | Linear -> 2 GCNII layers (alpha 0.1, lambda 0.5) -> ReLU -> dropout -> Linear |
| linear-ilr | ILR(X) | logistic regression (clf) / OLS (reg), no graph |
| constant | none | majority class / training mean. **Not in the paper** |

## Protocol and deviations from the paper

Followed as stated: Adam, Xavier init, max 10,000 epochs, early stopping on validation loss with
patience 50 and best-weight restore, gradient values clipped to [-1, 1]; synthetic splits of 30/40
labelled nodes per class (train/val), rest test; real-data splits of 80 train / 80 val, rest test.

Deviations / assumptions:
- **No grid search.** One fixed setting for all models, the one reported in the Fig. 6 caption:
  lr 0.005, L2 5e-4, hidden 8, dropout 0.2. The paper grid-searches and does not report the chosen values,
  nor whether the selection used validation or test accuracy.
- Fewer repetitions in the run behind `results/` (synthetic 5, multiclass 3, real data 10; paper: 20 / 10 / 10).
- Student split assumed class-balanced (40 fail + 40 pass for train, same for val); not stated in the paper.
- Stock graph is directed; normalized DGL-style (1/sqrt(outdeg(src) indeg(dst)), self-loops added).
  Use `--stock-adj sym` to symmetrize instead.
- `make_classification` settings beyond n_clusters_per_class=1 are sklearn defaults (n_informative=2), except n_informative=3 for C = 5-7 because the default cannot produce more than 4 classes; noise N(0, 0.8) added
  before the inverse ilr, as in Sec. 4.1.
- KENT not implemented.
- The student files contain `chengji_ave` (grade average): excluded, it leaks the label.

## Main findings (see results/summary.md for full tables)

1. **CGCN layer is implemented exactly**: `tests_check.py` verifies that the layer equals the paper's
   simplex formulation (inverse ilr of the propagated features, then Aitchison inner product with a
   simplex-valued weight).
2. **Synthetic (Table 1)**: COMP-GCN reproduces within roughly 0-5 accuracy points, usually higher than
   reported. At p = 100 it clearly beats GCN-scalar (e.g. 0.88 vs 0.58). At p = 10 GCN-scalar ties or
   beats it, unlike the paper.
3. **GCN-ilr ≈ COMP-GCN in every synthetic setting** (differences 0-3 points, both directions). The gain over
   raw-feature models comes from the ilr transform; the block-diagonal CGCN layer adds little.
4. **Stocks (Table 3)**: COMP-GCN reproduces within 2-4% MSE, but ties the predict-the-training-mean baseline
   in all three years; the paper's own COMP-GCN MSE equals the label variance. Their weaker neural baselines
   (MLP, SGC, GCNII) do worse than a constant; with fixed settings ours do not.
5. **Students (Table 5)**: every model, including the paper's, is at chance level on balanced accuracy
   (0.50-0.54). The majority class alone gets 0.73-0.88 plain accuracy.
6. Not matched: SGC, GCNII (architecture details unreported), logistic/OLS-ilr (fitting details unreported).
7. **Multiclass (Fig. 4)**: COMP-GCN lands near the paper's curves (about 0.72 down to 0.60 for p = 10, 0.70 down to 0.46 for p = 100),
   but **GCN-ilr beats it in 8 of 10 settings** (1 tie, 1 loss), by up to about 11 points. A plausible mechanism: the CGCN layer compresses each
   composition to one scalar before any mixing, which becomes a bottleneck as the number of classes grows. Only 3 repetitions, so treat
   individual gaps as noisy; the direction is consistent.
