# COMP-GCN: reproduction and follow-up

This repository does two things:

1. **Reproduces** the paper *"Graph convolutional network for compositional data"* (Lu, Wang & Zhao,
   *Information Fusion* 117, 2025, 102798), which proposes COMP-GCN. The authors shared their data but
   not their code, so everything here is rebuilt from the paper in plain PyTorch.
2. **Tests a follow-up idea** (CompMP, on the `compmp` branch): a graph neural network whose node states
   stay compositions in every layer, instead of being turned into ordinary numbers after the first one.

## Background in one minute

**Compositional data** are vectors of proportions that sum to 1: vote shares, budget splits, the
sentiment mix of news about a stock. Ordinary arithmetic on proportions is misleading, so they are usually
mapped to unconstrained coordinates first with a log-ratio transform (**ilr** or **clr**).

**COMP-GCN** feeds such features into a graph neural network. Its special first layer (CGCN) turns each
composition into one number. From the second layer on, it is an ordinary GCN.

## What the reproduction found

Full tables are in [`results/summary.md`](results/summary.md).

| Question | Answer |
|---|---|
| Is the CGCN layer implemented correctly? | Yes. `tests_check.py` checks it against the paper's own formula. |
| Do the synthetic results (Table 1) reproduce? | Yes, within about 0–5 accuracy points, usually higher than reported. |
| Is the special layer what makes it work? | **No.** A plain GCN on ilr features (GCN-ilr, which the paper never ran) ties it on binary tasks and beats it in 8 of 10 multiclass settings. The gain comes from the ilr transform. |
| Does it help on the real datasets? | **No.** On stocks, every model ties "predict the training mean", the paper's COMP-GCN included. On students, every model is at chance. The paper's wins there come from baselines that score worse than a constant. |

**In short:** the method is correct and works on synthetic data built to suit it, but the evidence that
the special layer matters, or that it helps on real data, does not hold up.

## The follow-up: CompMP (stopped; see REPORT.md)

COMP-GCN discards the compositional structure after one layer, and all its targets are single numbers.
Our hypothesis is that compositional geometry should matter more when (a) it is kept through the whole
network and (b) the thing being predicted is itself a composition.

CompMP keeps every node state as a composition. Each layer averages neighbours geometrically, mixes
compositions linearly, and applies one of three nonlinearities:

- **naive**: ReLU on ilr coordinates. This depends on an arbitrary choice of axes (the "basis").
- **norm**: rescales each composition by a function of its length. Independent of the basis.
- **part**: a per-part activation in clr space. Also independent of the basis.

Experiments, in order:

| | Question | Status |
|---|---|---|
| **E1** | Does the naive ReLU really change results when only the basis changes? | done: yes, but small |
| **E2** | On the paper's scalar targets, does CompMP beat GCN-ilr? (Expected: a tie.) | partial: tie (2 of 18 settings) |
| **E3** | On compositional targets (US county vote shares), does CompMP beat strong baselines? | partial: beats GCN-ilr, loses to a Laplacian-smoothed logit |

**Outcome: no-go for CompMP as a methods paper.** See [REPORT.md](REPORT.md) and [SUMMARY.md](SUMMARY.md).

All models get the same small hyperparameter grid, selected on validation data only, with 10 seeds.

## Running it

```
pip install torch numpy pandas scikit-learn scipy openpyxl tabulate

# reproduction
python prepare_data.py /path/to/COMP-GNN_data   # the real datasets, obtained from the authors
python tests_check.py
python run.py synthetic --reps 20 --workers 16
python run.py multiclass --reps 10 --workers 16
python run.py student --reps 10
python run.py stock --reps 10
python summarize.py                             # -> results/summary.md

# CompMP
python tests_compmp.py
python run_compmp.py e1
python run_compmp.py e2
python summarize_compmp.py                      # -> results/compmp_summary.md
python run_e3.py                                # downloads the county data into data/ on first use
python summarize_e3.py                          # -> results/compmp_e3_summary.md
```

The synthetic experiments need no downloaded data. The real datasets are not included here; ask the
paper's authors for a copy. Runs append to CSV files in `results/` and resume where they stopped if
interrupted.

## Files

```
compgcn/ilr.py       log-ratio transforms (ilr, clr) and random bases
compgcn/data.py      synthetic graphs, real-data loaders, splits
compgcn/models.py    COMP-GCN and the paper's baselines (GCN, MLP, SGC, GCNII)
compgcn/train.py     the paper's training protocol
compgcn/compmp.py    CompMP (follow-up model)
run.py, summarize.py                  reproduction experiments and tables
run_compmp.py, summarize_compmp.py    CompMP experiments and tables
tests_check.py, tests_compmp.py       correctness checks
prepare_data.py      converts the authors' .xlsx files to data/*.npz
```

## Reproduction details

**Models**

| name | input | architecture |
|---|---|---|
| COMP-GCN | ILR(X) | CGCN → GCN → ReLU → dropout → Linear (Fig. 3a) |
| GCN-ilr | ILR(X) | ordinary 2-layer GCN. Not in the paper: isolates the CGCN layer |
| GCN-scalar | raw proportions | GCN → ReLU → dropout → GCN (Fig. 3b) |
| GCN-graph | constant 1 | same as GCN-scalar |
| MLP / MLP-ilr | raw / ILR(X) | Linear → ReLU → dropout → Linear (Fig. 3c) |
| SGC | raw | A²XW, K = 2 |
| GCNII | raw | Linear → 2 GCNII layers (α 0.1, λ 0.5) → ReLU → dropout → Linear |
| linear-ilr | ILR(X) | logistic regression (classification) / OLS (regression), no graph |
| constant | none | majority class / training mean. Not in the paper |

**Protocol as stated in the paper:** Adam, Xavier initialization, at most 10,000 epochs, early stopping on
validation loss (patience 50, best weights restored), gradient values clipped to [−1, 1]. Synthetic: 30/40
labelled nodes per class for train/validation, the rest test. Real data: 80 train / 80 validation, the
rest test.

**Where we deviate or had to assume:**
- No grid search. One fixed setting for all models, the one in the Fig. 6 caption (lr 0.005, L2 5e-4,
  hidden 8, dropout 0.2). The paper grid-searches but does not report the chosen values, or whether
  selection used validation or test data.
- Fewer repetitions in `results/`: synthetic 5, multiclass 3, real data 10 (paper: 20 / 10 / 10).
- Student split assumed class-balanced (40 fail + 40 pass for train, the same for validation).
- The stock graph is directed and normalized as DGL does; `--stock-adj sym` symmetrizes it instead.
- `make_classification` uses sklearn defaults (n_informative = 2), except n_informative = 3 for C = 5–7,
  where the default cannot produce enough classes. Noise N(0, 0.8) is added before the inverse ilr, as in
  Sec. 4.1.
- KENT is not implemented.
- The student files contain `chengji_ave` (grade average). It is excluded because it leaks the label.
- Not matched: SGC, GCNII (architecture details unreported), logistic/OLS-ilr (fitting details unreported).
