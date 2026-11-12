# Graph learning on compositional data: a reproduction and a new model

A reimplementation and critical evaluation of **COMP-GCN** (Lu, Wang & Zhao, *"Graph convolutional network
for compositional data"*, *Information Fusion* 117, 2025, 102798), followed by **CompMP**, my own model in
which node states stay compositions through every layer. Plain PyTorch, no graph library needed.

**Compositional data** are vectors of proportions that sum to 1: vote shares, budget splits, the mix of
positive and negative news about a company. Ordinary arithmetic on proportions misleads, so they are
usually mapped to unconstrained coordinates with a log-ratio transform (**ilr** / **clr**) first.

## Highlights

- **Faithful reimplementation.** The compositional layer is verified against the paper's own formula
  (unit test); synthetic results reproduce within about 0–5 accuracy points.
- **The missing ablation.** An ordinary GCN on ilr features (GCN-ilr), which the paper does not run,
  matches COMP-GCN on binary tasks and outperforms it in 8 of 10 multiclass settings. On these settings the
  gain comes from the ilr transform rather than from the special layer.
- **Trivial baselines.** On the paper's two real datasets, no model (COMP-GCN included) outperforms a
  constant predictor: the training mean for stock co-movement, the majority class for student failure.
- **A small theorem, tested.** Without nonlinearities, any Aitchison-linear message passing is exactly a
  constrained GCN on ilr coordinates. So a compositional GNN can only differ from "transform, then an
  ordinary GNN" in its nonlinearities, outputs and loss.
- **My model, CompMP.** Compositional node states with basis-free nonlinearities, full zero-sum linear
  maps, skip connections, a Fisher–Rao (square-root sphere) variant that handles zeros, and four
  compositional losses. On US county vote shares (3,112 counties), it beats GCN-ilr on all four error
  metrics, but a Laplacian-regularized multinomial logit beats both.
- **Careful evaluation throughout.** Validation-only model selection, equal tuning budgets, 5–10 seeds,
  paired confidence intervals and several error metrics, with null results reported as such.

Write-ups: [docs/SUMMARY.md](docs/SUMMARY.md) (plain language) and [docs/REPORT.md](docs/REPORT.md)
(technical, one page). Research log and design decisions: [docs/PLAN.md](docs/PLAN.md).

## Reproduction results

Full tables: [`results/summary.md`](results/summary.md).

| Question | Answer |
|---|---|
| Is the CGCN layer implemented correctly? | Yes, checked against the paper's formula (`tests/check_reproduction.py`). |
| Do the synthetic results (Table 1) reproduce? | Yes, within about 0–5 accuracy points, usually higher than reported. |
| Is the special layer what makes it work? | Mostly not. GCN-ilr ties it on binary tasks and outperforms it in 8 of 10 multiclass settings (untuned, 3 seeds). A tuned rerun on the easiest setting puts COMP-GCN slightly ahead (+1.5 points), so the effect depends on the setting. |
| Does it help on the real datasets? | No model, COMP-GCN included, outperforms a constant predictor on either dataset. The margins in the paper's tables come from baselines that score below that constant. |

## CompMP results

| Experiment | Question | Result |
|---|---|---|
| **E1** | Does ReLU on ilr coordinates depend on the arbitrary choice of basis? | Yes, but by about a third of seed noise; the basis-free nonlinearities remove it exactly, without an accuracy gain. |
| **E2** | On the paper's scalar targets, does CompMP match GCN-ilr? | Yes (tie within ±0.004; 2 of 18 settings, tuned, 10 seeds). |
| **E3** | On compositional targets (US county vote shares), does CompMP beat strong baselines? | It beats every neural baseline (−10% share error vs GCN-ilr, CIs exclude 0), but not a Laplacian-regularized multinomial logit (1.8 vs 1.3 points). |

Tables: [`results/compmp_summary.md`](results/compmp_summary.md),
[`results/compmp_e3_summary.md`](results/compmp_e3_summary.md). Some runs were stopped early; what is
preliminary is listed in [docs/REPORT.md](docs/REPORT.md).

## Running it

```
pip install -r requirements.txt

# reproduction
python prepare_data.py /path/to/COMP-GNN_data   # the paper's real datasets (not included)
python tests/check_reproduction.py
python run.py synthetic --reps 20 --workers 16
python run.py multiclass --reps 10 --workers 16
python run.py student --reps 10
python run.py stock --reps 10
python summarize.py                             # -> results/summary.md

# CompMP
python tests/check_compmp.py
python run_compmp.py e1
python run_compmp.py e2
python summarize_compmp.py                      # -> results/compmp_summary.md
python run_e3.py                                # downloads the public county data into data/ on first use
python summarize_e3.py                          # -> results/compmp_e3_summary.md
```

The synthetic experiments and E3 need no private data. The paper's stock and student datasets are not
included; they are available from the paper's authors. Runs append to CSV files in `results/` and resume
where they stopped.

## Files

```
compgcn/ilr.py         log-ratio transforms (ilr, clr) and random bases
compgcn/data.py        synthetic graphs, real-data loaders, splits
compgcn/models.py      COMP-GCN and the paper's baselines (GCN, MLP, SGC, GCNII)
compgcn/train.py       the paper's training protocol
compgcn/compmp.py      CompMP layers and model
compgcn/e3.py          E3 models (CompMP, Fisher-Rao CompMP, GCN baselines), losses, classical baselines
compgcn/elections.py   US county election data (Jia & Benson, KDD 2020): loader and splits
run*.py, summarize*.py experiment runners and result tables
tests/                 correctness checks (layer equivalences, basis invariance, barycenter aggregation)
prepare_data.py        converts the paper's .xlsx files to data/*.npz
docs/                  plain-language summary, technical report, research log
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

**Clarified by the authors** (personal communication, 2026; the original code and the selected
configurations were not preserved):
- Hyperparameters were grid-searched over the grids in Sec. 4.3 and **selected on validation
  performance**.
- Student training and validation sets each contain **the same number of fail and not-fail** students.
  This is what we assumed (40 + 40 for train, the same for validation).
- Both real networks were treated as **undirected (symmetrized in DGL)** before graph convolution.
  Our stock runs in `results/` used the directed graph (`--stock-adj dgl`, then the default; `run.py` now
  defaults to `sym`), so the stock experiment is due a rerun. We don't expect the
  conclusion to change: the paper's own COMP-GCN MSE equals the label variance, independent of the graph.

**Where we deviate or had to assume:**
- No grid search. One fixed setting for all models, the one in the Fig. 6 caption (lr 0.005, L2 5e-4,
  hidden 8, dropout 0.2). The authors selected from the Sec. 4.3 grids on validation, but the chosen
  values are not available.
- Fewer repetitions in `results/`: synthetic 5, multiclass 3, real data 10 (paper: 20 / 10 / 10).
- Stock graph: directed in the runs in `results/` (see above; rerun pending).
- `make_classification` uses sklearn defaults (n_informative = 2), except n_informative = 3 for C = 5–7,
  where the default cannot produce enough classes. Noise N(0, 0.8) is added before the inverse ilr, as in
  Sec. 4.1.
- KENT is not implemented.
- The student files contain `chengji_ave` (grade average). It is excluded because it leaks the label.
- Not matched: SGC, GCNII (architecture details unreported), logistic/OLS-ilr (fitting details unreported).
