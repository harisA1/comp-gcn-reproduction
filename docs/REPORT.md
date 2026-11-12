# CompMP: final report (27 October 2026)

Follow-up to the COMP-GCN reproduction ([README](../README.md)). Plan: [PLAN.md](PLAN.md). Plain-language
version: [SUMMARY.md](SUMMARY.md). Tables: [E1, E2](../results/compmp_summary.md),
[E3b](../results/compmp_e3_summary.md). **The project was stopped
by decision before the runs completed**; what is preliminary is marked.

## Question 1: is naive ReLU on ilr coordinates basis-dependent, and do basis-free nonlinearities fix it?

**Yes, but the effect is small.** With seed, split and initialization fixed, changing only the ilr basis
moves test accuracy by a std of 0.009–0.023, against 0.03–0.07 across seeds (E1, 4 settings × 5 seeds ×
10 bases). The norm-gated and per-part nonlinearities remove the dependence exactly (unit-tested), but are
not more accurate. **Not a "clear basis effect"** in the sense of the go/no-go rule.

## Question 2: does keeping states compositional beat strong baselines on compositional targets?

**It beats the neural baselines but not the strongest baseline.** County vote shares 2016 (D = 3) from
2012 shares and 6 covariates, 3,112 counties, random split, 4–5 seeds (preliminary; 10 planned):

| model | Aitchison | MAE on shares |
|---|---|---|
| Laplacian-regularized multinomial logit | **0.179 ± 0.002** | **0.0130 ± 0.0004** |
| CompMP-full, Aitchison loss | 0.194 ± 0.004 | 0.0178 ± 0.0005 |
| GCN-ilr, Aitchison loss | 0.212 ± 0.008 | 0.0197 ± 0.0005 |
| CompMP Fisher–Rao, Hellinger loss | 0.272 ± 0.012 | 0.0218 ± 0.0009 |
| ridge on ilr (no graph) | 0.322 ± 0.006 | 0.0227 ± 0.0008 |
| persistence (2012 shares) | 0.993 ± 0.019 | 0.0517 ± 0.0012 |

Paired differences (95% CI over shared seeds):
- CompMP-full − Lap-MNL: MAE **+0.0048 [+0.0044, +0.0052]**, worse on all four metrics.
- CompMP-full − GCN-ilr (same loss): MAE **−0.0019 [−0.0027, −0.0011]**, better on all four metrics (n = 4).
- Fisher–Rao − Aitchison CompMP: MAE +0.0040 [+0.0018, +0.0061], worse.

## Other findings
- **Linear compositional layers are ordinary layers.** Without nonlinearities, CompMP equals a GCN on ilr
  coordinates with structured weights (unit-tested); COMP-GCN's layer is a restricted case.
- **Self-connections are essential on real data.** Without them every GNN lost to ridge regression
  (MAE ≈ 0.044 vs 0.022); the neighbour average dilutes a county's own 2012 result.
- **CompMP's original head overfit.** The 6–12 point E1 gap to GCN-ilr disappeared with full zero-sum maps
  and a propagation head; E2 (tuned, 10 seeds) puts CompMP-full level with GCN-ilr (±0.004).
- **Losses:** Aitchison > KL > count-weighted multinomial for both CompMP and GCN-ilr, on every metric.
- **Metric dependence:** persistence is far better than the constant in share error but far worse in
  Aitchison distance (third-party shares tripled from 2012 to 2016).
- **Correction to the reproduction story (preliminary):** tuned, on the easiest synthetic setting
  (p = 10, SNR 1), COMP-GCN beats GCN-ilr by +0.015 [+0.004, +0.025]. The "GCN-ilr ≥ COMP-GCN" claim
  rests on the multiclass and p = 100 settings, which the tuned E2 did not reach (2 of 18 settings run).

## Go / no-go: **no-go** for CompMP as a methods paper

Rule (PLAN.md 5.5): the validation-selected CompMP must beat the strongest baseline on a real dataset
beyond seed noise. It loses to the Laplacian logit by about 10 seed-standard-deviations. E1 shows no clear
basis effect.

## Not done (would be needed before publishing any of this)
- Reproduction: the stock experiment on the symmetrized graph (the authors confirmed both real graphs
  were undirected; our runs used the directed one).
- E3b state-blocked split, remaining seeds; multiparty data with structural zeros; E3a synthetic.
- E2 on the multiclass and p = 100 settings (tuned confirmation of the critical result).
- Second-tier baselines (SAR/SLX, graph GP), GNNs with neighbours' training labels as input.
- Caveat: hyperparameter selection used the mean validation loss over the seeds available per config,
  and not every config had the same number of seeds when the run stopped.
