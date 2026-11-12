# Plan: compositional message passing (CompMP)

Working plan for the follow-up to the COMP-GCN reproduction. Sections 1–4 are the original task;
section 5 is the update after the literature review, with amendments agreed afterwards. Where they
conflict, section 5 wins.

## 1. Context

The reproduction (README.md, results/summary.md) found:
- The CGCN layer = ilr-transform each composition, propagate with the normalized adjacency, inner product
  with one learned vector per composition. After layer 1 everything is scalar.
- GCN-ilr (ilr inputs + ordinary GCN, never run in the paper) ties COMP-GCN on binary synthetic data and
  beats it in 8/10 multiclass settings. The gain comes from the ilr transform, not the special layer.
- On the real datasets every model ties the trivial baseline (training mean for stocks, chance for students).
- All targets in the paper are scalars.

Hypothesis: compositional geometry inside the network should matter when the *target* is a composition,
and when nonlinearities act directly on compositional states.

Questions:
1. Is naive ReLU on ilr coordinates basis-dependent in practice, and do basis-free nonlinearities fix it?
2. Does keeping states compositional beat strong baselines on tasks with compositional targets?

## 2. Model (compgcn/compmp.py; existing reproduction code unchanged)

- States stored as clr vectors, shape (n, C, D): C channels (compositions) of D parts. P = I − 11ᵀ/D.
- Aggregation: A_hat @ clr. Row-normalized D⁻¹(A+I) (true Aitchison barycenter, default) or symmetric.
- Aitchison-linear map: mixing across channels out[k] = Σ_j α_kj · in[j]; optional within-composition
  map z ↦ P M z (shared or per channel); bias = perturbation by a learnable composition.
- Nonlinearities: (a) naive ReLU on ilr coordinates with a fixed basis (basis-dependent);
  (b) norm-gated z ↦ g(‖z‖) z (rotation-invariant); (c) centered pointwise z ↦ P f(z).
- Readout: scalars via Aitchison inner products → Linear; compositions directly (clr → softmax).
- Dropout drops whole compositions (sets them to the neutral element).

## 3. Experiments E1, E2 (done / running)

Protocol: training loop of compgcn/train.py (Adam, early stopping on validation loss, patience 50,
gradient value clipping). Grid lr ∈ {0.005, 0.01} × hidden ∈ {8, 16} × dropout ∈ {0.2, 0.5}, selected on
validation only, the same budget for every model; ≥ 10 seeds.

- **E1, basis dependence.** CompMP-(a) under 10 ilr bases with seed and split fixed, 5 seeds; controls
  GCN-ilr, CompMP-(b), CompMP-(c). **Done:** the effect is real but small (std across bases ≈ ⅓ of the std
  across seeds). Basis-free variants remove it exactly but are not more accurate.
- **E2, scalar targets.** Paper's synthetic setup (n = 2000, p ∈ {10, 100}, binary SNRs and C = 3..7);
  CompMP (a/b/c, c+M) vs GCN-ilr, COMP-GCN, GCN-scalar, MLP-ilr, linear-ilr, constant. Expected: a tie with
  GCN-ilr. **Running.**

**Checkpoint after E1 + E2:** short report, then wait for confirmation before building E3.

## 4. Rules

- Validation-only model selection; test data are never used for any decision.
- Every baseline gets the same tuning budget. A baseline that does worse than the constant predictor
  indicates a bug or bad training: investigate before reporting.
- New code is additive; the reproduction stays reproducible.
- Needed assumptions (data cleaning, missing counties, zeros): simplest option, documented in REPORT.md.
- Honest null results; a clean tie is a useful answer.
- Deliverables: code on branch `compmp`, results/compmp_summary.md, REPORT.md (≤ 1 page, go/no-go).

## 5. Update after the literature review (overrides the above where they conflict)

### 5.1 Equivalence result
Without nonlinearities, CompMP is exactly a linear GCN on ilr coordinates **with structured weights**
(α ⊗ ΨMΨᵀ, the same adjacency normalization). So any contribution must come from the nonlinear parts:
nonlinearities, multi-channel states, output geometry, losses, zeros. Unit test: CompMP without
nonlinearity == that structured linear GCN on ilr.

### 5.2 Prior work to position against (verify each before citing)
- AICoG (Nakis et al., ICML 2026, arXiv 2605.00716): Aitchison graph embeddings; lists compositional-state
  message passing as future work.
- Manifold GCN (Hanik, Steidl & von Tycowicz, IJCV 2026, arXiv 2401.14381): CompMP is its flat special case.
- Blondel, Drumetz & Chonavel (arXiv 2603.24108): ilr-pushforward Gaussian processes on the simplex.

### 5.3 Model changes
- Multi-channel states: already the case (hidden compositions = channels C).
- Residual connections in clr (z + layer(z)) against over-smoothing (convergence to the global barycenter):
  an option, not the default (2-layer models do not over-smooth).
- Default nonlinearity (c), which is S_D-equivariant (respects labelled parts). (b) kept as an ablation:
  its O(D−1) equivariance is too strong for labelled parts such as parties. Unit test: (c) is equivariant
  to permuting parts.
- Geometry as a flag: Aitchison (clr); Fisher–Rao (√p on the sphere, normalized weighted chordal mean,
  nonlinearity in the tangent space, allows zeros); learnable α-transform (α > 0).
- Loss as a flag: squared Aitchison, KL, Hellinger, multinomial NLL when counts exist. Always report
  Aitchison distance, KL, Hellinger and MAE on shares.

### 5.4 Experiment changes
- E3a synthetic: target passed through a nonlinear map (otherwise linear baselines win by construction);
  vary the fraction of near-zero parts.
- E3b real: multiparty US-ELECTION-2016 setup of Tomlinson & Benson (arXiv 2205.11365; 3,112 counties,
  geographic adjacency), which has small parts and structural zeros. Persistence baseline only if
  previous-election shares are available as features.
- E3c (spatial transcriptomics deconvolution): skipped for now (budget).
- Mandatory baselines: constant; label propagation in clr; Laplacian-regularized multinomial logit
  (Tomlinson & Benson); GCN-ilr with Aitchison loss; GCN with softmax output + KL (STdGCN-style).
  Second tier, only if the first round is promising: ilr-SAR / ilr-SLX (pysal/spreg); α-SAR via R
  CompositionalSR if R is available (otherwise documented skip); Gaussian process in ilr with a graph
  diffusion kernel.
- Label information: label propagation uses neighbours' training targets, which feature-only GNNs never
  see. For a fair comparison, GNNs also get a variant with neighbours' training labels as input (or a
  correct-and-smooth step).
- Zeros: multiplicative replacement for the Aitchison variants (report count and value); no replacement
  for the Fisher–Rao and α variants.

### 5.5 Go/no-go (revised)
Go if the **validation-selected** CompMP configuration (geometry, loss, hyperparameters chosen on
validation only) beats the strongest validation-selected baseline on compositional targets on at least
one real dataset, with a 95% CI of the paired difference over seeds excluding zero. A win only on
synthetic data is not enough. (The E1 clause is dropped: E1 already showed no clear basis effect.)

### 5.6 Step 1 findings (verification, 2026-10-27)

References (checked via web search; arXiv pages themselves are not reachable from this environment):
- **AICoG**: confirmed. "Aitchison Embeddings for Learning Compositional Graph Representations", Nakis,
  Kosma, Promponas, Chatzianastasis, Nikolentzos, arXiv 2605.00716, ICML 2026 (PMLR 306). Node embeddings
  as compositions over latent archetypes in ilr coordinates. **Not confirmed:** that it names
  compositional-state message passing as future work. The future work found is disconnected graphs and
  subcompositions. Do not cite that claim until checked in the full text.
- **Manifold GCN**: confirmed. Hanik, Steidl & von Tycowicz, arXiv 2401.14381. IJCV vol. 134 (2026)
  according to the group's page; DBLP lists only the preprint. Cite as IJCV 2026 once the DOI is found.
- **Blondel, Drumetz & Chonavel**: confirmed. "Aitchison Geometry on the Simplex for Uncertainty
  Quantification in Bayesian Hyperspectral Image Unmixing", arXiv 2603.24108 (March 2026), with an
  ilr-pushforward Gaussian-process prior.
- **Tomlinson & Benson**: confirmed. "Graph-Based Methods for Discrete Choice", arXiv 2205.11365,
  Network Science 12(1):21–40, 2024. Code: github.com/tomlinsonk/graph-based-discrete-choice.

Network access from the container: GitHub (git clone, raw files) works. arXiv, OSF, Zenodo, Harvard
Dataverse and census.gov are blocked.

Data:
- Tomlinson & Benson's processed data are on OSF (blocked). Their original sources are on GitHub and
  reachable:
  - **github.com/000Justin000/gnn-residual-correlation, datasets/election** (Jia & Benson, KDD 2020):
    3,112 counties, county adjacency, **vote counts** Dem / GOP / other for 2008, 2012, 2016, plus
    education, income, population and unemployment. Enables persistence (2012 → 2016), multinomial NLL
    and demographic features. No Alaska. No zero "other" counts (minimum county total 64).
  - **github.com/mkearney/presidential_election_county_results_2016**: per-candidate 2016 county
    **counts**, 25 candidates. Only 2,825 counties, so it is incomplete. Structural zeros: Stein is on the
    ballot in 2,314 counties, McMullin in 655, Castle in 1,166.
- Plan for E3b: the 3-party file as the main dataset (complete, has counts and a previous election); a
  multiparty variant (Trump / Clinton / Johnson / Stein / McMullin / other) on the 2,825 counties with
  coverage, for the zeros question. Document the coverage gap.
- R: not installed, so α-SAR (CompositionalSR) is skipped (documented).

### 5.7 CompMP-full (after E1, 2026-10-27)

E1 showed the original CompMP trailing GCN-ilr by 6–12 points at p = 100. Diagnosis, stepping from
GCN-ilr to CompMP one change at a time: the cause was the scalar head (an extra compositional linear
layer before the readout), which overfits the 60–150 training nodes within about 50 epochs. Neither the
nonlinearity nor the clr parametrization was responsible. CompMP-full uses:
- full zero-sum linear maps (one D×D block per channel pair; linear part == GCN on ilr, unit-tested);
- a propagation head (inner products with learned compositions, then A·, like GCN's second layer);
- hidden / (D−1) compositions, i.e. the same hidden width in ilr coordinates as GCN-ilr;
- the per-part nonlinearity (c); residual hidden layers available (layers > 2), unused in E2.

A 3-seed check put it level with GCN-ilr (0.82 vs 0.817 at p=100 binary, 0.62 vs 0.60 at p=100 C=5,
0.72 vs 0.74 at p=10 C=5). E2 was restarted with CompMP-c (old, reference), CompMP-full (sym adjacency)
and CompMP-full-row against the tuned baselines; the old CompMP-a/b/c-M rows already run are kept
but incomplete.

### 5.8 Order of work
1. Verify the references, the dataset contents (parties, counts, downloadability) and network access.
2. Unit tests: linear equivalence (5.1) and permutation equivariance (5.3).
3. E3b first, with Aitchison and Fisher–Rao and the mandatory baselines → go/no-go on real data.
4. Second-tier baselines, α-transform and E3a only if step 3 is promising.
