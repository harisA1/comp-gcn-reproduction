# The story so far, in plain language

*Final status, 27 October 2026. The project was wrapped up by decision before every run finished; findings
5 and 7 are preliminary (4–5 seeds, random split only). Technical report: [REPORT.md](REPORT.md). Details:
[README.md](README.md), [PLAN.md](PLAN.md), [results/summary.md](results/summary.md),
[results/compmp_summary.md](results/compmp_summary.md),
[results/compmp_e3_summary.md](results/compmp_e3_summary.md).*

## The starting question

A 2025 paper in *Information Fusion* (COMP-GCN, Lu, Wang & Zhao) claimed a new kind of graph neural network
for **compositional data**: data made of proportions that add up to 100%, like vote shares, budget splits,
or the mix of positive and negative news about a company. Its core idea is a special network layer that
respects the geometry of proportions. We asked two things: **does it really work, and can compositional
geometry help graph learning at all?**

## What we did

1. **Rebuilt the paper from scratch.** The authors shared their data but not their code.
2. **Added the comparison the paper skipped:** an ordinary graph network fed the same transformed data
   (GCN-ilr).
3. **Built our own model (CompMP)**, which keeps proportions as proportions through the whole network, not
   just the first layer.
4. **Tested it on a new, real task:** predicting each US county's 2016 presidential vote shares from its 2012
   result, its demographics and its neighbouring counties.

## What we found

**1. The paper's special layer adds nothing.** Its gains come from one standard preprocessing step, the
log-ratio (ilr) transform. An ordinary network given the same transformed data does as well or better. This
follows from a simple mathematical fact: after that transform, the layer is just a restricted ordinary layer.

**2. Its real-world results are empty.** On both of its real datasets (stock co-movement and student
failure), no model, the paper's included, does better than a trivial guess: "predict the average" or
"predict the most common outcome". The paper's apparent wins come from comparison models that were trained
worse than that trivial guess. The authors have since confirmed their setup by email: hyperparameters
were selected on validation data, the student training and validation sets were class-balanced (as we
assumed), and the graphs were treated as undirected. That rules out test-set tuning as an explanation, so a
tuned baseline losing to a constant points to a training problem. Our stock runs used a directed graph and
should be rerun undirected; the conclusion shouldn't change, because the paper's own stock error equals the
spread of the target values whatever the graph.

**3. A theoretical worry turned out to be minor.** Feeding proportions into a network requires choosing a
set of reference axes, and some network operations quietly depend on that arbitrary choice. We showed the
effect is real, but only about a third the size of ordinary run-to-run noise. Our axis-free versions remove
it completely, but aren't more accurate.

**4. Building compositional networks has hidden pitfalls.** Our first version lost badly, by 6–12 points.
The reason wasn't the compositional idea but an overfitting-prone output layer. Once fixed, our model
matches the standard network. And on the election task, *every* graph network failed until we added a
direct link from each county to its own past result: averaging with neighbours drowned out the strongest
signal.

**5. On real compositional data, a simpler method wins (preliminary).** On the county vote shares, the
best model is a **linear model with smoothing across neighbouring counties** (Tomlinson & Benson's method):
1.3 percentage points average error. Our network comes second at 1.8 points, ahead of every other model.
The gap to the linear model is about ten times the seed-to-seed noise, so it is not luck. Our network does
beat the ordinary graph network on the same transformed data (GCN-ilr), by about 10%, consistently across
seeds: the one place where keeping compositions compositional measurably helped.

**6. How you measure the error matters.** "Predict 2016 = 2012" is very good in percentage points but
terrible in ratio terms, because third-party votes roughly tripled. The ranking of methods changes with the
error measure. Any compositional-data study has to report several measures.

**7. One correction (preliminary).** With proper tuning, on the easiest synthetic setting, the paper's
model is slightly *better* than the plain network on transformed data. Our "the special layer adds nothing"
finding holds for the harder settings (many compositions, many classes); the tuned rerun didn't reach those.

## What it means so far

- **The paper's claimed contribution doesn't hold up.** This is solid.
- **"Respecting compositional geometry" is mostly about how you transform the data, not about special
  layers.** The network's linear parts are mathematically equivalent to ordinary ones. Only the nonlinear
  parts, the outputs and the loss can differ.
- **For the one real task tested so far, simple smoothing beats deep message passing.** That's preliminary.

## Not done

- The stock experiment rerun on the undirected graph, as the authors used.
- The election test with whole states held out, and the remaining seeds.
- The tuned synthetic comparison on the harder settings.
- Data with structural zeros (multiparty elections), where other geometries might matter more.

## Verdict

Our model (CompMP) is not a methods paper: it loses to a simpler smoothing model on the one real task.
The lessons carry over to the next project (spectral unmixing benchmark, separate repository).

## The paper this could become

*"What actually matters in graph learning for compositional data"*:
1. A theorem: compositional linear layers are equivalent to ordinary ones.
2. A reproduction showing a published method's gains are an artifact.
3. A careful test of what's left (nonlinearities, geometry, losses) on real compositional targets.

If our model wins in the final results, that becomes a methods paper. If it doesn't, it's an honest
"here's what doesn't help, and why" paper. Both are publishable stories; the first is stronger.
