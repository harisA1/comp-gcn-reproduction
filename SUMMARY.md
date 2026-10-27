# The story so far, in plain language

*Status as of 27 October 2026. Findings 5 and 6 are preliminary (one data split); the full runs are in
progress. Technical details: [README.md](README.md), [PLAN.md](PLAN.md),
[results/summary.md](results/summary.md), [results/compmp_summary.md](results/compmp_summary.md).*

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
worse than that trivial guess.

**3. A theoretical worry turned out to be minor.** Feeding proportions into a network requires choosing a
set of reference axes, and some network operations quietly depend on that arbitrary choice. We showed the
effect is real, but only about a third the size of ordinary run-to-run noise. Our axis-free versions remove
it completely, but aren't more accurate.

**4. Building compositional networks has hidden pitfalls.** Our first version lost badly, by 6–12 points.
The reason wasn't the compositional idea but an overfitting-prone output layer. Once fixed, our model
matches the standard network. And on the election task, *every* graph network failed until we added a
direct link from each county to its own past result: averaging with neighbours drowned out the strongest
signal.

**5. On real compositional data, a simpler method currently wins (preliminary).** In early results on the
county vote shares (one data split so far), the best model is a **linear model with smoothing across
neighbouring counties** (Tomlinson & Benson's method). Our network comes second, ahead of the other
baselines. It's best on the compositional error measure among the neural networks, but about 30% worse than
the linear model on plain percentage-point error.

**6. How you measure the error matters.** "Predict 2016 = 2012" is very good in percentage points but
terrible in ratio terms, because third-party votes roughly tripled. The ranking of methods changes with the
error measure. Any compositional-data study has to report several measures.

## What it means so far

- **The paper's claimed contribution doesn't hold up.** This is solid.
- **"Respecting compositional geometry" is mostly about how you transform the data, not about special
  layers.** The network's linear parts are mathematically equivalent to ordinary ones. Only the nonlinear
  parts, the outputs and the loss can differ.
- **For the one real task tested so far, simple smoothing beats deep message passing.** That's preliminary.

## Still running

- The full, fairly tuned comparison on the paper's synthetic tasks (E2).
- The full election experiment (E3): 10 random seeds, plus a harder test where whole states are held out,
  which could change the ranking.

## The paper this could become

*"What actually matters in graph learning for compositional data"*:
1. A theorem: compositional linear layers are equivalent to ordinary ones.
2. A reproduction showing a published method's gains are an artifact.
3. A careful test of what's left (nonlinearities, geometry, losses) on real compositional targets.

If our model wins in the final results, that becomes a methods paper. If it doesn't, it's an honest
"here's what doesn't help, and why" paper. Both are publishable stories; the first is stronger.
