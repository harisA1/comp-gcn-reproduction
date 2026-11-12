# Summary in plain language

*Final status, 27 October 2026. The project was wrapped up by decision before every run finished; findings
5 and 7 are preliminary (4–5 seeds, random split only). Technical report: [REPORT.md](REPORT.md). Details:
[README](../README.md), [research log](PLAN.md), [reproduction tables](../results/summary.md),
[CompMP tables](../results/compmp_summary.md), [vote-share tables](../results/compmp_e3_summary.md).*

## The starting question

A 2025 paper in *Information Fusion* (COMP-GCN, Lu, Wang & Zhao) proposed a new kind of graph neural network
for **compositional data**: data made of proportions that add up to 100%, like vote shares, budget splits,
or the mix of positive and negative news about a company. Its core idea is a special network layer that
respects the geometry of proportions. I asked two things: **what does its special layer contribute, and can compositional geometry help graph
learning at all?**

## What I did

1. **Rebuilt the paper from scratch** in PyTorch (the paper's code is not available).
2. **Added the comparison the paper skipped:** an ordinary graph network fed the same transformed data
   (GCN-ilr).
3. **Built my own model (CompMP)**, which keeps proportions as proportions through the whole network, not
   just the first layer.
4. **Tested it on a new, real task:** predicting each US county's 2016 presidential vote shares from its 2012
   result, its demographics and its neighbouring counties.

## What I found

**1. On most settings, the paper's special layer adds little.** The gains come mainly from one standard
preprocessing step, the log-ratio (ilr) transform: an ordinary network given the same transformed data does
as well or better on the harder settings (see finding 7 for the exception). This
follows from a simple mathematical fact: after that transform, the layer is just a restricted ordinary layer.

**2. The real datasets carry little signal for any model.** On both of the paper's real datasets (stock
co-movement and student failure), no model, COMP-GCN included, does better than a trivial guess: "predict
the average" or "predict the most common outcome". The margins in the paper's tables come from comparison
models that score below that trivial guess. The authors confirmed their setup (personal communication):
hyperparameters
were selected on validation data, the student training and validation sets were class-balanced (as I
assumed), and the graphs were treated as undirected. That rules out test-set tuning as an explanation, so a
tuned baseline losing to a constant points to a training problem. My stock runs used a directed graph and
should be rerun undirected; the conclusion shouldn't change, because the paper's own stock error equals the
spread of the target values whatever the graph.

**3. A theoretical worry turned out to be minor.** Feeding proportions into a network requires choosing a
set of reference axes, and some network operations quietly depend on that arbitrary choice. I showed the
effect is real, but only about a third the size of ordinary run-to-run noise. My axis-free versions remove
it completely, but aren't more accurate.

**4. Building compositional networks has hidden pitfalls.** My first version lost badly, by 6–12 points.
The reason wasn't the compositional idea but an overfitting-prone output layer. Once fixed, my model
matches the standard network. And on the election task, *every* graph network failed until I added a
direct link from each county to its own past result: averaging with neighbours drowned out the strongest
signal.

**5. On real compositional data, a simpler method wins (preliminary).** On the county vote shares, the
best model is a **linear model with smoothing across neighbouring counties** (Tomlinson & Benson's method):
1.3 percentage points average error. My network comes second at 1.8 points, ahead of every other model.
The gap to the linear model is about ten times the seed-to-seed noise, so it is not luck. My network does
beat the ordinary graph network on the same transformed data (GCN-ilr), by about 10%, consistently across
seeds: the one place where keeping compositions compositional measurably helped.

**6. How you measure the error matters.** "Predict 2016 = 2012" is very good in percentage points but
terrible in ratio terms, because third-party votes roughly tripled. The ranking of methods changes with the
error measure. Any compositional-data study has to report several measures.

**7. One correction (preliminary).** With proper tuning, on the easiest synthetic setting, the paper's
model is slightly *better* than the plain network on transformed data. The "the special layer adds little" finding
holds for the harder settings (many compositions, many classes); the tuned rerun didn't reach those.

## What it means

- **The paper's evidence for its special layer is weaker than presented:** the missing ablation and the
  trivial baselines account for most of its reported gains.
- **"Respecting compositional geometry" is mostly about how you transform the data, not about special
  layers.** The network's linear parts are mathematically equivalent to ordinary ones. Only the nonlinear
  parts, the outputs and the loss can differ.
- **On the one real compositional task tested, simple smoothing beats deep message passing**
  (preliminary).

## Not done

- The stock experiment rerun on the undirected graph, as the authors used.
- The election test with whole states held out, and the remaining seeds.
- The tuned synthetic comparison on the harder settings.
- Data with structural zeros (multiparty elections), where other geometries might matter more.

## Verdict

CompMP is a sound compositional GNN that beats the neural baselines, but not yet a methods contribution:
it loses to a simpler smoothing model on the one real task. The evaluation lessons carry over to my next
project, a spectral unmixing benchmark.

## Possible write-up

*"What actually matters in graph learning for compositional data"*:
1. A theorem: compositional linear layers are equivalent to ordinary ones.
2. A reproduction showing which parts of a published method's gains survive a full ablation.
3. A careful test of what's left (nonlinearities, geometry, losses) on real compositional targets.

With a win for CompMP this would be a methods paper; without one, a careful "what helps and what
doesn't" study.
