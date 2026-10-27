# CompMP results

## E1: basis dependence (test accuracy, fixed hyperparameters lr 0.005, hidden 8, dropout 0.2)

`sd_bases`: std across ilr bases with seed, split and initialization fixed, averaged over 5 seeds. `range_bases`: max - min across bases, averaged over seeds. `sd_seeds`: std across seeds of the basis-averaged accuracy. Basis 0 is Helmert, the others random. CompMP-b/c were run on 3 bases (they cannot depend on the basis).

| setting        | model      |   bases |   mean |   sd_bases |   range_bases |   sd_seeds |
|:---------------|:-----------|--------:|-------:|-----------:|--------------:|-----------:|
| p=10,snr=1.78  | CompMP-a   |      10 | 0.8821 |     0.0087 |        0.0248 |     0.0305 |
| p=10,snr=1.78  | CompMP-a-M |      10 | 0.8843 |     0.0060 |        0.0190 |     0.0316 |
| p=10,snr=1.78  | GCN-ilr    |      10 | 0.8818 |     0.0038 |        0.0115 |     0.0275 |
| p=10,snr=1.78  | CompMP-b   |       3 | 0.8741 |     0.0000 |        0.0000 |     0.0357 |
| p=10,snr=1.78  | CompMP-c   |       3 | 0.8812 |     0.0000 |        0.0000 |     0.0361 |
| p=100,snr=1.78 | CompMP-a   |      10 | 0.7242 |     0.0228 |        0.0639 |     0.0743 |
| p=100,snr=1.78 | CompMP-a-M |      10 | 0.7159 |     0.0174 |        0.0483 |     0.0533 |
| p=100,snr=1.78 | GCN-ilr    |      10 | 0.7815 |     0.0102 |        0.0313 |     0.0643 |
| p=100,snr=1.78 | CompMP-b   |       3 | 0.7187 |     0.0000 |        0.0000 |     0.0889 |
| p=100,snr=1.78 | CompMP-c   |       3 | 0.7240 |     0.0000 |        0.0000 |     0.0868 |
| p=10,C=5       | CompMP-a   |      10 | 0.6901 |     0.0161 |        0.0446 |     0.0295 |
| p=10,C=5       | CompMP-a-M |      10 | 0.6975 |     0.0144 |        0.0468 |     0.0347 |
| p=10,C=5       | GCN-ilr    |      10 | 0.7084 |     0.0077 |        0.0235 |     0.0338 |
| p=10,C=5       | CompMP-b   |       3 | 0.6876 |     0.0000 |        0.0000 |     0.0320 |
| p=10,C=5       | CompMP-c   |       3 | 0.6878 |     0.0000 |        0.0000 |     0.0456 |
| p=100,C=5      | CompMP-a   |      10 | 0.4798 |     0.0154 |        0.0439 |     0.0552 |
| p=100,C=5      | CompMP-a-M |      10 | 0.4691 |     0.0129 |        0.0361 |     0.0366 |
| p=100,C=5      | GCN-ilr    |      10 | 0.5977 |     0.0125 |        0.0400 |     0.0156 |
| p=100,C=5      | CompMP-b   |       3 | 0.4469 |     0.0000 |        0.0000 |     0.0544 |
| p=100,C=5      | CompMP-c   |       3 | 0.4777 |     0.0000 |        0.0000 |     0.0549 |


E2 was paused after 2 of 18 settings were complete; only those are shown.

## E2: scalar targets, n = 2000 (test accuracy, mean ± std over seeds)

Validation-selected config per model and setting (8-config grid: lr {0.005, 0.01} × hidden {8, 16} × dropout {0.2, 0.5}; linear-ilr: 4 values of C). Seeds per cell: 10–10.

### p=10

| model      | p=10,snr=1    | p=10,snr=1.78   |
|:-----------|:--------------|:----------------|
| CompMP-c   | 0.826 ± 0.034 | 0.895 ± 0.025   |
| GCN-ilr    | 0.820 ± 0.036 | 0.892 ± 0.024   |
| COMP-GCN   | 0.835 ± 0.035 | 0.895 ± 0.021   |
| GCN-scalar | 0.805 ± 0.112 | 0.905 ± 0.023   |
| MLP-ilr    | 0.770 ± 0.082 | 0.770 ± 0.082   |
| linear-ilr | 0.809 ± 0.070 | 0.809 ± 0.070   |
| constant   | 0.500 ± 0.002 | 0.500 ± 0.002   |


### p=100

| model      |
|:-----------|
| CompMP-c   |
| GCN-ilr    |
| COMP-GCN   |
| GCN-scalar |
| MLP-ilr    |
| linear-ilr |
| constant   |


### Paired difference to GCN-ilr (same seeds and splits), mean [95% CI]

#### p=10

| model      | p=10,snr=1              | p=10,snr=1.78           |
|:-----------|:------------------------|:------------------------|
| CompMP-c   | +0.006 [-0.006, +0.017] | +0.003 [-0.003, +0.008] |
| COMP-GCN   | +0.015 [+0.004, +0.025] | +0.003 [-0.004, +0.010] |
| GCN-scalar | -0.015 [-0.081, +0.050] | +0.012 [+0.007, +0.018] |
| MLP-ilr    | -0.050 [-0.089, -0.012] | -0.123 [-0.167, -0.078] |
| linear-ilr | -0.011 [-0.040, +0.017] | -0.084 [-0.120, -0.047] |
| constant   | -0.320 [-0.347, -0.293] | -0.392 [-0.410, -0.374] |


#### p=100

| model      |
|:-----------|
| CompMP-c   |
| COMP-GCN   |
| GCN-scalar |
| MLP-ilr    |
| linear-ilr |
| constant   |


Settings where the CI excludes 0 (win / loss vs GCN-ilr):

| model      |   wins |   losses |   ties |
|:-----------|-------:|---------:|-------:|
| CompMP-c   |      0 |        0 |      2 |
| COMP-GCN   |      1 |        0 |      1 |
| GCN-scalar |      1 |        0 |      1 |
| MLP-ilr    |      0 |        2 |      0 |
| linear-ilr |      0 |        1 |      1 |
| constant   |      0 |        2 |      0 |


### Selected configs

| setting       | model           | lr    | hidden   | dropout   | C   |
|:--------------|:----------------|:------|:---------|:----------|:----|
| p=10,snr=1    | COMP-GCN        | 0.005 | 16.0     | 0.5       | -   |
| p=10,snr=1    | CompMP-c        | 0.01  | 16.0     | 0.5       | -   |
| p=10,snr=1    | CompMP-full     | 0.01  | 16.0     | 0.5       | -   |
| p=10,snr=1    | CompMP-full-row | 0.01  | 16.0     | 0.5       | -   |
| p=10,snr=1    | GCN-ilr         | 0.01  | 16.0     | 0.2       | -   |
| p=10,snr=1    | GCN-scalar      | 0.01  | 16.0     | 0.2       | -   |
| p=10,snr=1    | MLP-ilr         | 0.01  | 16.0     | 0.5       | -   |
| p=10,snr=1    | constant        | -     | -        | -         | -   |
| p=10,snr=1    | linear-ilr      | -     | -        | -         | 0.1 |
| p=10,snr=1.78 | COMP-GCN        | 0.005 | 16.0     | 0.5       | -   |
| p=10,snr=1.78 | CompMP-c        | 0.01  | 16.0     | 0.5       | -   |
| p=10,snr=1.78 | CompMP-full     | 0.01  | 16.0     | 0.2       | -   |
| p=10,snr=1.78 | CompMP-full-row | 0.01  | 16.0     | 0.2       | -   |
| p=10,snr=1.78 | GCN-ilr         | 0.01  | 16.0     | 0.2       | -   |
| p=10,snr=1.78 | GCN-scalar      | 0.005 | 16.0     | 0.2       | -   |
| p=10,snr=1.78 | MLP-ilr         | 0.01  | 16.0     | 0.5       | -   |
| p=10,snr=1.78 | constant        | -     | -        | -         | -   |
| p=10,snr=1.78 | linear-ilr      | -     | -        | -         | 0.1 |

