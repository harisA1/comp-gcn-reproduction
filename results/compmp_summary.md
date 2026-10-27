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

