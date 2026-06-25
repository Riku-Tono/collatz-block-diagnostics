# Collatz Block Length Renormalization

- block lengths: 3, 4, 5, 6
- split: deterministic train/test sample index parity
- smoothing: probability mixture `(1-lambda) empirical + lambda uniform`, lambda=0.02
- stable threshold: train iid state mass >= 1e-07
- regression rows: deterministic weighted subsample capped at 50000
- bridge cluster z25 cuts: q_low=-1.5, q_high=-0.25

## AUC By Length

| L | marginal score AUC | logistic base AUC | logistic + score AUC | delta |
|---:|---:|---:|---:|---:|
| 3 | 0.5611 | 0.5025 | 0.5363 | 0.0337 |
| 4 | 0.5768 | 0.5025 | 0.5436 | 0.0411 |
| 5 | 0.5966 | 0.5025 | 0.5536 | 0.0511 |
| 6 | 0.6198 | 0.5025 | 0.5643 | 0.0618 |

## Focus Survival

For `late_growth|tail_64_95|even`, L=6 lowest score decile has S=`0.117`; the highest finite-ratio decile has S=`38.091`, while the top decile has zero iid mass. Read the high-score tail as sparse but directionally strong.

## Classification

Current classification: **B. AUC grows but residual remains**.

Best +score logistic AUC occurs at L=6 with AUC `0.5643`. L=3 to L=6 +score AUC gain is `0.0280`. At L=6, bridge abs residual is `1.1011` and parity residual is `0.3749`. The AUC trend is stable enough for B, but the extreme survival tail is sparse.
