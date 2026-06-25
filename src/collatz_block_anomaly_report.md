# Collatz Block Anomaly Score

- source script: `work/collatz_block_anomaly_score.py`
- seed: 20260625
- iid samples per h: 160000
- actual samples per p,h: 20000
- split: train/test by deterministic sample index parity; train estimates block probabilities, test receives word scores
- smoothing: add-alpha 0.5 over 3^m coarse blocks
- regression rows: deterministic weighted subsample capped at 60000; score distributions and survival bins use all scored test rows
- bridge cluster z25 cuts: q_low=-1.5, q_high=-0.25

## Main Reading

`late_growth|tail_64_95|even` has B4 actual median score `-0.047` versus iid median `-0.060`; weighted AUC(actual high score vs iid) is `0.719`.
In that same state, lowest B4-score decile has S=`0.035`, highest decile has S=`1.806`.

## Regression

| model | weighted AUC | delta vs base | B4 coef | bridge abs coef | parity coef |
|---|---:|---:|---:|---:|---:|
| xK_parity_bridge_z | 0.5023 | 0.0000 | 0.0000 | 1.1312 | 0.3682 |
| plus_B4 | 0.5260 | 0.0236 | 0.2330 | 1.1151 | 0.4268 |
| plus_B3_B4 | 0.5246 | 0.0223 | 0.1695 | 1.1132 | 0.4423 |

## Classification

Current classification: **B. B4 score helps but bridge/parity remain strong**.

B4 changes AUC by `0.0236` after x_K, parity, bridge_cluster, and z25/z50/z75. Bridge coefficient reduction is `0.0162` and parity coefficient reduction is `-0.0586`. Treat this as a split-sample sampled diagnostic, not an exact aggregation proof.
