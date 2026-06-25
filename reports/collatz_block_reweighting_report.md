# Collatz Finite-Block Reweighting Test

- iid test words are reweighted by `2^(alpha * block_log_score_L)`.
- block lengths: 3, 4, 5, 6
- alpha sweep: 0.0, 0.25, 0.5, 0.75, 1.0
- total predicted mass is scaled to actual total mass over stable full states.
- smoothing and stable thresholds inherit the L=3..6 renormalization script.

## Best Fit

Best full-state RMSE is L=3, alpha=0.25, RMSE=`0.000440978`, JS=`0.000921815`.
For L=6, best alpha is `0.0` with RMSE `0.000642649`.
Focus state `late_growth|tail_64_95|even` is best at L=6 alpha `0.5`: actual mass `0.00092052`, predicted mass `0.00029821`.
At L=6 alpha=1, focus predicted survival is `0.00416667` versus actual survival `0.472461`.

## Residuals

At the global best fit, bridge RMSE is `0.000427051` and parity RMSE is `0.00262763`.

## Classification

Current classification: **C. reweighting overcorrects or is not generative**.

The small best-fit improvement comes from damped short-block reweighting. Longer raw reweighting is over-sharp: useful diagnostically, weak as a generative finite-block model.
