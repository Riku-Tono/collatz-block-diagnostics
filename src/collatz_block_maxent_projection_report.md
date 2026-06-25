# Collatz Maximum-Entropy Block Projection Test

- projection: approximate regularized IPF/exponential-family update on iid test measure
- block lengths: 3, 4, 5, 6
- regularization sweep: 0.0, 0.5, 0.75, 0.9
- iterations per fit: 2
- deterministic iid evaluation cap: 40000

## Best Fit

Best maxent-style fit is L=3, regularization=0.75, RMSE=`0.000493147`, JS=`0.000868073`.
Best raw/damped finite-block RMSE from the previous test was `0.000440978`.
Focus state best: L=5, regularization=0.9, actual survival `0.472461`, predicted survival `0.444001`.

## Classification

Current classification: **C. maxent no better than raw/damped**.

This is an approximate projection, not a full exact IPF over all words. The test is meant to distinguish overcounted raw products from a regularized finite-block exponential family.