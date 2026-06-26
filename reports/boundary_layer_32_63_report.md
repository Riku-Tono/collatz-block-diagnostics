# Remaining_K 32-63 boundary-layer comparison

Scope: positions inside `remaining_K=32-63`. A layer point is a prefix position before the next valuation step; all in-layer positions are counted, with a first-entry diagnostic also emitted.

## Overall branch

Inside the layer, the largest next-step branch delta is `k_cat=1` with delta `-0.124919` and ratio `0.903`.
The largest next remaining_K transition delta is `32-63 -> 32-63` with delta `-0.170949` and ratio `0.911`.
The largest exit-distance bin delta is remaining_steps `32-63` with delta `-0.0936786`.

Conditioned on being inside the layer, actual has slightly less next `k_cat=1` than iid (`0.5936` vs `0.6020`) and more `k_cat=3+` (`0.1538` vs `0.1465`). The next remaining_K transition mostly stays in `32-63` for both, but actual drops to `16-31` slightly more often (`0.1062` vs `0.1016`).

## Exit distance

Mean remaining steps from layer points: actual `29.688`, iid `29.936`.
Mean remaining_K from layer points: actual `42.578`, iid `42.867`.
Mean next raw k: actual `1.631`, iid `1.617`.

## Focus states

- `late_growth|deep_32_63|even`: next_k top `1` delta `-0.0375341`, next remaining_K transition `32-63 -> 32-63` delta `-0.0537173`, mean exit steps actual/iid `32.51/32.63`.
- `late_growth|deep_32_63|odd`: next_k top `1` delta `-0.00594516`, next remaining_K transition `32-63 -> 32-63` delta `-0.00391813`, mean exit steps actual/iid `32.83/32.68`.
- `late_growth|exhaustion_0_31|odd`: next_k top `3+` delta `+0.00139112`, next remaining_K transition `32-63 -> 32-63` delta `+0.00256566`, mean exit steps actual/iid `27.36/27.33`.

## Outputs

- `boundary_layer_32_63_categorical_delta.csv`
- `boundary_layer_32_63_conditional_delta.csv`
- `boundary_layer_32_63_metric_summary.csv`
- `boundary_layer_32_63_focus_summary.csv`
- `boundary_layer_32_63_next_k.svg`
- `boundary_layer_32_63_next_remaining_K_transition.svg`
- `boundary_layer_32_63_exit_steps.svg`
- `boundary_layer_32_63_*_heatmap.svg`
