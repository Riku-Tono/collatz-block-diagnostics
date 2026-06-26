# Boundary delta report

Scope: actual - iid descriptive statistics only. No paradoxical sequence overlay, new mechanism, or generation claim is introduced.
Boundary bins use one representative prefix per u-decile per word. `remaining_K=0-1` and `remaining_steps=0` are nearest the escape boundary.

## Boundary localization

For ALL words, remaining_K L1 delta is largest at `32-63` with bin L1 `0.0339534`.
Remaining_steps L1 delta is largest at `16-31` with bin L1 `0.0320406`.
Nearest-boundary remaining_K bin `0-1` has delta `-0.00894702` and ratio `0.973`.
Normalized remaining_fraction is flat by construction under the one-point-per-decile sampling and is therefore not the main boundary coordinate here.

## Focus states

- `late_growth|deep_32_63|even`: largest remaining_K bin `32-63` delta `-0.011763`, ratio `0.826`.
- `late_growth|deep_32_63|odd`: largest remaining_K bin `16-31` delta `+0.000943822`, ratio `1.034`.
- `late_growth|exhaustion_0_31|odd`: largest remaining_K bin `16-31` delta `+0.00686272`, ratio `1.047`.

## Coordinate concentration comparison

The highest max-cell share among compared coordinates is `boundary_remaining_K` at `37.35%`.
Boundary remaining_K max-cell share is `37.35%`, versus state `28.12%` and transition `0.50%`.

## Short answer

1. Delta is stronger in late-internal boundary distance than at the final boundary itself: remaining_K `32-63` and remaining_steps `16-31` dominate ALL.
2. State-level delta is not explained away by boundary alone; focus states retain distinct boundary curves.
3. Boundary remaining_K is more concentrated than the fine transition map and also exceeds state by max-cell share, though it uses fewer coarse bins.
4. Delta is biased toward internal/late-internal distance-to-boundary bins, with a visible exit-side signal but not a single final-boundary spike.
5. At this point, the sharpest coordinate remains `boundary_remaining_K` by max-cell share in the comparison table.

## Outputs

- `boundary_delta.csv`, `boundary_delta_summary.csv`, `boundary_state_delta.csv`
- `coordinate_concentration_comparison.csv`
- `boundary_remaining_K_delta.svg`, `boundary_remaining_K_l1.svg`, `boundary_remaining_K_ratio.svg`
- `boundary_remaining_steps_delta.svg`, `boundary_remaining_steps_l1.svg`, `boundary_remaining_steps_ratio.svg`
- `boundary_remaining_fraction_delta.svg`, `boundary_remaining_fraction_l1.svg`, `boundary_remaining_fraction_ratio.svg`
- `boundary_focus_state_delta.svg`, `boundary_state_heatmap.svg`