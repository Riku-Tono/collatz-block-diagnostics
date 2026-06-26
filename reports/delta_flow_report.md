# Delta flow report

Scope: actual - iid descriptive statistics only. No paradoxical sequence comparison, hidden state, or new mechanism is introduced.

## State-transition delta

The largest overall transition delta is at `to_u_bin=0.9-1.0` with delta `-0.00202794` and ratio `0.880`.
Overall transition L1 is largest at `u_bin=0.3-0.4` with L1 `0.0578675`.
The top transition contributes `0.50%` of total transition L1 across all transition cells, so transition delta is distributed rather than single-edge concentrated.

## Prefix-growth delta

The largest overall prefix-growth mass delta is `1 -> 1,2` at m=1, delta `-0.00201933`, ratio `0.962`.
Overall prefix-growth L1 is largest at m=7 with L1 `0.0679581`.
The top growth branch contributes `1.08%` of total growth L1 across all m, again indicating dispersion as prefixes lengthen.

## Focus-state connection

- `late_growth|deep_32_63|even`: state delta `-0.00295258`; top inflow transition `0.8-0.9|16+|0:2|late_like|1|even -> 0.9-1.0|16+|2:4|balanced_like|1|even` with delta `-0.000769535`; top growth `1 -> 1,1` with delta `-0.00063364`.
- `late_growth|deep_32_63|odd`: state delta `+0.000118213`; top inflow transition `0.6-0.7|16+|0:2|late_like|1|odd -> 0.7-0.8|16+|2:4|balanced_like|1|odd` with delta `+0.000294118`; top growth `1 -> 1,3+` with delta `+0.000230683`.
- `late_growth|exhaustion_0_31|odd`: state delta `+0.00166185`; top inflow transition `0.8-0.9|0:4|0:2|balanced_like|1|odd -> 0.9-1.0|0:4|2:4|balanced_like|1|odd` with delta `+0.000343829`; top growth `1 -> 1,1` with delta `+0.000384507`.

## Conditional branch signs

Conditional branch rows are filtered to parent masses >= `1e-05` in both actual and iid to avoid thin-cylinder +/-1 artifacts.
- `late_growth|deep_32_63|even`: strongest conditional branch `2,2,3+,1,2,1,1 -> 2,2,3+,1,2,1,1,2`, conditional_delta `+0.93069`, mass delta `+9.475e-06`.
- `late_growth|deep_32_63|odd`: strongest conditional branch `2,2,1,2,3+,1,1 -> 2,2,1,2,3+,1,1,3+`, conditional_delta `+0.89367`, mass delta `+1.0545e-05`.
- `late_growth|exhaustion_0_31|odd`: strongest conditional branch `1,1,2,1,3+,2,3+ -> 1,1,2,1,3+,2,3+,2`, conditional_delta `+0.82411`, mass delta `+7.8873e-06`.

## Short answer

1. State-transition delta is not sharper than prefix-cylinder delta by single-cell concentration; it is more position-localized, especially by u_bin.
2. Delta is distributed across many transitions and growth branches rather than concentrated in one A -> B edge.
3. High-|delta| state correspondence is visible in inflow-transition maps, but the state total is not exhausted by a single inflow edge.
4. Prefix-growth mass-delta signs differ by state; supported conditional_delta extrema are positive in all three focus states, so sign reversal is clearer in mass-dominant growth and inflow-transition cells than in the supported conditional extrema.
5. Next descriptive target should be boundary delta before paradoxical-sequence overlay: boundary maps can say whether the observed flow localizes near entry/exit surfaces without importing sequence-specific interpretation.

## Outputs

- `state_transition_delta.csv`, `state_transition_top_positive.csv`, `state_transition_top_negative.csv`, `state_transition_top_abs.csv`, `state_transition_l1_by_u_bin.csv`
- `prefix_growth_delta.csv`, `prefix_growth_top_overall.csv`, `prefix_growth_top_by_focus_state.csv`, `prefix_growth_l1_by_m.csv`, `prefix_growth_conditional_top.csv`
- `delta_flow_focus_states.csv`, `delta_flow_summary.md`
- `transition_l1_by_u_bin.svg`, `transition_top_bar.svg`, `transition_top_heatmap.svg`, `prefix_growth_l1_by_m.svg`, `prefix_growth_top_bar.svg`, `prefix_growth_conditional_heatmap_*.svg`
