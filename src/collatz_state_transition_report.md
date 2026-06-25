# Collatz State-Conditioned Transition Diagnostic

- iid samples per h: 160000
- actual sample per p,h: 20000
- seed: 20260625
- focus: long tau only; x_K windows 0..31, 32..63, 64..95
- k coarse grain: 1, 2, 3+
- u bins: width 0.1 using transition current position
- bridge clusters: iid long-word z25 tertiles q_low=-1.5, q_high=-0.25

## Focus State Max JS

| state | max u_bin | max JS bits | actual transition mass | iid transition mass |
|---|---|---:|---:|---:|
| late_growth|tail_64_95|even | 0.8-0.9 | 0.004455 | 0.0120729 | 0.0254226 |
| late_growth|tail_64_95|odd | 0.4-0.5 | 0.006208 | 0.0108379 | 0.0161075 |
| late_growth|deep_32_63|even | 0.7-0.8 | 0.001849 | 0.0608025 | 0.0740779 |
| late_growth|deep_32_63|odd | 0.7-0.8 | 0.002112 | 0.0488623 | 0.0486629 |
| early_growth|tail_64_95|even | 0.8-0.9 | 0.009803 | 0.00682505 | 0.00839557 |
| early_growth|tail_64_95|odd | 0.7-0.8 | 0.004349 | 0.00526321 | 0.00535309 |

## Largest Stable JS Rows

| state | u_bin | k_current | JS bits | actual mass | iid mass |
|---|---|---|---:|---:|---:|
| early_growth|tail_64_95|even | 0.8-0.9 | ALL | 0.009803 | 0.00682505 | 0.00839557 |
| late_growth|tail_64_95|odd | 0.4-0.5 | ALL | 0.006208 | 0.0108379 | 0.0161075 |
| late_growth|tail_64_95|odd | 0.7-0.8 | ALL | 0.006081 | 0.0109284 | 0.0162528 |
| balanced|tail_64_95|even | 0.8-0.9 | ALL | 0.004645 | 0.00388642 | 0.00568201 |
| late_growth|tail_64_95|even | 0.8-0.9 | ALL | 0.004455 | 0.0120729 | 0.0254226 |
| balanced|tail_64_95|even | 0.1-0.2 | ALL | 0.004406 | 0.00386622 | 0.00562505 |
| early_growth|tail_64_95|odd | 0.7-0.8 | ALL | 0.004349 | 0.00526321 | 0.00535309 |
| late_growth|tail_64_95|odd | 0.6-0.7 | ALL | 0.004222 | 0.0110555 | 0.0163819 |
| balanced|tail_64_95|odd | 0.7-0.8 | ALL | 0.004194 | 0.00361636 | 0.0036395 |
| late_growth|tail_64_95|even | 0.6-0.7 | ALL | 0.003956 | 0.0119521 | 0.0250149 |
| balanced|tail_64_95|even | 0.9-1.0 | ALL | 0.003763 | 0.00355373 | 0.00522549 |
| early_growth|tail_64_95|even | 0.7-0.8 | ALL | 0.003440 | 0.00657108 | 0.00815708 |

## Focus Mean Next-k Delta Extremes

| state | min delta u | min delta | max delta u | max delta |
|---|---|---:|---|---:|
| late_growth|tail_64_95|even | 0.0-0.1 | -0.077682 | 0.4-0.5 | 0.076459 |
| late_growth|tail_64_95|odd | 0.7-0.8 | -0.117799 | 0.4-0.5 | 0.132145 |
| late_growth|deep_32_63|even | 0.6-0.7 | -0.039770 | 0.3-0.4 | 0.041141 |
| late_growth|deep_32_63|odd | 0.7-0.8 | -0.059764 | 0.2-0.3 | 0.052307 |
| early_growth|tail_64_95|even | 0.7-0.8 | -0.055904 | 0.9-1.0 | 0.044210 |
| early_growth|tail_64_95|odd | 0.7-0.8 | -0.110063 | 0.3-0.4 | 0.102824 |

## Reading

JS is computed both for k_current-specific transition distributions and for k_current=ALL. The ALL rows are the safer diagnostic when a conditioned current state is thin. This is still a sampled diagnostic: strong local JS supports transition-process distortion, while low JS with strong survival loss supports whole-word selection/weighting.

## Classification

Closest classification: **B with a weak transition-distortion signal**.

The depleted states do show local transition differences, especially in the `tail_64_95` window, but the `k_current=ALL` JS values are small in absolute size. The largest stable collapsed JS is about `0.0098` bits, and the key depleted state `late_growth|tail_64_95|even` peaks at about `0.00445` bits. This is much smaller than the survival loss previously observed for the same state.

The directional pattern is still informative: in `late_growth|tail_64_95`, actual tends to reduce `k_next=1` and increase `k_next>=3` around the middle of the word, then increase `k_next=1` in the later bins. This suggests a bridge-shape distortion, but not a large local Markov-kernel break.

Therefore the current evidence favors:

> survival depletion is mostly selection/weighting over whole long words, with small position-dependent transition distortions rather than a strong local transition-law failure.

The next useful diagnostic is not a larger scalar; it is a prefix/block transition test, e.g. comparing short blocks `(k_i,k_{i+1},k_{i+2})` or bridge-conditioned block frequencies inside the depleted state.
