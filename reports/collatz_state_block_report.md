# Collatz State-Conditioned Block Diagnostic

- iid samples per h: 160000
- actual sample per p,h: 20000
- seed: 20260625
- focus: long tau only; x_K windows 0..31, 32..63, 64..95
- k coarse grain: 1, 2, 3+
- block lengths: B3 and B4
- u bins: width 0.1 using block start position
- bridge clusters: iid long-word z25 tertiles q_low=-1.5, q_high=-0.25

## Focus State Max JS

| block | state | max u_bin | max JS bits | actual mass | iid mass |
|---:|---|---|---:|---:|---:|
| 3 | late_growth|tail_64_95|even | 0.4-0.5 | 0.041475 | 0.0117263 | 0.0246236 |
| 4 | late_growth|tail_64_95|even | 0.4-0.5 | 0.080936 | 0.0117263 | 0.0246236 |
| 3 | late_growth|tail_64_95|odd | 0.4-0.5 | 0.080330 | 0.0108379 | 0.0161075 |
| 4 | late_growth|tail_64_95|odd | 0.4-0.5 | 0.149491 | 0.0108379 | 0.0161075 |
| 3 | early_growth|tail_64_95|even | 0.8-0.9 | 0.033414 | 0.00682505 | 0.00839557 |
| 4 | early_growth|tail_64_95|even | 0.4-0.5 | 0.064327 | 0.00658379 | 0.00814234 |
| 3 | late_growth|deep_32_63|even | 0.9-1.0 | 0.010189 | 0.041715 | 0.0511053 |
| 4 | late_growth|deep_32_63|even | 0.9-1.0 | 0.022949 | 0.0274335 | 0.0338712 |

## Largest Stable Block JS Rows

| block | state | u_bin | JS bits | actual mass | iid mass |
|---:|---|---|---:|---:|---:|
| 4 | late_growth|tail_64_95|odd | 0.4-0.5 | 0.149491 | 0.0108379 | 0.0161075 |
| 4 | late_growth|tail_64_95|odd | 0.8-0.9 | 0.087112 | 0.0111763 | 0.0166205 |
| 4 | late_growth|tail_64_95|odd | 0.3-0.4 | 0.085782 | 0.0108502 | 0.016155 |
| 4 | late_growth|tail_64_95|odd | 0.6-0.7 | 0.084707 | 0.0110555 | 0.0163819 |
| 4 | late_growth|tail_64_95|even | 0.4-0.5 | 0.080936 | 0.0117263 | 0.0246236 |
| 3 | late_growth|tail_64_95|odd | 0.4-0.5 | 0.080330 | 0.0108379 | 0.0161075 |
| 4 | balanced|tail_64_95|odd | 0.7-0.8 | 0.079887 | 0.00361636 | 0.0036395 |
| 4 | late_growth|tail_64_95|odd | 0.9-1.0 | 0.076894 | 0.00684215 | 0.0102046 |
| 4 | late_growth|tail_64_95|odd | 0.5-0.6 | 0.076787 | 0.011733 | 0.0173525 |
| 4 | balanced|tail_64_95|odd | 0.4-0.5 | 0.076713 | 0.00359089 | 0.0036021 |
| 4 | late_growth|tail_64_95|odd | 0.7-0.8 | 0.075600 | 0.0109284 | 0.0162528 |
| 4 | balanced|tail_64_95|even | 0.9-1.0 | 0.071085 | 0.00233736 | 0.0034636 |
| 4 | late_growth|tail_64_95|even | 0.8-0.9 | 0.068174 | 0.0120729 | 0.0254226 |
| 4 | early_growth|tail_64_95|odd | 0.4-0.5 | 0.066960 | 0.00524204 | 0.00532571 |
| 4 | balanced|tail_64_95|even | 0.4-0.5 | 0.065265 | 0.00374776 | 0.00549826 |
| 4 | early_growth|tail_64_95|even | 0.4-0.5 | 0.064327 | 0.00658379 | 0.00814234 |

## Depleted Blocks In Main State

Top B3 depleted blocks for `late_growth|tail_64_95|even`, sorted within each u_bin by probability log ratio.

| u_bin | block | log2 ratio | actual prob | iid prob |
|---|---|---:|---:|---:|
| 0.0-0.1 | 2,3+,3+ | -1.566 | 0.00246399 | 0.00729301 |
| 0.0-0.1 | 3+,3+,2 | -0.951 | 0.00413729 | 0.00800039 |
| 0.0-0.1 | 3+,3+,3+ | -0.739 | 0.00248598 | 0.00415041 |
| 0.0-0.1 | 3+,2,3+ | -0.647 | 0.00539323 | 0.00844508 |
| 0.0-0.1 | 3+,3+,1 | -0.485 | 0.0149518 | 0.0209234 |
| 0.1-0.2 | 2,3+,3+ | -0.699 | 0.00493578 | 0.00801371 |
| 0.1-0.2 | 3+,3+,1 | -0.525 | 0.0134222 | 0.0193136 |
| 0.1-0.2 | 2,2,3+ | -0.522 | 0.00889065 | 0.0127708 |
| 0.1-0.2 | 1,2,3+ | -0.515 | 0.0204205 | 0.0291829 |
| 0.1-0.2 | 3+,1,1 | -0.414 | 0.0491065 | 0.0654466 |
| 0.2-0.3 | 3+,3+,3+ | -1.661 | 0.000634271 | 0.00200581 |
| 0.2-0.3 | 2,1,2 | -0.905 | 0.0189797 | 0.0355303 |
| 0.2-0.3 | 1,3+,3+ | -0.816 | 0.00634359 | 0.0111694 |
| 0.2-0.3 | 2,3+,1 | -0.765 | 0.0132001 | 0.0224364 |
| 0.2-0.3 | 2,2,3+ | -0.682 | 0.00431967 | 0.00693189 |
| 0.3-0.4 | 2,2,2 | -1.506 | 0.00427929 | 0.0121555 |
| 0.3-0.4 | 2,3+,2 | -1.236 | 0.0023998 | 0.00565214 |
| 0.3-0.4 | 2,2,3+ | -1.161 | 0.00230431 | 0.00515353 |
| 0.3-0.4 | 3+,2,1 | -1.107 | 0.00890685 | 0.0191885 |
| 0.3-0.4 | 3+,2,2 | -1.051 | 0.0032039 | 0.0066373 |
| 0.4-0.5 | 3+,3+,3+ | -4.177 | 8.18744e-05 | 0.00148057 |
| 0.4-0.5 | 2,2,3+ | -3.329 | 0.000573658 | 0.00576463 |
| 0.4-0.5 | 2,3+,3+ | -2.003 | 0.000829918 | 0.00332713 |
| 0.4-0.5 | 2,3+,1 | -1.877 | 0.00478477 | 0.0175771 |
| 0.4-0.5 | 1,2,3+ | -1.481 | 0.00618666 | 0.0172654 |
| 0.5-0.6 | 2,3+,3+ | -1.542 | 0.000615249 | 0.00179166 |
| 0.5-0.6 | 2,2,2 | -1.102 | 0.00557643 | 0.0119712 |
| 0.5-0.6 | 1,2,3+ | -0.939 | 0.00939425 | 0.0180095 |
| 0.5-0.6 | 2,3+,2 | -0.502 | 0.00395536 | 0.00560293 |
| 0.5-0.6 | 2,1,2 | -0.456 | 0.0253062 | 0.0347245 |
| 0.6-0.7 | 3+,2,3+ | -3.009 | 0.000242047 | 0.00194861 |
| 0.6-0.7 | 2,3+,2 | -2.961 | 0.000646415 | 0.00503236 |
| 0.6-0.7 | 2,3+,3+ | -1.660 | 0.000648078 | 0.00204824 |
| 0.6-0.7 | 1,3+,3+ | -1.526 | 0.00211426 | 0.00608775 |
| 0.6-0.7 | 3+,2,1 | -1.348 | 0.00615618 | 0.0156722 |
| 0.7-0.8 | 2,3+,3+ | -2.519 | 0.00016491 | 0.000945068 |
| 0.7-0.8 | 2,3+,2 | -1.857 | 0.000985688 | 0.00356974 |
| 0.7-0.8 | 3+,3+,1 | -1.666 | 0.00140133 | 0.00444559 |
| 0.7-0.8 | 2,2,2 | -1.456 | 0.00363352 | 0.00996838 |
| 0.7-0.8 | 1,3+,3+ | -1.300 | 0.00157139 | 0.00387048 |

## Classification

This diagnostic locates the first scale at which actual/iid differences become visible: one-step transition, B3/B4 short blocks, or whole-word bridge shape.

Interpretation should compare the block JS scale with the previous one-step JS scale. If B3/B4 JS is only modestly larger, the evidence still favors global whole-word/path selection. If B3/B4 JS sharply concentrates in `late_growth|tail_64_95|even`, then short-range block correlations are a plausible mediator.

Current classification: **block-scale difference is the first clear local signature, but it is not sufficient as the whole explanation**.

The one-step transition diagnostic was small: the largest collapsed `k_next` JS was about `0.0098` bits, and the key depleted state `late_growth|tail_64_95|even` peaked near `0.00445` bits. In contrast, B3 and B4 show materially larger divergences:

- `late_growth|tail_64_95|even`: B3 max `0.0415`, B4 max `0.0809`.
- `late_growth|tail_64_95|odd`: B3 max `0.0803`, B4 max `0.1495`.
- `early_growth|tail_64_95|even`: B3 max `0.0334`, B4 max `0.0643`.
- `late_growth|deep_32_63|even`: B3 max `0.0102`, B4 max `0.0229`.

Thus the first substantial actual/iid discrepancy appears at the **short block** level, not at the one-step transition level. However, it is not uniquely concentrated in the most depleted state. The high `tail_64_95` window broadly carries block-level distortion, including odd and early-growth states. This means B3/B4 correlations are a mediator or symptom of the finite-window selection, not yet a complete cause of the survival deficit.

The strongest current wording is:

> The local one-step valuation kernel remains close to iid, while short block frequencies already differ in high-x_K tail states. The survival deficit appears to arise from block-scale and bridge/whole-word selection rather than from a simple one-step transition-law failure.

This supports the hierarchy:

1. one-step transition: weak difference;
2. B3/B4 block frequencies: clear difference, especially in `tail_64_95`;
3. whole bridge/word state: still needed to explain why `late_growth|tail_64_95|even` is most depleted.
