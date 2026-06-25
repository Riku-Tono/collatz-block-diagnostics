# Collatz State Survival Operator

- source: outputs/collatz_bridge_long_word_features.csv
- focus: sampled long tau escape words
- state: tau_bin, bridge_cluster, x_K_bin, p_parity
- bridge clusters: iid z25 tertiles, q_low=-1.5, q_high=-0.25
- x_K bin width: 4
- stable rows require iid_mass >= 1e-05

## Cluster Marginal

| cluster | actual mass | iid mass | S |
|---|---:|---:|---:|
| balanced | 0.120809 | 0.123943 | 0.974714 |
| early_growth | 0.186446 | 0.187528 | 0.994232 |
| late_growth | 0.165606 | 0.173595 | 0.953982 |

## Full State Stability

- stable full states: 223
- min S among stable states: 0.000000
- max S among stable states: 1.224246
- mean |S-1| among stable states: 0.323286

## Most Depleted Stable States

| tau_bin | cluster | x_K_bin | parity | actual mass | iid mass | S |
|---|---|---|---|---:|---:|---:|
| long | balanced | +112..+115 | even | 0 | 1.32802e-05 | 0.000000 |
| long | balanced | +120..+123 | even | 0 | 1.23853e-05 | 0.000000 |
| long | early_growth | +108..+111 | odd | 0 | 1.7131e-05 | 0.000000 |
| long | early_growth | +112..+115 | even | 0 | 1.89834e-05 | 0.000000 |
| long | early_growth | +124..+127 | even | 0 | 1.22374e-05 | 0.000000 |
| long | late_growth | +124..+127 | odd | 0 | 3.31994e-05 | 0.000000 |
| long | late_growth | +128..+131 | odd | 0 | 4.14533e-05 | 0.000000 |
| long | late_growth | +132..+135 | even | 0 | 5.21883e-05 | 0.000000 |
| long | late_growth | +132..+135 | odd | 0 | 2.80279e-05 | 0.000000 |
| long | late_growth | +136..+139 | even | 0 | 4.47449e-05 | 0.000000 |

## Most Excess Stable States

| tau_bin | cluster | x_K_bin | parity | actual mass | iid mass | S |
|---|---|---|---|---:|---:|---:|
| long | balanced | +84..+87 | odd | 5.62739e-05 | 4.59661e-05 | 1.224246 |
| long | early_growth | +48..+51 | odd | 0.000552799 | 0.000456241 | 1.211637 |
| long | balanced | +76..+79 | odd | 8.07166e-05 | 6.72345e-05 | 1.200523 |
| long | early_growth | +76..+79 | odd | 0.000124502 | 0.000106158 | 1.172801 |
| long | late_growth | +40..+43 | odd | 0.00162199 | 0.00146954 | 1.103739 |
| long | early_growth | -12..-9 | odd | 0.0069987 | 0.00645259 | 1.084634 |
| long | late_growth | +44..+47 | odd | 0.00179042 | 0.00165207 | 1.083749 |
| long | late_growth | +16..+19 | odd | 0.00485912 | 0.0045159 | 1.076004 |
| long | balanced | +44..+47 | even | 0.000689133 | 0.000640477 | 1.075969 |
| long | balanced | +48..+51 | odd | 0.000308805 | 0.000289104 | 1.068144 |

## Reading

This table is a small-state survival operator rather than a proof of mechanism. It tests whether the late-growth depletion persists after splitting by cumulative bit-budget bin and p parity.