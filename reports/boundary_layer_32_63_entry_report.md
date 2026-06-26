# Remaining_K 32-63 first-entry analysis

Scope: first entry into the `remaining_K=32-63` boundary layer for each escape word.

## Entry mass

Total first-entry mass is actual `0.208145` vs iid `0.217407`, delta `-0.00926198`, ratio `0.957`.
The largest entry-kind delta is `inflow_step` with delta `-0.00605924`.
The largest entry transition delta is `64-95 -> 32-63` with delta `-0.00605924`, ratio `0.836`.

## Entry path

The largest entry-step k_cat delta is `START` with delta `-0.00320273`.
Conditioned on entry, the largest k_cat conditional shift is `START` with conditional_delta `+0.0215311`.
The largest entry_u_bin delta is `0.0-0.1` with delta `-0.00394228`.

## Short future after entry

The largest 3-step future mass delta is `1,1,1` with delta `-0.00191603`.
Conditioned on entry, the largest 3-step future conditional shift is `1,1,2` with conditional_delta `+0.00443845`.
Mean entry prefix length actual/iid: `1.370` / `1.717`.
Mean exit remaining steps after entry actual/iid: `31.510` / `31.935`.

## Focus states

- `late_growth|deep_32_63|even`: entry path `64-95 -> 32-63` delta `-0.00206815`, entry k `3+` delta `-0.000948799`, entry u `0.0-0.1` delta `-0.00169483`, future3 `1,1,1` delta `-0.000648989`.
- `late_growth|deep_32_63|odd`: entry path `START_IN_32-63 -> 32-63` delta `+0.000186829`, entry k `3+` delta `-0.00025961`, entry u `0.0-0.1` delta `+0.000374855`, future3 `3+,2,1` delta `+0.0001629`.
- `late_growth|exhaustion_0_31|odd`: entry path `START_IN_32-63 -> 32-63` delta `+0.000817314`, entry k `START` delta `+0.000817314`, entry u `0.0-0.1` delta `+0.000817314`, future3 `3+,1,1` delta `+0.000249407`.

## Short reading

1. The 32-63 layer deficit is strongly visible at first-entry mass, so entry-side thinning is a plausible description of the layer deficit.
2. The thinnest entry path is external inflow `64-95 -> 32-63`; `START_IN_32-63` is also slightly thin in mass, but conditionally occupies a larger share because external inflow is more depleted.
3. Focus states do not share one entry signature: even deep late-growth is entry-deficit, exhaustion odd is entry-excess, and odd deep late-growth is mixed/small.
4. The first 1-3 steps after entry show differences, but their conditional shifts are smaller than the entry-mass thinning.
5. Next look should be the 64-95 predecessor band before sequence overlay, because entry paths from `64-95` and `96+` separate external inflow from start-in-layer cases.
