# Remaining_K 64-95 layer analysis

Scope: all positions with `remaining_K=64-95`, and the subset of positions whose next step enters `32-63`.

## Layer mass

64-95 layer mass is actual `0.266435` vs iid `0.341662`, delta `-0.0752273`, ratio `0.780`.

## Exit transition

`64-95 -> 32-63` mass is actual `0.0309729` vs iid `0.0370321`, delta `-0.00605924`, ratio `0.836`.
Conditioned on being in 64-95, `64-95 -> 32-63` is actual `0.1162` vs iid `0.1084`, conditional_delta `+0.00786124`.
`64-95 -> 64-95` mass delta is `-0.069168`.

## Next k

Largest next k_cat delta is `1` with delta `-0.0413526`.
`raw k >= 3` delta `-0.0152769`; `raw k >= 4` delta `-0.00496292`.
Mean next raw k actual/iid is `1.661` / `1.688`.

## Focus states

- `late_growth|deep_32_63|even`: layer delta `-0.0123009`, 64-95->32-63 delta `-0.00206815`, conditional `0.1771/0.1751`.
- `late_growth|deep_32_63|odd`: layer delta `-0.00159669`, 64-95->32-63 delta `-6.86167e-05`, conditional `0.1800/0.1740`.
- `late_growth|exhaustion_0_31|odd`: layer delta `+0`, 64-95->32-63 delta `+0`, conditional `nan/nan`.

## Short reading

1. The 64-95 band itself is thin in actual.
2. The 64-95 -> 32-63 deficit is mostly mass-level thinning; the conditional transition probability should be read separately from the mass delta.
3. k_cat/raw-k differences identify which valuation sizes are underrepresented in the band and in the fall to 32-63.
4. Focus states differ in sign and route; the even deep late-growth deficit is the clearest connection to the previous 32-63 entry deficit.
5. Next look should be the 96+ predecessor band before any sequence overlay, because it is the upstream side of the depleted 64-95 inflow.