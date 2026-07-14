# ESCAPE selection check for finite-block actual vs iid

## Fixed comparison

- power: `24`
- h: `2`
- depth: `4`
- word alphabet: `1`, `2`, `3+` (the original finite-block `k_cat` definition)
- iid samples for the ESCAPE-conditioned side: `500,000`
- seed: `20260625`

## Code-confirmed definitions

- `ESCAPE`: in `compute_status`, the status becomes `ESCAPE` when the odd Syracuse orbit first has `cur > 2^power`.
- `trace_escape`: it appends each `k=v2(3n+1)` while `cur <= 2^power`; it stops immediately after the update that makes `cur > 2^power`.
- iid condition: yes. `iid_escape_sample` stops its iid walk when `position > h-log2(y)`, i.e. first passage of the matched escape boundary, and reweights that stopped sample. Therefore the original iid side is also ESCAPE-conditioned; it is not an unconditional iid prefix sample.

For the finite-block comparison, only words with at least four valuations are eligible on the ESCAPE side, matching the original B4 requirement. ESCAPE paths shorter than four: `0` of `162,938`. The all-starting-values side records exactly four Syracuse valuations for every odd start in the same finite layer.

## Comparison table

| sample | eligible n | actual-iid TV | max absolute standardized residual | word at max | sum abs deviation, top 10 |
|---|---:|---:|---:|:---|---:|
| ESCAPE | 162,938 | 0.006276567 | 2.682537 | `2,2,3+,2` | 0.006769434 |
| all_starting_values | 1,048,576 | 0.000020027 | 0.046967 | `3+,3+,3+,2` | 0.000017166 |

The standardized residual is `(observed - n*q) / sqrt(n*q*(1-q))` on the fixed 81-word support.

## Top 10 word deviations

See `escape_vs_all_depth4_top10_words.csv`. It contains, for each sample, actual count, actual probability, iid probability, signed deviation, and standardized residual for the ten largest absolute probability deviations.

## Verdict

`escape_selection_supported`

This verdict says only whether the difference is stronger with the ESCAPE condition. It is not a causal claim.

## Source files used

- `C:\Users\yauki\Documents\Codex\2026-06-25\new-chat\work\collatz_escape_word_deficit.py`
- `C:\Users\yauki\Documents\Codex\2026-06-25\new-chat-2\work\status_cache\odd_only_status_p24.bin`
- `C:\Users\yauki\Documents\design\Collatz\log\一時README\Collatz-finite-block_README.md`
- `C:\Users\yauki\Documents\Codex\2026-06-25\new-chat\outputs\collatz_escape_prefix_metrics.csv`

## Command

```powershell
python work/compare_escape_selection_depth4.py --power 24 --h 2 --depth 4 --iid-samples 500000 --seed 20260625 --out-dir outputs
```
