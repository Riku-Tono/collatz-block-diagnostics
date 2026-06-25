# Collatz finite-block diagnostics vs. an iid 2-adic reference

A measurement study of **how finite-integer Collatz escape-word statistics deviate
from an iid 2-adic word model** — and, just as importantly, of how far simple
finite-block descriptions can and *cannot* go in reproducing that deviation.

This repository contains four diagnostic experiments, their reports, the figures
they produce, and a short paper written as standalone HTML chapters.

---

## What this study is

We compare two families of *escape words* — the sequences of 2-adic valuations
`k_i = v2(3n+1)` recorded along accelerated (Syracuse) Collatz trajectories:

- **actual**: words from finite integers, enumerated from exhaustive residue-status
  caches over odd residues up to `2^power`;
- **iid**: words sampled from an iid 2-adic reference model (geometric-like tilted
  valuations).

Both are compared **inside conditioning cells** called *states*, defined as
`bridge_cluster | x_K_window | parity` (see [Method](paper/03_method.html)). Within
each state we ask a sequence of increasingly demanding questions:

1. Do short blocks of valuation categories (`B3`, `B4`) carry a *diagnostic* signal
   that separates actual from iid? — `collatz_block_anomaly_score.py`
2. Does that signal *grow* with block length `L = 3..6`, and does the structural
   residual shrink? — `collatz_block_length_renormalization.py`
3. Can a finite-block reweighting of the iid measure *generate* the actual mass? —
   `collatz_block_reweighting_report.md` (reweighting test)
4. Can an approximate maximum-entropy block projection do better than raw/damped
   reweighting? — `collatz_block_maxent_projection.py`

## What this study is **not**

This is a measurement and model-comparison study. In particular:

- **It does not prove (or bear on the truth of) the Collatz conjecture.**
- **It does not identify a mechanism.** We do not claim the deviation "is" a Doob
  *h*-transform, a hidden semi-Markov process, a Gibbs measure, a first-passage
  effect, or any other specific object. Such classes appear **only as candidates**
  in the Discussion.
- **It does not present a generative model of the finite-integer word measure.** The
  central finding is the opposite: finite-block features are useful *diagnostics* but
  *weak generators*.

The honest one-line summary:

> Finite-block features have measurable diagnostic power that increases with block
> length, but as generative models they do not reproduce the whole-word deficit, and
> bridge- and parity-indexed residuals persist.

---

## Experiment pipeline

```
collatz_escape_word_deficit.py        (upstream source: trajectories, status caches)
        │   provides trace_escape, layer_bounds, tilted_k, ESCAPE status
        ▼
collatz_block_anomaly_score.py        (1) shared definitions + B3/B4 anomaly test
        │   defines: k_cat, state_key, bridge clusters, x_K windows, u-bins,
        │            iid sampling, weighted AUC / logistic helpers
        ▼
collatz_block_length_renormalization.py  (2) extends blocks to L=3..6; AUC vs L
        ▼
collatz_block_maxent_projection.py    (4) approximate regularized IPF projection
                                          (reuses the renormalization machinery)

reweighting test                      (3) reweights iid words by 2^(alpha * score)
                                          (report: collatz_block_reweighting_report.md)
```

Each stage imports definitions from the stage above it, so the **state definition,
smoothing, train/test split, and stability thresholds are shared** rather than
re-derived. This keeps the four tests comparable.

### Shared conventions (defined in `collatz_block_anomaly_score.py`)

| Object | Definition |
|---|---|
| `k_cat(k)` | valuation bucket: `1`, `2`, or `3+` |
| block / `B_L` | length-`L` window of consecutive `k_cat` values (`3^L` blocks) |
| `bridge_cluster` | tertile of a path-shape feature `z25` → `early_growth` / `balanced` / `late_growth` |
| `x_K_window` | window of `x_K = k_tau - (power - h)` → `exhaustion_0_31` / `deep_32_63` / `tail_64_95` |
| `parity` | parity of `power` (`even` / `odd`) |
| `state` | `bridge_cluster | x_K_window | parity` |
| block score | sum over windows of `log2(actual_p / iid_p)`, estimated on the train split |
| focus state | `late_growth | tail_64_95 | even` — a deep-tail stress cell |

Default constants: `POWERS = [24,25,26,27,28]`, `HS = [2,3,4,5,6]`,
`IID_SAMPLES_PER_H = 160000`, `ACTUAL_SAMPLE_PER_PH = 20000`, `SEED = 20260625`.

---

## Directory layout

```
.
├── README.md
├── collatz_block_anomaly_score.py            # test (1) + shared definitions
├── collatz_block_length_renormalization.py   # test (2)
├── collatz_block_maxent_projection.py        # test (4)
│   # test (3) reweighting is reported in collatz_block_reweighting_report.md
│
├── reports/
│   ├── collatz_block_anomaly_report.md
│   ├── collatz_block_length_renormalization_report.md
│   ├── collatz_block_reweighting_report.md
│   └── collatz_block_maxent_projection_report.md
│
├── figures/
│   ├── maxent_vs_raw_rmse.svg
│   ├── residuals_vs_regularization.svg
│   └── focus_state_maxent_fit.svg
│
└── paper/
    ├── style.css
    ├── 01_introduction.html
    ├── 02_background.html
    ├── 03_method.html
    ├── 04_results.html
    ├── 05_negative_results.html
    ├── 06_discussion.html
    └── 07_appendix.html
```

> Note: the scripts as written expect an upstream module
> `collatz_escape_word_deficit.py` and binary status caches
> (`odd_only_status_p{power}.bin`). Paths are currently hard-coded near the top of
> `collatz_block_anomaly_score.py` (`SRC`, `CACHE_DIRS`) and must be edited to point
> at your local copies before running.

---

## Role of each report

- **`collatz_block_anomaly_report.md`** — Test (1). Establishes that the `B4`
  log-ratio score is a *diagnostic* that adds a small amount of separation
  (`AUC +0.0236`) on top of a baseline built from `x_K`, parity, bridge cluster, and
  path-shape features — while the bridge and parity coefficients remain large.
  Self-classification **B**.
- **`collatz_block_length_renormalization_report.md`** — Test (2). Shows the
  diagnostic AUC grows monotonically with `L` (`0.5363 → 0.5643` for the `+score`
  logistic; `0.5611 → 0.6198` for the marginal score), but bridge/parity residuals do
  not vanish. Self-classification **B**.
- **`collatz_block_reweighting_report.md`** — Test (3). A *generative* attempt:
  reweighting iid words by `2^(alpha * score)`. The best fit is barely-damped short
  blocks; stronger/longer reweighting *overcorrects*. Self-classification **C**.
- **`collatz_block_maxent_projection_report.md`** — Test (4). A second *generative*
  attempt via approximate regularized IPF. Its best RMSE (`0.000493147`) does **not**
  beat raw/damped reweighting (`0.000440978`). Self-classification **C**.

The meaning of the **B / C** letters is explained honestly in
[Negative Results](paper/05_negative_results.html): each test emits its own coarse
self-verdict; `B` = "diagnostic signal present, structural residual remains";
`C` = "the generative attempt does not beat the simple/damped baseline (or
overcorrects)". The verdict `A` ("finite blocks reconstruct the whole-word deficit")
was never reached.

---

## Reproducing

Requirements: Python 3.10+ and `numpy`.

```bash
python -m pip install numpy
```

1. Obtain or build the upstream `collatz_escape_word_deficit.py` and the
   `odd_only_status_p{24..28}.bin` status caches.
2. Edit the `SRC` and `CACHE_DIRS` paths at the top of
   `collatz_block_anomaly_score.py`.
3. Run the tests (each writes CSVs, an SVG or two, and a Markdown report into
   `outputs/`):

```bash
python collatz_block_anomaly_score.py
python collatz_block_length_renormalization.py
python collatz_block_maxent_projection.py
```

All randomness is seeded (`SEED = 20260625`) and the train/test split is by
deterministic sample-index parity, so reruns on the same caches are reproducible up
to floating-point summation order.

---

## Hypotheses negated so far

Within the scope of these four tests (and the prior steps that motivated them), the
following descriptions **fail to account** for the finite-vs-iid discrepancy:

- the cumulative-valuation window `x_K` alone (baseline logistic AUC ≈ `0.50`);
- escape-word length `tau` alone;
- mean valuation / cumulative drift alone;
- a near-iid one-step picture (local transitions are close to iid);
- **finite-block generative models**: reweighting overcorrects (Test 3); maximum-
  entropy block projection is no better than raw/damped (Test 4);
- accumulating block anomalies does not reproduce the whole-word deficit, and
  bridge / parity residuals persist (Tests 1–2).

What *survives* as organizing structure: block features become more discriminative
as `L` grows, and a residual that is indexed by **bridge shape** and **parity**
remains after the block score is included.

## Open problems

- The discriminative signal in `B_L` increases with `L`, but no finite `L` tested
  closes the gap. Whether it closes at all is open.
- The persistent bridge/parity residual is *measured* but not *explained*.
- The focus state's extreme survival tail is **sparse** (some deciles have zero iid
  mass), so the strongest-looking effects are also the least sampled.
- These are split-sample, sampled diagnostics — **not** exact aggregation proofs.

---

## License and caveats

- **License:** code under MIT (see `LICENSE`); reports, figures, and the paper text
  under CC BY 4.0.
- The hard-coded Windows paths in the scripts are environment-specific and must be
  changed.
- The classifications (`A/B/C/D`) are **coarse self-diagnostics produced by the
  scripts**, with thresholds chosen by the authors; they are reported verbatim and
  should not be read as external benchmarks.
- Sampled AUC/RMSE figures carry sampling error that is not formally quantified here;
  treat small differences as suggestive, not decisive.
