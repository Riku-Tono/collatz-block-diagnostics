from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


SOURCE = Path(r"C:\Users\yauki\Documents\Codex\2026-06-25\new-chat\work\collatz_escape_word_deficit.py")
CACHE = Path(r"C:\Users\yauki\Documents\Codex\2026-06-25\new-chat-2\work\status_cache\odd_only_status_p24.bin")
CATEGORIES = ("1", "2", "3+")


def load_source():
    spec = importlib.util.spec_from_file_location("collatz_escape_word_deficit", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def k_cat(k: int) -> str:
    if k == 1:
        return "1"
    if k == 2:
        return "2"
    return "3+"


def cat_word(word: tuple[int, ...], depth: int) -> tuple[str, ...]:
    return tuple(k_cat(k) for k in word[:depth])


def fixed_prefix(module, n: int, depth: int) -> tuple[int, ...]:
    out = []
    cur = n
    for _ in range(depth):
        raw = 3 * cur + 1
        k = module.v2(raw)
        out.append(k)
        cur = raw >> k
    return tuple(out)


def summarize(
    sample: str,
    counts: Counter[tuple[str, ...]],
    iid_probability: dict[tuple[str, ...], float],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    total = sum(counts.values())
    support = list(itertools.product(CATEGORIES, repeat=4))
    rows = []
    tv = 0.0
    max_abs_z = -1.0
    max_z_word = None
    max_z = math.nan
    for word in support:
        p = counts[word] / total
        q = iid_probability.get(word, 0.0)
        delta = p - q
        tv += abs(delta)
        denom = math.sqrt(total * q * (1.0 - q)) if 0.0 < q < 1.0 else 0.0
        z = (counts[word] - total * q) / denom if denom else math.nan
        if math.isfinite(z) and abs(z) > max_abs_z:
            max_abs_z = abs(z)
            max_z_word = word
            max_z = z
        rows.append(
            {
                "sample": sample,
                "word": ",".join(word),
                "actual_count": counts[word],
                "actual_probability": p,
                "iid_probability": q,
                "deviation_actual_minus_iid": delta,
                "standardized_residual": z,
            }
        )
    rows.sort(key=lambda row: abs(float(row["deviation_actual_minus_iid"])), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["absolute_deviation_rank"] = rank
    summary = {
        "sample": sample,
        "sample_count": total,
        "TV_distance": 0.5 * tv,
        "max_absolute_standardized_residual": max_abs_z,
        "max_residual_word": ",".join(max_z_word) if max_z_word else "",
        "signed_standardized_residual_at_max": max_z,
        "top10_absolute_deviation_sum": sum(
            abs(float(row["deviation_actual_minus_iid"])) for row in rows[:10]
        ),
    }
    return summary, rows[:10]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--power", type=int, default=24)
    parser.add_argument("--h", type=int, default=2)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--iid-samples", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if (args.power, args.depth) != (24, 4):
        raise ValueError("This lightweight audit is fixed to power=24 and depth=4")

    module = load_source()
    expected = 1 << (args.power - 1)
    if CACHE.stat().st_size != expected:
        raise RuntimeError(f"cache size mismatch: {CACHE}")
    status = bytearray(CACHE.read_bytes())
    lo, hi, layer_total = module.layer_bounds(args.power, args.h)

    all_counts: Counter[tuple[str, ...]] = Counter()
    escape_counts: Counter[tuple[str, ...]] = Counter()
    escape_total = 0
    escape_too_short = 0
    for idx in range(lo >> 1, (hi >> 1) + 1):
        n = 2 * idx + 1
        all_counts[cat_word(fixed_prefix(module, n, args.depth), args.depth)] += 1
        if status[idx] == module.ESCAPE:
            escape_total += 1
            word = module.trace_escape(n, 1 << args.power)
            if len(word) >= args.depth:
                escape_counts[cat_word(word, args.depth)] += 1
            else:
                escape_too_short += 1

    # Original upstream iid sampler: the random walk is stopped at first passage
    # of the same h-dependent boundary and is reweighted to iid escape mass.
    rng = random.Random(args.seed)
    iid_counts, _budget, _iid_q, _iid_q_se = module.iid_escape_sample(
        args.h, args.iid_samples, (args.depth,), rng
    )
    iid_escape_cat: defaultdict[tuple[str, ...], float] = defaultdict(float)
    for word, weight in iid_counts[args.depth].items():
        if len(word) == args.depth and (not word or word[-1] != 0):
            iid_escape_cat[cat_word(word, args.depth)] += weight
    iid_escape_total = sum(iid_escape_cat.values())
    iid_escape_probability = {word: weight / iid_escape_total for word, weight in iid_escape_cat.items()}

    one_step = {"1": 0.5, "2": 0.25, "3+": 0.25}
    iid_all_probability = {
        word: math.prod(one_step[item] for item in word)
        for word in itertools.product(CATEGORIES, repeat=args.depth)
    }

    escape_summary, escape_top = summarize("ESCAPE", escape_counts, iid_escape_probability)
    all_summary, all_top = summarize("all_starting_values", all_counts, iid_all_probability)
    summaries = [escape_summary, all_summary]
    top_rows = escape_top + all_top

    if (
        float(escape_summary["TV_distance"]) > float(all_summary["TV_distance"])
        and float(escape_summary["max_absolute_standardized_residual"])
        > float(all_summary["max_absolute_standardized_residual"])
        and float(escape_summary["top10_absolute_deviation_sum"])
        > float(all_summary["top10_absolute_deviation_sum"])
    ):
        verdict = "escape_selection_supported"
    elif (
        float(all_summary["TV_distance"]) >= 0.5 * float(escape_summary["TV_distance"])
        and float(all_summary["max_absolute_standardized_residual"])
        >= 0.5 * float(escape_summary["max_absolute_standardized_residual"])
    ):
        verdict = "difference_survives_without_escape_filter"
    else:
        verdict = "unresolved"

    write_csv(args.out_dir / "escape_vs_all_depth4_summary.csv", summaries)
    write_csv(args.out_dir / "escape_vs_all_depth4_top10_words.csv", top_rows)

    def fmt_summary(row: dict[str, object]) -> str:
        return (
            f"| {row['sample']} | {int(row['sample_count']):,} | "
            f"{float(row['TV_distance']):.9f} | "
            f"{float(row['max_absolute_standardized_residual']):.6f} | "
            f"`{row['max_residual_word']}` | "
            f"{float(row['top10_absolute_deviation_sum']):.9f} |"
        )

    report = f"""# ESCAPE selection check for finite-block actual vs iid

## Fixed comparison

- power: `{args.power}`
- h: `{args.h}`
- depth: `{args.depth}`
- word alphabet: `1`, `2`, `3+` (the original finite-block `k_cat` definition)
- iid samples for the ESCAPE-conditioned side: `{args.iid_samples:,}`
- seed: `{args.seed}`

## Code-confirmed definitions

- `ESCAPE`: in `compute_status`, the status becomes `ESCAPE` when the odd Syracuse orbit first has `cur > 2^power`.
- `trace_escape`: it appends each `k=v2(3n+1)` while `cur <= 2^power`; it stops immediately after the update that makes `cur > 2^power`.
- iid condition: yes. `iid_escape_sample` stops its iid walk when `position > h-log2(y)`, i.e. first passage of the matched escape boundary, and reweights that stopped sample. Therefore the original iid side is also ESCAPE-conditioned; it is not an unconditional iid prefix sample.

For the finite-block comparison, only words with at least four valuations are eligible on the ESCAPE side, matching the original B4 requirement. ESCAPE paths shorter than four: `{escape_too_short:,}` of `{escape_total:,}`. The all-starting-values side records exactly four Syracuse valuations for every odd start in the same finite layer.

## Comparison table

| sample | eligible n | actual-iid TV | max absolute standardized residual | word at max | sum abs deviation, top 10 |
|---|---:|---:|---:|:---|---:|
{fmt_summary(escape_summary)}
{fmt_summary(all_summary)}

The standardized residual is `(observed - n*q) / sqrt(n*q*(1-q))` on the fixed 81-word support.

## Top 10 word deviations

See `escape_vs_all_depth4_top10_words.csv`. It contains, for each sample, actual count, actual probability, iid probability, signed deviation, and standardized residual for the ten largest absolute probability deviations.

## Verdict

`{verdict}`

This verdict says only whether the difference is stronger with the ESCAPE condition. It is not a causal claim.

## Source files used

- `{SOURCE}`
- `{CACHE}`
- `C:\\Users\\yauki\\Documents\\design\\Collatz\\log\\一時README\\Collatz-finite-block_README.md`
- `C:\\Users\\yauki\\Documents\\Codex\\2026-06-25\\new-chat\\outputs\\collatz_escape_prefix_metrics.csv`

## Command

```powershell
python work/compare_escape_selection_depth4.py --power 24 --h 2 --depth 4 --iid-samples 500000 --seed 20260625 --out-dir outputs
```
"""
    (args.out_dir / "escape_selection_depth4_report.md").write_text(report, encoding="utf-8")
    print(verdict)
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
