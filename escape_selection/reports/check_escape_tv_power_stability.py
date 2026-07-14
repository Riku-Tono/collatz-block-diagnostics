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
EXISTING_CACHE = Path(r"C:\Users\yauki\Documents\Codex\2026-06-25\new-chat-2\work\status_cache")
CATEGORIES = ("1", "2", "3+")


def load_source():
    spec = importlib.util.spec_from_file_location("collatz_escape_word_deficit", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def k_cat(k: int) -> str:
    return "1" if k == 1 else "2" if k == 2 else "3+"


def tv(counts: Counter[tuple[str, ...]], q: dict[tuple[str, ...], float]) -> float:
    total = sum(counts.values())
    support = itertools.product(CATEGORIES, repeat=4)
    return 0.5 * sum(abs(counts[word] / total - q[word]) for word in support)


def load_or_compute_status(module, power: int, local_cache: Path) -> tuple[bytearray, Path]:
    name = f"odd_only_status_p{power}.bin"
    expected = 1 << (power - 1)
    for directory in (local_cache, EXISTING_CACHE):
        path = directory / name
        if path.exists() and path.stat().st_size == expected:
            print(f"loading {path}", flush=True)
            return bytearray(path.read_bytes()), path
    print(f"computing status p={power}", flush=True)
    status = module.compute_status(power)
    local_cache.mkdir(parents=True, exist_ok=True)
    path = local_cache / name
    path.write_bytes(status)
    return status, path


def matched_iid_probability(module, h: int, depth: int, samples: int, seed: int):
    rng = random.Random(seed)
    weighted, _budget, _q, _q_se = module.iid_escape_sample(h, samples, (depth,), rng)
    by_cat: defaultdict[tuple[str, ...], float] = defaultdict(float)
    for word, weight in weighted[depth].items():
        if len(word) == depth and word[-1] != 0:
            by_cat[tuple(k_cat(k) for k in word)] += weight
    total = sum(by_cat.values())
    return {
        word: by_cat[word] / total
        for word in itertools.product(CATEGORIES, repeat=depth)
    }


def count_power(module, power: int, h: int, depth: int, status: bytearray):
    lo, hi, layer_total = module.layer_bounds(power, h)
    n_max = 1 << power
    all_counts: Counter[tuple[str, ...]] = Counter()
    escape_counts: Counter[tuple[str, ...]] = Counter()
    escape_total = 0
    escape_short = 0
    for idx in range(lo >> 1, (hi >> 1) + 1):
        cur = 2 * idx + 1
        cats = []
        first_cross_step = 0
        for step in range(1, depth + 1):
            raw = 3 * cur + 1
            k = module.v2(raw)
            cats.append(k_cat(k))
            cur = raw >> k
            if not first_cross_step and cur > n_max:
                first_cross_step = step
        word = tuple(cats)
        all_counts[word] += 1
        if status[idx] == module.ESCAPE:
            escape_total += 1
            if first_cross_step and first_cross_step < depth:
                escape_short += 1
            else:
                escape_counts[word] += 1
    return layer_total, escape_total, escape_short, all_counts, escape_counts


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--powers", nargs="+", type=int, default=[25, 26, 27, 28])
    parser.add_argument("--h", type=int, default=2)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--iid-samples", type=int, default=500_000)
    parser.add_argument("--iid-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--cache-dir", type=Path, default=Path("work/status_cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.h != 2 or args.depth != 4:
        raise ValueError("This stability check is fixed to h=2 and depth=4")

    module = load_source()
    seeds = [args.seed + i for i in range(args.iid_repeats)]
    print(f"building {len(seeds)} matched iid references", flush=True)
    iid_refs = [
        matched_iid_probability(module, args.h, args.depth, args.iid_samples, seed)
        for seed in seeds
    ]
    iid_all = {
        word: math.prod({"1": 0.5, "2": 0.25, "3+": 0.25}[item] for item in word)
        for word in itertools.product(CATEGORIES, repeat=args.depth)
    }

    rows = []
    cache_paths = []
    for power in args.powers:
        status, cache_path = load_or_compute_status(module, power, args.cache_dir)
        cache_paths.append(cache_path)
        layer_total, escape_total, escape_short, all_counts, escape_counts = count_power(
            module, power, args.h, args.depth, status
        )
        all_tv = tv(all_counts, iid_all)
        escape_tvs = [tv(escape_counts, ref) for ref in iid_refs]
        row = {
            "power": power,
            "h": args.h,
            "depth": args.depth,
            "all_starting_values": layer_total,
            "escape_sample_count": escape_total,
            "escape_depth4_eligible_count": sum(escape_counts.values()),
            "escape_shorter_than_depth4": escape_short,
            "TV_all": all_tv,
            "TV_escape_mean": sum(escape_tvs) / len(escape_tvs),
            "TV_escape_min": min(escape_tvs),
            "TV_escape_max": max(escape_tvs),
            "TV_escape_minus_TV_all_at_min": min(escape_tvs) - all_tv,
            "TV_escape_gt_TV_all_all_repeats": all(value > all_tv for value in escape_tvs),
        }
        rows.append(row)
        print(row, flush=True)
        del status, all_counts, escape_counts

    stable = all(bool(row["TV_escape_gt_TV_all_all_repeats"]) for row in rows)
    conclusion = (
        "ESCAPE条件付けと差の増幅には関連がある。原因・機構は未同定"
        if stable
        else "方向は全powerで安定しなかったため未解決"
    )
    write_csv(args.out_dir / "escape_tv_power_stability.csv", rows)

    table_rows = "\n".join(
        f"| {r['power']} | {int(r['escape_sample_count']):,} | "
        f"{float(r['TV_all']):.9f} | {float(r['TV_escape_mean']):.9f} | "
        f"{float(r['TV_escape_min']):.9f}–{float(r['TV_escape_max']):.9f} | "
        f"{'yes' if r['TV_escape_gt_TV_all_all_repeats'] else 'no'} |"
        for r in rows
    )
    command = (
        "python work/check_escape_tv_power_stability.py --powers 25 26 27 28 "
        "--h 2 --depth 4 --iid-samples 500000 --iid-repeats 5 "
        "--seed 20260625 --cache-dir work/status_cache --out-dir outputs"
    )
    report = f"""# ESCAPE-conditioned TV power stability check

Fixed throughout: `h=2`, `depth=4`, word alphabet `1 / 2 / 3+`. Each power uses the complete finite layer. The ESCAPE iid reference uses the same matched first-passage sampler as the previous check.

| power | ESCAPE samples | TV_all | TV_escape mean | TV_escape range ({args.iid_repeats} iid runs) | TV_escape > TV_all in every run |
|---:|---:|---:|---:|---:|:---:|
{table_rows}

The range is only the minimum–maximum across matched-iid runs with seeds `{seeds[0]}..{seeds[-1]}`. Each run used `{args.iid_samples:,}` samples. No new classification or additional statistic was introduced.

## Direction-only judgment

`TV_escape > TV_all` is {'stable for every tested power and every iid repeat' if stable else 'not stable across all tested powers and repeats'}.

**{conclusion}。**

This closes the present discrepancy check at the requested descriptive level.

## Source files

- `{SOURCE}`
- status caches: {', '.join(f'`{path}`' for path in cache_paths)}
- `C:\\Users\\yauki\\Documents\\design\\Collatz\\log\\一時README\\Collatz-finite-block_README.md`

## Command

```powershell
{command}
```
"""
    (args.out_dir / "escape_tv_power_stability_report.md").write_text(report, encoding="utf-8")
    print(conclusion, flush=True)


if __name__ == "__main__":
    main()
