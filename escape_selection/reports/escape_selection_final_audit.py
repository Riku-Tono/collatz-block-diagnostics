from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


SOURCE_SCRIPT = Path(
    r"C:\Users\yauki\Documents\design\Collatz\collatz finite-block\py\2026-06-24_Collatz\python\collatz_escape_word_deficit.py"
)
SOURCE_PREFIX_METRICS = Path(
    r"C:\Users\yauki\Documents\design\Collatz\collatz finite-block\py\2026-06-24_Collatz\csv\collatz_escape_prefix_metrics.csv"
)
CACHE_PATHS = {
    24: Path(r"C:\Users\yauki\Documents\Codex\2026-06-25\new-chat-2\work\status_cache\odd_only_status_p24.bin"),
    25: Path(r"C:\Users\yauki\Documents\Codex\2026-06-25\new-chat\work\status_cache\odd_only_status_p25.bin"),
    26: Path(r"C:\Users\yauki\Documents\Codex\2026-06-25\new-chat-2\work\status_cache\odd_only_status_p26.bin"),
    27: Path(r"C:\Users\yauki\Documents\Codex\2026-06-25\new-chat\work\status_cache\odd_only_status_p27.bin"),
    28: Path(r"C:\Users\yauki\Documents\Codex\2026-06-25\new-chat-2\work\status_cache\odd_only_status_p28.bin"),
}

ESCAPE = 2
LOG2_3 = math.log2(3.0)
CATEGORY_LABELS = ("1", "2", "3+")
BLOCK_SIZE = 4
BLOCK_SUPPORT = 3**BLOCK_SIZE


def load_source_module():
    spec = importlib.util.spec_from_file_location("finite_block_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def block_label(code: int) -> str:
    digits = []
    for divisor in (27, 9, 3, 1):
        digit, code = divmod(code, divisor)
        digits.append(CATEGORY_LABELS[digit])
    return ",".join(digits)


def exact_iid_block_probs() -> np.ndarray:
    cat_p = np.array([0.5, 0.25, 0.25], dtype=np.float64)
    probs = np.zeros(BLOCK_SUPPORT, dtype=np.float64)
    for code in range(BLOCK_SUPPORT):
        remainder = code
        probability = 1.0
        for divisor in (27, 9, 3, 1):
            digit, remainder = divmod(remainder, divisor)
            probability *= cat_p[digit]
        probs[code] = probability
    if not np.isclose(probs.sum(), 1.0):
        raise AssertionError(probs.sum())
    return probs


def source_escape_counts(powers: list[int], hs: list[int]) -> dict[tuple[int, int], tuple[int, int]]:
    result: dict[tuple[int, int], tuple[int, int]] = {}
    with SOURCE_PREFIX_METRICS.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            power = int(row["power"])
            h = int(row["h"])
            if power not in powers or h not in hs or int(row["depth"]) != 4:
                continue
            result[(power, h)] = (int(row["layer_total"]), int(row["actual_escape_count"]))
    return result


def new_accumulator(lengths: list[int]) -> dict[int, dict[str, np.ndarray | int]]:
    return {
        length: {
            "trajectory_total": 0,
            "trajectory_eligible": 0,
            "position_counts": np.zeros((length, 3), dtype=np.int64),
            "block_counts": np.zeros(BLOCK_SUPPORT, dtype=np.int64),
        }
        for length in lengths
    }


def accumulate_actual(
    accumulator: dict[int, dict[str, np.ndarray | int]],
    cats: np.ndarray,
    first_reach_step: np.ndarray,
    group_mask: np.ndarray,
    lengths: list[int],
) -> None:
    for length in lengths:
        data = accumulator[length]
        eligible = group_mask & ((first_reach_step == 0) | (first_reach_step >= length))
        data["trajectory_total"] = int(data["trajectory_total"]) + int(group_mask.sum())
        data["trajectory_eligible"] = int(data["trajectory_eligible"]) + int(eligible.sum())
        if not eligible.any():
            continue

        position_counts = data["position_counts"]
        assert isinstance(position_counts, np.ndarray)
        for position in range(length):
            position_counts[position] += np.bincount(
                cats[eligible, position], minlength=3
            ).astype(np.int64)

        block_counts = data["block_counts"]
        assert isinstance(block_counts, np.ndarray)
        for position in range(length - BLOCK_SIZE + 1):
            code = (
                cats[:, position].astype(np.int16) * 27
                + cats[:, position + 1].astype(np.int16) * 9
                + cats[:, position + 2].astype(np.int16) * 3
                + cats[:, position + 3].astype(np.int16)
            )
            block_counts += np.bincount(code[eligible], minlength=BLOCK_SUPPORT).astype(
                np.int64
            )


def build_prefix_categories(indices: np.ndarray, max_length: int) -> tuple[np.ndarray, np.ndarray]:
    current = indices.astype(np.uint64) * np.uint64(2) + np.uint64(1)
    categories = np.empty((len(indices), max_length), dtype=np.uint8)
    first_reach_step = np.zeros(len(indices), dtype=np.uint8)
    for position in range(max_length):
        raw = current * np.uint64(3) + np.uint64(1)
        lowbit = raw & (~raw + np.uint64(1))
        k = np.log2(lowbit).astype(np.uint8)
        categories[:, position] = np.where(k == 1, 0, np.where(k == 2, 1, 2))
        current = np.right_shift(raw, k)
        newly_reached = (first_reach_step == 0) & (current == 1)
        first_reach_step[newly_reached] = position + 1
    return categories, first_reach_step


def aggregate_actual(
    powers: list[int],
    hs: list[int],
    lengths: list[int],
    chunk_size: int,
) -> dict[tuple[int, int, str], dict[int, dict[str, np.ndarray | int]]]:
    status = {
        power: np.memmap(CACHE_PATHS[power], mode="r", dtype=np.uint8)
        for power in powers
    }
    result: dict[
        tuple[int, int, str], dict[int, dict[str, np.ndarray | int]]
    ] = {}
    combinations_by_bits: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for power in powers:
        for h in hs:
            combinations_by_bits[power - h].append((power, h))
            result[(power, h, "escape_only")] = new_accumulator(lengths)
            result[(power, h, "all_starts")] = new_accumulator(lengths)

    max_length = max(lengths)
    for bit_length in sorted(combinations_by_bits):
        lo = (1 << (bit_length - 1)) + 1
        hi = (1 << bit_length) - 1
        start = lo >> 1
        end = (hi >> 1) + 1
        combinations = combinations_by_bits[bit_length]
        print(
            f"actual bit_length={bit_length}: starts={end-start:,}, combinations={combinations}",
            flush=True,
        )
        for chunk_start in range(start, end, chunk_size):
            chunk_end = min(end, chunk_start + chunk_size)
            indices = np.arange(chunk_start, chunk_end, dtype=np.uint64)
            cats, first_reach_step = build_prefix_categories(indices, max_length)
            all_mask = np.ones(len(indices), dtype=bool)

            # ALL depends only on the starting layer, so compute once and copy to
            # every (power, h) sharing that bit length after the chunk.
            all_accumulator = new_accumulator(lengths)
            accumulate_actual(
                all_accumulator, cats, first_reach_step, all_mask, lengths
            )
            for power, h in combinations:
                target_all = result[(power, h, "all_starts")]
                for length in lengths:
                    target_all[length]["trajectory_total"] = int(
                        target_all[length]["trajectory_total"]
                    ) + int(all_accumulator[length]["trajectory_total"])
                    target_all[length]["trajectory_eligible"] = int(
                        target_all[length]["trajectory_eligible"]
                    ) + int(all_accumulator[length]["trajectory_eligible"])
                    for key in ("position_counts", "block_counts"):
                        target = target_all[length][key]
                        source = all_accumulator[length][key]
                        assert isinstance(target, np.ndarray)
                        assert isinstance(source, np.ndarray)
                        target += source

                escape_mask = status[power][chunk_start:chunk_end] == ESCAPE
                accumulate_actual(
                    result[(power, h, "escape_only")],
                    cats,
                    first_reach_step,
                    escape_mask,
                    lengths,
                )
    return result


def tilted_k(rng: random.Random) -> int:
    k = 1
    while rng.random() < 0.25:
        k += 1
    return k


def base_iid_k(rng: random.Random) -> int:
    k = 1
    while rng.random() < 0.5:
        k += 1
    return k


def aggregate_iid_escape(
    hs: list[int], lengths: list[int], samples: int, seed: int
) -> dict[tuple[int, int], dict[str, np.ndarray | float]]:
    escape_rng = random.Random(seed)
    continuation_rng = random.Random(seed + 1)
    max_length = max(lengths)
    result: dict[tuple[int, int], dict[str, np.ndarray | float]] = {}
    for h in hs:
        print(f"iid escape h={h}: samples={samples:,}", flush=True)
        accumulators = {
            length: {
                "weight": 0.0,
                "position": np.zeros((length, 3), dtype=np.float64),
                "block": np.zeros(BLOCK_SUPPORT, dtype=np.float64),
                "half_position": np.zeros((2, length, 3), dtype=np.float64),
                "half_block": np.zeros((2, BLOCK_SUPPORT), dtype=np.float64),
                "half_weight": np.zeros(2, dtype=np.float64),
            }
            for length in lengths
        }
        for sample_index in range(samples):
            y = 0.5 + 0.5 * escape_rng.random()
            distance = h - math.log2(y)
            position = 0.0
            word: list[int] = []
            while position <= distance:
                k = tilted_k(escape_rng)
                word.append(k)
                position += LOG2_3 - k
            overshoot = position - distance
            weight = y * (2.0 ** (-overshoot))
            while len(word) < max_length:
                word.append(base_iid_k(continuation_rng))
            cats = [0 if k == 1 else 1 if k == 2 else 2 for k in word[:max_length]]
            half = sample_index & 1
            for length in lengths:
                acc = accumulators[length]
                acc["weight"] = float(acc["weight"]) + weight
                half_weight = acc["half_weight"]
                assert isinstance(half_weight, np.ndarray)
                half_weight[half] += weight
                position_counts = acc["position"]
                half_position = acc["half_position"]
                block_counts = acc["block"]
                half_block = acc["half_block"]
                assert isinstance(position_counts, np.ndarray)
                assert isinstance(half_position, np.ndarray)
                assert isinstance(block_counts, np.ndarray)
                assert isinstance(half_block, np.ndarray)
                for i in range(length):
                    position_counts[i, cats[i]] += weight
                    half_position[half, i, cats[i]] += weight
                for i in range(length - BLOCK_SIZE + 1):
                    code = cats[i] * 27 + cats[i + 1] * 9 + cats[i + 2] * 3 + cats[i + 3]
                    block_counts[code] += weight
                    half_block[half, code] += weight
        for length in lengths:
            result[(h, length)] = accumulators[length]
    return result


def distributions_from_actual(
    data: dict[str, np.ndarray | int], length: int
) -> tuple[np.ndarray, np.ndarray]:
    position_counts = data["position_counts"]
    block_counts = data["block_counts"]
    assert isinstance(position_counts, np.ndarray)
    assert isinstance(block_counts, np.ndarray)
    position_denominator = position_counts.sum(axis=1, keepdims=True)
    if np.any(position_denominator == 0) or block_counts.sum() == 0:
        raise ValueError("Empty actual distribution")
    return position_counts / position_denominator, block_counts / block_counts.sum()


def distributions_from_iid_escape(
    data: dict[str, np.ndarray | float], length: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    position_counts = data["position"]
    block_counts = data["block"]
    half_position = data["half_position"]
    half_block = data["half_block"]
    assert isinstance(position_counts, np.ndarray)
    assert isinstance(block_counts, np.ndarray)
    assert isinstance(half_position, np.ndarray)
    assert isinstance(half_block, np.ndarray)
    position_dist = position_counts / position_counts.sum(axis=1, keepdims=True)
    block_dist = block_counts / block_counts.sum()
    half_position_dist = half_position / half_position.sum(axis=2, keepdims=True)
    half_dist = half_block / half_block.sum(axis=1, keepdims=True)
    half_tv = 0.5 * np.abs(half_dist[0] - half_dist[1]).sum()
    half_position_tv_mean = float(
        (0.5 * np.abs(half_position_dist[0] - half_position_dist[1]).sum(axis=1)).mean()
    )
    return position_dist, block_dist, float(half_tv), half_position_tv_mean


def metric_row(
    power: int,
    h: int,
    length: int,
    group: str,
    actual_data: dict[str, np.ndarray | int],
    iid_position: np.ndarray,
    iid_block: np.ndarray,
    iid_mc_half_tv: float,
    iid_mc_half_position_tv_mean: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    actual_position, actual_block = distributions_from_actual(actual_data, length)
    position_tvs = 0.5 * np.abs(actual_position - iid_position).sum(axis=1)
    abs_block_difference = np.abs(actual_block - iid_block)
    block_tv = 0.5 * abs_block_difference.sum()
    order = np.argsort(abs_block_difference)[::-1]
    top5_share = (
        float(abs_block_difference[order[:5]].sum() / abs_block_difference.sum())
        if abs_block_difference.sum()
        else 0.0
    )
    total = int(actual_data["trajectory_total"])
    eligible = int(actual_data["trajectory_eligible"])
    row = {
        "power": power,
        "h": h,
        "start_bit_length": power - h,
        "L": length,
        "group": group,
        "trajectory_total": total,
        "trajectory_eligible": eligible,
        "trajectory_excluded_before_L": total - eligible,
        "eligible_share": eligible / total if total else float("nan"),
        "block4_tv": float(block_tv),
        "position_tv_mean": float(position_tvs.mean()),
        "position_tv_max": float(position_tvs.max()),
        "iid_mc_half_tv": iid_mc_half_tv,
        "iid_mc_half_position_tv_mean": iid_mc_half_position_tv_mean,
        "top5_block_abs_difference_share": top5_share,
        "largest_block_difference": block_label(int(order[0])),
        "largest_block_actual_probability": float(actual_block[order[0]]),
        "largest_block_iid_probability": float(iid_block[order[0]]),
        "largest_block_signed_difference": float(actual_block[order[0]] - iid_block[order[0]]),
    }
    detail = []
    for rank, code in enumerate(order[:10], 1):
        detail.append(
            {
                "power": power,
                "h": h,
                "L": length,
                "group": group,
                "rank": rank,
                "block": block_label(int(code)),
                "actual_probability": float(actual_block[code]),
                "iid_probability": float(iid_block[code]),
                "signed_difference": float(actual_block[code] - iid_block[code]),
                "absolute_difference": float(abs_block_difference[code]),
                "share_of_total_absolute_difference": float(
                    abs_block_difference[code] / abs_block_difference.sum()
                )
                if abs_block_difference.sum()
                else 0.0,
            }
        )
    return row, detail


def summarize(
    metric_rows: list[dict[str, object]], powers: list[int], hs: list[int], lengths: list[int]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    lookup = {
        (int(row["power"]), int(row["h"]), int(row["L"]), str(row["group"])): row
        for row in metric_rows
    }
    contrast_rows = []
    for power in powers:
        for h in hs:
            for length in lengths:
                escape = lookup[(power, h, length, "escape_only")]
                all_starts = lookup[(power, h, length, "all_starts")]
                a = float(escape["block4_tv"])
                b = float(all_starts["block4_tv"])
                contrast_rows.append(
                    {
                        "power": power,
                        "h": h,
                        "L": length,
                        "escape_only_block4_tv": a,
                        "all_starts_block4_tv": b,
                        "all_over_escape_tv_ratio": b / a if a else float("nan"),
                        "escape_minus_all_tv": a - b,
                        "escape_larger": int(a > b),
                    }
                )

    power_rows = []
    for power in powers:
        selected = [row for row in contrast_rows if int(row["power"]) == power]
        a_values = [float(row["escape_only_block4_tv"]) for row in selected]
        b_values = [float(row["all_starts_block4_tv"]) for row in selected]
        ratios = [float(row["all_over_escape_tv_ratio"]) for row in selected]
        power_rows.append(
            {
                "power": power,
                "cells": len(selected),
                "median_escape_only_block4_tv": float(np.median(a_values)),
                "median_all_starts_block4_tv": float(np.median(b_values)),
                "median_all_over_escape_tv_ratio": float(np.median(ratios)),
                "escape_larger_cells": sum(int(row["escape_larger"]) for row in selected),
                "escape_larger_share": sum(int(row["escape_larger"]) for row in selected)
                / len(selected),
            }
        )
    return contrast_rows, power_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--powers", nargs="+", type=int, default=[24, 25, 26, 27, 28])
    parser.add_argument("--hs", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--lengths", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--iid-samples", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"C:\Users\yauki\Documents\Codex\2026-07-14\new-chat\outputs\escape_selection_final_audit"),
    )
    args = parser.parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for power in args.powers:
        path = CACHE_PATHS[power]
        expected = 1 << (power - 1)
        if not path.exists() or path.stat().st_size != expected:
            raise FileNotFoundError(f"Invalid status cache for p={power}: {path}")

    source_module = load_source_module()
    if source_module.ESCAPE != ESCAPE:
        raise AssertionError("ESCAPE constant mismatch")

    actual = aggregate_actual(args.powers, args.hs, args.lengths, args.chunk_size)
    expected_counts = source_escape_counts(args.powers, args.hs)
    validation_rows = []
    for power in args.powers:
        for h in args.hs:
            layer_total, escape_count = expected_counts[(power, h)]
            observed_all = int(actual[(power, h, "all_starts")][args.lengths[0]]["trajectory_total"])
            observed_escape = int(
                actual[(power, h, "escape_only")][args.lengths[0]]["trajectory_total"]
            )
            validation_rows.append(
                {
                    "power": power,
                    "h": h,
                    "source_layer_total": layer_total,
                    "observed_layer_total": observed_all,
                    "source_escape_count": escape_count,
                    "observed_escape_count": observed_escape,
                    "validation": "PASS"
                    if layer_total == observed_all and escape_count == observed_escape
                    else "FAIL",
                }
            )
    if any(row["validation"] != "PASS" for row in validation_rows):
        raise AssertionError("Source count validation failed")

    iid_escape = aggregate_iid_escape(args.hs, args.lengths, args.iid_samples, args.seed)
    iid_block_unconditional = exact_iid_block_probs()
    iid_position_unconditional = {
        length: np.tile(np.array([0.5, 0.25, 0.25]), (length, 1))
        for length in args.lengths
    }

    metric_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for power in args.powers:
        for h in args.hs:
            for length in args.lengths:
                (
                    iid_position_escape,
                    iid_block_escape,
                    half_tv,
                    half_position_tv,
                ) = distributions_from_iid_escape(iid_escape[(h, length)], length)
                row, detail = metric_row(
                    power,
                    h,
                    length,
                    "escape_only",
                    actual[(power, h, "escape_only")][length],
                    iid_position_escape,
                    iid_block_escape,
                    half_tv,
                    half_position_tv,
                )
                metric_rows.append(row)
                detail_rows.extend(detail)
                row, detail = metric_row(
                    power,
                    h,
                    length,
                    "all_starts",
                    actual[(power, h, "all_starts")][length],
                    iid_position_unconditional[length],
                    iid_block_unconditional,
                    0.0,
                    0.0,
                )
                metric_rows.append(row)
                detail_rows.extend(detail)

    contrast_rows, power_rows = summarize(
        metric_rows, args.powers, args.hs, args.lengths
    )
    write_csv(args.out_dir / "definition_and_count_validation.csv", validation_rows)
    write_csv(args.out_dir / "actual_iid_fixed_prefix_metrics.csv", metric_rows)
    write_csv(args.out_dir / "escape_vs_all_contrast.csv", contrast_rows)
    write_csv(args.out_dir / "power_stability.csv", power_rows)
    write_csv(args.out_dir / "top_block_differences.csv", detail_rows)

    run_summary = {
        "powers": args.powers,
        "hs": args.hs,
        "lengths": args.lengths,
        "iid_samples_per_h": args.iid_samples,
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "source_script": str(SOURCE_SCRIPT),
        "source_prefix_metrics": str(SOURCE_PREFIX_METRICS),
        "status_caches": {str(power): str(CACHE_PATHS[power]) for power in args.powers},
        "elapsed_seconds": time.time() - started,
    }
    (args.out_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
