from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np


LOG2_3 = math.log2(3.0)
LENGTHS = (4, 8, 16)


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


def run_h(h: int, samples: int, seed: int) -> dict[str, object]:
    escape_rng = random.Random(seed)
    continuation_rng = random.Random(seed + 1)
    max_length = max(LENGTHS)
    position_counts = {length: np.zeros((length, 3)) for length in LENGTHS}
    block_counts = {length: np.zeros(81) for length in LENGTHS}
    proposal_k_draws = 0
    continuation_k_draws = 0
    sum_weight = 0.0
    sum_weight_sq = 0.0
    started = time.perf_counter()
    for _ in range(samples):
        y = 0.5 + 0.5 * escape_rng.random()
        distance = h - math.log2(y)
        position = 0.0
        word: list[int] = []
        while position <= distance:
            k = tilted_k(escape_rng)
            proposal_k_draws += 1
            word.append(k)
            position += LOG2_3 - k
        overshoot = position - distance
        weight = y * (2.0 ** (-overshoot))
        sum_weight += weight
        sum_weight_sq += weight * weight
        while len(word) < max_length:
            word.append(base_iid_k(continuation_rng))
            continuation_k_draws += 1
        cats = [0 if k == 1 else 1 if k == 2 else 2 for k in word[:max_length]]
        for length in LENGTHS:
            for i in range(length):
                position_counts[length][i, cats[i]] += weight
            for i in range(length - 3):
                code = cats[i] * 27 + cats[i + 1] * 9 + cats[i + 2] * 3 + cats[i + 3]
                block_counts[length][code] += weight
    seconds = time.perf_counter() - started
    ess = sum_weight * sum_weight / sum_weight_sq
    return {
        "h": h,
        "candidates_generated": samples,
        "candidates_accepted": samples,
        "acceptance_rate": 1.0,
        "proposal_k_draws": proposal_k_draws,
        "mean_proposal_k_draws_per_candidate": proposal_k_draws / samples,
        "continuation_k_draws_to_L16": continuation_k_draws,
        "mean_continuation_k_draws_per_candidate": continuation_k_draws / samples,
        "importance_weight_ess": ess,
        "ess_share": ess / samples,
        "seconds": seconds,
        "candidates_per_second": samples / seconds,
        "implementation": "pure_python_loop_with_numpy_accumulator_arrays",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [run_h(h, args.samples, args.seed + h * 1000) for h in range(2, 7)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
