from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, deque
from pathlib import Path


def emit_k(
    k: int,
    k_counts: Counter[int],
    transitions: Counter[tuple[int, int]],
    blocks: dict[int, Counter[tuple[int, ...]]],
    history: deque[int],
) -> None:
    k_counts[k] += 1
    if history:
        transitions[(history[-1], k)] += 1
    history.append(k)
    for depth, counter in blocks.items():
        if len(history) >= depth:
            counter[tuple(list(history)[-depth:])] += 1


def mutual_information(transitions: Counter[tuple[int, int]]) -> float:
    total = sum(transitions.values())
    if not total:
        return float("nan")
    left: Counter[int] = Counter()
    right: Counter[int] = Counter()
    for (a, b), count in transitions.items():
        left[a] += count
        right[b] += count
    mi = 0.0
    for (a, b), count in transitions.items():
        p_ab = count / total
        mi += p_ab * math.log2(p_ab / ((left[a] / total) * (right[b] / total)))
    return mi


def analyze_file(path: Path):
    with path.open("r", encoding="ascii", newline="") as handle:
        start_bits = handle.readline().strip()
        dynamics = handle.readline().strip()
        summary = handle.readline().strip().split()
        extra = handle.readline()

    if extra:
        raise ValueError(f"{path.name}: expected exactly three lines")
    if len(summary) != 6:
        raise ValueError(f"{path.name}: malformed summary line")
    if not start_bits or start_bits[0] != "1" or start_bits[-1] != "1":
        raise ValueError(f"{path.name}: starting integer is not an odd binary integer")
    if not dynamics or dynamics[0] != "I" or set(dynamics) - {"I", "O"}:
        raise ValueError(f"{path.name}: malformed I/O sequence")

    declared_bit_length, peak_bit_length, declared_steps, declared_i, declared_o = map(
        int, summary[:5]
    )
    declared_ratio = float(summary[5])
    observed_i = dynamics.count("I")
    observed_o = dynamics.count("O")
    if (
        len(start_bits) != declared_bit_length
        or len(dynamics) != declared_steps
        or observed_i != declared_i
        or observed_o != declared_o
    ):
        raise ValueError(f"{path.name}: summary does not match the two data lines")

    k_counts: Counter[int] = Counter()
    transitions: Counter[tuple[int, int]] = Counter()
    blocks = {3: Counter(), 4: Counter()}
    history: deque[int] = deque(maxlen=4)
    current_k: int | None = None
    for symbol in dynamics:
        if symbol == "I":
            if current_k is not None:
                emit_k(current_k, k_counts, transitions, blocks, history)
            current_k = 1
        else:
            if current_k is None:
                raise ValueError(f"{path.name}: O appeared before the first I")
            current_k += 1
    if current_k is not None:
        emit_k(current_k, k_counts, transitions, blocks, history)

    k_total = sum(k_counts.values())
    if k_total != observed_i or sum(k * n for k, n in k_counts.items()) != len(dynamics):
        raise ValueError(f"{path.name}: I/O to k conversion failed validation")

    max_k = max(k_counts)
    tv_geometric = 0.5 * (
        sum(abs(k_counts[k] / k_total - 2.0 ** (-k)) for k in range(1, max_k + 1))
        + 2.0 ** (-max_k)
    )
    k1 = k_counts[1] / k_total
    triple_total = sum(blocks[3].values())
    triple_111 = blocks[3][(1, 1, 1)] / triple_total if triple_total else float("nan")
    mean_k = len(dynamics) / k_total
    row = {
        "file": path.name,
        "start_bit_length": declared_bit_length,
        "peak_bit_length": peak_bit_length,
        "io_steps": len(dynamics),
        "odd_steps_I": observed_i,
        "even_steps_O": observed_o,
        "O_over_I": observed_o / observed_i,
        "declared_O_over_I": declared_ratio,
        "mean_k": mean_k,
        "mean_approx_log2_multiplier_per_odd_step": math.log2(3.0) - mean_k,
        "share_k_eq_1": k1,
        "share_k_triple_111": triple_111,
        "max_k": max_k,
        "tv_distance_from_geometric_half_power": tv_geometric,
        "adjacent_k_mutual_information_bits": mutual_information(transitions),
        "validation": "PASS",
    }
    return row, k_counts, transitions, blocks


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dynamics_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        args.dynamics_dir.glob("txpo46dynamics*"),
        key=lambda p: int(p.name.removeprefix("txpo46dynamics")),
    )
    if not files:
        raise SystemExit("No txpo46dynamics* files found")

    summaries: list[dict] = []
    frequencies: list[dict] = []
    transition_rows: list[dict] = []
    block_rows: list[dict] = []
    for path in files:
        row, k_counts, transitions, blocks = analyze_file(path)
        summaries.append(row)
        k_total = sum(k_counts.values())
        for k in sorted(k_counts):
            frequencies.append(
                {
                    "file": path.name,
                    "start_bit_length": row["start_bit_length"],
                    "k": k,
                    "count": k_counts[k],
                    "empirical_probability": k_counts[k] / k_total,
                    "geometric_probability_2^-k": 2.0 ** (-k),
                    "difference": k_counts[k] / k_total - 2.0 ** (-k),
                }
            )
        transition_total = sum(transitions.values())
        for (source_k, target_k), count in sorted(transitions.items()):
            transition_rows.append(
                {
                    "file": path.name,
                    "start_bit_length": row["start_bit_length"],
                    "source_k": source_k,
                    "target_k": target_k,
                    "count": count,
                    "joint_probability": count / transition_total,
                }
            )
        for depth, counts in blocks.items():
            total = sum(counts.values())
            for word, count in counts.most_common(200):
                block_rows.append(
                    {
                        "file": path.name,
                        "start_bit_length": row["start_bit_length"],
                        "depth": depth,
                        "word": ",".join(map(str, word)),
                        "count": count,
                        "empirical_probability": count / total,
                        "iid_geometric_probability": 2.0 ** (-sum(word)),
                        "difference": count / total - 2.0 ** (-sum(word)),
                    }
                )

    write_csv(args.output_dir / "wei_ren_txpo_summary.csv", summaries, list(summaries[0]))
    write_csv(args.output_dir / "wei_ren_k_frequency.csv", frequencies, list(frequencies[0]))
    write_csv(args.output_dir / "wei_ren_k_transition.csv", transition_rows, list(transition_rows[0]))
    write_csv(args.output_dir / "wei_ren_top_k_blocks.csv", block_rows, list(block_rows[0]))


if __name__ == "__main__":
    main()
