from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


OUT_DIR = Path("outputs")
MIN_IID = 1e-6


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT_DIR / name).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    with (OUT_DIR / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def weighted_cv_across_groups(rows: list[dict[str, str]], axis: str, group: str, ratio_field: str, iid_field: str) -> float:
    by_axis: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if float(row[iid_field]) < MIN_IID or row[ratio_field] == "":
            continue
        by_axis[row[axis]].append(float(row[ratio_field]))
    vals = []
    weights = []
    for _axis, ratios in by_axis.items():
        if len(ratios) < 2:
            continue
        m = mean(ratios)
        if m:
            vals.append(pstdev(ratios) / abs(m))
            weights.append(len(ratios))
    return sum(v * w for v, w in zip(vals, weights)) / sum(weights) if weights else float("nan")


def range_for(rows: list[dict[str, str]], ratio_field: str, iid_field: str) -> tuple[int, float, float, float]:
    vals = [float(r[ratio_field]) for r in rows if r[ratio_field] != "" and float(r[iid_field]) >= MIN_IID]
    return len(vals), min(vals), max(vals), max(vals) - min(vals)


def monotone(rows: list[dict[str, str]], axis: str, ratio_field: str, iid_field: str, decreasing: bool) -> float:
    pts = sorted(
        (float(r[axis]), float(r[ratio_field]))
        for r in rows
        if r[ratio_field] != "" and float(r[iid_field]) >= MIN_IID
    )
    if len(pts) < 3:
        return float("nan")
    good = 0
    total = 0
    for (_x0, y0), (_x1, y1) in zip(pts, pts[1:]):
        good += int(y1 <= y0 if decreasing else y1 >= y0)
        total += 1
    return good / total


def add(rows_out: list[dict[str, object]], label: str, rows: list[dict[str, str]], axis: str, ratio: str, iid: str, decreasing: bool, group: str | None = "h") -> None:
    n, mn, mx, rg = range_for(rows, ratio, iid)
    rows_out.append(
        {
            "axis": label,
            "stable_bin_count": n,
            "ratio_min": mn,
            "ratio_max": mx,
            "ratio_range": rg,
            "monotonicity_score": monotone(rows, axis, ratio, iid, decreasing),
            "collapse_cv_across_h": weighted_cv_across_groups(rows, axis, group, ratio, iid) if group else "",
        }
    )


def main() -> None:
    out: list[dict[str, object]] = []
    add(out, "drift_delta_point_by_h", read_csv("collatz_tail_by_drift_bins_by_h.csv"), "drift_bin", "survival_ratio", "iid_mass", False)
    add(out, "drift_delta_tail_by_h", read_csv("collatz_tail_by_drift_tail_by_h.csv"), "drift_delta_cut", "tail_survival_ratio", "iid_tail_mass", False)
    add(out, "mean_k_point_by_h", read_csv("collatz_tail_by_mean_k_bins_by_h.csv"), "mean_k_bin", "survival_ratio", "iid_mass", False)
    add(out, "mean_k_tail_by_h", read_csv("collatz_tail_by_mean_k_tail_by_h.csv"), "mean_k_cut", "tail_survival_ratio", "iid_tail_mass", False)
    add(out, "tau_bin_tail", read_csv("collatz_tail_by_tau_collapse.csv"), "relative_K_cut", "tail_survival_ratio", "iid_tail_mass", True, "tau_bin")
    add(out, "xK_tail_by_h", read_csv("collatz_survival_tail_collapse_by_h.csv"), "relative_K_cut", "tail_survival_ratio", "iid_tail_mass", True)
    write_csv("collatz_axis_comparison_refined.csv", out)
    print(OUT_DIR / "collatz_axis_comparison_refined.csv")


if __name__ == "__main__":
    main()
