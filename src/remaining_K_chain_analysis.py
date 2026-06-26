from __future__ import annotations

import importlib.util
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(r"C:\Users\yauki\Documents\design\Collatz")
SRC = BASE / "py" / "py" / "collatz_escape_word_deficit.py"
OUT = Path("outputs")

POWERS = [24, 25, 26, 27, 28]
HS = [2, 3, 4, 5, 6]
IID_SAMPLES_PER_H = 160_000
ACTUAL_SAMPLE_PER_PH = 20_000
SEED = 20260625
LOG2_3 = math.log2(3.0)
Q_LOW = -1.5
Q_HIGH = -0.25
FOCUS_STATES = [
    "late_growth|deep_32_63|even",
    "late_growth|deep_32_63|odd",
    "late_growth|exhaustion_0_31|odd",
]
BIN_LABELS = ["0-1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-95", "96-127", "128-191", "192+"]
CACHE_DIRS = [
    Path(r"C:\Users\yauki\Documents\Codex\2026-06-25\new-chat-3\work\status_cache"),
    Path(r"C:\Users\yauki\Documents\Codex\2026-06-25\new-chat-2\work\status_cache"),
    Path(r"C:\Users\yauki\Documents\Codex\2026-06-25\new-chat\work\status_cache"),
]


def load_source():
    spec = importlib.util.spec_from_file_location("collatz_escape_word_deficit", SRC)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SRC}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_status(power: int) -> bytearray:
    name = f"odd_only_status_p{power}.bin"
    expected = 1 << (power - 1)
    for directory in CACHE_DIRS:
        path = directory / name
        if path.exists() and path.stat().st_size == expected:
            return bytearray(path.read_bytes())
    raise FileNotFoundError(f"missing cached status for p={power}")


def evenly_spaced(items: list[int], limit: int) -> list[int]:
    if len(items) <= limit:
        return items
    if limit <= 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (limit - 1))] for i in range(limit)]


def tilted_k(rng: random.Random) -> int:
    k = 1
    while rng.random() < 0.25:
        k += 1
    return k


def sample_iid(module, h: int, rng: random.Random) -> list[tuple[tuple[int, ...], float]]:
    out = []
    for _ in range(IID_SAMPLES_PER_H):
        y = 0.5 + 0.5 * rng.random()
        distance = h - math.log2(y)
        position = 0.0
        word: list[int] = []
        while position <= distance:
            k = module.tilted_k(rng) if hasattr(module, "tilted_k") else tilted_k(rng)
            word.append(k)
            position += LOG2_3 - k
        overshoot = position - distance
        weight = (2.0 ** (-h)) * y * (2.0 ** (-overshoot)) / IID_SAMPLES_PER_H
        out.append((tuple(word), weight))
    return out


def weighted_tau_cuts(words: list[tuple[tuple[int, ...], float]]) -> tuple[int, int]:
    by_tau: Counter[int] = Counter()
    for word, weight in words:
        by_tau[len(word)] += weight
    total = sum(by_tau.values())
    cuts = []
    acc = 0.0
    targets = [total / 3, 2 * total / 3]
    for tau, weight in sorted(by_tau.items()):
        acc += weight
        while targets and acc >= targets[0]:
            cuts.append(tau)
            targets.pop(0)
    while len(cuts) < 2:
        cuts.append(max(by_tau))
    return cuts[0], cuts[1]


def path_xs(word: tuple[int, ...]) -> list[float]:
    xs = [0.0]
    cur = 0.0
    for k in word:
        cur += LOG2_3 - k
        xs.append(cur)
    return xs


def interp(values: list[float], u: float) -> float:
    tau = len(values) - 1
    pos = u * tau
    lo = int(math.floor(pos))
    hi = min(tau, lo + 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def z25_feature(word: tuple[int, ...]) -> float:
    xs = path_xs(word)
    final = xs[-1]
    return interp(xs, 0.25) - 0.25 * final


def cluster_from_z(z: float) -> str:
    if z <= Q_LOW:
        return "late_growth"
    if z >= Q_HIGH:
        return "early_growth"
    return "balanced"


def xk_window(x_k: int) -> str | None:
    if 0 <= x_k < 32:
        return "exhaustion_0_31"
    if 32 <= x_k < 64:
        return "deep_32_63"
    if 64 <= x_k < 96:
        return "tail_64_95"
    return None


def state_info(word: tuple[int, ...], power: int, h: int) -> tuple[str, str, str, str] | None:
    x_k = sum(word) - (power - h)
    window = xk_window(x_k)
    if window is None:
        return None
    bridge = cluster_from_z(z25_feature(word))
    parity = "even" if power % 2 == 0 else "odd"
    return f"{bridge}|{window}|{parity}", bridge, window, parity


def remaining_k_bin(value: int) -> tuple[int, str]:
    bounds = [0, 2, 4, 8, 16, 32, 64, 96, 128, 192]
    for i, lo in enumerate(bounds):
        hi = bounds[i + 1] if i + 1 < len(bounds) else math.inf
        if lo <= value < hi:
            return i, BIN_LABELS[i]
    return 0, "<0"


def add_count(counts, table: str, scope: str, key: str, source: str, weight: float) -> None:
    counts[(table, scope, key, source)] += weight


def add_word(counts, source: str, word: tuple[int, ...], power: int, h: int, weight: float) -> None:
    info = state_info(word, power, h)
    if info is None:
        return
    state, _bridge, _window, _parity = info
    scopes = ["ALL", state]
    total_k = sum(word)
    prefix_k = 0
    for k in word:
        rem = total_k - prefix_k
        next_rem = rem - k
        from_idx, from_bin = remaining_k_bin(rem)
        to_idx, to_bin = remaining_k_bin(next_rem)
        trans = f"{from_bin} -> {to_bin}"
        for scope in scopes:
            add_count(counts, "bin_mass", scope, from_bin, source, weight)
            add_count(counts, "transition", scope, trans, source, weight)
            add_count(counts, f"transition_from_{from_bin}", scope, trans, source, weight)
            if to_idx < from_idx:
                add_count(counts, "downstream_transition", scope, trans, source, weight)
        prefix_k += k


def rows_from_counts(counts) -> pd.DataFrame:
    keys = sorted({(table, scope, key) for table, scope, key, _source in counts})
    rows = []
    l1 = defaultdict(float)
    for table, scope, key in keys:
        actual = counts[(table, scope, key, "actual")]
        iid = counts[(table, scope, key, "iid")]
        delta = actual - iid
        if table == "bin_mass":
            order = BIN_LABELS.index(key) if key in BIN_LABELS else -1
        elif " -> " in key:
            order = BIN_LABELS.index(key.split(" -> ")[0]) if key.split(" -> ")[0] in BIN_LABELS else -1
        else:
            order = -1
        rows.append(
            {
                "table": table,
                "scope": scope,
                "key": key,
                "bin_order": order,
                "actual_mass": actual,
                "iid_mass": iid,
                "delta": delta,
                "ratio": actual / iid if iid else math.nan,
                "abs_delta": abs(delta),
            }
        )
        l1[(table, scope)] += abs(delta)
    for row in rows:
        denom = l1[(row["table"], row["scope"])]
        row["L1_share"] = row["abs_delta"] / denom if denom else math.nan
    return pd.DataFrame(rows)


def conditional_transition_rows(df: pd.DataFrame) -> pd.DataFrame:
    mass = df[df["table"] == "bin_mass"][["scope", "key", "actual_mass", "iid_mass"]].rename(
        columns={"key": "from_bin", "actual_mass": "from_actual_mass", "iid_mass": "from_iid_mass"}
    )
    trans = df[df["table"] == "transition"].copy()
    trans["from_bin"] = trans["key"].str.split(" -> ").str[0]
    out = trans.merge(mass, on=["scope", "from_bin"], how="left")
    out["conditional_actual"] = out["actual_mass"] / out["from_actual_mass"]
    out["conditional_iid"] = out["iid_mass"] / out["from_iid_mass"]
    out["conditional_delta"] = out["conditional_actual"] - out["conditional_iid"]
    out["abs_conditional_delta"] = out["conditional_delta"].abs()
    return out


def mass_vs_conditional(mass: pd.DataFrame, cond: pd.DataFrame) -> pd.DataFrame:
    top_cond = cond.sort_values("abs_conditional_delta", ascending=False).groupby(["scope", "from_bin"], as_index=False).first()
    out = mass.rename(columns={"key": "bin"})[
        ["scope", "bin", "bin_order", "actual_mass", "iid_mass", "delta", "ratio", "abs_delta", "L1_share"]
    ].merge(
        top_cond[
            [
                "scope",
                "from_bin",
                "key",
                "conditional_actual",
                "conditional_iid",
                "conditional_delta",
                "abs_conditional_delta",
            ]
        ].rename(columns={"from_bin": "bin", "key": "largest_conditional_transition"}),
        on=["scope", "bin"],
        how="left",
    )
    out["mass_delta_sign"] = np.sign(out["delta"])
    out["conditional_delta_sign"] = np.sign(out["conditional_delta"])
    out["sign_relation"] = np.where(
        out["conditional_delta"].isna(),
        "none",
        np.where(out["mass_delta_sign"] == out["conditional_delta_sign"], "same", "opposite"),
    )
    return out.sort_values(["scope", "bin_order"])


def line_svg(path: Path, rows: pd.DataFrame, y_col: str, title: str) -> None:
    sub = rows[rows["scope"] == "ALL"].sort_values("bin_order")
    width, height = 900, 430
    left, right, top, bottom = 75, 35, 50, 70
    vals = [float(v) for v in sub[y_col]]
    if y_col == "ratio":
        ymin, ymax = min(vals + [1.0]) * 0.95, max(vals + [1.0]) * 1.05
    elif y_col == "L1_share":
        ymin, ymax = 0.0, max(vals) * 1.15 if vals else 1.0
    else:
        m = max(abs(v) for v in vals) if vals else 1.0
        ymin, ymax = -m * 1.15, m * 1.15
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="16">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
    ]
    if ymin < 0 < ymax:
        y0 = top + (ymax - 0) * (height - top - bottom) / (ymax - ymin)
        parts.append(f'<line x1="{left}" y1="{y0:.1f}" x2="{width-right}" y2="{y0:.1f}" stroke="#9ca3af" stroke-dasharray="4 4"/>')
    pts = []
    for idx, r in enumerate(sub.itertuples(index=False)):
        x = left + idx * (width - left - right) / max(1, len(sub) - 1)
        y = top + (ymax - float(getattr(r, y_col))) * (height - top - bottom) / (ymax - ymin if ymax != ymin else 1.0)
        pts.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="Arial" font-size="10">{r.key}</text>')
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#111827" stroke-width="2.4"/>')
    for pt in pts:
        x, y = pt.split(",")
        parts.append(f'<circle cx="{x}" cy="{y}" r="3.3" fill="#111827"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def transition_heatmap_svg(path: Path, cond: pd.DataFrame) -> None:
    sub = cond[cond["scope"] == "ALL"].copy()
    sub["to_bin"] = sub["key"].str.split(" -> ").str[1]
    pivot = sub.pivot(index="from_bin", columns="to_bin", values="delta").reindex(index=BIN_LABELS, columns=BIN_LABELS).fillna(0.0)
    vals = pivot.to_numpy()
    max_abs = float(np.nanmax(np.abs(vals))) or 1.0
    cell = 58
    left, top = 95, 72
    width = left + cell * len(BIN_LABELS) + 30
    height = top + cell * len(BIN_LABELS) + 40

    def color(v: float) -> str:
        t = max(-1.0, min(1.0, v / max_abs))
        if t >= 0:
            return f"rgb(185,{int(220 - 90*t)},{int(220 - 90*t)})"
        return f"rgb({int(220 + 25*t)},{int(235 + 8*t)},254)"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="16">remaining_K transition delta heatmap</text>',
    ]
    for j, col in enumerate(BIN_LABELS):
        x = left + j * cell + cell / 2
        parts.append(f'<text x="{x:.1f}" y="{top-10}" text-anchor="middle" font-family="Arial" font-size="9">{col}</text>')
    for i, row in enumerate(BIN_LABELS):
        y = top + i * cell
        parts.append(f'<text x="{left-8}" y="{y+cell/2+4:.1f}" text-anchor="end" font-family="Arial" font-size="9">{row}</text>')
        for j, col in enumerate(BIN_LABELS):
            x = left + j * cell
            v = float(pivot.loc[row, col])
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color(v)}" stroke="white"/>')
            if abs(v) > max_abs * 0.04:
                parts.append(f'<text x="{x+cell/2:.1f}" y="{y+cell/2+4:.1f}" text-anchor="middle" font-family="Arial" font-size="8">{v:+.2g}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def focus_heatmap_svg(path: Path, rows: pd.DataFrame) -> None:
    sub = rows[rows["scope"].isin(FOCUS_STATES)].copy()
    pivot = sub.pivot(index="scope", columns="key", values="delta").reindex(index=FOCUS_STATES, columns=BIN_LABELS).fillna(0.0)
    vals = pivot.to_numpy()
    max_abs = float(np.nanmax(np.abs(vals))) or 1.0
    cell_w, cell_h = 72, 45
    left, top = 260, 66
    width = left + cell_w * len(BIN_LABELS) + 30
    height = top + cell_h * len(FOCUS_STATES) + 42

    def color(v: float) -> str:
        t = max(-1.0, min(1.0, v / max_abs))
        if t >= 0:
            return f"rgb(185,{int(220 - 90*t)},{int(220 - 90*t)})"
        return f"rgb({int(220 + 25*t)},{int(235 + 8*t)},254)"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="16">focus state x remaining_K bin delta</text>',
    ]
    for j, col in enumerate(BIN_LABELS):
        x = left + j * cell_w + cell_w / 2
        parts.append(f'<text x="{x:.1f}" y="{top-12}" text-anchor="middle" font-family="Arial" font-size="9">{col}</text>')
    for i, row in enumerate(FOCUS_STATES):
        y = top + i * cell_h
        parts.append(f'<text x="{left-8}" y="{y+cell_h/2+4:.1f}" text-anchor="end" font-family="Arial" font-size="10">{row}</text>')
        for j, col in enumerate(BIN_LABELS):
            x = left + j * cell_w
            v = float(pivot.loc[row, col])
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color(v)}" stroke="white"/>')
            parts.append(f'<text x="{x+cell_w/2:.1f}" y="{y+cell_h/2+4:.1f}" text-anchor="middle" font-family="Arial" font-size="8">{v:+.2g}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def build_report(mass: pd.DataFrame, cond: pd.DataFrame, mvc: pd.DataFrame, focus: pd.DataFrame) -> str:
    all_mass = mass[mass["scope"] == "ALL"].sort_values("bin_order")
    max_delta = all_mass.sort_values("abs_delta", ascending=False).iloc[0]
    min_ratio = all_mass[all_mass["iid_mass"] > 0].sort_values("ratio").iloc[0]
    upstream = all_mass[all_mass["key"].isin(["32-63", "64-95", "96-127", "128-191", "192+"])]
    opp = mvc[(mvc["scope"] == "ALL") & (mvc["sign_relation"] == "opposite")].sort_values("abs_conditional_delta", ascending=False)
    focus_lines = []
    for state in FOCUS_STATES:
        sub = focus[focus["scope"] == state].sort_values("abs_delta", ascending=False)
        if not sub.empty:
            r = sub.iloc[0]
            focus_lines.append(f"- `{state}`: max |delta| bin `{r['key']}`, delta `{r['delta']:+.6g}`, ratio `{r['ratio']:.3f}`.")
    upstream_lines = [
        f"- `{r.key}`: delta `{r.delta:+.6g}`, ratio `{r.ratio:.3f}`."
        for r in upstream.itertuples(index=False)
    ]
    opp_lines = [
        f"- `{r.bin}`: mass delta `{r.delta:+.6g}`, largest conditional `{r.largest_conditional_transition}` conditional_delta `{r.conditional_delta:+.6g}`."
        for r in opp.head(5).itertuples(index=False)
    ]
    lines = [
        "# remaining_K chain report",
        "",
        "Scope: same long-word, final-state-defined universe as the previous state/boundary-layer analyses.",
        "",
        "## Mass placement",
        "",
        f"Largest ALL |delta| bin is `{max_delta['key']}` with delta `{max_delta['delta']:+.6g}`, ratio `{max_delta['ratio']:.3f}`, L1 share `{max_delta['L1_share']:.2%}`.",
        f"Lowest ALL ratio bin is `{min_ratio['key']}` with ratio `{min_ratio['ratio']:.3f}`, delta `{min_ratio['delta']:+.6g}`.",
        "",
        "Upstream-side bins:",
        "",
        *upstream_lines,
        "",
        "## Mass delta vs conditional transition delta",
        "",
        "Bins where the mass delta and the largest conditional transition delta have opposite signs:",
        "",
        *(opp_lines if opp_lines else ["- none"]),
        "",
        "## Focus states",
        "",
        *focus_lines,
        "",
        "## Short reading",
        "",
        "1. The 32-63 and 64-95 deficits should be read against the whole remaining_K chain, not as isolated boundary events.",
        "2. If the negative mass delta keeps extending into larger bins, the observation is a mass-placement shift along remaining_K rather than a single stopping point.",
        "3. Conditional transition deltas are separated from bin mass deltas here; opposite signs mark places where being in a bin and moving out of it tell different stories.",
        "4. Focus-state heatmaps show whether the global chain keeps drifting upstream or whether a state-conditioned band holds the contrast.",
        "5. If no state-conditioned stopping band appears, this is a good place to stop boundary digging and reconnect the result to the state-level/prefix-level maps.",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    module = load_source()
    rng = random.Random(SEED)
    counts = defaultdict(float)
    tau_cuts: dict[int, tuple[int, int]] = {}

    print("sampling iid", flush=True)
    for h in HS:
        sampled = sample_iid(module, h, rng)
        tau_cuts[h] = weighted_tau_cuts(sampled)
        for word, weight in sampled:
            if len(word) <= tau_cuts[h][1]:
                continue
            for power in POWERS:
                add_word(counts, "iid", word, power, h, weight)

    print("sampling actual", flush=True)
    for power in POWERS:
        status = load_status(power)
        for h in HS:
            _cut1, cut2 = tau_cuts[h]
            lo, hi, total = module.layer_bounds(power, h)
            escape_indices = [idx for idx in range(lo >> 1, (hi >> 1) + 1) if status[idx] == module.ESCAPE]
            chosen = evenly_spaced(escape_indices, ACTUAL_SAMPLE_PER_PH)
            sample_weight = len(escape_indices) / (len(chosen) * total) if chosen else 0.0
            for idx in chosen:
                word = module.trace_escape(2 * idx + 1, 1 << power)
                if len(word) <= cut2:
                    continue
                add_word(counts, "actual", word, power, h, sample_weight)
        del status

    df = rows_from_counts(counts)
    mass = df[df["table"] == "bin_mass"].sort_values(["scope", "bin_order"])
    trans = df[df["table"] == "transition"].sort_values(["scope", "bin_order", "key"])
    cond = conditional_transition_rows(df).sort_values(["scope", "bin_order", "key"])
    mvc = mass_vs_conditional(mass, cond)
    focus = mass[mass["scope"].isin(FOCUS_STATES)].copy()
    focus_max = focus.sort_values("abs_delta", ascending=False).groupby("scope", as_index=False).first()

    mass.to_csv(OUT / "remaining_K_chain_mass.csv", index=False, encoding="utf-8-sig")
    trans.to_csv(OUT / "remaining_K_chain_transition.csv", index=False, encoding="utf-8-sig")
    cond.to_csv(OUT / "remaining_K_chain_conditional_transition.csv", index=False, encoding="utf-8-sig")
    mvc.to_csv(OUT / "remaining_K_chain_mass_vs_conditional.csv", index=False, encoding="utf-8-sig")
    focus.to_csv(OUT / "remaining_K_chain_by_focus_state.csv", index=False, encoding="utf-8-sig")
    focus_max.to_csv(OUT / "remaining_K_chain_focus_state_max_bin.csv", index=False, encoding="utf-8-sig")

    line_svg(OUT / "remaining_K_chain_delta.svg", mass, "delta", "remaining_K bin delta")
    line_svg(OUT / "remaining_K_chain_ratio.svg", mass, "ratio", "remaining_K bin actual/iid ratio")
    line_svg(OUT / "remaining_K_chain_l1_share.svg", mass, "L1_share", "remaining_K bin L1 share")
    transition_heatmap_svg(OUT / "remaining_K_chain_transition_heatmap.svg", cond)
    focus_heatmap_svg(OUT / "remaining_K_chain_focus_state_heatmap.svg", focus)

    (OUT / "remaining_K_chain_report.md").write_text(build_report(mass, cond, mvc, focus), encoding="utf-8")
    print(OUT / "remaining_K_chain_report.md", flush=True)


if __name__ == "__main__":
    main()
