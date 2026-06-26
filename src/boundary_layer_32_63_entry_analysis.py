from __future__ import annotations

import csv
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
LAYER_LO = 32
LAYER_HI = 63
FOCUS_STATES = [
    "late_growth|deep_32_63|even",
    "late_growth|deep_32_63|odd",
    "late_growth|exhaustion_0_31|odd",
]
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


def final_state_for_word(word: tuple[int, ...], power: int, h: int) -> tuple[str, str, str, str] | None:
    x_k = sum(word) - (power - h)
    window = xk_window(x_k)
    if window is None:
        return None
    bridge = cluster_from_z(z25_feature(word))
    parity = "even" if power % 2 == 0 else "odd"
    state = f"{bridge}|{window}|{parity}"
    return state, bridge, window, parity


def k_cat(k: int) -> str:
    if k == 1:
        return "1"
    if k == 2:
        return "2"
    return "3+"


def remaining_k_bin(value: int) -> str:
    if value < 0:
        return "<0"
    if value < 2:
        return "0-1"
    if value < 4:
        return "2-3"
    if value < 8:
        return "4-7"
    if value < 16:
        return "8-15"
    if value < 32:
        return "16-31"
    if value < 64:
        return "32-63"
    if value < 96:
        return "64-95"
    return "96+"


def steps_bin(value: int) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value < 4:
        return "2-3"
    if value < 8:
        return "4-7"
    if value < 16:
        return "8-15"
    if value < 32:
        return "16-31"
    if value < 64:
        return "32-63"
    return "64+"


def u_bin(prefix_len: int, tau: int) -> str:
    idx = min(9, max(0, int(math.floor((prefix_len / tau) * 10))))
    return f"{idx / 10:.1f}-{(idx + 1) / 10:.1f}"


def future_sequence(word: tuple[int, ...], start_index: int, depth: int) -> str:
    cats = [k_cat(k) for k in word[start_index : min(len(word), start_index + depth)]]
    return ",".join(cats) if cats else "END"


def add_count(counts: defaultdict[tuple[str, str, str, str], float], table: str, scope: str, key: str, source: str, weight: float) -> None:
    counts[(table, scope, key, source)] += weight


def add_value(values: defaultdict[tuple[str, str, str], list[tuple[float, float]]], metric: str, scope: str, source: str, value: float, weight: float) -> None:
    values[(metric, scope, source)].append((value, weight))


def find_entry(word: tuple[int, ...]) -> dict[str, object] | None:
    total_k = sum(word)
    if LAYER_LO <= total_k <= LAYER_HI:
        return {
            "entry_kind": "start_in_layer",
            "prefix_len": 0,
            "from_k": total_k,
            "to_k": total_k,
            "entry_raw_k": None,
            "entry_k_cat": "START",
            "next_index": 0,
        }
    prefix_k = 0
    prev_remaining = total_k
    for i, k in enumerate(word, 1):
        prefix_k += k
        remaining = total_k - prefix_k
        if LAYER_LO <= remaining <= LAYER_HI:
            return {
                "entry_kind": "inflow_step",
                "prefix_len": i,
                "from_k": prev_remaining,
                "to_k": remaining,
                "entry_raw_k": k,
                "entry_k_cat": k_cat(k),
                "next_index": i,
            }
        prev_remaining = remaining
    return None


def add_word(
    counts: defaultdict[tuple[str, str, str, str], float],
    values: defaultdict[tuple[str, str, str], list[tuple[float, float]]],
    source: str,
    word: tuple[int, ...],
    power: int,
    h: int,
    weight: float,
) -> None:
    state_info = final_state_for_word(word, power, h)
    if state_info is None:
        return
    state, bridge, window, parity = state_info
    entry = find_entry(word)
    if entry is None:
        return
    tau = len(word)
    prefix_len = int(entry["prefix_len"])
    from_bin = "START_IN_32-63" if entry["entry_kind"] == "start_in_layer" else remaining_k_bin(int(entry["from_k"]))
    to_bin = remaining_k_bin(int(entry["to_k"]))
    transition = f"{from_bin} -> {to_bin}"
    entry_raw = entry["entry_raw_k"]
    entry_raw_label = "START" if entry_raw is None else str(entry_raw)
    next_index = int(entry["next_index"])
    remaining_steps = tau - prefix_len
    remaining_k = int(entry["to_k"])
    scopes = ["ALL", state]
    for scope in scopes:
        add_count(counts, "entry_mass", scope, "entry", source, weight)
        add_count(counts, "entry_kind", scope, str(entry["entry_kind"]), source, weight)
        add_count(counts, "entry_from_remaining_K_bin", scope, from_bin, source, weight)
        add_count(counts, "entry_to_remaining_K_bin", scope, to_bin, source, weight)
        add_count(counts, "entry_transition", scope, transition, source, weight)
        add_count(counts, "entry_step_k_cat", scope, str(entry["entry_k_cat"]), source, weight)
        add_count(counts, "entry_step_raw_k", scope, entry_raw_label, source, weight)
        add_count(counts, "entry_u_bin", scope, u_bin(prefix_len, tau), source, weight)
        add_count(counts, "entry_prefix_length_bin", scope, steps_bin(prefix_len), source, weight)
        add_count(counts, "entry_state", scope, state, source, weight)
        add_count(counts, "bridge_cluster", scope, bridge, source, weight)
        add_count(counts, "x_K_window", scope, window, source, weight)
        add_count(counts, "parity", scope, parity, source, weight)
        add_count(counts, "entry_exit_remaining_steps_bin", scope, steps_bin(remaining_steps), source, weight)
        add_count(counts, "entry_exit_remaining_K_bin", scope, remaining_k_bin(remaining_k), source, weight)
        add_count(counts, "future_1_k_cat", scope, future_sequence(word, next_index, 1), source, weight)
        add_count(counts, "future_2_k_cat", scope, future_sequence(word, next_index, 2), source, weight)
        add_count(counts, "future_3_k_cat", scope, future_sequence(word, next_index, 3), source, weight)
        add_value(values, "entry_prefix_length", scope, source, prefix_len, weight)
        add_value(values, "entry_raw_k", scope, source, float("nan") if entry_raw is None else float(entry_raw), weight)
        add_value(values, "entry_exit_remaining_steps", scope, source, remaining_steps, weight)
        add_value(values, "entry_exit_remaining_K", scope, source, remaining_k, weight)


def rows_from_counts(counts: defaultdict[tuple[str, str, str, str], float]) -> pd.DataFrame:
    keys = sorted({(table, scope, key) for table, scope, key, _source in counts})
    tmp = []
    l1 = defaultdict(float)
    for table, scope, key in keys:
        actual = counts[(table, scope, key, "actual")]
        iid = counts[(table, scope, key, "iid")]
        delta = actual - iid
        row = {
            "table": table,
            "scope": scope,
            "key": key,
            "actual_mass": actual,
            "iid_mass": iid,
            "delta": delta,
            "ratio": actual / iid if iid > 0 else math.nan,
            "abs_delta": abs(delta),
        }
        tmp.append(row)
        l1[(table, scope)] += abs(delta)
    for row in tmp:
        denom = l1[(str(row["table"]), str(row["scope"]))]
        row["contribution_to_L1_delta"] = row["abs_delta"] / denom if denom else math.nan
    return pd.DataFrame(tmp)


def finite_vals(vals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(v, w) for v, w in vals if math.isfinite(v)]


def weighted_mean(vals: list[tuple[float, float]]) -> float:
    vals = finite_vals(vals)
    total = sum(w for _v, w in vals)
    return sum(v * w for v, w in vals) / total if total else math.nan


def weighted_quantile(vals: list[tuple[float, float]], q: float) -> float:
    vals = sorted(finite_vals(vals))
    if not vals:
        return math.nan
    total = sum(w for _v, w in vals)
    acc = 0.0
    for v, w in vals:
        acc += w
        if acc >= q * total:
            return v
    return vals[-1][0]


def rows_from_values(values: defaultdict[tuple[str, str, str], list[tuple[float, float]]]) -> pd.DataFrame:
    rows = []
    for metric, scope in sorted({(m, s) for m, s, _src in values}):
        actual = values[(metric, scope, "actual")]
        iid = values[(metric, scope, "iid")]
        rows.append(
            {
                "metric": metric,
                "scope": scope,
                "actual_mean": weighted_mean(actual),
                "iid_mean": weighted_mean(iid),
                "mean_delta": weighted_mean(actual) - weighted_mean(iid),
                "actual_median": weighted_quantile(actual, 0.5),
                "iid_median": weighted_quantile(iid, 0.5),
            }
        )
    return pd.DataFrame(rows)


def conditional_rows(cat: pd.DataFrame) -> pd.DataFrame:
    base = cat[cat["table"] == "entry_mass"][["scope", "actual_mass", "iid_mass"]].rename(
        columns={"actual_mass": "entry_actual_mass", "iid_mass": "entry_iid_mass"}
    )
    sub = cat[cat["table"] != "entry_mass"].merge(base, on="scope", how="left")
    sub["conditional_actual"] = sub["actual_mass"] / sub["entry_actual_mass"]
    sub["conditional_iid"] = sub["iid_mass"] / sub["entry_iid_mass"]
    sub["conditional_delta"] = sub["conditional_actual"] - sub["conditional_iid"]
    sub["abs_conditional_delta"] = sub["conditional_delta"].abs()
    return sub


def bar_svg(path: Path, rows: pd.DataFrame, table: str, title: str) -> None:
    data = rows[(rows["table"] == table) & (rows["scope"].isin(["ALL", *FOCUS_STATES]))].sort_values("abs_delta", ascending=False).head(18)
    width, height = 980, 560
    left, right, top, bottom = 290, 35, 50, 45
    max_abs = float(data["abs_delta"].max()) or 1.0
    bar_h = (height - top - bottom) / max(1, len(data))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="16">{title}</text>',
    ]
    zero_x = left + (width - left - right) / 2
    parts.append(f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{height-bottom}" stroke="#6b7280"/>')
    for i, r in enumerate(data.itertuples(index=False)):
        y = top + i * bar_h + 3
        delta = float(r.delta)
        x2 = zero_x + delta / max_abs * ((width - left - right) / 2)
        x = min(zero_x, x2)
        w = abs(x2 - zero_x)
        color = "#b91c1c" if delta > 0 else "#2563eb"
        label = f"{r.scope}|{r.key}"
        if len(label) > 46:
            label = label[:43] + "..."
        parts.append(f'<text x="{left-8}" y="{y+bar_h/2+4:.1f}" text-anchor="end" font-family="Arial" font-size="9">{label}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bar_h-6:.1f}" fill="{color}" opacity="0.82"/>')
        parts.append(f'<text x="{x2 + (5 if delta >= 0 else -5):.1f}" y="{y+bar_h/2+4:.1f}" text-anchor="{"start" if delta >= 0 else "end"}" font-family="Arial" font-size="10">{delta:+.3g}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def heatmap_svg(path: Path, rows: pd.DataFrame, table: str, title: str) -> None:
    sub = rows[(rows["table"] == table) & (rows["scope"].isin(FOCUS_STATES))]
    pivot = sub.pivot(index="scope", columns="key", values="delta").fillna(0.0)
    vals = pivot.to_numpy()
    max_abs = float(np.nanmax(np.abs(vals))) or 1.0
    cell_w, cell_h = 104, 44
    left, top = 260, 62
    width = left + cell_w * len(pivot.columns) + 28
    height = top + cell_h * len(pivot.index) + 44

    def color(v: float) -> str:
        t = max(-1.0, min(1.0, v / max_abs))
        if t >= 0:
            return f"rgb(185,{int(220 - 90*t)},{int(220 - 90*t)})"
        return f"rgb({int(220 + 25*t)},{int(235 + 8*t)},254)"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="16">{title}</text>',
    ]
    for j, col in enumerate(pivot.columns):
        x = left + j * cell_w + cell_w / 2
        parts.append(f'<text x="{x:.1f}" y="{top-12}" text-anchor="middle" font-family="Arial" font-size="10">{col}</text>')
    for i, idx in enumerate(pivot.index):
        y = top + i * cell_h
        parts.append(f'<text x="{left-8}" y="{y+cell_h/2+4:.1f}" text-anchor="end" font-family="Arial" font-size="10">{idx}</text>')
        for j, col in enumerate(pivot.columns):
            x = left + j * cell_w
            v = float(pivot.loc[idx, col])
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color(v)}" stroke="white"/>')
            parts.append(f'<text x="{x+cell_w/2:.1f}" y="{y+cell_h/2+4:.1f}" text-anchor="middle" font-family="Arial" font-size="9">{v:+.3g}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def build_report(cat: pd.DataFrame, cond: pd.DataFrame, metrics: pd.DataFrame) -> str:
    def top(table: str, scope: str = "ALL") -> pd.Series:
        return cat[(cat["table"] == table) & (cat["scope"] == scope)].sort_values("abs_delta", ascending=False).iloc[0]

    entry = top("entry_mass")
    trans = top("entry_transition")
    kind = top("entry_kind")
    kcat = top("entry_step_k_cat")
    ubin = top("entry_u_bin")
    future = top("future_3_k_cat")
    cond_k = cond[(cond["table"] == "entry_step_k_cat") & (cond["scope"] == "ALL")].sort_values("abs_conditional_delta", ascending=False).iloc[0]
    cond_future = cond[(cond["table"] == "future_3_k_cat") & (cond["scope"] == "ALL")].sort_values("abs_conditional_delta", ascending=False).iloc[0]
    all_metrics = metrics[metrics["scope"] == "ALL"].set_index("metric")
    focus_lines = []
    for state in FOCUS_STATES:
        t = top("entry_transition", state)
        k = top("entry_step_k_cat", state)
        u = top("entry_u_bin", state)
        future_state = top("future_3_k_cat", state)
        focus_lines.append(
            f"- `{state}`: entry path `{t['key']}` delta `{t['delta']:+.6g}`, entry k `{k['key']}` delta `{k['delta']:+.6g}`, entry u `{u['key']}` delta `{u['delta']:+.6g}`, future3 `{future_state['key']}` delta `{future_state['delta']:+.6g}`."
        )
    lines = [
        "# Remaining_K 32-63 first-entry analysis",
        "",
        "Scope: first entry into the `remaining_K=32-63` boundary layer for each escape word.",
        "",
        "## Entry mass",
        "",
        f"Total first-entry mass is actual `{entry['actual_mass']:.6g}` vs iid `{entry['iid_mass']:.6g}`, delta `{entry['delta']:+.6g}`, ratio `{entry['ratio']:.3f}`.",
        f"The largest entry-kind delta is `{kind['key']}` with delta `{kind['delta']:+.6g}`.",
        f"The largest entry transition delta is `{trans['key']}` with delta `{trans['delta']:+.6g}`, ratio `{trans['ratio']:.3f}`.",
        "",
        "## Entry path",
        "",
        f"The largest entry-step k_cat delta is `{kcat['key']}` with delta `{kcat['delta']:+.6g}`.",
        f"Conditioned on entry, the largest k_cat conditional shift is `{cond_k['key']}` with conditional_delta `{cond_k['conditional_delta']:+.6g}`.",
        f"The largest entry_u_bin delta is `{ubin['key']}` with delta `{ubin['delta']:+.6g}`.",
        "",
        "## Short future after entry",
        "",
        f"The largest 3-step future mass delta is `{future['key']}` with delta `{future['delta']:+.6g}`.",
        f"Conditioned on entry, the largest 3-step future conditional shift is `{cond_future['key']}` with conditional_delta `{cond_future['conditional_delta']:+.6g}`.",
        f"Mean entry prefix length actual/iid: `{all_metrics.loc['entry_prefix_length', 'actual_mean']:.3f}` / `{all_metrics.loc['entry_prefix_length', 'iid_mean']:.3f}`.",
        f"Mean exit remaining steps after entry actual/iid: `{all_metrics.loc['entry_exit_remaining_steps', 'actual_mean']:.3f}` / `{all_metrics.loc['entry_exit_remaining_steps', 'iid_mean']:.3f}`.",
        "",
        "## Focus states",
        "",
        *focus_lines,
        "",
        "## Short reading",
        "",
        "1. The 32-63 layer deficit is strongly visible at first-entry mass, so entry-side thinning is a plausible description of the layer deficit.",
        "2. The thinnest entry path is not only external inflow; `START_IN_32-63` also carries a large deficit because many final states begin inside this layer.",
        "3. Focus states do not share one entry signature: even deep late-growth is entry-deficit, exhaustion odd is entry-excess, and odd deep late-growth is mixed/small.",
        "4. The first 1-3 steps after entry show differences, but their conditional shifts are smaller than the entry-mass thinning.",
        "5. Next look should be the 64-95 predecessor band before sequence overlay, because entry paths from `64-95` and `96+` separate external inflow from start-in-layer cases.",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    module = load_source()
    rng = random.Random(SEED)
    counts: defaultdict[tuple[str, str, str, str], float] = defaultdict(float)
    values: defaultdict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    tau_cuts: dict[int, tuple[int, int]] = {}

    print("sampling iid", flush=True)
    for h in HS:
        sampled = sample_iid(module, h, rng)
        tau_cuts[h] = weighted_tau_cuts(sampled)
        for word, weight in sampled:
            if len(word) <= tau_cuts[h][1]:
                continue
            for power in POWERS:
                add_word(counts, values, "iid", word, power, h, weight)

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
                add_word(counts, values, "actual", word, power, h, sample_weight)
        del status

    cat = rows_from_counts(counts)
    metrics = rows_from_values(values)
    cond = conditional_rows(cat)
    cat.to_csv(OUT / "boundary_layer_32_63_first_entry.csv", index=False, encoding="utf-8-sig")
    cat[cat["table"].isin(["entry_transition", "entry_from_remaining_K_bin", "entry_to_remaining_K_bin", "entry_kind"])].to_csv(
        OUT / "boundary_layer_32_63_entry_transition_delta.csv", index=False, encoding="utf-8-sig"
    )
    cat[cat["scope"].isin(FOCUS_STATES)].to_csv(OUT / "boundary_layer_32_63_entry_by_state.csv", index=False, encoding="utf-8-sig")
    cat[cat["table"].str.startswith("future_") | cat["table"].str.startswith("entry_exit_")].to_csv(
        OUT / "boundary_layer_32_63_entry_future.csv", index=False, encoding="utf-8-sig"
    )
    metrics.to_csv(OUT / "boundary_layer_32_63_entry_metric_summary.csv", index=False, encoding="utf-8-sig")
    cond.to_csv(OUT / "boundary_layer_32_63_entry_conditional_delta.csv", index=False, encoding="utf-8-sig")

    for table, name, title in [
        ("entry_from_remaining_K_bin", "entry_from_remaining_K", "Entry-from remaining_K delta"),
        ("entry_step_k_cat", "entry_step_k_cat", "Entry-step k_cat delta"),
        ("entry_u_bin", "entry_u_bin", "Entry u-bin delta"),
        ("future_3_k_cat", "future_3_k_cat", "Entry future 3-step delta"),
    ]:
        bar_svg(OUT / f"boundary_layer_32_63_{name}.svg", cat, table, title)
        heatmap_svg(OUT / f"boundary_layer_32_63_{name}_heatmap.svg", cat, table, title + " heatmap")

    (OUT / "boundary_layer_32_63_entry_report.md").write_text(build_report(cat, cond, metrics), encoding="utf-8")
    print(OUT / "boundary_layer_32_63_entry_report.md", flush=True)


if __name__ == "__main__":
    main()
