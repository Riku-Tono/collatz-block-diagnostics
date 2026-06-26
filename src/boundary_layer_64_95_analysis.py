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
LAYER_LO = 64
LAYER_HI = 95
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
    return [items[round(i * (len(items) - 1) / (limit - 1))] for i in range(limit)]


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


def tilted_k(rng: random.Random) -> int:
    k = 1
    while rng.random() < 0.25:
        k += 1
    return k


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


def k_cat(k: int) -> str:
    if k == 1:
        return "1"
    if k == 2:
        return "2"
    return "3+"


def remaining_k_bin(value: int) -> str:
    if value < 0:
        return "<0"
    if value < 16:
        return "0-15"
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


def future_sequence(word: tuple[int, ...], start: int, depth: int) -> str:
    cats = [k_cat(k) for k in word[start : min(len(word), start + depth)]]
    return ",".join(cats) if cats else "END"


def add_count(counts, table: str, scope: str, key: str, source: str, weight: float) -> None:
    counts[(table, scope, key, source)] += weight


def add_value(values, metric: str, scope: str, source: str, value: float, weight: float) -> None:
    values[(metric, scope, source)].append((value, weight))


def previous_prefix_cat(word: tuple[int, ...], i: int) -> str:
    if i <= 0:
        return "START"
    return k_cat(word[i - 1])


def add_word(counts, values, source: str, word: tuple[int, ...], power: int, h: int, weight: float) -> None:
    info = state_info(word, power, h)
    if info is None:
        return
    state, bridge, window, parity = info
    scopes = ["ALL", state]
    total_k = sum(word)
    tau = len(word)
    prefix_k = 0
    for i in range(tau):
        remaining_k = total_k - prefix_k
        if LAYER_LO <= remaining_k <= LAYER_HI:
            next_k = word[i]
            next_remaining_k = remaining_k - next_k
            transition = f"64-95 -> {remaining_k_bin(next_remaining_k)}"
            next_k_label = k_cat(next_k)
            raw_tail_3 = "raw_k>=3" if next_k >= 3 else "raw_k<3"
            raw_tail_4 = "raw_k>=4" if next_k >= 4 else "raw_k<4"
            for scope in scopes:
                add_count(counts, "layer_mass", scope, "remaining_K_64_95", source, weight)
                add_count(counts, "bridge_cluster", scope, bridge, source, weight)
                add_count(counts, "x_K_window", scope, window, source, weight)
                add_count(counts, "parity", scope, parity, source, weight)
                add_count(counts, "exit_transition", scope, transition, source, weight)
                add_count(counts, "next_remaining_K_bin", scope, remaining_k_bin(next_remaining_k), source, weight)
                add_count(counts, "next_k_cat", scope, next_k_label, source, weight)
                add_count(counts, "raw_k_tail_ge3", scope, raw_tail_3, source, weight)
                add_count(counts, "raw_k_tail_ge4", scope, raw_tail_4, source, weight)
                add_count(counts, "raw_k", scope, str(next_k if next_k <= 8 else "9+"), source, weight)
                add_value(values, "next_raw_k", scope, source, float(next_k), weight)
            if remaining_k_bin(next_remaining_k) == "32-63":
                for scope in scopes:
                    add_count(counts, "to_32_63_mass", scope, "64-95_to_32-63", source, weight)
                    add_count(counts, "to_32_63_bridge_cluster", scope, bridge, source, weight)
                    add_count(counts, "to_32_63_x_K_window", scope, window, source, weight)
                    add_count(counts, "to_32_63_parity", scope, parity, source, weight)
                    add_count(counts, "to_32_63_u_bin", scope, u_bin(i + 1, tau), source, weight)
                    add_count(counts, "to_32_63_prefix_length_bin", scope, steps_bin(i + 1), source, weight)
                    add_count(counts, "to_32_63_previous_prefix_k_cat", scope, previous_prefix_cat(word, i), source, weight)
                    add_count(counts, "to_32_63_next_k_cat", scope, next_k_label, source, weight)
                    add_count(counts, "to_32_63_raw_k", scope, str(next_k if next_k <= 8 else "9+"), source, weight)
                    add_count(counts, "to_32_63_future_1_k_cat", scope, future_sequence(word, i + 1, 1), source, weight)
                    add_count(counts, "to_32_63_future_2_k_cat", scope, future_sequence(word, i + 1, 2), source, weight)
                    add_count(counts, "to_32_63_future_3_k_cat", scope, future_sequence(word, i + 1, 3), source, weight)
                    add_value(values, "to_32_63_prefix_length", scope, source, float(i + 1), weight)
                    add_value(values, "to_32_63_raw_k", scope, source, float(next_k), weight)
        prefix_k += word[i]


def rows_from_counts(counts) -> pd.DataFrame:
    keys = sorted({(table, scope, key) for table, scope, key, _source in counts})
    tmp = []
    l1 = defaultdict(float)
    for table, scope, key in keys:
        actual = counts[(table, scope, key, "actual")]
        iid = counts[(table, scope, key, "iid")]
        delta = actual - iid
        tmp.append(
            {
                "table": table,
                "scope": scope,
                "key": key,
                "actual_mass": actual,
                "iid_mass": iid,
                "delta": delta,
                "ratio": actual / iid if iid else math.nan,
                "abs_delta": abs(delta),
            }
        )
        l1[(table, scope)] += abs(delta)
    for row in tmp:
        denom = l1[(row["table"], row["scope"])]
        row["contribution_to_L1_delta"] = row["abs_delta"] / denom if denom else math.nan
    return pd.DataFrame(tmp)


def conditional_rows(cat: pd.DataFrame) -> pd.DataFrame:
    base = cat[cat["table"] == "layer_mass"][["scope", "actual_mass", "iid_mass"]].rename(
        columns={"actual_mass": "layer_actual_mass", "iid_mass": "layer_iid_mass"}
    )
    out = cat[cat["table"] != "layer_mass"].merge(base, on="scope", how="left")
    out["conditional_actual"] = out["actual_mass"] / out["layer_actual_mass"]
    out["conditional_iid"] = out["iid_mass"] / out["layer_iid_mass"]
    out["conditional_delta"] = out["conditional_actual"] - out["conditional_iid"]
    out["abs_conditional_delta"] = out["conditional_delta"].abs()
    return out


def weighted_mean(vals: list[tuple[float, float]]) -> float:
    total = sum(w for _v, w in vals)
    return sum(v * w for v, w in vals) / total if total else math.nan


def values_rows(values) -> pd.DataFrame:
    rows = []
    for metric, scope in sorted({(m, s) for m, s, _src in values}):
        a = values[(metric, scope, "actual")]
        i = values[(metric, scope, "iid")]
        rows.append({"metric": metric, "scope": scope, "actual_mean": weighted_mean(a), "iid_mean": weighted_mean(i), "mean_delta": weighted_mean(a) - weighted_mean(i)})
    return pd.DataFrame(rows)


def bar_svg(path: Path, rows: pd.DataFrame, table: str, title: str, value_col: str = "delta") -> None:
    data = rows[(rows["table"] == table) & (rows["scope"].isin(["ALL", *FOCUS_STATES]))].copy()
    data["plot_abs"] = data[value_col].abs()
    data = data.sort_values("plot_abs", ascending=False).head(18)
    width, height = 980, 560
    left, right, top, bottom = 300, 35, 50, 45
    max_abs = float(data["plot_abs"].max()) or 1.0
    bar_h = (height - top - bottom) / max(1, len(data))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="16">{title}</text>',
    ]
    zero_x = left + (width - left - right) / 2
    parts.append(f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{height-bottom}" stroke="#6b7280"/>')
    for idx, r in enumerate(data.itertuples(index=False)):
        val = float(getattr(r, value_col))
        y = top + idx * bar_h + 3
        x2 = zero_x + val / max_abs * ((width - left - right) / 2)
        x = min(zero_x, x2)
        color = "#b91c1c" if val > 0 else "#2563eb"
        label = f"{r.scope}|{r.key}"
        if len(label) > 48:
            label = label[:45] + "..."
        parts.append(f'<text x="{left-8}" y="{y+bar_h/2+4:.1f}" text-anchor="end" font-family="Arial" font-size="9">{label}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{abs(x2-zero_x):.1f}" height="{bar_h-6:.1f}" fill="{color}" opacity="0.82"/>')
        parts.append(f'<text x="{x2 + (5 if val >= 0 else -5):.1f}" y="{y+bar_h/2+4:.1f}" text-anchor="{"start" if val >= 0 else "end"}" font-family="Arial" font-size="10">{val:+.3g}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def heatmap_svg(path: Path, rows: pd.DataFrame, table: str, title: str) -> None:
    sub = rows[(rows["table"] == table) & (rows["scope"].isin(FOCUS_STATES))]
    pivot = sub.pivot(index="scope", columns="key", values="delta").fillna(0.0)
    vals = pivot.to_numpy()
    max_abs = float(np.nanmax(np.abs(vals))) or 1.0
    cell_w, cell_h = 110, 44
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
    for i, row in enumerate(pivot.index):
        y = top + i * cell_h
        parts.append(f'<text x="{left-8}" y="{y+cell_h/2+4:.1f}" text-anchor="end" font-family="Arial" font-size="10">{row}</text>')
        for j, col in enumerate(pivot.columns):
            x = left + j * cell_w
            v = float(pivot.loc[row, col])
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color(v)}" stroke="white"/>')
            parts.append(f'<text x="{x+cell_w/2:.1f}" y="{y+cell_h/2+4:.1f}" text-anchor="middle" font-family="Arial" font-size="9">{v:+.3g}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def pick(cat: pd.DataFrame, table: str, scope: str, key: str) -> pd.Series:
    sub = cat[(cat["table"] == table) & (cat["scope"] == scope) & (cat["key"] == key)]
    if sub.empty:
        return pd.Series({"actual_mass": 0, "iid_mass": 0, "delta": 0, "ratio": math.nan})
    return sub.iloc[0]


def build_report(cat: pd.DataFrame, cond: pd.DataFrame, values: pd.DataFrame) -> str:
    layer = pick(cat, "layer_mass", "ALL", "remaining_K_64_95")
    to32 = pick(cat, "exit_transition", "ALL", "64-95 -> 32-63")
    stay = pick(cat, "exit_transition", "ALL", "64-95 -> 64-95")
    cond_to32 = cond[(cond["table"] == "exit_transition") & (cond["scope"] == "ALL") & (cond["key"] == "64-95 -> 32-63")].iloc[0]
    next_k = cat[(cat["table"] == "next_k_cat") & (cat["scope"] == "ALL")].sort_values("abs_delta", ascending=False).iloc[0]
    raw3 = pick(cat, "raw_k_tail_ge3", "ALL", "raw_k>=3")
    raw4 = pick(cat, "raw_k_tail_ge4", "ALL", "raw_k>=4")
    mean_k = values[(values["metric"] == "next_raw_k") & (values["scope"] == "ALL")].iloc[0]
    focus_lines = []
    for state in FOCUS_STATES:
        lm = pick(cat, "layer_mass", state, "remaining_K_64_95")
        tr = pick(cat, "exit_transition", state, "64-95 -> 32-63")
        c = cond[(cond["table"] == "exit_transition") & (cond["scope"] == state) & (cond["key"] == "64-95 -> 32-63")]
        p = float(c["conditional_actual"].iloc[0]) if not c.empty else math.nan
        q = float(c["conditional_iid"].iloc[0]) if not c.empty else math.nan
        focus_lines.append(f"- `{state}`: layer delta `{lm['delta']:+.6g}`, 64-95->32-63 delta `{tr['delta']:+.6g}`, conditional `{p:.4f}/{q:.4f}`.")
    lines = [
        "# Remaining_K 64-95 layer analysis",
        "",
        "Scope: all positions with `remaining_K=64-95`, and the subset of positions whose next step enters `32-63`.",
        "",
        "## Layer mass",
        "",
        f"64-95 layer mass is actual `{layer['actual_mass']:.6g}` vs iid `{layer['iid_mass']:.6g}`, delta `{layer['delta']:+.6g}`, ratio `{layer['ratio']:.3f}`.",
        "",
        "## Exit transition",
        "",
        f"`64-95 -> 32-63` mass is actual `{to32['actual_mass']:.6g}` vs iid `{to32['iid_mass']:.6g}`, delta `{to32['delta']:+.6g}`, ratio `{to32['ratio']:.3f}`.",
        f"Conditioned on being in 64-95, `64-95 -> 32-63` is actual `{cond_to32['conditional_actual']:.4f}` vs iid `{cond_to32['conditional_iid']:.4f}`, conditional_delta `{cond_to32['conditional_delta']:+.6g}`.",
        f"`64-95 -> 64-95` mass delta is `{stay['delta']:+.6g}`.",
        "",
        "## Next k",
        "",
        f"Largest next k_cat delta is `{next_k['key']}` with delta `{next_k['delta']:+.6g}`.",
        f"`raw k >= 3` delta `{raw3['delta']:+.6g}`; `raw k >= 4` delta `{raw4['delta']:+.6g}`.",
        f"Mean next raw k actual/iid is `{mean_k['actual_mean']:.3f}` / `{mean_k['iid_mean']:.3f}`.",
        "",
        "## Focus states",
        "",
        *focus_lines,
        "",
        "## Short reading",
        "",
        "1. The 64-95 band itself is thin in actual.",
        "2. The 64-95 -> 32-63 deficit is mostly mass-level thinning; the conditional transition probability should be read separately from the mass delta.",
        "3. k_cat/raw-k differences identify which valuation sizes are underrepresented in the band and in the fall to 32-63.",
        "4. Focus states differ in sign and route; the even deep late-growth deficit is the clearest connection to the previous 32-63 entry deficit.",
        "5. Next look should be the 96+ predecessor band before any sequence overlay, because it is the upstream side of the depleted 64-95 inflow.",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    module = load_source()
    rng = random.Random(SEED)
    counts = defaultdict(float)
    values = defaultdict(list)
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
    cond = conditional_rows(cat)
    val = values_rows(values)

    cat[cat["table"].isin(["layer_mass", "bridge_cluster", "x_K_window", "parity"])].to_csv(OUT / "boundary_layer_64_95_mass.csv", index=False, encoding="utf-8-sig")
    cat[cat["scope"] != "ALL"].to_csv(OUT / "boundary_layer_64_95_by_state.csv", index=False, encoding="utf-8-sig")
    cat[cat["table"] == "exit_transition"].to_csv(OUT / "boundary_layer_64_95_transition_delta.csv", index=False, encoding="utf-8-sig")
    cond[cond["table"] == "exit_transition"].to_csv(OUT / "boundary_layer_64_95_conditional_transition.csv", index=False, encoding="utf-8-sig")
    cat[cat["table"].isin(["next_k_cat", "next_remaining_K_bin", "raw_k"])].to_csv(OUT / "boundary_layer_64_95_next_k.csv", index=False, encoding="utf-8-sig")
    cat[cat["table"].isin(["raw_k_tail_ge3", "raw_k_tail_ge4"])].to_csv(OUT / "boundary_layer_64_95_raw_k_tail.csv", index=False, encoding="utf-8-sig")
    cat[cat["table"].str.startswith("to_32_63_") | (cat["table"] == "to_32_63_mass")].to_csv(OUT / "boundary_layer_64_95_to_32_63_entries.csv", index=False, encoding="utf-8-sig")
    cat[(cat["scope"] != "ALL") & (cat["table"].str.startswith("to_32_63_") | (cat["table"] == "to_32_63_mass"))].to_csv(
        OUT / "boundary_layer_64_95_to_32_63_by_state.csv", index=False, encoding="utf-8-sig"
    )
    cat[cat["table"].str.startswith("to_32_63_future_")].to_csv(OUT / "boundary_layer_64_95_to_32_63_future.csv", index=False, encoding="utf-8-sig")

    focus_rows = []
    for state in FOCUS_STATES:
        layer = pick(cat, "layer_mass", state, "remaining_K_64_95")
        to32 = pick(cat, "exit_transition", state, "64-95 -> 32-63")
        cond_to32 = cond[(cond["scope"] == state) & (cond["table"] == "exit_transition") & (cond["key"] == "64-95 -> 32-63")]
        next_top = cat[(cat["scope"] == state) & (cat["table"] == "next_k_cat")].sort_values("abs_delta", ascending=False).head(1)
        future_top = cat[(cat["scope"] == state) & (cat["table"] == "to_32_63_future_3_k_cat")].sort_values("abs_delta", ascending=False).head(1)
        mean = val[(val["scope"] == state) & (val["metric"] == "next_raw_k")]
        focus_rows.append(
            {
                "state": state,
                "layer_actual_mass": layer["actual_mass"],
                "layer_iid_mass": layer["iid_mass"],
                "layer_delta": layer["delta"],
                "to_32_63_actual_mass": to32["actual_mass"],
                "to_32_63_iid_mass": to32["iid_mass"],
                "to_32_63_delta": to32["delta"],
                "conditional_to_32_63_actual": cond_to32["conditional_actual"].iloc[0] if not cond_to32.empty else math.nan,
                "conditional_to_32_63_iid": cond_to32["conditional_iid"].iloc[0] if not cond_to32.empty else math.nan,
                "top_next_k_cat": next_top["key"].iloc[0] if not next_top.empty else "",
                "top_next_k_delta": next_top["delta"].iloc[0] if not next_top.empty else math.nan,
                "mean_next_raw_k_actual": mean["actual_mean"].iloc[0] if not mean.empty else math.nan,
                "mean_next_raw_k_iid": mean["iid_mean"].iloc[0] if not mean.empty else math.nan,
                "top_future3": future_top["key"].iloc[0] if not future_top.empty else "",
                "top_future3_delta": future_top["delta"].iloc[0] if not future_top.empty else math.nan,
            }
        )
    pd.DataFrame(focus_rows).to_csv(OUT / "boundary_layer_64_95_focus_summary.csv", index=False, encoding="utf-8-sig")

    bar_svg(OUT / "boundary_layer_64_95_exit_transition_ratio.svg", cond, "exit_transition", "64-95 exit transition conditional delta", "conditional_delta")
    bar_svg(OUT / "boundary_layer_64_95_exit_transition_delta.svg", cat, "exit_transition", "64-95 exit transition mass delta")
    heatmap_svg(OUT / "boundary_layer_64_95_to_32_63_state_heatmap.svg", cat, "exit_transition", "State heatmap: 64-95 exit transitions")
    bar_svg(OUT / "boundary_layer_64_95_next_k_delta.svg", cat, "next_k_cat", "64-95 next k_cat delta")
    bar_svg(OUT / "boundary_layer_64_95_to_32_63_future3_delta.svg", cat, "to_32_63_future_3_k_cat", "64-95 -> 32-63 future3 delta")

    (OUT / "boundary_layer_64_95_report.md").write_text(build_report(cat, cond, val), encoding="utf-8")
    print(OUT / "boundary_layer_64_95_report.md", flush=True)


if __name__ == "__main__":
    main()
