from __future__ import annotations

import csv
import importlib.util
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SRC = Path(r"C:\Users\yauki\Documents\design\Collatz\py\py\collatz_escape_word_deficit.py")
OUT_DIR = Path("outputs")
POWERS = [24, 25, 26, 27, 28]
HS = [2, 3, 4, 5, 6]
IID_SAMPLES_PER_H = 160_000
ACTUAL_SAMPLE_PER_PH = 20_000
SEED = 20260625
LOG2_3 = math.log2(3.0)
U_BIN_WIDTH = 0.1
SMOOTH_ALPHA = 0.5
MIN_STATE_IID_MASS = 1e-7
MAX_REGRESSION_ROWS = 60_000
CATEGORY_ORDER = ["1", "2", "3+"]
SUPPORT = {
    3: [f"{a},{b},{c}" for a in CATEGORY_ORDER for b in CATEGORY_ORDER for c in CATEGORY_ORDER],
    4: [f"{a},{b},{c},{d}" for a in CATEGORY_ORDER for b in CATEGORY_ORDER for c in CATEGORY_ORDER for d in CATEGORY_ORDER],
}
FOCUS_STATE = ("late_growth", "tail_64_95", "even")
CACHE_DIRS = [
    Path(r"C:\Users\yauki\Documents\Codex\2026-06-25\new-chat-3\work\status_cache"),
    Path(r"C:\Users\yauki\Documents\Codex\2026-06-25\new-chat-2\work\status_cache"),
    Path(r"C:\Users\yauki\Documents\Codex\2026-06-25\new-chat\work\status_cache"),
]


@dataclass(frozen=True)
class WordRec:
    source: str
    split: str
    power: int
    h: int
    parity: str
    weight: float
    word: tuple[int, ...]
    tau: int
    k_tau: int
    x_k: int
    x_k_window: str
    bridge_cluster: str
    z25: float
    z50: float
    z75: float
    drift_delta: float


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
            print(f"loading {path}", flush=True)
            return bytearray(path.read_bytes())
    raise FileNotFoundError(f"missing cache for p={power}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def z_features(word: tuple[int, ...]) -> tuple[float, float, float, float]:
    xs = path_xs(word)
    final = xs[-1]
    return (
        interp(xs, 0.25) - 0.25 * final,
        interp(xs, 0.50) - 0.50 * final,
        interp(xs, 0.75) - 0.75 * final,
        final,
    )


def weighted_quantiles(values: list[tuple[float, float]], probs: tuple[float, ...]) -> list[float]:
    if not values:
        return [math.nan for _ in probs]
    vals = sorted(values)
    total = sum(w for _v, w in vals)
    out: list[float] = []
    acc = 0.0
    pi = 0
    for value, weight in vals:
        acc += weight
        while pi < len(probs) and acc >= probs[pi] * total:
            out.append(value)
            pi += 1
    while len(out) < len(probs):
        out.append(vals[-1][0])
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


def sample_iid(module, h: int, rng: random.Random) -> list[tuple[tuple[int, ...], float]]:
    out = []
    for _ in range(IID_SAMPLES_PER_H):
        y = 0.5 + 0.5 * rng.random()
        distance = h - math.log2(y)
        position = 0.0
        word: list[int] = []
        while position <= distance:
            k = module.tilted_k(rng)
            word.append(k)
            position += LOG2_3 - k
        overshoot = position - distance
        weight = (2.0 ** (-h)) * y * (2.0 ** (-overshoot)) / IID_SAMPLES_PER_H
        out.append((tuple(word), weight))
    return out


def evenly_spaced(items: list[int], limit: int) -> list[int]:
    if len(items) <= limit:
        return items
    if limit <= 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (limit - 1))] for i in range(limit)]


def cluster_from_z(z: float, q_low: float, q_high: float) -> str:
    if z <= q_low:
        return "late_growth"
    if z >= q_high:
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


def k_cat(k: int) -> str:
    if k == 1:
        return "1"
    if k == 2:
        return "2"
    return "3+"


def u_bin(i: int, tau: int) -> str:
    u = (i + 1) / tau
    lo = math.floor(u / U_BIN_WIDTH) * U_BIN_WIDTH
    if lo >= 1.0:
        lo = 0.9
    return f"{lo:.1f}-{lo + U_BIN_WIDTH:.1f}"


def state_key(rec: WordRec) -> str:
    return f"{rec.bridge_cluster}|{rec.x_k_window}|{rec.parity}"


def make_rec(
    source: str,
    split: str,
    power: int,
    h: int,
    weight: float,
    word: tuple[int, ...],
    q_low: float,
    q_high: float,
) -> WordRec | None:
    k_tau = sum(word)
    x_k = k_tau - (power - h)
    window = xk_window(x_k)
    if window is None:
        return None
    z25, z50, z75, drift = z_features(word)
    cluster = cluster_from_z(z25, q_low, q_high)
    return WordRec(
        source=source,
        split=split,
        power=power,
        h=h,
        parity="even" if power % 2 == 0 else "odd",
        weight=weight,
        word=word,
        tau=len(word),
        k_tau=k_tau,
        x_k=x_k,
        x_k_window=window,
        bridge_cluster=cluster,
        z25=z25,
        z50=z50,
        z75=z75,
        drift_delta=drift,
    )


def add_block_counts(rec: WordRec, counts: defaultdict[tuple[int, str, str, str, str], float]) -> None:
    cats = [k_cat(k) for k in rec.word]
    st = state_key(rec)
    for block_len in [3, 4]:
        if rec.tau < block_len:
            continue
        for i in range(rec.tau - block_len + 1):
            block = ",".join(cats[i : i + block_len])
            counts[(block_len, st, u_bin(i, rec.tau), rec.source, block)] += rec.weight


def make_log_ratio_lookup(
    counts: defaultdict[tuple[int, str, str, str, str], float],
) -> tuple[dict[tuple[int, str, str, str], float], set[str]]:
    lookup: dict[tuple[int, str, str, str], float] = {}
    stable_states: set[str] = set()
    u_bins = [f"{i / 10:.1f}-{(i + 1) / 10:.1f}" for i in range(10)]
    states = sorted({state for _bl, state, _ub, _src, _block in counts})
    for block_len in [3, 4]:
        support = SUPPORT[block_len]
        v = len(support)
        for st in states:
            state_iid_mass = 0.0
            for ub in u_bins:
                actual_total = sum(counts[(block_len, st, ub, "actual", b)] for b in support)
                iid_total = sum(counts[(block_len, st, ub, "iid", b)] for b in support)
                state_iid_mass += iid_total
                for block in support:
                    actual_p = (counts[(block_len, st, ub, "actual", block)] + SMOOTH_ALPHA) / (actual_total + SMOOTH_ALPHA * v)
                    iid_p = (counts[(block_len, st, ub, "iid", block)] + SMOOTH_ALPHA) / (iid_total + SMOOTH_ALPHA * v)
                    lookup[(block_len, st, ub, block)] = math.log2(actual_p / iid_p)
            if block_len == 4 and state_iid_mass >= MIN_STATE_IID_MASS:
                stable_states.add(st)
    return lookup, stable_states


def block_score(rec: WordRec, lookup: dict[tuple[int, str, str, str], float], block_len: int) -> float:
    cats = [k_cat(k) for k in rec.word]
    st = state_key(rec)
    score = 0.0
    if rec.tau < block_len:
        return score
    for i in range(rec.tau - block_len + 1):
        ub = u_bin(i, rec.tau)
        block = ",".join(cats[i : i + block_len])
        score += lookup.get((block_len, st, ub, block), 0.0)
    return score


def weighted_mean(values: list[tuple[float, float]]) -> float:
    total = sum(w for _v, w in values)
    return sum(v * w for v, w in values) / total if total else math.nan


def weighted_auc(pos: list[tuple[float, float]], neg: list[tuple[float, float]]) -> float:
    if not pos or not neg:
        return math.nan
    events: dict[float, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for score, weight in pos:
        events[score][0] += weight
    for score, weight in neg:
        events[score][1] += weight
    neg_seen = 0.0
    numer = 0.0
    for score in sorted(events):
        pw, nw = events[score]
        numer += pw * neg_seen + 0.5 * pw * nw
        neg_seen += nw
    ptotal = sum(w for _s, w in pos)
    ntotal = sum(w for _s, w in neg)
    return numer / (ptotal * ntotal) if ptotal and ntotal else math.nan


def weighted_wasserstein(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    if not a or not b:
        return math.nan
    xs = sorted({v for v, _w in a} | {v for v, _w in b})
    if len(xs) < 2:
        return 0.0
    ta = sum(w for _v, w in a)
    tb = sum(w for _v, w in b)
    amap = defaultdict(float)
    bmap = defaultdict(float)
    for v, w in a:
        amap[v] += w / ta
    for v, w in b:
        bmap[v] += w / tb
    ca = cb = 0.0
    dist = 0.0
    for i in range(len(xs) - 1):
        ca += amap[xs[i]]
        cb += bmap[xs[i]]
        dist += abs(ca - cb) * (xs[i + 1] - xs[i])
    return dist


def distribution_rows(score_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for block_len in [3, 4]:
        field = f"block_log_score_B{block_len}"
        states = sorted({str(r["state"]) for r in score_rows})
        for st in states:
            actual = [(float(r[field]), float(r["weight"])) for r in score_rows if r["state"] == st and r["source"] == "actual"]
            iid = [(float(r[field]), float(r["weight"])) for r in score_rows if r["state"] == st and r["source"] == "iid"]
            for source, vals in [("actual", actual), ("iid", iid)]:
                qs = weighted_quantiles(vals, (0.05, 0.25, 0.5, 0.75, 0.95))
                out.append(
                    {
                        "block_len": block_len,
                        "state": st,
                        "source": source,
                        "mass": sum(w for _v, w in vals),
                        "mean": weighted_mean(vals),
                        "q05": qs[0],
                        "q25": qs[1],
                        "median": qs[2],
                        "q75": qs[3],
                        "q95": qs[4],
                        "wasserstein_actual_iid": weighted_wasserstein(actual, iid),
                        "auc_actual_high_score_vs_iid": weighted_auc(actual, iid),
                    }
                )
    return out


def survival_by_score_rows(score_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for block_len in [3, 4]:
        field = f"block_log_score_B{block_len}"
        for st in sorted({str(r["state"]) for r in score_rows}):
            subset = [r for r in score_rows if r["state"] == st]
            vals = [(float(r[field]), float(r["weight"])) for r in subset]
            cuts = weighted_quantiles(vals, tuple(i / 10 for i in range(1, 10)))
            for r in subset:
                score = float(r[field])
                idx = sum(score > c for c in cuts)
                actual = "actual" if r["source"] == "actual" else "iid"
                while len(out) <= 0:
                    break
            buckets: dict[int, dict[str, float]] = defaultdict(lambda: {"actual": 0.0, "iid": 0.0})
            score_ranges: dict[int, list[float]] = defaultdict(list)
            for r in subset:
                score = float(r[field])
                idx = sum(score > c for c in cuts)
                buckets[idx][str(r["source"])] += float(r["weight"])
                score_ranges[idx].append(score)
            for idx in range(10):
                mass = buckets[idx]
                iid = mass["iid"]
                out.append(
                    {
                        "block_len": block_len,
                        "state": st,
                        "score_bin": idx,
                        "score_min": min(score_ranges[idx]) if score_ranges[idx] else "",
                        "score_max": max(score_ranges[idx]) if score_ranges[idx] else "",
                        "actual_mass": mass["actual"],
                        "iid_mass": iid,
                        "survival_ratio": mass["actual"] / iid if iid else "",
                        "iid_excess_score_bin_high": 9 - idx,
                    }
                )
    return out


def design_matrix(rows: list[dict[str, object]], features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    names: list[str] = []
    cols: list[np.ndarray] = []
    n = len(rows)
    for feat in features:
        if feat in {"x_K", "z25", "z50", "z75", "block_log_score_B3", "block_log_score_B4"}:
            arr = np.array([float(r[feat]) for r in rows], dtype=float)
            sd = arr.std()
            arr = (arr - arr.mean()) / (sd if sd > 0 else 1.0)
            cols.append(arr)
            names.append(feat)
        elif feat == "parity":
            arr = np.array([1.0 if r["p_parity"] == "even" else 0.0 for r in rows])
            cols.append(arr)
            names.append("parity_even")
        elif feat == "bridge_cluster":
            for value in ["balanced", "early_growth", "late_growth"]:
                cols.append(np.array([1.0 if r["bridge_cluster"] == value else 0.0 for r in rows]))
                names.append(f"bridge_{value}")
        elif feat == "x_K_window":
            for value in ["deep_32_63", "exhaustion_0_31", "tail_64_95"]:
                cols.append(np.array([1.0 if r["x_K_window"] == value else 0.0 for r in rows]))
                names.append(f"xKwin_{value}")
        else:
            raise ValueError(feat)
    x = np.column_stack([np.ones(n)] + cols)
    y = np.array([1.0 if r["source"] == "actual" else 0.0 for r in rows])
    w = np.array([float(r["weight"]) for r in rows])
    return x, y, w, ["intercept"] + names


def fit_logistic(rows: list[dict[str, object]], features: list[str]) -> tuple[np.ndarray, list[str], float]:
    x, y, w, names = design_matrix(rows, features)
    # Normalize weights for stable gradient scale while preserving relative mass.
    w = w / w.mean()
    beta = np.zeros(x.shape[1], dtype=float)
    lr = 0.08
    l2 = 1e-4
    for _ in range(350):
        z = np.clip(x @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = (x.T @ (w * (p - y))) / len(y)
        grad[1:] += l2 * beta[1:]
        beta -= lr * grad
    scores = x @ beta
    auc = weighted_auc(
        [(float(s), float(r["weight"])) for s, r in zip(scores, rows) if r["source"] == "actual"],
        [(float(s), float(r["weight"])) for s, r in zip(scores, rows) if r["source"] == "iid"],
    )
    return beta, names, auc


def regression_rows(score_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [r for r in score_rows if r["split"] == "test"]
    if len(rows) > MAX_REGRESSION_ROWS:
        actual = [r for r in rows if r["source"] == "actual"]
        iid = [r for r in rows if r["source"] == "iid"]
        half = MAX_REGRESSION_ROWS // 2

        def thin(items: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
            if len(items) <= limit:
                return items
            stride = len(items) / limit
            return [items[int(i * stride)] for i in range(limit)]

        rows = thin(actual, half) + thin(iid, MAX_REGRESSION_ROWS - min(half, len(actual)))
    models = [
        ("xK_parity_bridge_z", ["x_K", "x_K_window", "parity", "bridge_cluster", "z25", "z50", "z75"]),
        ("plus_B3", ["x_K", "x_K_window", "parity", "bridge_cluster", "z25", "z50", "z75", "block_log_score_B3"]),
        ("plus_B4", ["x_K", "x_K_window", "parity", "bridge_cluster", "z25", "z50", "z75", "block_log_score_B4"]),
        ("plus_B3_B4", ["x_K", "x_K_window", "parity", "bridge_cluster", "z25", "z50", "z75", "block_log_score_B3", "block_log_score_B4"]),
        ("score_only_B4", ["block_log_score_B4"]),
    ]
    out: list[dict[str, object]] = []
    base_auc = None
    base_bridge_abs = None
    base_parity_abs = None
    for name, features in models:
        beta, names, auc = fit_logistic(rows, features)
        coef = dict(zip(names, beta))
        bridge_abs = sum(abs(coef.get(f"bridge_{v}", 0.0)) for v in ["balanced", "early_growth", "late_growth"])
        parity_abs = abs(coef.get("parity_even", 0.0))
        if base_auc is None:
            base_auc = auc
            base_bridge_abs = bridge_abs
            base_parity_abs = parity_abs
        out.append(
            {
                "model": name,
                "features": ";".join(features),
                "weighted_auc": auc,
                "auc_delta_vs_base": auc - base_auc,
                "coef_B3_score": coef.get("block_log_score_B3", ""),
                "coef_B4_score": coef.get("block_log_score_B4", ""),
                "bridge_abs_coef_sum": bridge_abs,
                "bridge_abs_reduction_vs_base": (base_bridge_abs - bridge_abs) if base_bridge_abs is not None else "",
                "parity_even_coef": coef.get("parity_even", ""),
                "parity_abs_reduction_vs_base": (base_parity_abs - parity_abs) if base_parity_abs is not None else "",
                "coef_late_growth": coef.get("bridge_late_growth", ""),
                "coef_tail_64_95": coef.get("xKwin_tail_64_95", ""),
            }
        )
    # Add marginal weighted AUCs so feature_importance_or_auc.svg can be read without the model.
    for field in ["x_K", "z25", "z50", "z75", "block_log_score_B3", "block_log_score_B4"]:
        pos = [(float(r[field]), float(r["weight"])) for r in rows if r["source"] == "actual"]
        neg = [(float(r[field]), float(r["weight"])) for r in rows if r["source"] == "iid"]
        out.append(
            {
                "model": f"marginal_{field}",
                "features": field,
                "weighted_auc": weighted_auc(pos, neg),
                "auc_delta_vs_base": "",
                "coef_B3_score": "",
                "coef_B4_score": "",
                "bridge_abs_coef_sum": "",
                "bridge_abs_reduction_vs_base": "",
                "parity_even_coef": "",
                "parity_abs_reduction_vs_base": "",
                "coef_late_growth": "",
                "coef_tail_64_95": "",
            }
        )
    return out


def simple_bar_svg(path: Path, rows: list[dict[str, object]], title: str) -> None:
    model_rows = [r for r in rows if not str(r["model"]).startswith("marginal_")]
    width, height = 760, 380
    left, right, top, bottom = 120, 40, 46, 90
    vals = [float(r["weighted_auc"]) for r in model_rows]
    ymin, ymax = 0.45, max(vals + [0.55])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-family="Arial" font-size="17">{title}</text>',
    ]
    bar_w = (width - left - right) / max(1, len(model_rows))
    for i, r in enumerate(model_rows):
        v = float(r["weighted_auc"])
        x = left + i * bar_w + 8
        y = top + (ymax - v) * (height - top - bottom) / (ymax - ymin)
        h = height - bottom - y
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-16:.1f}" height="{h:.1f}" fill="#2563eb"/>')
        parts.append(f'<text x="{x + (bar_w-16)/2:.1f}" y="{y-5:.1f}" text-anchor="middle" font-family="Arial" font-size="10">{v:.3f}</text>')
        parts.append(f'<text transform="translate({x + (bar_w-16)/2:.1f},{height-bottom+14}) rotate(38)" text-anchor="start" font-family="Arial" font-size="10">{r["model"]}</text>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def distribution_svg(path: Path, rows: list[dict[str, object]]) -> None:
    focus_states = [
        "late_growth|tail_64_95|even",
        "late_growth|tail_64_95|odd",
        "early_growth|tail_64_95|even",
        "late_growth|deep_32_63|even",
    ]
    width, height = 980, 460
    left, right, top, bottom = 100, 40, 48, 95
    subset = [r for r in rows if int(r["block_len"]) == 4 and r["state"] in focus_states]
    vals = [float(r["median"]) for r in subset] + [float(r["q25"]) for r in subset] + [float(r["q75"]) for r in subset]
    xmin, xmax = min(vals), max(vals)
    pad = (xmax - xmin) * 0.15 or 1.0
    xmin -= pad
    xmax += pad

    def sx(v: float) -> float:
        return left + (v - xmin) * (width - left - right) / (xmax - xmin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-family="Arial" font-size="17">B4 score distribution by state</text>',
    ]
    y = top
    colors = {"actual": "#b91c1c", "iid": "#2563eb"}
    for state in focus_states:
        for source in ["actual", "iid"]:
            r = next((x for x in subset if x["state"] == state and x["source"] == source), None)
            if not r:
                continue
            y += 34
            q25, q50, q75 = float(r["q25"]), float(r["median"]), float(r["q75"])
            color = colors[source]
            parts.append(f'<line x1="{sx(q25):.1f}" y1="{y}" x2="{sx(q75):.1f}" y2="{y}" stroke="{color}" stroke-width="7" opacity="0.55"/>')
            parts.append(f'<circle cx="{sx(q50):.1f}" cy="{y}" r="5" fill="{color}"/>')
            parts.append(f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-family="Arial" font-size="10">{state} {source}</text>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def survival_svg(path: Path, rows: list[dict[str, object]]) -> None:
    subset = [r for r in rows if int(r["block_len"]) == 4 and r["state"] == "late_growth|tail_64_95|even" and r["survival_ratio"] != ""]
    width, height = 760, 420
    left, right, top, bottom = 70, 35, 45, 65
    vals = [float(r["survival_ratio"]) for r in subset]
    ymax = max(vals + [1.0]) * 1.15
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-family="Arial" font-size="17">Survival by B4 block score: late_growth tail even</text>',
    ]
    pts = []
    for r in subset:
        i = int(r["score_bin"])
        x = left + i * (width - left - right) / 9
        y = top + (ymax - float(r["survival_ratio"])) * (height - top - bottom) / ymax
        pts.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#b91c1c"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+20}" text-anchor="middle" font-family="Arial" font-size="10">{i}</text>')
    if len(pts) >= 2:
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#b91c1c" stroke-width="2"/>')
    y1 = top + (ymax - 1.0) * (height - top - bottom) / ymax
    parts.append(f'<line x1="{left}" y1="{y1:.1f}" x2="{width-right}" y2="{y1:.1f}" stroke="#6b7280" stroke-dasharray="4 4"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def build_report(
    score_rows: list[dict[str, object]],
    dist_rows: list[dict[str, object]],
    survival_rows_: list[dict[str, object]],
    regression: list[dict[str, object]],
    q_low: float,
    q_high: float,
) -> None:
    main_dist = [
        r
        for r in dist_rows
        if int(r["block_len"]) == 4 and r["state"] == "late_growth|tail_64_95|even" and r["source"] == "actual"
    ][0]
    iid_main_dist = [
        r
        for r in dist_rows
        if int(r["block_len"]) == 4 and r["state"] == "late_growth|tail_64_95|even" and r["source"] == "iid"
    ][0]
    reg_base = next(r for r in regression if r["model"] == "xK_parity_bridge_z")
    reg_b4 = next(r for r in regression if r["model"] == "plus_B4")
    reg_both = next(r for r in regression if r["model"] == "plus_B3_B4")
    focus_surv = [
        r
        for r in survival_rows_
        if int(r["block_len"]) == 4 and r["state"] == "late_growth|tail_64_95|even" and r["survival_ratio"] != ""
    ]
    low_bin = min(focus_surv, key=lambda r: int(r["score_bin"]))
    high_bin = max(focus_surv, key=lambda r: int(r["score_bin"]))
    auc_delta = float(reg_b4["auc_delta_vs_base"])
    bridge_reduction = float(reg_b4["bridge_abs_reduction_vs_base"])
    parity_reduction = float(reg_b4["parity_abs_reduction_vs_base"])
    if auc_delta > 0.03 and bridge_reduction > 0.25 * float(reg_base["bridge_abs_coef_sum"]):
        classification = "A. B4 score explains most survival depletion"
    elif auc_delta > 0.005 and float(reg_b4["bridge_abs_coef_sum"]) > 0.5 * float(reg_base["bridge_abs_coef_sum"]):
        classification = "B. B4 score helps but bridge/parity remain strong"
    elif auc_delta <= 0.005:
        classification = "C. B4 score adds little after bridge_cluster and x_K"
    else:
        classification = "D. unstable / leakage-sensitive"
    lines = [
        "# Collatz Block Anomaly Score",
        "",
        f"- source script: `work/collatz_block_anomaly_score.py`",
        f"- seed: {SEED}",
        f"- iid samples per h: {IID_SAMPLES_PER_H}",
        f"- actual samples per p,h: {ACTUAL_SAMPLE_PER_PH}",
        "- split: train/test by deterministic sample index parity; train estimates block probabilities, test receives word scores",
        f"- smoothing: add-alpha {SMOOTH_ALPHA} over 3^m coarse blocks",
        f"- regression rows: deterministic weighted subsample capped at {MAX_REGRESSION_ROWS}; score distributions and survival bins use all scored test rows",
        f"- bridge cluster z25 cuts: q_low={q_low:.6g}, q_high={q_high:.6g}",
        "",
        "## Main Reading",
        "",
        f"`late_growth|tail_64_95|even` has B4 actual median score `{float(main_dist['median']):.3f}` versus iid median `{float(iid_main_dist['median']):.3f}`; weighted AUC(actual high score vs iid) is `{float(main_dist['auc_actual_high_score_vs_iid']):.3f}`.",
        f"In that same state, lowest B4-score decile has S=`{float(low_bin['survival_ratio']):.3f}`, highest decile has S=`{float(high_bin['survival_ratio']):.3f}`.",
        "",
        "## Regression",
        "",
        "| model | weighted AUC | delta vs base | B4 coef | bridge abs coef | parity coef |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in [reg_base, reg_b4, reg_both]:
        lines.append(
            f"| {r['model']} | {float(r['weighted_auc']):.4f} | {float(r['auc_delta_vs_base']):.4f} | "
            f"{float(r['coef_B4_score']) if r['coef_B4_score'] != '' else 0.0:.4f} | "
            f"{float(r['bridge_abs_coef_sum']):.4f} | {float(r['parity_even_coef']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Classification",
            "",
            f"Current classification: **{classification}**.",
            "",
            f"B4 changes AUC by `{auc_delta:.4f}` after x_K, parity, bridge_cluster, and z25/z50/z75. Bridge coefficient reduction is `{bridge_reduction:.4f}` and parity coefficient reduction is `{parity_reduction:.4f}`. Treat this as a split-sample sampled diagnostic, not an exact aggregation proof.",
        ]
    )
    (OUT_DIR / "collatz_block_anomaly_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    module = load_source()
    rng = random.Random(SEED)
    iid_by_h: dict[int, list[tuple[tuple[int, ...], float]]] = {}
    tau_cuts: dict[int, tuple[int, int]] = {}
    iid_z_for_cluster: list[tuple[float, float]] = []

    print("sampling iid words", flush=True)
    for h in HS:
        sampled = sample_iid(module, h, rng)
        iid_by_h[h] = sampled
        tau_cuts[h] = weighted_tau_cuts(sampled)
        long_words = [(word, weight) for word, weight in sampled if len(word) > tau_cuts[h][1]]
        iid_z_for_cluster.extend((z_features(word)[0], weight) for word, weight in long_words)
        print(f"iid h={h}: tau cuts={tau_cuts[h]}, long={len(long_words)}", flush=True)

    q_low, q_high = weighted_quantiles(iid_z_for_cluster, (1 / 3, 2 / 3))
    print(f"z25 cuts: {q_low:.6g}, {q_high:.6g}", flush=True)

    records: list[WordRec] = []
    block_counts: defaultdict[tuple[int, str, str, str, str], float] = defaultdict(float)

    print("building iid records", flush=True)
    for h, sampled in iid_by_h.items():
        for sample_idx, (word, weight) in enumerate(sampled):
            if len(word) <= tau_cuts[h][1]:
                continue
            split = "train" if sample_idx % 2 == 0 else "test"
            for power in POWERS:
                rec = make_rec("iid", split, power, h, weight, word, q_low, q_high)
                if rec is None:
                    continue
                records.append(rec)
                if split == "train":
                    add_block_counts(rec, block_counts)

    print("building actual records", flush=True)
    for power in POWERS:
        status = load_status(power)
        for h in HS:
            _cut1, cut2 = tau_cuts[h]
            lo, hi, total = module.layer_bounds(power, h)
            escape_indices = [idx for idx in range(lo >> 1, (hi >> 1) + 1) if status[idx] == module.ESCAPE]
            chosen = evenly_spaced(escape_indices, ACTUAL_SAMPLE_PER_PH)
            sample_weight = len(escape_indices) / (len(chosen) * total) if chosen else 0.0
            kept = 0
            for sample_idx, idx in enumerate(chosen):
                word = module.trace_escape(2 * idx + 1, 1 << power)
                if len(word) <= cut2:
                    continue
                split = "train" if sample_idx % 2 == 0 else "test"
                rec = make_rec("actual", split, power, h, sample_weight, word, q_low, q_high)
                if rec is None:
                    continue
                records.append(rec)
                kept += 1
                if split == "train":
                    add_block_counts(rec, block_counts)
            print(f"actual p={power} h={h}: sampled={len(chosen)}, kept={kept}", flush=True)
        del status

    print("building log-ratio lookup", flush=True)
    log_lookup, stable_states = make_log_ratio_lookup(block_counts)

    print("scoring test words", flush=True)
    score_rows: list[dict[str, object]] = []
    for rec in records:
        if rec.split != "test":
            continue
        st = state_key(rec)
        if st not in stable_states:
            continue
        b3 = block_score(rec, log_lookup, 3)
        b4 = block_score(rec, log_lookup, 4)
        score_rows.append(
            {
                "source": rec.source,
                "split": rec.split,
                "power": rec.power,
                "h": rec.h,
                "p_parity": rec.parity,
                "weight": rec.weight,
                "tau": rec.tau,
                "K_tau": rec.k_tau,
                "x_K": rec.x_k,
                "x_K_window": rec.x_k_window,
                "bridge_cluster": rec.bridge_cluster,
                "state": st,
                "z25": rec.z25,
                "z50": rec.z50,
                "z75": rec.z75,
                "drift_delta": rec.drift_delta,
                "block_log_score_B3": b3,
                "iid_excess_score_B3": -b3,
                "block_log_score_B4": b4,
                "iid_excess_score_B4": -b4,
            }
        )

    print(f"scored test rows: {len(score_rows)}", flush=True)
    print("building score distributions", flush=True)
    dist = distribution_rows(score_rows)
    print("building survival by score", flush=True)
    survival = survival_by_score_rows(score_rows)
    print("running regression summaries", flush=True)
    regression = regression_rows(score_rows)

    print("writing outputs", flush=True)
    write_csv(OUT_DIR / "collatz_block_anomaly_word_scores.csv", score_rows)
    write_csv(OUT_DIR / "collatz_block_anomaly_score_distribution.csv", dist)
    write_csv(OUT_DIR / "collatz_block_anomaly_survival_by_score.csv", survival)
    write_csv(OUT_DIR / "collatz_block_anomaly_regression_summary.csv", regression)
    distribution_svg(OUT_DIR / "score_distribution_by_state.svg", dist)
    survival_svg(OUT_DIR / "survival_by_block_score.svg", survival)
    simple_bar_svg(OUT_DIR / "feature_importance_or_auc.svg", regression, "Weighted AUC by logistic feature set")
    build_report(score_rows, dist, survival, regression, q_low, q_high)
    print(OUT_DIR / "collatz_block_anomaly_report.md", flush=True)


if __name__ == "__main__":
    main()
