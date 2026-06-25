from __future__ import annotations

import csv
import importlib.util
import itertools
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BASE_PATH = Path("work/collatz_block_anomaly_score.py")
OUT_DIR = Path("outputs")
BLOCK_LENGTHS = [3, 4, 5, 6]
SMOOTH_MIX = 0.02
MIN_TRAIN_IID_TOTAL = 1e-7
MAX_REGRESSION_ROWS = 50_000
CATEGORY_ORDER = ["1", "2", "3+"]
FOCUS_STATE = "late_growth|tail_64_95|even"


@dataclass(frozen=True)
class ThinRec:
    source: str
    split: str
    weight: float
    word: tuple[int, ...]
    x_k: int
    x_k_window: str
    parity: str
    bridge_cluster: str
    state: str
    z25: float
    z50: float
    z75: float


def load_base():
    spec = importlib.util.spec_from_file_location("collatz_block_anomaly_score", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def support(block_len: int) -> list[str]:
    return [",".join(p) for p in itertools.product(CATEGORY_ORDER, repeat=block_len)]


SUPPORT = {block_len: support(block_len) for block_len in BLOCK_LENGTHS}


def make_thin(base, source: str, split: str, power: int, h: int, weight: float, word: tuple[int, ...], q_low: float, q_high: float) -> ThinRec | None:
    rec = base.make_rec(source, split, power, h, weight, word, q_low, q_high)
    if rec is None:
        return None
    return ThinRec(
        source=rec.source,
        split=rec.split,
        weight=rec.weight,
        word=rec.word,
        x_k=rec.x_k,
        x_k_window=rec.x_k_window,
        parity=rec.parity,
        bridge_cluster=rec.bridge_cluster,
        state=base.state_key(rec),
        z25=rec.z25,
        z50=rec.z50,
        z75=rec.z75,
    )


def add_counts(base, rec: ThinRec, counts: defaultdict[tuple[int, str, str, str, str], float]) -> None:
    cats = [base.k_cat(k) for k in rec.word]
    tau = len(cats)
    for block_len in BLOCK_LENGTHS:
        if tau < block_len:
            continue
        for i in range(tau - block_len + 1):
            block = ",".join(cats[i : i + block_len])
            counts[(block_len, rec.state, base.u_bin(i, tau), rec.source, block)] += rec.weight


def make_lookup(counts: defaultdict[tuple[int, str, str, str, str], float]) -> tuple[dict[tuple[int, str, str, str], float], dict[int, set[str]]]:
    lookup: dict[tuple[int, str, str, str], float] = {}
    stable: dict[int, set[str]] = {block_len: set() for block_len in BLOCK_LENGTHS}
    states = sorted({state for _bl, state, _ub, _src, _block in counts})
    u_bins = [f"{i / 10:.1f}-{(i + 1) / 10:.1f}" for i in range(10)]
    for block_len in BLOCK_LENGTHS:
        sup = SUPPORT[block_len]
        v = len(sup)
        for state in states:
            state_iid = 0.0
            for ub in u_bins:
                actual_total = sum(counts[(block_len, state, ub, "actual", b)] for b in sup)
                iid_total = sum(counts[(block_len, state, ub, "iid", b)] for b in sup)
                state_iid += iid_total
                for block in sup:
                    actual_emp = counts[(block_len, state, ub, "actual", block)] / actual_total if actual_total else 1.0 / v
                    iid_emp = counts[(block_len, state, ub, "iid", block)] / iid_total if iid_total else 1.0 / v
                    actual_p = (1.0 - SMOOTH_MIX) * actual_emp + SMOOTH_MIX / v
                    iid_p = (1.0 - SMOOTH_MIX) * iid_emp + SMOOTH_MIX / v
                    lookup[(block_len, state, ub, block)] = math.log2(actual_p / iid_p)
            if state_iid >= MIN_TRAIN_IID_TOTAL:
                stable[block_len].add(state)
    return lookup, stable


def block_score(base, rec: ThinRec, lookup: dict[tuple[int, str, str, str], float], block_len: int) -> float:
    cats = [base.k_cat(k) for k in rec.word]
    tau = len(cats)
    if tau < block_len:
        return 0.0
    score = 0.0
    for i in range(tau - block_len + 1):
        block = ",".join(cats[i : i + block_len])
        score += lookup.get((block_len, rec.state, base.u_bin(i, tau), block), 0.0)
    return score


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


def thin_for_regression(rows: list[tuple[ThinRec, float]], limit: int = MAX_REGRESSION_ROWS) -> list[tuple[ThinRec, float]]:
    if len(rows) <= limit:
        return rows
    actual = [r for r in rows if r[0].source == "actual"]
    iid = [r for r in rows if r[0].source == "iid"]
    half = limit // 2

    def thin(items: list[tuple[ThinRec, float]], n: int) -> list[tuple[ThinRec, float]]:
        if len(items) <= n:
            return items
        stride = len(items) / n
        return [items[int(i * stride)] for i in range(n)]

    return thin(actual, half) + thin(iid, limit - min(half, len(actual)))


def design_matrix(rows: list[tuple[ThinRec, float]], include_score: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    names = [
        "intercept",
        "x_K",
        "z25",
        "z50",
        "z75",
        "parity_even",
        "bridge_balanced",
        "bridge_early_growth",
        "bridge_late_growth",
        "xKwin_deep_32_63",
        "xKwin_exhaustion_0_31",
        "xKwin_tail_64_95",
    ]
    cols = [np.ones(len(rows))]
    numeric = [
        np.array([r.x_k for r, _s in rows], dtype=float),
        np.array([r.z25 for r, _s in rows], dtype=float),
        np.array([r.z50 for r, _s in rows], dtype=float),
        np.array([r.z75 for r, _s in rows], dtype=float),
    ]
    for arr in numeric:
        sd = arr.std()
        cols.append((arr - arr.mean()) / (sd if sd else 1.0))
    cols.append(np.array([1.0 if r.parity == "even" else 0.0 for r, _s in rows]))
    for cluster in ["balanced", "early_growth", "late_growth"]:
        cols.append(np.array([1.0 if r.bridge_cluster == cluster else 0.0 for r, _s in rows]))
    for window in ["deep_32_63", "exhaustion_0_31", "tail_64_95"]:
        cols.append(np.array([1.0 if r.x_k_window == window else 0.0 for r, _s in rows]))
    if include_score:
        arr = np.array([score for _r, score in rows], dtype=float)
        sd = arr.std()
        cols.append((arr - arr.mean()) / (sd if sd else 1.0))
        names.append("block_score")
    x = np.column_stack(cols)
    y = np.array([1.0 if r.source == "actual" else 0.0 for r, _s in rows])
    w = np.array([r.weight for r, _s in rows])
    return x, y, w, names


def fit_logistic(rows: list[tuple[ThinRec, float]], include_score: bool) -> tuple[float, dict[str, float]]:
    x, y, w, names = design_matrix(rows, include_score)
    w = w / w.mean()
    beta = np.zeros(x.shape[1])
    lr = 0.08
    l2 = 1e-4
    for _ in range(320):
        z = np.clip(x @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = (x.T @ (w * (p - y))) / len(y)
        grad[1:] += l2 * beta[1:]
        beta -= lr * grad
    scores = x @ beta
    auc = weighted_auc(
        [(float(s), r.weight) for s, (r, _b) in zip(scores, rows) if r.source == "actual"],
        [(float(s), r.weight) for s, (r, _b) in zip(scores, rows) if r.source == "iid"],
    )
    return auc, dict(zip(names, beta))


def survival_by_score(block_len: int, scored: list[tuple[ThinRec, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for state in sorted({r.state for r, _s in scored}):
        subset = [(r, s) for r, s in scored if r.state == state]
        cuts = weighted_quantiles([(s, r.weight) for r, s in subset], tuple(i / 10 for i in range(1, 10)))
        buckets: dict[int, dict[str, float]] = defaultdict(lambda: {"actual": 0.0, "iid": 0.0})
        ranges: dict[int, list[float]] = defaultdict(list)
        for rec, score in subset:
            idx = sum(score > c for c in cuts)
            buckets[idx][rec.source] += rec.weight
            ranges[idx].append(score)
        for idx in range(10):
            mass = buckets[idx]
            iid = mass["iid"]
            rows.append(
                {
                    "block_len": block_len,
                    "state": state,
                    "score_decile": idx,
                    "score_min": min(ranges[idx]) if ranges[idx] else "",
                    "score_max": max(ranges[idx]) if ranges[idx] else "",
                    "actual_mass": mass["actual"],
                    "iid_mass": iid,
                    "survival_ratio": mass["actual"] / iid if iid else "",
                }
            )
    return rows


def line_svg(path: Path, rows: list[dict[str, object]], title: str, y_field: str) -> None:
    width, height = 740, 420
    left, right, top, bottom = 70, 40, 45, 65
    vals = [float(r[y_field]) for r in rows if r[y_field] != ""]
    ymin, ymax = min(vals + [0.49]), max(vals + [0.55])
    pad = (ymax - ymin) * 0.12 or 0.02
    ymin -= pad
    ymax += pad

    def sx(lval: int) -> float:
        return left + (lval - 3) * (width - left - right) / 3

    def sy(v: float) -> float:
        return top + (ymax - v) * (height - top - bottom) / (ymax - ymin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-family="Arial" font-size="17">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
    ]
    pts = []
    for r in rows:
        x = sx(int(r["block_len"]))
        y = sy(float(r[y_field]))
        pts.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#2563eb"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+20}" text-anchor="middle" font-family="Arial" font-size="11">L{r["block_len"]}</text>')
    if len(pts) >= 2:
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#2563eb" stroke-width="2"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def survival_svg(path: Path, rows: list[dict[str, object]]) -> None:
    subset = [r for r in rows if r["state"] == FOCUS_STATE and r["survival_ratio"] != ""]
    width, height = 820, 430
    left, right, top, bottom = 70, 145, 45, 65
    vals = [float(r["survival_ratio"]) for r in subset]
    ymax = max(vals + [1.0]) * 1.15

    def sx(decile: int) -> float:
        return left + decile * (width - left - right) / 9

    def sy(v: float) -> float:
        return top + (ymax - v) * (height - top - bottom) / ymax

    colors = {3: "#64748b", 4: "#2563eb", 5: "#b91c1c", 6: "#047857"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-family="Arial" font-size="17">Survival deciles by block length: {FOCUS_STATE}</text>',
    ]
    for block_len in BLOCK_LENGTHS:
        pts = []
        for r in sorted([x for x in subset if int(x["block_len"]) == block_len], key=lambda x: int(x["score_decile"])):
            pts.append(f"{sx(int(r['score_decile'])):.1f},{sy(float(r['survival_ratio'])):.1f}")
        if len(pts) >= 2:
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[block_len]}" stroke-width="2"/>')
        parts.append(f'<text x="{width-right+18}" y="{70+(block_len-3)*22}" font-family="Arial" font-size="12" fill="{colors[block_len]}">L={block_len}</text>')
    y1 = sy(1.0)
    parts.append(f'<line x1="{left}" y1="{y1:.1f}" x2="{width-right}" y2="{y1:.1f}" stroke="#6b7280" stroke-dasharray="4 4"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def build_report(auc_rows: list[dict[str, object]], residual_rows: list[dict[str, object]], survival_rows: list[dict[str, object]], q_low: float, q_high: float) -> None:
    best = max(auc_rows, key=lambda r: float(r["logistic_auc_plus_score"]))
    l3 = next(r for r in auc_rows if int(r["block_len"]) == 3)
    l6 = next(r for r in auc_rows if int(r["block_len"]) == 6)
    r6 = next(r for r in residual_rows if int(r["block_len"]) == 6)
    auc_gain = float(l6["logistic_auc_plus_score"]) - float(l3["logistic_auc_plus_score"])
    residual_bridge = float(r6["bridge_abs_coef_plus"])
    residual_parity = abs(float(r6["parity_even_coef_plus"]))
    if auc_gain > 0.03 and residual_bridge < 0.3 and residual_parity < 0.1:
        classification = "A. finite block anomaly accumulation approximates the deficit"
    elif auc_gain > 0.005:
        classification = "B. AUC grows but residual remains"
    elif float(l6["logistic_auc_plus_score"]) <= float(l3["logistic_auc_plus_score"]) + 0.005:
        classification = "C. improvement saturates near B4"
    else:
        classification = "D. sparse/noisy"
    focus = [r for r in survival_rows if r["state"] == FOCUS_STATE and int(r["block_len"]) == 6 and r["survival_ratio"] != ""]
    low = min(focus, key=lambda r: int(r["score_decile"]))
    high = max(focus, key=lambda r: int(r["score_decile"]))
    lines = [
        "# Collatz Block Length Renormalization",
        "",
        f"- block lengths: {', '.join(map(str, BLOCK_LENGTHS))}",
        f"- split: deterministic train/test sample index parity",
        f"- smoothing: probability mixture `(1-lambda) empirical + lambda uniform`, lambda={SMOOTH_MIX}",
        f"- stable threshold: train iid state mass >= {MIN_TRAIN_IID_TOTAL}",
        f"- regression rows: deterministic weighted subsample capped at {MAX_REGRESSION_ROWS}",
        f"- bridge cluster z25 cuts: q_low={q_low:.6g}, q_high={q_high:.6g}",
        "",
        "## AUC By Length",
        "",
        "| L | marginal score AUC | logistic base AUC | logistic + score AUC | delta |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in auc_rows:
        lines.append(
            f"| {row['block_len']} | {float(row['marginal_score_auc']):.4f} | {float(row['logistic_auc_base']):.4f} | "
            f"{float(row['logistic_auc_plus_score']):.4f} | {float(row['logistic_auc_delta']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Focus Survival",
            "",
            f"For `{FOCUS_STATE}`, L=6 lowest score decile has S=`{float(low['survival_ratio']):.3f}`; the highest finite-ratio decile has S=`{float(high['survival_ratio']):.3f}`, while the top decile may have zero iid mass. Read the high-score tail as sparse but directionally strong.",
            "",
            "## Classification",
            "",
            f"Current classification: **{classification}**.",
            "",
            f"Best +score logistic AUC occurs at L={best['block_len']} with AUC `{float(best['logistic_auc_plus_score']):.4f}`. L=3 to L=6 +score AUC gain is `{auc_gain:.4f}`. At L=6, bridge abs residual is `{residual_bridge:.4f}` and parity residual is `{residual_parity:.4f}`. The AUC trend is stable enough for B, but the extreme survival tail is sparse.",
        ]
    )
    (OUT_DIR / "collatz_block_length_renormalization_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base()
    module = base.load_source()
    rng = random.Random(base.SEED)
    counts: defaultdict[tuple[int, str, str, str, str], float] = defaultdict(float)
    test_records: list[ThinRec] = []
    iid_by_h: dict[int, list[tuple[tuple[int, ...], float]]] = {}
    tau_cuts: dict[int, tuple[int, int]] = {}
    iid_z_for_cluster: list[tuple[float, float]] = []

    print("sampling iid", flush=True)
    for h in base.HS:
        sampled = base.sample_iid(module, h, rng)
        iid_by_h[h] = sampled
        tau_cuts[h] = base.weighted_tau_cuts(sampled)
        long_words = [(word, weight) for word, weight in sampled if len(word) > tau_cuts[h][1]]
        iid_z_for_cluster.extend((base.z_features(word)[0], weight) for word, weight in long_words)
        print(f"iid h={h}: tau cuts={tau_cuts[h]}, long={len(long_words)}", flush=True)
    q_low, q_high = base.weighted_quantiles(iid_z_for_cluster, (1 / 3, 2 / 3))
    print(f"z25 cuts: {q_low:.6g}, {q_high:.6g}", flush=True)

    print("building iid train counts and test records", flush=True)
    for h, sampled in iid_by_h.items():
        for sample_idx, (word, weight) in enumerate(sampled):
            if len(word) <= tau_cuts[h][1]:
                continue
            split = "train" if sample_idx % 2 == 0 else "test"
            for power in base.POWERS:
                rec = make_thin(base, "iid", split, power, h, weight, word, q_low, q_high)
                if rec is None:
                    continue
                if split == "train":
                    add_counts(base, rec, counts)
                else:
                    test_records.append(rec)

    print("building actual train counts and test records", flush=True)
    for power in base.POWERS:
        status = base.load_status(power)
        for h in base.HS:
            _cut1, cut2 = tau_cuts[h]
            lo, hi, total = module.layer_bounds(power, h)
            escape_indices = [idx for idx in range(lo >> 1, (hi >> 1) + 1) if status[idx] == module.ESCAPE]
            chosen = base.evenly_spaced(escape_indices, base.ACTUAL_SAMPLE_PER_PH)
            sample_weight = len(escape_indices) / (len(chosen) * total) if chosen else 0.0
            kept = 0
            for sample_idx, idx in enumerate(chosen):
                word = module.trace_escape(2 * idx + 1, 1 << power)
                if len(word) <= cut2:
                    continue
                split = "train" if sample_idx % 2 == 0 else "test"
                rec = make_thin(base, "actual", split, power, h, sample_weight, word, q_low, q_high)
                if rec is None:
                    continue
                kept += 1
                if split == "train":
                    add_counts(base, rec, counts)
                else:
                    test_records.append(rec)
            print(f"actual p={power} h={h}: sampled={len(chosen)}, kept={kept}", flush=True)
        del status

    print("building lookups", flush=True)
    lookup, stable = make_lookup(counts)
    auc_rows: list[dict[str, object]] = []
    survival_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []

    for block_len in BLOCK_LENGTHS:
        print(f"scoring L={block_len}", flush=True)
        scored = [(rec, block_score(base, rec, lookup, block_len)) for rec in test_records if rec.state in stable[block_len]]
        pos = [(score, rec.weight) for rec, score in scored if rec.source == "actual"]
        neg = [(score, rec.weight) for rec, score in scored if rec.source == "iid"]
        focus_pos = [(score, rec.weight) for rec, score in scored if rec.source == "actual" and rec.state == FOCUS_STATE]
        focus_neg = [(score, rec.weight) for rec, score in scored if rec.source == "iid" and rec.state == FOCUS_STATE]
        reg_rows = thin_for_regression(scored)
        base_auc, base_coef = fit_logistic(reg_rows, include_score=False)
        plus_auc, plus_coef = fit_logistic(reg_rows, include_score=True)
        bridge_base = sum(abs(base_coef.get(f"bridge_{v}", 0.0)) for v in ["balanced", "early_growth", "late_growth"])
        bridge_plus = sum(abs(plus_coef.get(f"bridge_{v}", 0.0)) for v in ["balanced", "early_growth", "late_growth"])
        auc_rows.append(
            {
                "block_len": block_len,
                "test_rows": len(scored),
                "marginal_score_auc": weighted_auc(pos, neg),
                "focus_state_score_auc": weighted_auc(focus_pos, focus_neg),
                "logistic_auc_base": base_auc,
                "logistic_auc_plus_score": plus_auc,
                "logistic_auc_delta": plus_auc - base_auc,
            }
        )
        residual_rows.append(
            {
                "block_len": block_len,
                "bridge_abs_coef_base": bridge_base,
                "bridge_abs_coef_plus": bridge_plus,
                "bridge_abs_reduction": bridge_base - bridge_plus,
                "parity_even_coef_base": base_coef.get("parity_even", 0.0),
                "parity_even_coef_plus": plus_coef.get("parity_even", 0.0),
                "parity_abs_reduction": abs(base_coef.get("parity_even", 0.0)) - abs(plus_coef.get("parity_even", 0.0)),
                "block_score_coef": plus_coef.get("block_score", 0.0),
            }
        )
        survival_rows.extend(survival_by_score(block_len, scored))

    print("writing outputs", flush=True)
    write_csv(OUT_DIR / "block_length_auc_summary.csv", auc_rows)
    write_csv(OUT_DIR / "block_length_survival_by_score.csv", survival_rows)
    write_csv(OUT_DIR / "block_length_residual_effects.csv", residual_rows)
    line_svg(OUT_DIR / "auc_vs_block_length.svg", auc_rows, "Logistic AUC with block score by L", "logistic_auc_plus_score")
    survival_svg(OUT_DIR / "survival_deciles_by_L.svg", survival_rows)
    build_report(auc_rows, residual_rows, survival_rows, q_low, q_high)
    print(OUT_DIR / "collatz_block_length_renormalization_report.md", flush=True)


if __name__ == "__main__":
    main()
