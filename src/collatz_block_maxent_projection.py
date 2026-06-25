from __future__ import annotations

import csv
import importlib.util
import math
import random
import sys
from collections import defaultdict
from pathlib import Path


RENORM_PATH = Path("work/collatz_block_length_renormalization.py")
OUT_DIR = Path("outputs")
BLOCK_LENGTHS = [3, 4, 5, 6]
REGULARIZATION = [0.0, 0.5, 0.75, 0.9]
ITERATIONS = 2
STEP = 0.35
SMOOTH = 0.02
MAX_EVAL_IID = 40_000
FOCUS_STATE = "late_growth|tail_64_95|even"
EPS = 1e-300


def load_renorm():
    spec = importlib.util.spec_from_file_location("collatz_block_length_renormalization", RENORM_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {RENORM_PATH}")
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


def thin(items: list[object], limit: int) -> list[object]:
    if len(items) <= limit:
        return items
    stride = len(items) / limit
    return [items[int(i * stride)] for i in range(limit)]


def split_state(state: str) -> tuple[str, str, str]:
    return tuple(state.split("|"))  # type: ignore[return-value]


def aggregate(full: dict[str, float], level: str) -> dict[str, float]:
    out: defaultdict[str, float] = defaultdict(float)
    for state, mass in full.items():
        cluster, xk, parity = split_state(state)
        if level == "cluster":
            key = cluster
        elif level == "parity":
            key = parity
        elif level == "cluster_xK":
            key = f"{cluster}|{xk}"
        elif level == "full":
            key = state
        else:
            raise ValueError(level)
        out[key] += mass
    return dict(out)


def rmse(actual: dict[str, float], pred: dict[str, float], keys: list[str]) -> float:
    return math.sqrt(sum((pred.get(k, 0.0) - actual.get(k, 0.0)) ** 2 for k in keys) / max(1, len(keys)))


def normalize(values: dict[str, float], keys: list[str]) -> dict[str, float]:
    total = sum(values.get(k, 0.0) for k in keys)
    if total <= 0:
        return {k: 0.0 for k in keys}
    return {k: values.get(k, 0.0) / total for k in keys}


def kl_js(actual: dict[str, float], pred: dict[str, float], keys: list[str]) -> tuple[float, float]:
    p = normalize(actual, keys)
    q = normalize(pred, keys)
    m = {k: 0.5 * (p[k] + q[k]) for k in keys}
    kl = sum(p[k] * math.log2(p[k] / max(q[k], EPS)) for k in keys if p[k] > 0)
    js = 0.5 * sum(p[k] * math.log2(p[k] / max(m[k], EPS)) for k in keys if p[k] > 0)
    js += 0.5 * sum(q[k] * math.log2(q[k] / max(m[k], EPS)) for k in keys if q[k] > 0)
    return kl, js


def features(base, rec, block_len: int) -> list[tuple[str, str, str]]:
    cats = [base.k_cat(k) for k in rec.word]
    tau = len(cats)
    if tau < block_len:
        return []
    out = []
    for i in range(tau - block_len + 1):
        out.append((rec.state, base.u_bin(i, tau), ",".join(cats[i : i + block_len])))
    return out


def build_target_probs(counts, support: list[str], block_len: int) -> dict[tuple[str, str, str], float]:
    targets = {}
    states_ub = sorted({(state, ub) for bl, state, ub, _src, _block in counts if bl == block_len})
    v = len(support)
    for state, ub in states_ub:
        total = sum(counts[(block_len, state, ub, "actual", b)] for b in support)
        for block in support:
            emp = counts[(block_len, state, ub, "actual", block)] / total if total else 1.0 / v
            targets[(state, ub, block)] = (1.0 - SMOOTH) * emp + SMOOTH / v
    return targets


def predict_feature_probs(base, iid_records, theta, block_len: int, support: list[str]) -> tuple[dict[tuple[str, str, str], float], dict[str, float]]:
    counts: defaultdict[tuple[str, str, str], float] = defaultdict(float)
    totals: defaultdict[tuple[str, str], float] = defaultdict(float)
    state_mass: defaultdict[str, float] = defaultdict(float)
    for rec in iid_records:
        feats = features(base, rec, block_len)
        score = sum(theta.get(f, 0.0) for f in feats)
        weight = rec.weight * (2.0 ** max(-80.0, min(80.0, score)))
        state_mass[rec.state] += weight
        for f in feats:
            counts[f] += weight
            totals[(f[0], f[1])] += weight
    probs = {}
    v = len(support)
    for state, ub in sorted({(f[0], f[1]) for f in counts}):
        total = totals[(state, ub)]
        for block in support:
            emp = counts[(state, ub, block)] / total if total else 1.0 / v
            probs[(state, ub, block)] = (1.0 - SMOOTH) * emp + SMOOTH / v
    return probs, dict(state_mass)


def scale_to_actual(pred: dict[str, float], actual_total: float) -> dict[str, float]:
    total = sum(pred.values())
    scale = actual_total / total if total > 0 else 0.0
    return {k: v * scale for k, v in pred.items()}


def line_svg(path: Path, rows: list[dict[str, object]], title: str, y_field: str) -> None:
    width, height = 780, 430
    left, right, top, bottom = 70, 150, 45, 65
    vals = [float(r[y_field]) for r in rows]
    ymax = max(vals + [1e-6]) * 1.15
    colors = {3: "#64748b", 4: "#2563eb", 5: "#b91c1c", 6: "#047857"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-family="Arial" font-size="17">{title}</text>',
    ]

    def sx(reg: float) -> float:
        return left + reg / 0.9 * (width - left - right)

    def sy(v: float) -> float:
        return top + (ymax - v) * (height - top - bottom) / ymax

    for block_len in BLOCK_LENGTHS:
        pts = []
        for r in sorted([x for x in rows if int(x["block_len"]) == block_len], key=lambda x: float(x["regularization"])):
            pts.append(f"{sx(float(r['regularization'])):.1f},{sy(float(r[y_field])):.1f}")
        if len(pts) >= 2:
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[block_len]}" stroke-width="2"/>')
        parts.append(f'<text x="{width-right+18}" y="{70+(block_len-3)*22}" font-family="Arial" font-size="12" fill="{colors[block_len]}">L={block_len}</text>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def compare_svg(path: Path, raw_rmse: float, best_maxent: float) -> None:
    width, height = 520, 340
    vals = [raw_rmse, best_maxent]
    ymax = max(vals) * 1.2
    labels = ["raw/damped best", "maxent best"]
    colors = ["#64748b", "#2563eb"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="260" y="25" text-anchor="middle" font-family="Arial" font-size="17">RMSE comparison</text>',
    ]
    for i, v in enumerate(vals):
        x = 120 + i * 150
        h = v / ymax * 220
        y = 280 - h
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="80" height="{h:.1f}" fill="{colors[i]}"/>')
        parts.append(f'<text x="{x+40}" y="{y-6:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{v:.6g}</text>')
        parts.append(f'<text x="{x+40}" y="306" text-anchor="middle" font-family="Arial" font-size="10">{labels[i]}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def build_report(summary_rows, focus_rows, raw_best_rmse: float) -> None:
    best = min(summary_rows, key=lambda r: float(r["full_rmse"]))
    best_focus = min(focus_rows, key=lambda r: abs(float(r["prediction_error"])))
    raw_better = raw_best_rmse <= float(best["full_rmse"])
    if raw_better:
        classification = "C. maxent no better than raw/damped"
    elif float(best["full_rmse"]) < 0.5 * raw_best_rmse:
        classification = "A. maxent block projection reproduces most state structure"
    else:
        classification = "B. maxent improves but bridge/parity residual remains"
    lines = [
        "# Collatz Maximum-Entropy Block Projection Test",
        "",
        "- projection: approximate regularized IPF/exponential-family update on iid test measure",
        f"- block lengths: {', '.join(map(str, BLOCK_LENGTHS))}",
        f"- regularization sweep: {', '.join(map(str, REGULARIZATION))}",
        f"- iterations per fit: {ITERATIONS}",
        f"- deterministic iid evaluation cap: {MAX_EVAL_IID}",
        "",
        "## Best Fit",
        "",
        f"Best maxent-style fit is L={best['block_len']}, regularization={best['regularization']}, RMSE=`{float(best['full_rmse']):.6g}`, JS=`{float(best['full_js_bits']):.6g}`.",
        f"Best raw/damped finite-block RMSE from the previous test was `{raw_best_rmse:.6g}`.",
        f"Focus state best: L={best_focus['block_len']}, regularization={best_focus['regularization']}, actual survival `{float(best_focus['actual_survival_ratio']):.6g}`, predicted survival `{float(best_focus['predicted_survival_ratio']):.6g}`.",
        "",
        "## Classification",
        "",
        f"Current classification: **{classification}**.",
        "",
        "This is an approximate projection, not a full exact IPF over all words. The test is meant to distinguish overcounted raw products from a regularized finite-block exponential family.",
    ]
    (OUT_DIR / "collatz_block_maxent_projection_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    renorm = load_renorm()
    base = renorm.load_base()
    module = base.load_source()
    rng = random.Random(base.SEED)
    counts: defaultdict[tuple[int, str, str, str, str], float] = defaultdict(float)
    actual_test = []
    iid_test = []
    iid_by_h = {}
    tau_cuts = {}
    iid_z_for_cluster = []

    print("sampling iid", flush=True)
    for h in base.HS:
        sampled = base.sample_iid(module, h, rng)
        iid_by_h[h] = sampled
        tau_cuts[h] = base.weighted_tau_cuts(sampled)
        long_words = [(word, weight) for word, weight in sampled if len(word) > tau_cuts[h][1]]
        iid_z_for_cluster.extend((base.z_features(word)[0], weight) for word, weight in long_words)
        print(f"iid h={h}: tau cuts={tau_cuts[h]}, long={len(long_words)}", flush=True)
    q_low, q_high = base.weighted_quantiles(iid_z_for_cluster, (1 / 3, 2 / 3))

    print("building records", flush=True)
    for h, sampled in iid_by_h.items():
        for sample_idx, (word, weight) in enumerate(sampled):
            if len(word) <= tau_cuts[h][1]:
                continue
            split = "train" if sample_idx % 2 == 0 else "test"
            for power in base.POWERS:
                rec = renorm.make_thin(base, "iid", split, power, h, weight, word, q_low, q_high)
                if rec is None:
                    continue
                if split == "train":
                    renorm.add_counts(base, rec, counts)
                else:
                    iid_test.append(rec)
    iid_eval = thin(iid_test, MAX_EVAL_IID)

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
                rec = renorm.make_thin(base, "actual", split, power, h, sample_weight, word, q_low, q_high)
                if rec is None:
                    continue
                kept += 1
                if split == "train":
                    renorm.add_counts(base, rec, counts)
                else:
                    actual_test.append(rec)
            print(f"actual p={power} h={h}: sampled={len(chosen)}, kept={kept}", flush=True)
        del status

    print("building stable state set", flush=True)
    _lookup, stable = renorm.make_lookup(counts)
    all_stable = set.intersection(*(stable[l] for l in BLOCK_LENGTHS))
    actual_full: defaultdict[str, float] = defaultdict(float)
    iid_full: defaultdict[str, float] = defaultdict(float)
    for rec in actual_test:
        if rec.state in all_stable:
            actual_full[rec.state] += rec.weight
    iid_full_all: defaultdict[str, float] = defaultdict(float)
    for rec in iid_test:
        if rec.state in all_stable:
            iid_full_all[rec.state] += rec.weight
    for rec in iid_eval:
        if rec.state in all_stable:
            iid_full[rec.state] += rec.weight
    full_keys = sorted(k for k in all_stable if iid_full.get(k, 0.0) > 0 and iid_full_all.get(k, 0.0) > 0)
    actual_full = defaultdict(float, {k: actual_full.get(k, 0.0) for k in full_keys})
    actual_total = sum(actual_full.values())

    summary_rows = []
    state_rows = []
    focus_rows = []
    residual_rows = []

    for block_len in BLOCK_LENGTHS:
        print(f"projecting L={block_len}", flush=True)
        sup = renorm.SUPPORT[block_len]
        target = build_target_probs(counts, sup, block_len)
        for reg in REGULARIZATION:
            theta: dict[tuple[str, str, str], float] = {}
            for _ in range(ITERATIONS):
                pred_probs, _state_mass = predict_feature_probs(base, iid_eval, theta, block_len, sup)
                for feat, targ in target.items():
                    pred = pred_probs.get(feat)
                    if pred is None:
                        continue
                    delta = max(-3.0, min(3.0, math.log2(targ / max(pred, EPS))))
                    theta[feat] = (1.0 - reg) * theta.get(feat, 0.0) + STEP * (1.0 - reg) * delta
            _pred_probs, raw_state = predict_feature_probs(base, iid_eval, theta, block_len, sup)
            pred_full = scale_to_actual({k: raw_state.get(k, 0.0) for k in full_keys}, actual_total)
            keys = full_keys
            full_rmse = rmse(actual_full, pred_full, keys)
            full_kl, full_js = kl_js(actual_full, pred_full, keys)
            cluster_keys = sorted(set(aggregate(actual_full, "cluster")) | set(aggregate(pred_full, "cluster")))
            parity_keys = sorted(set(aggregate(actual_full, "parity")) | set(aggregate(pred_full, "parity")))
            cluster_rmse = rmse(aggregate(actual_full, "cluster"), aggregate(pred_full, "cluster"), cluster_keys)
            parity_rmse = rmse(aggregate(actual_full, "parity"), aggregate(pred_full, "parity"), parity_keys)
            focus_actual = actual_full.get(FOCUS_STATE, 0.0)
            focus_iid = iid_full_all.get(FOCUS_STATE, 0.0)
            focus_pred = pred_full.get(FOCUS_STATE, 0.0)
            summary_rows.append(
                {
                    "block_len": block_len,
                    "regularization": reg,
                    "full_rmse": full_rmse,
                    "full_kl_bits": full_kl,
                    "full_js_bits": full_js,
                    "cluster_rmse": cluster_rmse,
                    "parity_rmse": parity_rmse,
                    "theta_nonzero": len(theta),
                }
            )
            focus_rows.append(
                {
                    "block_len": block_len,
                    "regularization": reg,
                    "state": FOCUS_STATE,
                    "actual_mass": focus_actual,
                    "iid_mass": focus_iid,
                    "predicted_mass": focus_pred,
                    "prediction_error": focus_pred - focus_actual,
                    "actual_survival_ratio": focus_actual / focus_iid if focus_iid else "",
                    "predicted_survival_ratio": focus_pred / focus_iid if focus_iid else "",
                }
            )
            residual_rows.append(
                {
                    "block_len": block_len,
                    "regularization": reg,
                    "cluster_rmse": cluster_rmse,
                    "parity_rmse": parity_rmse,
                }
            )
            for state in keys:
                state_rows.append(
                    {
                        "block_len": block_len,
                        "regularization": reg,
                        "state": state,
                        "actual_mass": actual_full.get(state, 0.0),
                        "iid_eval_mass": iid_full.get(state, 0.0),
                        "iid_test_mass": iid_full_all.get(state, 0.0),
                        "predicted_mass": pred_full.get(state, 0.0),
                        "prediction_error": pred_full.get(state, 0.0) - actual_full.get(state, 0.0),
                    }
                )

    raw_rows = list(csv.DictReader(open(OUT_DIR / "block_reweighting_alpha_sweep.csv", encoding="utf-8-sig")))
    raw_best_rmse = min(float(r["full_rmse"]) for r in raw_rows)
    print("writing outputs", flush=True)
    write_csv(OUT_DIR / "block_maxent_fit_summary.csv", summary_rows)
    write_csv(OUT_DIR / "block_maxent_state_fit.csv", state_rows)
    write_csv(OUT_DIR / "block_maxent_regularization_sweep.csv", summary_rows)
    write_csv(OUT_DIR / "block_maxent_focus_state.csv", focus_rows)
    compare_svg(OUT_DIR / "maxent_vs_raw_rmse.svg", raw_best_rmse, min(float(r["full_rmse"]) for r in summary_rows))
    line_svg(OUT_DIR / "residuals_vs_regularization.svg", residual_rows, "Parity residual RMSE vs regularization", "parity_rmse")
    line_svg(OUT_DIR / "focus_state_maxent_fit.svg", focus_rows, "Focus predicted survival vs regularization", "predicted_survival_ratio")
    build_report(summary_rows, focus_rows, raw_best_rmse)
    print(OUT_DIR / "collatz_block_maxent_projection_report.md", flush=True)


if __name__ == "__main__":
    main()
