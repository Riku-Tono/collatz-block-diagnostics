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
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
FOCUS_STATE = "late_growth|tail_64_95|even"
STABLE_IID_MASS = 1e-7
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


def split_state(state: str) -> tuple[str, str, str]:
    cluster, xk, parity = state.split("|")
    return cluster, xk, parity


def add_mass(bucket: defaultdict[str, float], key: str, value: float) -> None:
    bucket[key] += value


def aggregate(full: dict[str, float], level: str) -> dict[str, float]:
    out: defaultdict[str, float] = defaultdict(float)
    for state, mass in full.items():
        cluster, xk, parity = split_state(state)
        if level == "cluster":
            key = cluster
        elif level == "cluster_xK":
            key = f"{cluster}|{xk}"
        elif level == "cluster_xK_parity":
            key = state
        elif level == "parity":
            key = parity
        else:
            raise ValueError(level)
        out[key] += mass
    return dict(out)


def normalize_on_keys(values: dict[str, float], keys: list[str], total: float | None = None) -> dict[str, float]:
    if total is None:
        total = sum(values.get(k, 0.0) for k in keys)
    if total <= 0:
        return {k: 0.0 for k in keys}
    return {k: values.get(k, 0.0) / total for k in keys}


def rmse(actual: dict[str, float], pred: dict[str, float], keys: list[str]) -> float:
    return math.sqrt(sum((pred.get(k, 0.0) - actual.get(k, 0.0)) ** 2 for k in keys) / len(keys))


def kl_js(actual: dict[str, float], pred: dict[str, float], keys: list[str]) -> tuple[float, float]:
    p = normalize_on_keys(actual, keys)
    q = normalize_on_keys(pred, keys)
    m = {k: 0.5 * (p[k] + q[k]) for k in keys}
    kl = sum(p[k] * math.log2(p[k] / max(q[k], EPS)) for k in keys if p[k] > 0)
    js = 0.5 * sum(p[k] * math.log2(p[k] / max(m[k], EPS)) for k in keys if p[k] > 0)
    js += 0.5 * sum(q[k] * math.log2(q[k] / max(m[k], EPS)) for k in keys if q[k] > 0)
    return kl, js


def scaled_reweight(
    iid_scored: list[tuple[object, float]],
    alpha: float,
    states: set[str],
    actual_total: float,
) -> tuple[dict[str, float], float]:
    raw: defaultdict[str, float] = defaultdict(float)
    for rec, score in iid_scored:
        if rec.state not in states:
            continue
        weight = rec.weight * (2.0 ** (alpha * score))
        raw[rec.state] += weight
    raw_total = sum(raw.values())
    scale = actual_total / raw_total if raw_total > 0 else 0.0
    return {k: v * scale for k, v in raw.items()}, scale


def state_fit_rows_for(
    block_len: int,
    alpha: float,
    actual_full: dict[str, float],
    iid_full: dict[str, float],
    pred_full: dict[str, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for level in ["cluster", "cluster_xK", "cluster_xK_parity"]:
        actual = aggregate(actual_full, level)
        iid = aggregate(iid_full, level)
        pred = aggregate(pred_full, level)
        for key in sorted(set(actual) | set(iid) | set(pred)):
            iid_mass = iid.get(key, 0.0)
            rows.append(
                {
                    "block_len": block_len,
                    "alpha": alpha,
                    "level": level,
                    "state": key,
                    "actual_mass": actual.get(key, 0.0),
                    "iid_mass": iid_mass,
                    "predicted_mass": pred.get(key, 0.0),
                    "actual_survival_ratio": actual.get(key, 0.0) / iid_mass if iid_mass else "",
                    "predicted_survival_ratio": pred.get(key, 0.0) / iid_mass if iid_mass else "",
                    "prediction_error": pred.get(key, 0.0) - actual.get(key, 0.0),
                }
            )
    return rows


def simple_heat_svg(path: Path, rows: list[dict[str, object]]) -> None:
    data = [r for r in rows if r["level"] == "cluster_xK_parity" and int(r["block_len"]) == 6 and float(r["alpha"]) == 1.0]
    data = sorted(data, key=lambda r: abs(float(r["prediction_error"])), reverse=True)[:18]
    width, height = 920, 520
    left, top = 260, 50
    row_h = 24
    x0, x1 = left, width - 60
    max_mass = max([float(r["actual_mass"]) for r in data] + [float(r["predicted_mass"]) for r in data] + [1e-12])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-family="Arial" font-size="17">Predicted vs actual state mass, L=6 alpha=1</text>',
    ]
    for i, r in enumerate(data):
        y = top + i * row_h
        actual_w = float(r["actual_mass"]) / max_mass * (x1 - x0) * 0.45
        pred_w = float(r["predicted_mass"]) / max_mass * (x1 - x0) * 0.45
        parts.append(f'<text x="{left-8}" y="{y+15}" text-anchor="end" font-family="Arial" font-size="10">{r["state"]}</text>')
        parts.append(f'<rect x="{x0}" y="{y+3}" width="{actual_w:.1f}" height="8" fill="#b91c1c"/>')
        parts.append(f'<rect x="{x0}" y="{y+13}" width="{pred_w:.1f}" height="8" fill="#2563eb"/>')
    parts.append(f'<text x="{x0}" y="{height-20}" font-family="Arial" font-size="11" fill="#b91c1c">actual</text>')
    parts.append(f'<text x="{x0+70}" y="{height-20}" font-family="Arial" font-size="11" fill="#2563eb">predicted</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def rmse_svg(path: Path, rows: list[dict[str, object]]) -> None:
    width, height = 760, 420
    left, right, top, bottom = 70, 135, 45, 65
    vals = [float(r["full_rmse"]) for r in rows]
    ymax = max(vals + [1e-6]) * 1.15
    colors = {3: "#64748b", 4: "#2563eb", 5: "#b91c1c", 6: "#047857"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-family="Arial" font-size="17">Full-state RMSE by L and alpha</text>',
    ]

    def sx(alpha: float) -> float:
        return left + alpha * (width - left - right)

    def sy(v: float) -> float:
        return top + (ymax - v) * (height - top - bottom) / ymax

    for block_len in BLOCK_LENGTHS:
        pts = []
        for r in sorted([x for x in rows if int(x["block_len"]) == block_len], key=lambda x: float(x["alpha"])):
            pts.append(f"{sx(float(r['alpha'])):.1f},{sy(float(r['full_rmse'])):.1f}")
        if len(pts) >= 2:
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[block_len]}" stroke-width="2"/>')
        parts.append(f'<text x="{width-right+18}" y="{70+(block_len-3)*22}" font-family="Arial" font-size="12" fill="{colors[block_len]}">L={block_len}</text>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def focus_svg(path: Path, rows: list[dict[str, object]]) -> None:
    width, height = 760, 420
    left, right, top, bottom = 70, 135, 45, 65
    vals = [float(r["predicted_survival_ratio"]) for r in rows if r["predicted_survival_ratio"] != ""]
    vals += [float(r["actual_survival_ratio"]) for r in rows if r["actual_survival_ratio"] != ""]
    ymax = max(vals + [1.0]) * 1.15
    colors = {3: "#64748b", 4: "#2563eb", 5: "#b91c1c", 6: "#047857"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-family="Arial" font-size="17">Focus state S_pred by alpha</text>',
    ]

    def sx(alpha: float) -> float:
        return left + alpha * (width - left - right)

    def sy(v: float) -> float:
        return top + (ymax - v) * (height - top - bottom) / ymax

    actual = next(float(r["actual_survival_ratio"]) for r in rows if r["actual_survival_ratio"] != "")
    y_actual = sy(actual)
    parts.append(f'<line x1="{left}" y1="{y_actual:.1f}" x2="{width-right}" y2="{y_actual:.1f}" stroke="#111827" stroke-dasharray="4 4"/>')
    for block_len in BLOCK_LENGTHS:
        pts = []
        for r in sorted([x for x in rows if int(x["block_len"]) == block_len and x["predicted_survival_ratio"] != ""], key=lambda x: float(x["alpha"])):
            pts.append(f"{sx(float(r['alpha'])):.1f},{sy(float(r['predicted_survival_ratio'])):.1f}")
        if len(pts) >= 2:
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[block_len]}" stroke-width="2"/>')
        parts.append(f'<text x="{width-right+18}" y="{70+(block_len-3)*22}" font-family="Arial" font-size="12" fill="{colors[block_len]}">L={block_len}</text>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def build_report(alpha_rows: list[dict[str, object]], focus_rows: list[dict[str, object]], residual_rows: list[dict[str, object]]) -> None:
    best = min(alpha_rows, key=lambda r: float(r["full_rmse"]))
    l6_best = min([r for r in alpha_rows if int(r["block_len"]) == 6], key=lambda r: float(r["full_rmse"]))
    l6_alpha0 = next(r for r in alpha_rows if int(r["block_len"]) == 6 and float(r["alpha"]) == 0.0)
    l6_alpha1 = next(r for r in alpha_rows if int(r["block_len"]) == 6 and float(r["alpha"]) == 1.0)
    focus_l6 = [r for r in focus_rows if int(r["block_len"]) == 6]
    focus_best = min(focus_l6, key=lambda r: abs(float(r["prediction_error"])))
    focus_l6_alpha1 = next(r for r in focus_rows if int(r["block_len"]) == 6 and float(r["alpha"]) == 1.0)
    residual_best = next(r for r in residual_rows if int(r["block_len"]) == int(best["block_len"]) and float(r["alpha"]) == float(best["alpha"]))
    baseline_rmse = float(next(r for r in alpha_rows if int(r["block_len"]) == 3 and float(r["alpha"]) == 0.0)["full_rmse"])
    if float(l6_alpha1["full_rmse"]) > 2.0 * float(l6_alpha0["full_rmse"]) or float(best["alpha"]) < 0.5:
        classification = "C. reweighting overcorrects or is not generative"
    elif float(best["full_rmse"]) < 0.5 * baseline_rmse:
        classification = "A. finite block reweighting reproduces most actual state structure"
    elif float(best["full_rmse"]) < baseline_rmse:
        classification = "B. finite block reweighting improves fit but leaves bridge/parity residual"
    else:
        classification = "D. sparse/noisy"
    lines = [
        "# Collatz Finite-Block Reweighting Test",
        "",
        "- iid test words are reweighted by `2^(alpha * block_log_score_L)`.",
        f"- block lengths: {', '.join(map(str, BLOCK_LENGTHS))}",
        f"- alpha sweep: {', '.join(map(str, ALPHAS))}",
        "- total predicted mass is scaled to actual total mass over stable full states.",
        "- smoothing and stable thresholds inherit the L=3..6 renormalization script.",
        "",
        "## Best Fit",
        "",
        f"Best full-state RMSE is L={best['block_len']}, alpha={best['alpha']}, RMSE=`{float(best['full_rmse']):.6g}`, JS=`{float(best['full_js_bits']):.6g}`.",
        f"For L=6, best alpha is `{l6_best['alpha']}` with RMSE `{float(l6_best['full_rmse']):.6g}`.",
        f"Focus state `{FOCUS_STATE}` is best at L=6 alpha `{focus_best['alpha']}`: actual mass `{float(focus_best['actual_mass']):.6g}`, predicted mass `{float(focus_best['predicted_mass']):.6g}`.",
        f"At L=6 alpha=1, focus predicted survival is `{float(focus_l6_alpha1['predicted_survival_ratio']):.6g}` versus actual survival `{float(focus_l6_alpha1['actual_survival_ratio']):.6g}`.",
        "",
        "## Residuals",
        "",
        f"At the global best fit, bridge RMSE is `{float(residual_best['cluster_rmse']):.6g}` and parity RMSE is `{float(residual_best['parity_rmse']):.6g}`.",
        "",
        "## Classification",
        "",
        f"Current classification: **{classification}**.",
        "",
        "The small best-fit improvement comes from damped short-block reweighting. Longer raw reweighting is over-sharp: useful diagnostically, weak as a generative finite-block model.",
    ]
    (OUT_DIR / "collatz_block_reweighting_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    renorm = load_renorm()
    base = renorm.load_base()
    module = base.load_source()
    rng = random.Random(base.SEED)
    counts: defaultdict[tuple[int, str, str, str, str], float] = defaultdict(float)
    actual_test: list[object] = []
    iid_test: list[object] = []
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
                rec = renorm.make_thin(base, "iid", split, power, h, weight, word, q_low, q_high)
                if rec is None:
                    continue
                if split == "train":
                    renorm.add_counts(base, rec, counts)
                else:
                    iid_test.append(rec)

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

    print("building lookups", flush=True)
    lookup, stable = renorm.make_lookup(counts)
    all_stable = set.intersection(*(stable[l] for l in BLOCK_LENGTHS))
    actual_full = defaultdict(float)
    iid_full = defaultdict(float)
    for rec in actual_test:
        if rec.state in all_stable:
            actual_full[rec.state] += rec.weight
    for rec in iid_test:
        if rec.state in all_stable:
            iid_full[rec.state] += rec.weight
    full_keys = sorted(k for k in all_stable if iid_full.get(k, 0.0) >= STABLE_IID_MASS)
    actual_full = {k: actual_full.get(k, 0.0) for k in full_keys}
    iid_full = {k: iid_full.get(k, 0.0) for k in full_keys}
    actual_total = sum(actual_full.values())

    state_fit_rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    focus_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []

    base_cluster_rmse = rmse(aggregate(actual_full, "cluster"), aggregate(iid_full, "cluster"), sorted(set(aggregate(actual_full, "cluster")) | set(aggregate(iid_full, "cluster"))))
    base_parity_rmse = rmse(aggregate(actual_full, "parity"), aggregate(iid_full, "parity"), sorted(set(aggregate(actual_full, "parity")) | set(aggregate(iid_full, "parity"))))

    for block_len in BLOCK_LENGTHS:
        print(f"scoring iid L={block_len}", flush=True)
        iid_scored = [(rec, renorm.block_score(base, rec, lookup, block_len)) for rec in iid_test if rec.state in full_keys]
        for alpha in ALPHAS:
            pred_full, scale = scaled_reweight(iid_scored, alpha, set(full_keys), actual_total)
            state_fit_rows.extend(state_fit_rows_for(block_len, alpha, actual_full, iid_full, pred_full))
            pred_on_keys = {k: pred_full.get(k, 0.0) for k in full_keys}
            full_rmse = rmse(actual_full, pred_on_keys, full_keys)
            full_kl, full_js = kl_js(actual_full, pred_on_keys, full_keys)
            focus_actual = actual_full.get(FOCUS_STATE, 0.0)
            focus_iid = iid_full.get(FOCUS_STATE, 0.0)
            focus_pred = pred_full.get(FOCUS_STATE, 0.0)
            cluster_keys = sorted(set(aggregate(actual_full, "cluster")) | set(aggregate(pred_on_keys, "cluster")))
            parity_keys = sorted(set(aggregate(actual_full, "parity")) | set(aggregate(pred_on_keys, "parity")))
            cluster_rmse = rmse(aggregate(actual_full, "cluster"), aggregate(pred_on_keys, "cluster"), cluster_keys)
            parity_rmse = rmse(aggregate(actual_full, "parity"), aggregate(pred_on_keys, "parity"), parity_keys)
            alpha_rows.append(
                {
                    "block_len": block_len,
                    "alpha": alpha,
                    "scale_to_actual_total": scale,
                    "full_rmse": full_rmse,
                    "full_kl_bits": full_kl,
                    "full_js_bits": full_js,
                    "focus_actual_mass": focus_actual,
                    "focus_iid_mass": focus_iid,
                    "focus_predicted_mass": focus_pred,
                    "focus_error": focus_pred - focus_actual,
                    "focus_predicted_survival_ratio": focus_pred / focus_iid if focus_iid else "",
                    "focus_actual_survival_ratio": focus_actual / focus_iid if focus_iid else "",
                    "cluster_rmse": cluster_rmse,
                    "parity_rmse": parity_rmse,
                }
            )
            focus_rows.append(
                {
                    "block_len": block_len,
                    "alpha": alpha,
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
                    "alpha": alpha,
                    "cluster_rmse": cluster_rmse,
                    "cluster_rmse_baseline_iid": base_cluster_rmse,
                    "cluster_rmse_reduction_vs_iid": base_cluster_rmse - cluster_rmse,
                    "parity_rmse": parity_rmse,
                    "parity_rmse_baseline_iid": base_parity_rmse,
                    "parity_rmse_reduction_vs_iid": base_parity_rmse - parity_rmse,
                }
            )

    print("writing outputs", flush=True)
    write_csv(OUT_DIR / "block_reweighting_state_fit.csv", state_fit_rows)
    write_csv(OUT_DIR / "block_reweighting_alpha_sweep.csv", alpha_rows)
    write_csv(OUT_DIR / "block_reweighting_focus_state.csv", focus_rows)
    write_csv(OUT_DIR / "block_reweighting_residuals.csv", residual_rows)
    rmse_svg(OUT_DIR / "reweighting_rmse_by_L_alpha.svg", alpha_rows)
    simple_heat_svg(OUT_DIR / "predicted_vs_actual_state_mass.svg", state_fit_rows)
    focus_svg(OUT_DIR / "focus_state_fit_by_alpha.svg", focus_rows)
    build_report(alpha_rows, focus_rows, residual_rows)
    print(OUT_DIR / "collatz_block_reweighting_report.md", flush=True)


if __name__ == "__main__":
    main()
