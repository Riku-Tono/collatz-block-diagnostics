from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections import deque
from pathlib import Path


ROOT = Path(r"C:\Users\yauki\Documents\Codex")
CARD_PATH = ROOT / r"2026-06-25\new-chat\outputs\collatz_escape_word_deficits.csv"
WEI_PATH = ROOT / r"2026-07-14\new-chat\outputs\wei_ren_top_k_blocks.csv"
OUT_DIR = ROOT / r"2026-07-14\new-chat\outputs"
DYNAMICS_DIR = ROOT / r"2026-07-14\new-chat\work\TXPO_inspect\Dynamics"
FOCUS_BITS = 6_000_000


def direction(ratio: float, tolerance: float = 0.0) -> str:
    if ratio > 1.0 + tolerance:
        return "above_iid"
    if ratio < 1.0 - tolerance:
        return "below_iid"
    return "tie"


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for pos in order[i:j]:
            ranks[pos] = rank
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else float("nan")


def read_card_rows() -> list[dict[str, str]]:
    with CARD_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row["power"] == "28"
        and row["h"] == "2"
        and row["depth"] == "4"
        and row["terminal_before_depth"] == "0"
    ]
    if len(selected) != 50 or len({row["word"] for row in selected}) != 50:
        raise ValueError(f"Expected 50 unique finite-block words, got {len(selected)}")
    return selected


def read_wei_rows() -> tuple[list[int], dict[tuple[int, str], dict[str, str]]]:
    with WEI_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["depth"] == "4"]
    bit_lengths = sorted({int(row["start_bit_length"]) for row in rows})
    lookup = {(int(row["start_bit_length"]), row["word"]): row for row in rows}
    return bit_lengths, lookup


def count_missing_word(bits: int, word_text: str) -> dict[str, str]:
    target = tuple(map(int, word_text.split(",")))
    path = DYNAMICS_DIR / f"txpo46dynamics{bits}"
    with path.open("r", encoding="ascii") as handle:
        handle.readline()
        dynamics = handle.readline().strip()

    history: deque[int] = deque(maxlen=len(target))
    current_k: int | None = None
    count = 0
    windows = 0

    def consume(k: int) -> None:
        nonlocal count, windows
        history.append(k)
        if len(history) == len(target):
            windows += 1
            if tuple(history) == target:
                count += 1

    for symbol in dynamics:
        if symbol == "I":
            if current_k is not None:
                consume(current_k)
            current_k = 1
        else:
            if current_k is None:
                raise ValueError(f"Malformed dynamics in {path}")
            current_k += 1
    if current_k is not None:
        consume(current_k)

    empirical = count / windows
    iid = 2.0 ** (-sum(target))
    return {
        "file": path.name,
        "start_bit_length": str(bits),
        "depth": str(len(target)),
        "word": word_text,
        "count": str(count),
        "empirical_probability": str(empirical),
        "iid_geometric_probability": str(iid),
        "difference": str(empirical - iid),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cards = read_card_rows()
    bit_lengths, wei = read_wei_rows()
    missing = [
        (bits, row["word"])
        for bits in bit_lengths
        for row in cards
        if (bits, row["word"]) not in wei
    ]
    for bits, word in missing:
        wei[(bits, word)] = count_missing_word(bits, word)

    detail_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    scale_agreement: dict[int, list[bool]] = defaultdict(list)

    for card in cards:
        word = card["word"]
        card_ratio = float(card["actual_to_iid_ratio"])
        card_direction = direction(card_ratio)
        agreements: list[bool] = []
        large_agreements: list[bool] = []
        per_scale_ratios: dict[int, float] = {}

        for bits in bit_lengths:
            wei_row = wei[(bits, word)]
            empirical = float(wei_row["empirical_probability"])
            iid = float(wei_row["iid_geometric_probability"])
            wei_ratio = empirical / iid
            agree = direction(wei_ratio) == card_direction
            agreements.append(agree)
            scale_agreement[bits].append(agree)
            per_scale_ratios[bits] = wei_ratio
            if bits >= 1_000_000:
                large_agreements.append(agree)

        focus = wei[(FOCUS_BITS, word)]
        focus_empirical = float(focus["empirical_probability"])
        focus_iid = float(focus["iid_geometric_probability"])
        focus_ratio = focus_empirical / focus_iid
        focus_agree = direction(focus_ratio) == card_direction
        conditional_ratio = float(card["actual_conditional_probability"]) / float(
            card["iid_conditional_probability"]
        )

        detail_rows.append(
            {
                "word": word,
                "finite_block_rank_kind": card["rank_kind"],
                "finite_block_actual_to_iid_ratio": card_ratio,
                "finite_block_conditional_ratio": conditional_ratio,
                "finite_block_direction": card_direction,
                "wei_start_bit_length": FOCUS_BITS,
                "wei_count": focus["count"],
                "wei_empirical_probability": focus_empirical,
                "wei_iid_probability": focus_iid,
                "wei_empirical_to_iid_ratio": focus_ratio,
                "wei_direction": direction(focus_ratio),
                "same_direction": int(focus_agree),
                "finite_block_log2_ratio": math.log2(card_ratio),
                "wei_log2_ratio": math.log2(focus_ratio),
            }
        )
        stability_rows.append(
            {
                "word": word,
                "finite_block_direction": card_direction,
                "agreement_count_all_10_scales": sum(agreements),
                "agreement_share_all_10_scales": sum(agreements) / len(agreements),
                "agreement_count_1M_to_6M": sum(large_agreements),
                "agreement_share_1M_to_6M": sum(large_agreements) / len(large_agreements),
                "same_direction_at_6M": int(focus_agree),
                **{f"wei_ratio_{bits}": per_scale_ratios[bits] for bits in bit_lengths},
            }
        )

    detail_rows.sort(key=lambda row: (str(row["finite_block_rank_kind"]), str(row["word"])))
    stability_rows.sort(key=lambda row: str(row["word"]))
    write_csv(OUT_DIR / "wei_ren_vs_finite_block_50_words.csv", detail_rows)
    write_csv(OUT_DIR / "wei_ren_vs_finite_block_scale_stability.csv", stability_rows)

    deficit_rows = [row for row in detail_rows if row["finite_block_rank_kind"] == "largest_deficit"]
    excess_rows = [row for row in detail_rows if row["finite_block_rank_kind"] == "largest_excess"]
    agreements = sum(int(row["same_direction"]) for row in detail_rows)
    deficit_agreements = sum(int(row["same_direction"]) for row in deficit_rows)
    excess_agreements = sum(int(row["same_direction"]) for row in excess_rows)
    card_logs = [float(row["finite_block_log2_ratio"]) for row in detail_rows]
    wei_logs = [float(row["wei_log2_ratio"]) for row in detail_rows]
    log_pearson = pearson(card_logs, wei_logs)
    spearman = pearson(average_ranks(card_logs), average_ranks(wei_logs))

    scale_lines = []
    for bits in bit_lengths:
        count = sum(scale_agreement[bits])
        scale_lines.append(f"| {bits:,} | {count}/50 | {count / 50:.1%} |")

    most_stable = sorted(
        stability_rows,
        key=lambda row: (-int(row["agreement_count_1M_to_6M"]), str(row["word"])),
    )[:10]
    unstable = sorted(
        stability_rows,
        key=lambda row: (int(row["agreement_count_1M_to_6M"]), str(row["word"])),
    )[:10]

    def word_list(rows: list[dict[str, object]]) -> str:
        return ", ".join(
            f"`{row['word']}` ({row['agreement_count_1M_to_6M']}/6)" for row in rows
        )

    report = f"""# Wei Ren TXPO と finite-block 50語の方向比較

## 結論

600万ビット軌道では、finite-block側の不足・過剰方向とWei Ren側の無条件4語頻度の方向が一致したのは **{agreements}/50 ({agreements / 50:.1%})** でした。

- finite-block不足25語のうち、Wei Renでもiid未満：**{deficit_agreements}/25 ({deficit_agreements / 25:.1%})**
- finite-block過剰25語のうち、Wei Renでもiid超過：**{excess_agreements}/25 ({excess_agreements / 25:.1%})**
- 50語のlog2比どうしのPearson相関：**{log_pearson:.4f}**
- 50語の順位相関（Spearman）：**{spearman:.4f}**

この比較では、はっきりした同方向の再現は見えません。また、一致率が仮に50%付近になっても、50語はfinite-block側で極端な語として選ばれた集合であり、語どうしも独立ではありません。したがって単純な符号検定や外部再現の主張には使いません。

## 比較の固定条件

- finite-block：`power=28, h=2, depth=4, terminal_before_depth=0`
- 対象：`largest_deficit` 25語＋`largest_excess` 25語
- finite-block方向：`actual_unconditional_mass / iid_unconditional_mass` が1より上か下か
- Wei Ren方向：600万ビット軌道の4語頻度を `2^(-sum(k))` で割り、1より上か下か

finite-blockのconditional比は50語すべてでunconditional比と同じ符号でした。

## ビット長ごとの一致

| Wei Ren開始値のビット長 | 一致語数 | 一致率 |
|---:|---:|---:|
{chr(10).join(scale_lines)}

これら10本は独立反復とは扱いません。公開Cコードでは乱数seed設定が無効で、各長さの開始ビット列が同じ疑似乱数列を共有する可能性が高いためです。ここではスケール安定性の点検としてだけ読みます。

## 100万〜600万ビットで方向が比較的安定した語

{word_list(most_stable)}

## 100万〜600万ビットで方向が安定しなかった語

{word_list(unstable)}

## 読み方

今回否定的だったのは、finite-blockで選ばれた50個の短い語の不足・過剰が、Wei Renの巨大軌道の無条件平均にもそのまま現れる、という単純な対応です。

これはfinite-blockの結果そのものを否定しません。finite-block側は有限boxからescapeした語、Wei Ren側は1本の巨大軌道全体です。母集団と切り取り条件が違います。

次に進むなら、短い語を増やすより、Wei Ren軌道にもfinite-blockと同じ「有限boxからのescape窓」を置く必要があります。そこで初めて、同じ条件を付けた比較になります。
"""
    (OUT_DIR / "Wei_Ren_vs_finite_block_direction_report.md").write_text(
        report, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
