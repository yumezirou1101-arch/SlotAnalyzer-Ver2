import csv
import math
from pathlib import Path
from collections import defaultdict


# ============================================================
# 投入パターン解析 V3
#
# 目的:
#   「過去に出ていた機種」だけではなく、
#   次回ホールが投入しそうな機種を評価する。
#
# 主な評価要素:
#   1. 長期実績
#   2. 直近実績
#   3. 前回実績
#   4. 凹み状況
#   5. プラス率
#   6. +1000枚率
#   7. +2000枚率
#   8. データ信頼度
#   9. 過熱補正
#  10. 凹み狙い補正
#
# all_data.csv は変更しない。
# ============================================================


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    BASE_DIR
    / "data"
    / "maruhan_maebashi"
)

INPUT_FILE = DATA_DIR / "all_data.csv"

OUTPUT_DIR = (
    DATA_DIR
    / "machine_number"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "investment_pattern_v3.csv"
)


# ============================================================
# 基本関数
# ============================================================

def to_float(value):

    if value is None:
        return 0.0

    value = str(value).strip()

    if value == "":
        return 0.0

    value = value.replace(",", "")
    value = value.replace("+", "")

    try:
        return float(value)

    except ValueError:
        return 0.0


def average(values):

    if not values:
        return 0.0

    return sum(values) / len(values)


def clamp(value):

    return max(
        0.0,
        min(100.0, value)
    )


def save_csv(path, fields, rows):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("★ CSV保存成功")
    print(path)


# ============================================================
# 開始
# ============================================================

print("=" * 70)
print("投入パターン解析 V3")
print("=" * 70)

print()
print("入力ファイル:")
print(INPUT_FILE)


if not INPUT_FILE.exists():

    print()
    print("[エラー]")
    print("all_data.csv が見つかりません。")

    input("Enterキーで終了...")
    raise SystemExit


# ============================================================
# CSV読み込み
# ============================================================

print()
print("all_data.csv を読み込みます...")

with open(
    INPUT_FILE,
    "r",
    newline="",
    encoding="utf-8-sig"
) as f:

    reader = csv.DictReader(f)

    rows = list(reader)

    fieldnames = reader.fieldnames


required = [
    "日付",
    "機種名",
    "台番号",
    "G数",
    "差枚"
]


missing = [
    x for x in required
    if x not in fieldnames
]


if missing:

    print()
    print("[エラー]")
    print("必要な列がありません:")

    for x in missing:
        print(x)

    input("Enterキーで終了...")
    raise SystemExit


print(
    f"読み込みデータ: {len(rows):,}行"
)

print("必要な列: OK")


# ============================================================
# 日付
# ============================================================

dates = sorted(
    set(
        row["日付"]
        for row in rows
        if row["日付"]
    )
)


print()
print(
    f"解析期間: "
    f"{dates[0]} ～ {dates[-1]}"
)

print(
    f"解析日数: {len(dates)}日"
)


# ============================================================
# 機種別データ
# ============================================================

machine_data = defaultdict(list)

machine_daily = defaultdict(
    lambda: defaultdict(list)
)


for row in rows:

    machine = row["機種名"]

    diff = to_float(
        row["差枚"]
    )

    machine_data[machine].append(
        diff
    )

    machine_daily[machine][
        row["日付"]
    ].append(
        diff
    )


# ============================================================
# 機種ごとの解析
# ============================================================

results = []


for machine, values in machine_data.items():

    count = len(values)

    if count == 0:
        continue


    # --------------------------------------------------------
    # 基本実績
    # --------------------------------------------------------

    avg_diff = average(values)

    positive_rate = (
        sum(
            1
            for x in values
            if x > 0
        )
        / count
        * 100
    )

    plus_1000_rate = (
        sum(
            1
            for x in values
            if x >= 1000
        )
        / count
        * 100
    )

    plus_2000_rate = (
        sum(
            1
            for x in values
            if x >= 2000
        )
        / count
        * 100
    )


    # --------------------------------------------------------
    # 各日の機種平均
    # --------------------------------------------------------

    daily_avg = {}

    for date in dates:

        day_values = (
            machine_daily[machine]
            .get(date, [])
        )

        if day_values:

            daily_avg[date] = average(
                day_values
            )


    # --------------------------------------------------------
    # 直近3日
    # --------------------------------------------------------

    recent3_dates = dates[-3:]

    recent3 = [
        daily_avg[d]
        for d in recent3_dates
        if d in daily_avg
    ]

    recent3_avg = average(
        recent3
    )


    # --------------------------------------------------------
    # 直近5日
    # --------------------------------------------------------

    recent5_dates = dates[-5:]

    recent5 = [
        daily_avg[d]
        for d in recent5_dates
        if d in daily_avg
    ]

    recent5_avg = average(
        recent5
    )


    # --------------------------------------------------------
    # 前回実績
    # --------------------------------------------------------

    previous_avg = 0.0

    for date in reversed(dates):

        if date in daily_avg:

            previous_avg = daily_avg[
                date
            ]

            break


    # --------------------------------------------------------
    # 直近3日以外の過去平均
    # --------------------------------------------------------

    old_values = []

    for date in dates[:-3]:

        if date in daily_avg:

            old_values.append(
                daily_avg[date]
            )


    old_avg = average(
        old_values
    )


    # --------------------------------------------------------
    # 直近と過去の差
    #
    # マイナスなら最近凹んでいる。
    # --------------------------------------------------------

    recent_change = (
        recent3_avg
        - old_avg
    )


    # ========================================================
    # ① 長期実績スコア
    # ========================================================

    long_score = (
        50
        + 50
        * math.tanh(
            avg_diff / 1800
        )
    )

    long_score = clamp(
        long_score
    )


    # ========================================================
    # ② 直近実績スコア
    # ========================================================

    recent_score = (
        50
        + 50
        * math.tanh(
            recent3_avg / 1800
        )
    )

    recent_score = clamp(
        recent_score
    )


    # ========================================================
    # ③ 前回実績スコア
    #
    # 前回出過ぎている台を少し抑える。
    # ========================================================

    previous_score = (
        50
        + 50
        * math.tanh(
            previous_avg / 2000
        )
    )

    previous_score = clamp(
        previous_score
    )


    # ========================================================
    # ④ プラス率
    # ========================================================

    positive_score = clamp(
        positive_rate
    )


    # ========================================================
    # ⑤ +1000枚率
    # ========================================================

    plus1000_score = clamp(
        plus_1000_rate
    )


    # ========================================================
    # ⑥ +2000枚率
    # ========================================================

    plus2000_score = clamp(
        plus_2000_rate
    )


    # ========================================================
    # ⑦ 凹み狙いスコア
    #
    # 最近大きく凹んでいるが、
    # 過去実績が悪くない機種を評価。
    #
    # ただし、単なる低設定機種を
    # 「凹んでいるから狙い目」と誤認しないよう、
    # 長期実績を条件にする。
    # ========================================================

    if avg_diff > 300:

        if recent_change < 0:

            dip_score = clamp(
                50
                + (
                    abs(recent_change)
                    / 2500
                    * 50
                )
            )

        else:

            dip_score = 50

    else:

        dip_score = 30


    # ========================================================
    # ⑧ 過熱補正
    #
    # 直近3日で大きく出ている場合、
    # 「出ているからさらに出る」とは
    # 考えず少し抑える。
    # ========================================================

    if recent3_avg > 1500:

        heat_penalty = min(
            15,
            (
                recent3_avg - 1500
            )
            / 500
            * 5
        )

    else:

        heat_penalty = 0


    # ========================================================
    # ⑨ 信頼度
    # ========================================================

    confidence = (
        1
        - math.exp(
            -count / 50
        )
    ) * 100

    confidence = clamp(
        confidence
    )


    # ========================================================
    # ⑩ 機種データ量補正
    # ========================================================

    sample_factor = (
        0.60
        + (
            confidence / 100
        ) * 0.40
    )


    # ========================================================
    # V3総合スコア
    #
    # 長期実績       20%
    # 直近実績       15%
    # 前回実績       10%
    # プラス率       15%
    # +1000率        10%
    # +2000率        5%
    # 凹み狙い       20%
    # 基本安定性     5%
    #
    # その後、
    # 過熱補正を引く。
    # ========================================================

    base_score = (

        long_score * 0.20

        + recent_score * 0.15

        + previous_score * 0.10

        + positive_score * 0.15

        + plus1000_score * 0.10

        + plus2000_score * 0.05

        + dip_score * 0.20

        + 50 * 0.05
    )


    # ========================================================
    # 信頼度補正
    # ========================================================

    final_score = (
        50
        + (
            base_score
            - 50
        )
        * sample_factor
    )


    # 過熱補正

    final_score -= heat_penalty


    final_score = clamp(
        final_score
    )


    # ========================================================
    # ランク
    # ========================================================

    if final_score >= 75:

        rank = "S"

    elif final_score >= 68:

        rank = "A"

    elif final_score >= 60:

        rank = "B"

    elif final_score >= 50:

        rank = "C"

    elif final_score >= 40:

        rank = "D"

    else:

        rank = "E"


    # ========================================================
    # 判定理由
    # ========================================================

    reasons = []


    if avg_diff >= 500:

        reasons.append(
            "長期実績○"
        )


    if recent3_avg >= 500:

        reasons.append(
            "直近実績○"
        )


    if previous_avg >= 1000:

        reasons.append(
            "前回強"
        )


    if recent_change <= -500:

        reasons.append(
            "直近凹み"
        )


    if recent_change >= 1000:

        reasons.append(
            "直近好調"
        )


    if positive_rate >= 50:

        reasons.append(
            "プラス率50%以上"
        )


    if plus_1000_rate >= 30:

        reasons.append(
            "+1000率30%以上"
        )


    if heat_penalty > 0:

        reasons.append(
            "過熱補正"
        )


    reason_text = (
        " / ".join(reasons)
        if reasons
        else "特徴弱"
    )


    # ========================================================
    # 結果
    # ========================================================

    results.append({

        "順位": 0,

        "機種名":
            machine,

        "データ台数":
            count,

        "平均差枚":
            round(
                avg_diff,
                1
            ),

        "プラス率":
            round(
                positive_rate,
                1
            ),

        "+1000枚率":
            round(
                plus_1000_rate,
                1
            ),

        "+2000枚率":
            round(
                plus_2000_rate,
                1
            ),

        "直近3日平均差枚":
            round(
                recent3_avg,
                1
            ),

        "直近5日平均差枚":
            round(
                recent5_avg,
                1
            ),

        "前回平均差枚":
            round(
                previous_avg,
                1
            ),

        "直近変化":
            round(
                recent_change,
                1
            ),

        "長期実績スコア":
            round(
                long_score,
                1
            ),

        "直近実績スコア":
            round(
                recent_score,
                1
            ),

        "凹み狙いスコア":
            round(
                dip_score,
                1
            ),

        "過熱補正":
            round(
                heat_penalty,
                1
            ),

        "信頼度":
            round(
                confidence,
                1
            ),

        "V3スコア":
            round(
                final_score,
                1
            ),

        "ランク":
            rank,

        "判定理由":
            reason_text
    })


# ============================================================
# スコア順
# ============================================================

results.sort(
    key=lambda x: (
        x["V3スコア"],
        x["平均差枚"]
    ),
    reverse=True
)


for i, row in enumerate(
    results,
    1
):

    row["順位"] = i


# ============================================================
# TOP30表示
# ============================================================

print()
print("=" * 70)
print("【次回投入パターン候補 TOP30】")
print("=" * 70)


for row in results[:30]:

    print(
        f"{row['順位']:2d}. "
        f"{row['機種名']} / "
        f"{row['データ台数']}台日 / "
        f"平均差枚 "
        f"{row['平均差枚']:+.1f}枚 / "
        f"プラス率 "
        f"{row['プラス率']:.1f}% / "
        f"直近3日 "
        f"{row['直近3日平均差枚']:+.1f}枚 / "
        f"前回 "
        f"{row['前回平均差枚']:+.1f}枚 / "
        f"凹み "
        f"{row['直近変化']:+.1f} / "
        f"V3 "
        f"{row['V3スコア']:.1f} / "
        f"{row['ランク']}"
    )


# ============================================================
# ランク別
# ============================================================

print()
print("=" * 70)
print("【ランク別機種数】")
print("=" * 70)


rank_counts = defaultdict(int)


for row in results:

    rank_counts[
        row["ランク"]
    ] += 1


for rank in [
    "S",
    "A",
    "B",
    "C",
    "D",
    "E"
]:

    print(
        f"{rank}: "
        f"{rank_counts[rank]}機種"
    )


# ============================================================
# 凹み狙いTOP10
# ============================================================

print()
print("=" * 70)
print("【凹み狙い候補 TOP10】")
print("=" * 70)


dip_candidates = [

    row
    for row in results

    if row["平均差枚"] > 300
]


dip_candidates.sort(
    key=lambda x: (
        x["凹み狙いスコア"],
        x["平均差枚"]
    ),
    reverse=True
)


for i, row in enumerate(
    dip_candidates[:10],
    1
):

    print(
        f"{i:2d}. "
        f"{row['機種名']} / "
        f"平均差枚 "
        f"{row['平均差枚']:+.1f}枚 / "
        f"直近変化 "
        f"{row['直近変化']:+.1f} / "
        f"凹みスコア "
        f"{row['凹み狙いスコア']:.1f}"
    )


# ============================================================
# 過熱機種
# ============================================================

print()
print("=" * 70)
print("【直近過熱機種】")
print("=" * 70)


hot_candidates = [

    row
    for row in results

    if row["過熱補正"] > 0
]


hot_candidates.sort(
    key=lambda x: x["過熱補正"],
    reverse=True
)


if hot_candidates:

    for row in hot_candidates[:10]:

        print(
            f"{row['機種名']} / "
            f"直近3日 "
            f"{row['直近3日平均差枚']:+.1f}枚 / "
            f"過熱補正 "
            f"-{row['過熱補正']:.1f}"
        )

else:

    print(
        "現在、強い過熱補正対象はありません。"
    )


# ============================================================
# CSV保存
# ============================================================

output_fields = [

    "順位",
    "機種名",
    "データ台数",
    "平均差枚",
    "プラス率",
    "+1000枚率",
    "+2000枚率",
    "直近3日平均差枚",
    "直近5日平均差枚",
    "前回平均差枚",
    "直近変化",
    "長期実績スコア",
    "直近実績スコア",
    "凹み狙いスコア",
    "過熱補正",
    "信頼度",
    "V3スコア",
    "ランク",
    "判定理由"
]


save_csv(
    OUTPUT_FILE,
    output_fields,
    results
)


# ============================================================
# 完了
# ============================================================

print()
print("=" * 70)
print("★★★★★ 投入パターン解析 V3 完了 ★★★★★")
print("=" * 70)

print()
print("保存ファイル:")
print(OUTPUT_FILE)

print()
print("all_data.csv は変更していません。")

input("Enterキーで終了...")