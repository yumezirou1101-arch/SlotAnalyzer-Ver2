import csv
import math
from pathlib import Path
from collections import defaultdict


# ============================================================
# 信頼度補正付き 投入傾向スコア V2.1
# ============================================================
#
# 改良点
# ・少数サンプルを過大評価しない
# ・9日程度のデータは適切に評価
# ・平均差枚の高い台を適切に評価
# ・直近3日/5日の実績を評価
# ・標準偏差による過剰なペナルティを緩和
# ・0～100点に張り付かない
#
# all_data.csv は変更しません。
# ============================================================


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    BASE_DIR
    / "data"
    / "maruhan_maebashi"
)

INPUT_FILE = (
    DATA_DIR
    / "all_data.csv"
)

OUTPUT_FILE = (
    DATA_DIR
    / "machine_number"
    / "investment_score_v2_1.csv"
)


# ============================================================
# 数値変換
# ============================================================

def to_int(value):

    if value is None:
        return 0

    value = str(value).strip()

    if value == "":
        return 0

    value = value.replace(",", "")
    value = value.replace("+", "")

    try:
        return int(value)

    except ValueError:
        return 0


# ============================================================
# 平均
# ============================================================

def average(values):

    if not values:
        return 0

    return sum(values) / len(values)


# ============================================================
# 標準偏差
# ============================================================

def standard_deviation(values):

    if len(values) <= 1:
        return 0

    avg = average(values)

    variance = sum(
        (value - avg) ** 2
        for value in values
    ) / len(values)

    return math.sqrt(variance)


# ============================================================
# 0～100に制限
# ============================================================

def clamp(value, minimum=0, maximum=100):

    return max(
        minimum,
        min(
            maximum,
            value
        )
    )


# ============================================================
# CSV保存
# ============================================================

def save_csv(
    path,
    fieldnames,
    rows
):

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
            fieldnames=fieldnames
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
print("信頼度補正付き 投入傾向スコア V2.1")
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


print(
    f"読み込みデータ: {len(rows):,}行"
)


required_columns = [
    "日付",
    "機種名",
    "台番号",
    "G数",
    "差枚"
]


missing = [
    column
    for column in required_columns
    if column not in fieldnames
]


if missing:

    print()
    print("[エラー]")
    print("必要な列がありません。")

    for column in missing:

        print(
            f"  {column}"
        )

    input("Enterキーで終了...")
    raise SystemExit


print("必要な列: OK")


# ============================================================
# 日付＋台番号で整理
# ============================================================

daily_data = defaultdict(dict)


for row in rows:

    try:

        number = int(
            row["台番号"]
        )

    except ValueError:

        continue


    row["_number"] = number

    row["_difference"] = to_int(
        row["差枚"]
    )

    row["_games"] = to_int(
        row["G数"]
    )


    daily_data[
        row["日付"]
    ][number] = row


dates = sorted(
    daily_data.keys()
)


total_days = len(dates)


print()

if dates:

    print(
        f"解析期間: "
        f"{dates[0]} ～ {dates[-1]}"
    )

print(
    f"解析日数: {total_days}日"
)


# ============================================================
# 台番号別データ
# ============================================================

number_records = defaultdict(list)


for date in dates:

    for number, row in daily_data[date].items():

        number_records[
            number
        ].append(row)


# ============================================================
# 台番号解析
# ============================================================

results = []


for number, records in number_records.items():

    sample_days = len(records)


    # --------------------------------------------------------
    # 基本データ
    # --------------------------------------------------------

    differences = [
        row["_difference"]
        for row in records
    ]

    games = [
        row["_games"]
        for row in records
    ]


    avg_difference = average(
        differences
    )


    std_difference = standard_deviation(
        differences
    )


    avg_games = average(
        games
    )


    # --------------------------------------------------------
    # プラス率
    # --------------------------------------------------------

    positive_count = sum(
        1
        for value in differences
        if value > 0
    )


    positive_rate = (
        positive_count
        / sample_days
        * 100
    )


    # --------------------------------------------------------
    # +1000枚率
    # --------------------------------------------------------

    plus_1000_count = sum(
        1
        for value in differences
        if value >= 1000
    )


    plus_1000_rate = (
        plus_1000_count
        / sample_days
        * 100
    )


    # --------------------------------------------------------
    # +2000枚率
    # --------------------------------------------------------

    plus_2000_count = sum(
        1
        for value in differences
        if value >= 2000
    )


    plus_2000_rate = (
        plus_2000_count
        / sample_days
        * 100
    )


    # --------------------------------------------------------
    # -1000枚以下率
    # --------------------------------------------------------

    minus_1000_count = sum(
        1
        for value in differences
        if value <= -1000
    )


    minus_1000_rate = (
        minus_1000_count
        / sample_days
        * 100
    )


    # --------------------------------------------------------
    # 直近3日
    # --------------------------------------------------------

    recent_3 = records[-3:]


    recent_3_differences = [
        row["_difference"]
        for row in recent_3
    ]


    recent_3_difference = average(
        recent_3_differences
    )


    recent_3_positive_rate = (
        sum(
            1
            for value in recent_3_differences
            if value > 0
        )
        / len(recent_3_differences)
        * 100
    )


    # --------------------------------------------------------
    # 直近5日
    # --------------------------------------------------------

    recent_5 = records[-5:]


    recent_5_differences = [
        row["_difference"]
        for row in recent_5
    ]


    recent_5_difference = average(
        recent_5_differences
    )


    recent_5_positive_rate = (
        sum(
            1
            for value in recent_5_differences
            if value > 0
        )
        / len(recent_5_differences)
        * 100
    )


    # ========================================================
    # 信頼度
    #
    # 2日  → 約36%
    # 3日  → 約49%
    # 5日  → 約67%
    # 7日  → 約79%
    # 9日  → 約86%
    # 14日 → 約95%
    # 30日 → 約100%
    #
    # 少数サンプルを強く抑制しつつ、
    # 9日程度あれば実績をかなり反映。
    # ========================================================

    confidence = (
        1
        - math.exp(
            -sample_days / 4.5
        )
    )


    confidence_score = (
        confidence * 100
    )


    # ========================================================
    # 平均差枚スコア
    #
    # +3000枚以上を高評価。
    # ただし+10000枚などで100点に張り付かない。
    # ========================================================

    avg_score = (
        50
        + 50
        * math.tanh(
            avg_difference / 2200
        )
    )


    avg_score = clamp(
        avg_score
    )


    # ========================================================
    # プラス率スコア
    # ========================================================

    positive_score = clamp(
        positive_rate
    )


    # ========================================================
    # +1000枚率スコア
    # ========================================================

    plus_1000_score = clamp(
        plus_1000_rate
    )


    # ========================================================
    # +2000枚率スコア
    # ========================================================

    plus_2000_score = clamp(
        plus_2000_rate
    )


    # ========================================================
    # 直近3日差枚スコア
    # ========================================================

    recent_3_difference_score = (
        50
        + 50
        * math.tanh(
            recent_3_difference
            / 2200
        )
    )


    recent_3_difference_score = clamp(
        recent_3_difference_score
    )


    # ========================================================
    # 直近5日差枚スコア
    # ========================================================

    recent_5_difference_score = (
        50
        + 50
        * math.tanh(
            recent_5_difference
            / 2200
        )
    )


    recent_5_difference_score = clamp(
        recent_5_difference_score
    )


    # ========================================================
    # 直近実績スコア
    #
    # 直近3日をやや重視。
    # ========================================================

    recent_score = (
        recent_3_difference_score * 0.55
        + recent_5_difference_score * 0.25
        + recent_3_positive_rate * 0.10
        + recent_5_positive_rate * 0.10
    )


    recent_score = clamp(
        recent_score
    )


    # ========================================================
    # 安定性
    #
    # V2よりペナルティを弱める。
    #
    # 荒い台だから即低評価、
    # とはしない。
    # ========================================================

    if abs(avg_difference) > 100:

        coefficient_variation = (
            std_difference
            / abs(avg_difference)
        )

        stability = (
            1
            / (
                1
                + coefficient_variation
                * 0.35
            )
        )

        stability_score = (
            stability * 100
        )

    else:

        stability_score = 50


    stability_score = clamp(
        stability_score
    )


    # ========================================================
    # マイナス補正
    #
    # -1000枚以下が多い台には軽いペナルティ。
    # ========================================================

    negative_penalty = (
        minus_1000_rate
        * 0.15
    )


    # ========================================================
    # 基礎スコア
    #
    # 平均差枚        35%
    # プラス率        20%
    # +1000枚率       10%
    # +2000枚率       10%
    # 直近実績        15%
    # 安定性          10%
    # ========================================================

    raw_score = (

        avg_score * 0.35

        + positive_score * 0.20

        + plus_1000_score * 0.10

        + plus_2000_score * 0.10

        + recent_score * 0.15

        + stability_score * 0.10

        - negative_penalty
    )


    raw_score = clamp(
        raw_score
    )


    # ========================================================
    # 信頼度補正
    #
    # データが少ない場合、
    # 基礎スコアを50点方向へ戻す。
    #
    # これにより、
    #
    # 2日だけ+5000枚
    #
    # のような台が、
    #
    # 9日間安定して強い台
    #
    # を簡単に上回らないようにする。
    # ========================================================

    final_score = (
        50
        + (
            raw_score
            - 50
        )
        * confidence
    )


    final_score = clamp(
        final_score
    )


    # ========================================================
    # ランク
    # ========================================================

    if final_score >= 80:

        rank = "S"

    elif final_score >= 70:

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
    # 現在機種
    # ========================================================

    current_machine = records[-1][
        "機種名"
    ]


    # ========================================================
    # 結果
    # ========================================================

    results.append({

        "台番号":
            number,

        "現在機種":
            current_machine,

        "分析日数":
            sample_days,

        "平均G数":
            round(
                avg_games,
                1
            ),

        "平均差枚":
            round(
                avg_difference,
                1
            ),

        "差枚標準偏差":
            round(
                std_difference,
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

        "-1000枚以下率":
            round(
                minus_1000_rate,
                1
            ),

        "直近3日平均差枚":
            round(
                recent_3_difference,
                1
            ),

        "直近3日プラス率":
            round(
                recent_3_positive_rate,
                1
            ),

        "直近5日平均差枚":
            round(
                recent_5_difference,
                1
            ),

        "直近5日プラス率":
            round(
                recent_5_positive_rate,
                1
            ),

        "安定性":
            round(
                stability_score,
                1
            ),

        "信頼度":
            round(
                confidence_score,
                1
            ),

        "基礎スコア":
            round(
                raw_score,
                1
            ),

        "総合スコア":
            round(
                final_score,
                1
            ),

        "ランク":
            rank
    })


# ============================================================
# 総合スコア順
# ============================================================

results.sort(
    key=lambda x: (
        x["総合スコア"],
        x["信頼度"],
        x["平均差枚"]
    ),
    reverse=True
)


# ============================================================
# TOP30表示
# ============================================================

print()
print("=" * 70)
print("【信頼度補正付き 台番号ランキング TOP30】")
print("=" * 70)


for rank_number, row in enumerate(
    results[:30],
    1
):

    print(
        f"{rank_number:2d}. "
        f"台{row['台番号']} / "
        f"{row['現在機種']} / "
        f"{row['分析日数']}日 / "
        f"平均差枚 "
        f"{row['平均差枚']:+.1f}枚 / "
        f"プラス率 "
        f"{row['プラス率']:.1f}% / "
        f"直近3日 "
        f"{row['直近3日平均差枚']:+.1f}枚 / "
        f"信頼度 "
        f"{row['信頼度']:.1f} / "
        f"総合 "
        f"{row['総合スコア']:.1f} / "
        f"{row['ランク']}"
    )


# ============================================================
# 分析日数別
# ============================================================

print()
print("=" * 70)
print("【分析日数別 台数】")
print("=" * 70)


sample_counts = defaultdict(int)


for row in results:

    sample_counts[
        row["分析日数"]
    ] += 1


for days in sorted(
    sample_counts.keys()
):

    print(
        f"{days}日分析: "
        f"{sample_counts[days]}台"
    )


# ============================================================
# ランク別
# ============================================================

print()
print("=" * 70)
print("【ランク別台数】")
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
        f"{rank_counts[rank]}台"
    )


# ============================================================
# 重要台の比較表示
# ============================================================

print()
print("=" * 70)
print("【高実績台チェック】")
print("=" * 70)


high_performance = sorted(
    results,
    key=lambda x: x["平均差枚"],
    reverse=True
)


for row in high_performance[:10]:

    print(
        f"台{row['台番号']} / "
        f"{row['現在機種']} / "
        f"{row['分析日数']}日 / "
        f"平均差枚 "
        f"{row['平均差枚']:+.1f}枚 / "
        f"総合 "
        f"{row['総合スコア']:.1f} / "
        f"{row['ランク']}"
    )


# ============================================================
# CSV保存
# ============================================================

output_fields = [

    "台番号",
    "現在機種",
    "分析日数",
    "平均G数",
    "平均差枚",
    "差枚標準偏差",
    "プラス率",
    "+1000枚率",
    "+2000枚率",
    "-1000枚以下率",
    "直近3日平均差枚",
    "直近3日プラス率",
    "直近5日平均差枚",
    "直近5日プラス率",
    "安定性",
    "信頼度",
    "基礎スコア",
    "総合スコア",
    "ランク"
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
print("★★★★★ 投入傾向スコア V2.1 完了 ★★★★★")
print("=" * 70)

print()
print("保存ファイル:")
print(OUTPUT_FILE)

print()
print("all_data.csv は変更していません。")

input("Enterキーで終了...")