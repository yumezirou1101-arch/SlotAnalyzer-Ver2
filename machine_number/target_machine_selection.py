import csv
import math
from pathlib import Path
from collections import defaultdict


# ============================================================
# おすすめ機種判定プログラム
#
# 入力:
#   all_data.csv
#
# 出力:
#   target_machine_selection.csv
#
# 目的:
#   機種単位で過去実績・直近実績・安定性を分析し、
#   次回狙う価値の高い機種をランキングする。
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
    / "target_machine_selection.csv"
)


# ============================================================
# 数値変換
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


def to_int(value):

    return int(
        to_float(value)
    )


# ============================================================
# 平均
# ============================================================

def average(values):

    if not values:
        return 0.0

    return sum(values) / len(values)


# ============================================================
# 標準偏差
# ============================================================

def standard_deviation(values):

    if len(values) <= 1:
        return 0.0

    avg = average(values)

    variance = sum(
        (value - avg) ** 2
        for value in values
    ) / len(values)

    return math.sqrt(variance)


# ============================================================
# 0～100
# ============================================================

def clamp(value):

    return max(
        0.0,
        min(
            100.0,
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
print("おすすめ機種判定プログラム")
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
# 日付一覧
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

    difference = to_float(
        row["差枚"]
    )

    machine_data[
        machine
    ].append(
        difference
    )

    machine_daily[
        machine
    ][
        row["日付"]
    ].append(
        difference
    )


# ============================================================
# 機種別解析
# ============================================================

results = []


for machine, differences in machine_data.items():

    total_count = len(
        differences
    )

    analysis_days = len(
        machine_daily[machine]
    )


    # --------------------------------------------------------
    # 基本統計
    # --------------------------------------------------------

    avg_difference = average(
        differences
    )

    std_difference = standard_deviation(
        differences
    )


    positive_rate = (
        sum(
            1
            for value in differences
            if value > 0
        )
        / total_count
        * 100
    )


    plus_1000_rate = (
        sum(
            1
            for value in differences
            if value >= 1000
        )
        / total_count
        * 100
    )


    plus_2000_rate = (
        sum(
            1
            for value in differences
            if value >= 2000
        )
        / total_count
        * 100
    )


    minus_1000_rate = (
        sum(
            1
            for value in differences
            if value <= -1000
        )
        / total_count
        * 100
    )


    # --------------------------------------------------------
    # 日平均差枚
    # --------------------------------------------------------

    daily_averages = []

    for date in dates:

        values = (
            machine_daily[machine]
            .get(date, [])
        )

        if values:

            daily_averages.append(
                average(values)
            )


    # --------------------------------------------------------
    # 直近3日
    # --------------------------------------------------------

    recent_dates = dates[-3:]

    recent_3_values = []

    for date in recent_dates:

        values = (
            machine_daily[machine]
            .get(date, [])
        )

        recent_3_values.extend(
            values
        )


    if recent_3_values:

        recent_3_average = average(
            recent_3_values
        )

        recent_3_positive_rate = (
            sum(
                1
                for value
                in recent_3_values
                if value > 0
            )
            / len(recent_3_values)
            * 100
        )

    else:

        recent_3_average = 0
        recent_3_positive_rate = 0


    # --------------------------------------------------------
    # 直近5日
    # --------------------------------------------------------

    recent_dates_5 = dates[-5:]

    recent_5_values = []

    for date in recent_dates_5:

        values = (
            machine_daily[machine]
            .get(date, [])
        )

        recent_5_values.extend(
            values
        )


    if recent_5_values:

        recent_5_average = average(
            recent_5_values
        )

        recent_5_positive_rate = (
            sum(
                1
                for value
                in recent_5_values
                if value > 0
            )
            / len(recent_5_values)
            * 100
        )

    else:

        recent_5_average = 0
        recent_5_positive_rate = 0


    # ========================================================
    # 平均差枚スコア
    # ========================================================

    average_score = (
        50
        + 50
        * math.tanh(
            avg_difference / 1800
        )
    )

    average_score = clamp(
        average_score
    )


    # ========================================================
    # プラス率スコア
    # ========================================================

    positive_score = clamp(
        positive_rate
    )


    # ========================================================
    # +1000枚率
    # ========================================================

    plus_1000_score = clamp(
        plus_1000_rate
    )


    # ========================================================
    # +2000枚率
    # ========================================================

    plus_2000_score = clamp(
        plus_2000_rate
    )


    # ========================================================
    # 直近実績スコア
    # ========================================================

    recent_difference_score = (
        50
        + 50
        * math.tanh(
            recent_3_average / 1800
        )
    )

    recent_difference_score = clamp(
        recent_difference_score
    )


    recent_score = (
        recent_difference_score * 0.60
        + recent_3_positive_rate * 0.20
        + recent_5_positive_rate * 0.20
    )

    recent_score = clamp(
        recent_score
    )


    # ========================================================
    # 安定性
    # ========================================================

    if abs(avg_difference) > 100:

        variation = (
            std_difference
            / abs(avg_difference)
        )

        stability_score = (
            100
            / (
                1
                + variation * 0.35
            )
        )

    else:

        stability_score = 50


    stability_score = clamp(
        stability_score
    )


    # ========================================================
    # マイナス補正
    # ========================================================

    negative_penalty = (
        minus_1000_rate
        * 0.15
    )


    # ========================================================
    # 機種スコア
    #
    # 平均差枚       30%
    # プラス率       20%
    # +1000枚率      10%
    # +2000枚率      10%
    # 直近実績       20%
    # 安定性         10%
    # ========================================================

    machine_score = (

        average_score * 0.30

        + positive_score * 0.20

        + plus_1000_score * 0.10

        + plus_2000_score * 0.10

        + recent_score * 0.20

        + stability_score * 0.10

        - negative_penalty
    )


    machine_score = clamp(
        machine_score
    )


    # ========================================================
    # 信頼度
    #
    # 機種全体のデータ量を考慮。
    # ========================================================

    confidence = (
        1
        - math.exp(
            -total_count / 50
        )
    ) * 100


    confidence = clamp(
        confidence
    )


    # ========================================================
    # 最終スコア
    #
    # データが少ない機種は50点方向へ戻す。
    # ========================================================

    final_score = (
        50
        + (
            machine_score
            - 50
        )
        * (
            confidence / 100
        )
    )


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
    # 注意事項
    # ========================================================

    notes = []


    if total_count < 10:

        notes.append(
            "データ不足"
        )


    if total_count >= 50:

        notes.append(
            "十分なデータ"
        )


    if avg_difference >= 1000:

        notes.append(
            "平均差枚強"
        )


    if positive_rate >= 50:

        notes.append(
            "プラス率50%以上"
        )


    if recent_3_average >= 1000:

        notes.append(
            "直近3日強"
        )


    if recent_3_average <= -1000:

        notes.append(
            "直近3日弱"
        )


    if minus_1000_rate >= 50:

        notes.append(
            "マイナス多"
        )


    note_text = (
        " / ".join(notes)
        if notes
        else ""
    )


    # ========================================================
    # 結果
    # ========================================================

    results.append({

        "順位": 0,

        "機種名":
            machine,

        "分析日数":
            analysis_days,

        "データ台数":
            total_count,

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
                recent_3_average,
                1
            ),

        "直近3日プラス率":
            round(
                recent_3_positive_rate,
                1
            ),

        "直近5日平均差枚":
            round(
                recent_5_average,
                1
            ),

        "直近5日プラス率":
            round(
                recent_5_positive_rate,
                1
            ),

        "信頼度":
            round(
                confidence,
                1
            ),

        "機種スコア":
            round(
                machine_score,
                1
            ),

        "最終スコア":
            round(
                final_score,
                1
            ),

        "ランク":
            rank,

        "注意事項":
            note_text
    })


# ============================================================
# 最終スコア順
# ============================================================

results.sort(
    key=lambda x: (
        x["最終スコア"],
        x["信頼度"],
        x["平均差枚"]
    ),
    reverse=True
)


# ============================================================
# 順位
# ============================================================

for index, row in enumerate(
    results,
    1
):

    row["順位"] = index


# ============================================================
# TOP30表示
# ============================================================

print()
print("=" * 70)
print("【次回おすすめ機種 TOP30】")
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
        f"信頼度 "
        f"{row['信頼度']:.1f} / "
        f"スコア "
        f"{row['最終スコア']:.1f} / "
        f"{row['ランク']}"
    )


# ============================================================
# ランク別集計
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
# 特に強い機種
# ============================================================

print()
print("=" * 70)
print("【高実績機種チェック】")
print("=" * 70)


high_performance = sorted(
    results,
    key=lambda x: x["平均差枚"],
    reverse=True
)


for row in high_performance[:10]:

    print(
        f"{row['機種名']} / "
        f"{row['データ台数']}台日 / "
        f"平均差枚 "
        f"{row['平均差枚']:+.1f}枚 / "
        f"プラス率 "
        f"{row['プラス率']:.1f}% / "
        f"スコア "
        f"{row['最終スコア']:.1f}"
    )


# ============================================================
# CSV保存
# ============================================================

output_fields = [

    "順位",
    "機種名",
    "分析日数",
    "データ台数",
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
    "信頼度",
    "機種スコア",
    "最終スコア",
    "ランク",
    "注意事項"
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
print("★★★★★ おすすめ機種判定 完了 ★★★★★")
print("=" * 70)

print()
print("保存ファイル:")
print(OUTPUT_FILE)

print()
print("all_data.csv は変更していません。")

input("Enterキーで終了...")