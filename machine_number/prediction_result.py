import csv
import math
from pathlib import Path
from collections import defaultdict


# ============================================================
# 予測結果・バックテスト解析
#
# 目的:
#   過去データだけを使って次の収録日の機種を予測し、
#   実際の結果と比較する。
#
# 重要:
#   予測対象日のデータは、予測スコア計算には一切使用しない。
#   未来データによる情報漏洩（データリーク）を防止する。
#
# 入力:
#   all_data.csv
#
# 出力:
#   prediction_result.csv
#   prediction_summary.csv
#
# all_data.csv は変更しない。
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

OUTPUT_DIR = (
    DATA_DIR
    / "machine_number"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "prediction_result.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "prediction_summary.csv"
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
        (x - avg) ** 2
        for x in values
    ) / len(values)

    return math.sqrt(variance)


# ============================================================
# 0～100
# ============================================================

def clamp(value):

    return max(
        0.0,
        min(100.0, value)
    )


# ============================================================
# 機種予測スコア
#
# ここでは「予測基準日以前」のデータだけを使用。
# ============================================================

def calculate_machine_scores(
    historical_rows,
    prediction_date
):

    machine_data = defaultdict(list)

    machine_daily = defaultdict(
        lambda: defaultdict(list)
    )

    dates = sorted(
        set(
            row["日付"]
            for row in historical_rows
            if row["日付"] < prediction_date
        )
    )

    for row in historical_rows:

        if row["日付"] >= prediction_date:
            continue

        machine = row["機種名"]

        difference = to_float(
            row["差枚"]
        )

        machine_data[machine].append(
            difference
        )

        machine_daily[machine][
            row["日付"]
        ].append(
            difference
        )

    results = []

    for machine, values in machine_data.items():

        count = len(values)

        if count == 0:
            continue

        avg_difference = average(
            values
        )

        std_difference = standard_deviation(
            values
        )

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

        minus_1000_rate = (
            sum(
                1
                for x in values
                if x <= -1000
            )
            / count
            * 100
        )

        # ----------------------------------------------------
        # 直近3日
        # ----------------------------------------------------

        recent_dates = dates[-3:]

        recent_values = []

        for date in recent_dates:

            recent_values.extend(
                machine_daily[machine]
                .get(date, [])
            )

        if recent_values:

            recent_average = average(
                recent_values
            )

            recent_positive_rate = (
                sum(
                    1
                    for x in recent_values
                    if x > 0
                )
                / len(recent_values)
                * 100
            )

        else:

            recent_average = 0
            recent_positive_rate = 0

        # ----------------------------------------------------
        # 直近5日
        # ----------------------------------------------------

        recent_5_dates = dates[-5:]

        recent_5_values = []

        for date in recent_5_dates:

            recent_5_values.extend(
                machine_daily[machine]
                .get(date, [])
            )

        if recent_5_values:

            recent_5_positive_rate = (
                sum(
                    1
                    for x in recent_5_values
                    if x > 0
                )
                / len(recent_5_values)
                * 100
            )

        else:

            recent_5_positive_rate = 0

        # ----------------------------------------------------
        # 平均差枚スコア
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 直近スコア
        # ----------------------------------------------------

        recent_difference_score = (
            50
            + 50
            * math.tanh(
                recent_average / 1800
            )
        )

        recent_difference_score = clamp(
            recent_difference_score
        )

        recent_score = (
            recent_difference_score * 0.60
            + recent_positive_rate * 0.20
            + recent_5_positive_rate * 0.20
        )

        recent_score = clamp(
            recent_score
        )

        # ----------------------------------------------------
        # 安定性
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # マイナス補正
        # ----------------------------------------------------

        negative_penalty = (
            minus_1000_rate * 0.15
        )

        # ----------------------------------------------------
        # 基本スコア
        # ----------------------------------------------------

        machine_score = (

            average_score * 0.30

            + positive_rate * 0.20

            + plus_1000_rate * 0.10

            + plus_2000_rate * 0.10

            + recent_score * 0.20

            + stability_score * 0.10

            - negative_penalty
        )

        machine_score = clamp(
            machine_score
        )

        # ----------------------------------------------------
        # 信頼度
        #
        # データ量が多いほど高くする。
        # ----------------------------------------------------

        confidence = (
            1
            - math.exp(
                -count / 50
            )
        ) * 100

        confidence = clamp(
            confidence
        )

        # ----------------------------------------------------
        # 信頼度補正
        # ----------------------------------------------------

        final_score = (
            50
            + (
                machine_score - 50
            )
            * (
                confidence / 100
            )
        )

        final_score = clamp(
            final_score
        )

        # ----------------------------------------------------
        # ランク
        # ----------------------------------------------------

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

        results.append({

            "機種名": machine,

            "データ数": count,

            "平均差枚": avg_difference,

            "プラス率": positive_rate,

            "+1000枚率": plus_1000_rate,

            "+2000枚率": plus_2000_rate,

            "直近3日平均差枚":
                recent_average,

            "直近3日プラス率":
                recent_positive_rate,

            "信頼度":
                confidence,

            "予測スコア":
                final_score,

            "ランク":
                rank
        })

    results.sort(
        key=lambda x: (
            x["予測スコア"],
            x["平均差枚"]
        ),
        reverse=True
    )

    return results


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

        writer.writerows(
            rows
        )

    print()
    print("★ CSV保存成功")
    print(path)


# ============================================================
# 開始
# ============================================================

print("=" * 70)
print("予測結果・バックテスト解析")
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


required_columns = [
    "日付",
    "機種名",
    "台番号",
    "差枚"
]


missing = [
    x
    for x in required_columns
    if x not in fieldnames
]


if missing:

    print()
    print("[エラー]")
    print("必要な列がありません。")

    for x in missing:
        print(x)

    input("Enterキーで終了...")
    raise SystemExit


print(
    f"読み込みデータ: {len(rows):,}行"
)

print("必要な列: OK")


# ============================================================
# 収録日
# ============================================================

all_dates = sorted(
    set(
        row["日付"]
        for row in rows
        if row["日付"]
    )
)


print()
print(
    f"収録日数: {len(all_dates)}日"
)

print(
    "収録日:"
)

print(
    " / ".join(all_dates)
)


# ============================================================
# バックテスト可能な日を決定
#
# 最初の3日程度は過去データが少なすぎるため除外。
# ============================================================

prediction_dates = all_dates[3:]


if not prediction_dates:

    print()
    print("[エラー]")
    print("バックテスト可能な日がありません。")

    input("Enterキーで終了...")
    raise SystemExit


# ============================================================
# バックテスト
# ============================================================

all_results = []

summary_results = []


for prediction_date in prediction_dates:

    # --------------------------------------------------------
    # 予測日より前だけを使用
    # --------------------------------------------------------

    historical_rows = [
        row
        for row in rows
        if row["日付"] < prediction_date
    ]

    # --------------------------------------------------------
    # 予測
    # --------------------------------------------------------

    predictions = calculate_machine_scores(
        historical_rows,
        prediction_date
    )

    if not predictions:
        continue

    # --------------------------------------------------------
    # 実際の予測対象日のデータ
    # --------------------------------------------------------

    actual_rows = [
        row
        for row in rows
        if row["日付"] == prediction_date
    ]

    actual_machine = defaultdict(list)

    for row in actual_rows:

        actual_machine[
            row["機種名"]
        ].append(
            to_float(
                row["差枚"]
            )
        )

    # --------------------------------------------------------
    # 各予測順位を実績と比較
    # --------------------------------------------------------

    for rank, prediction in enumerate(
        predictions[:30],
        1
    ):

        machine = prediction["機種名"]

        actual_values = (
            actual_machine
            .get(machine, [])
        )

        if actual_values:

            actual_average = average(
                actual_values
            )

            actual_positive_rate = (
                sum(
                    1
                    for x in actual_values
                    if x > 0
                )
                / len(actual_values)
                * 100
            )

            actual_plus_1000_rate = (
                sum(
                    1
                    for x in actual_values
                    if x >= 1000
                )
                / len(actual_values)
                * 100
            )

            actual_plus_2000_rate = (
                sum(
                    1
                    for x in actual_values
                    if x >= 2000
                )
                / len(actual_values)
                * 100
            )

            actual_data_count = len(
                actual_values
            )

        else:

            actual_average = 0
            actual_positive_rate = 0
            actual_plus_1000_rate = 0
            actual_plus_2000_rate = 0
            actual_data_count = 0

        # ----------------------------------------------------
        # 結果判定
        # ----------------------------------------------------

        if actual_data_count == 0:

            result = "対象外"

        elif actual_average >= 1000:

            result = "◎"

        elif actual_average > 0:

            result = "○"

        elif actual_average == 0:

            result = "△"

        else:

            result = "×"

        all_results.append({

            "予測基準日":
                (
                    max(
                        r["日付"]
                        for r in historical_rows
                    )
                    if historical_rows
                    else ""
                ),

            "評価日":
                prediction_date,

            "予測順位":
                rank,

            "機種名":
                machine,

            "予測スコア":
                round(
                    prediction["予測スコア"],
                    1
                ),

            "予測ランク":
                prediction["ランク"],

            "過去平均差枚":
                round(
                    prediction["平均差枚"],
                    1
                ),

            "過去プラス率":
                round(
                    prediction["プラス率"],
                    1
                ),

            "過去直近3日平均":
                round(
                    prediction["直近3日平均差枚"],
                    1
                ),

            "信頼度":
                round(
                    prediction["信頼度"],
                    1
                ),

            "評価データ台数":
                actual_data_count,

            "実績平均差枚":
                round(
                    actual_average,
                    1
                ),

            "実績プラス率":
                round(
                    actual_positive_rate,
                    1
                ),

            "実績+1000枚率":
                round(
                    actual_plus_1000_rate,
                    1
                ),

            "実績+2000枚率":
                round(
                    actual_plus_2000_rate,
                    1
                ),

            "結果":
                result
        })


    # ========================================================
    # TOP5 / TOP10 / TOP30 集計
    # ========================================================

    print()
    print("=" * 70)
    print(
        f"【バックテスト】"
        f"{prediction_date}"
    )
    print("=" * 70)

    for top_n in [5, 10, 30]:

        selected = predictions[:top_n]

        actual_avgs = []

        positive_rates = []

        hit_1000 = 0

        valid_count = 0

        for prediction in selected:

            machine = prediction["機種名"]

            values = actual_machine.get(
                machine,
                []
            )

            if not values:
                continue

            valid_count += 1

            avg = average(values)

            actual_avgs.append(
                avg
            )

            positive_rates.append(
                sum(
                    1
                    for x in values
                    if x > 0
                )
                / len(values)
                * 100
            )

            if avg >= 1000:

                hit_1000 += 1


        if valid_count:

            top_average = average(
                actual_avgs
            )

            top_positive = average(
                positive_rates
            )

            hit_rate = (
                hit_1000
                / valid_count
                * 100
            )

        else:

            top_average = 0
            top_positive = 0
            hit_rate = 0


        print(
            f"TOP{top_n:2d} / "
            f"実績平均差枚 "
            f"{top_average:+.1f}枚 / "
            f"平均プラス率 "
            f"{top_positive:.1f}% / "
            f"+1000枚以上 "
            f"{hit_rate:.1f}%"
        )


    # --------------------------------------------------------
    # 日別サマリー
    # --------------------------------------------------------

    summary_results.append({

        "評価日":
            prediction_date,

        "予測元データ日数":
            len(
                set(
                    row["日付"]
                    for row in historical_rows
                )
            ),

        "TOP5平均差枚":
            round(
                (
                    average([
                        average(
                            actual_machine[
                                p["機種名"]
                            ]
                        )
                        for p in predictions[:5]
                        if p["機種名"]
                        in actual_machine
                    ])
                ),
                1
            ),

        "TOP10平均差枚":
            round(
                (
                    average([
                        average(
                            actual_machine[
                                p["機種名"]
                            ]
                        )
                        for p in predictions[:10]
                        if p["機種名"]
                        in actual_machine
                    ])
                ),
                1
            ),

        "TOP30平均差枚":
            round(
                (
                    average([
                        average(
                            actual_machine[
                                p["機種名"]
                            ]
                        )
                        for p in predictions[:30]
                        if p["機種名"]
                        in actual_machine
                    ])
                ),
                1
            )
    })


# ============================================================
# 全体評価
# ============================================================

print()
print("=" * 70)
print("【バックテスト総合結果】")
print("=" * 70)


for top_n in [5, 10, 30]:

    target_rows = [
        row
        for row in all_results
        if row["予測順位"] <= top_n
        and row["結果"] != "対象外"
    ]

    if not target_rows:
        continue

    avg_actual = average(
        [
            row["実績平均差枚"]
            for row in target_rows
        ]
    )

    positive_rate = average(
        [
            row["実績プラス率"]
            for row in target_rows
        ]
    )

    hit_rate = (
        sum(
            1
            for row in target_rows
            if row["実績平均差枚"] >= 1000
        )
        / len(target_rows)
        * 100
    )

    print(
        f"TOP{top_n:2d} / "
        f"実績平均差枚 "
        f"{avg_actual:+.1f}枚 / "
        f"平均プラス率 "
        f"{positive_rate:.1f}% / "
        f"+1000枚以上 "
        f"{hit_rate:.1f}%"
    )


# ============================================================
# CSV保存
# ============================================================

result_fields = [

    "予測基準日",
    "評価日",
    "予測順位",
    "機種名",
    "予測スコア",
    "予測ランク",
    "過去平均差枚",
    "過去プラス率",
    "過去直近3日平均",
    "信頼度",
    "評価データ台数",
    "実績平均差枚",
    "実績プラス率",
    "実績+1000枚率",
    "実績+2000枚率",
    "結果"
]


summary_fields = [

    "評価日",
    "予測元データ日数",
    "TOP5平均差枚",
    "TOP10平均差枚",
    "TOP30平均差枚"
]


save_csv(
    OUTPUT_FILE,
    result_fields,
    all_results
)


save_csv(
    SUMMARY_FILE,
    summary_fields,
    summary_results
)


# ============================================================
# 完了
# ============================================================

print()
print("=" * 70)
print("★★★★★ バックテスト解析 完了 ★★★★★")
print("=" * 70)

print()
print("保存ファイル:")
print(OUTPUT_FILE)
print(SUMMARY_FILE)

print()
print("all_data.csv は変更していません。")

input("Enterキーで終了...")