import csv
from pathlib import Path
from collections import defaultdict


# ============================================================
# 基本設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "maruhan_maebashi"

INPUT_FILE = DATA_DIR / "all_data.csv"

OUTPUT_MACHINE = DATA_DIR / "machine_strength_analysis.csv"
OUTPUT_MACHINE_NUMBER = DATA_DIR / "machine_number_strength.csv"
OUTPUT_NEIGHBOR = DATA_DIR / "neighbor_strength_analysis.csv"


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
# CSV読み込み
# ============================================================

print("=" * 70)
print("機種別・台番号別 強弱分析")
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


print()
print("all_data.csv を読み込みます...")


try:

    with open(
        INPUT_FILE,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        rows = list(reader)

except Exception as e:

    print()
    print("[エラー]")
    print(f"CSV読み込み失敗: {e}")

    input("Enterキーで終了...")
    raise SystemExit


print(f"読み込みデータ: {len(rows):,}行")


if not rows:

    print()
    print("[エラー]")
    print("データがありません。")

    input("Enterキーで終了...")
    raise SystemExit


# ============================================================
# 必要列確認
# ============================================================

required_columns = [
    "日付",
    "機種名",
    "台番号",
    "G数",
    "差枚",
]


missing_columns = [
    column
    for column in required_columns
    if column not in rows[0]
]


if missing_columns:

    print()
    print("[エラー]")
    print("必要な列がありません。")

    for column in missing_columns:
        print(f"  {column}")

    input("Enterキーで終了...")
    raise SystemExit


print("必要な列: OK")


# ============================================================
# 解析期間
# ============================================================

dates = sorted(
    set(
        row["日付"]
        for row in rows
    )
)

print()
print(f"収録日数: {len(dates)}日")


# ============================================================
# 機種別データ
# ============================================================

machine_data = defaultdict(list)

for row in rows:

    machine_data[row["機種名"]].append(row)


# ============================================================
# 機種別集計
# ============================================================

machine_results = []


for machine, records in machine_data.items():

    days = len(
        set(
            row["日付"]
            for row in records
        )
    )

    machine_units = len(
        set(
            row["台番号"]
            for row in records
        )
    )

    total_games = sum(
        to_int(row["G数"])
        for row in records
    )

    total_difference = sum(
        to_int(row["差枚"])
        for row in records
    )

    positive = sum(
        1
        for row in records
        if to_int(row["差枚"]) > 0
    )

    negative = sum(
        1
        for row in records
        if to_int(row["差枚"]) < 0
    )

    zero = sum(
        1
        for row in records
        if to_int(row["差枚"]) == 0
    )

    average_difference = (
        total_difference / len(records)
        if records
        else 0
    )

    average_games = (
        total_games / len(records)
        if records
        else 0
    )

    positive_rate = (
        positive / len(records) * 100
        if records
        else 0
    )

    machine_results.append({

        "機種名": machine,

        "台数": machine_units,

        "データ行数": len(records),

        "稼働日数": days,

        "平均G数": round(
            average_games,
            1
        ),

        "総差枚": total_difference,

        "平均差枚": round(
            average_difference,
            1
        ),

        "プラス日数": positive,

        "マイナス日数": negative,

        "差枚0": zero,

        "プラス率": round(
            positive_rate,
            2
        ),
    })


# ============================================================
# 機種別ランキング
# ============================================================

machine_results.sort(
    key=lambda x: x["平均差枚"],
    reverse=True
)


print()
print("=" * 70)
print("【機種別 平均差枚ランキング】")
print("=" * 70)


for rank, result in enumerate(
    machine_results[:30],
    1
):

    print(
        f"{rank:2d}. "
        f"{result['機種名']} / "
        f"{result['台数']}台 / "
        f"{result['稼働日数']}日 / "
        f"平均G数 "
        f"{result['平均G数']:.0f}G / "
        f"平均差枚 "
        f"{result['平均差枚']:+.1f}枚 / "
        f"プラス率 "
        f"{result['プラス率']:.1f}%"
    )


# ============================================================
# 台番号 × 機種
# ============================================================

machine_number_data = defaultdict(list)


for row in rows:

    key = (
        row["機種名"],
        row["台番号"]
    )

    machine_number_data[key].append(row)


machine_number_results = []


for (
    machine,
    number
), records in machine_number_data.items():

    days = len(
        set(
            row["日付"]
            for row in records
        )
    )

    total_games = sum(
        to_int(row["G数"])
        for row in records
    )

    total_difference = sum(
        to_int(row["差枚"])
        for row in records
    )

    positive = sum(
        1
        for row in records
        if to_int(row["差枚"]) > 0
    )

    average_games = (
        total_games / days
        if days
        else 0
    )

    average_difference = (
        total_difference / days
        if days
        else 0
    )

    positive_rate = (
        positive / days * 100
        if days
        else 0
    )

    machine_number_results.append({

        "機種名": machine,

        "台番号": number,

        "日数": days,

        "平均G数": round(
            average_games,
            1
        ),

        "総差枚": total_difference,

        "平均差枚": round(
            average_difference,
            1
        ),

        "プラス日数": positive,

        "プラス率": round(
            positive_rate,
            2
        ),
    })


# ============================================================
# 機種ごとの台番号ランキング
# ============================================================

print()
print("=" * 70)
print("【機種別 強い台番号】")
print("=" * 70)


for machine in sorted(machine_data.keys()):

    machine_numbers = [

        result

        for result in machine_number_results

        if result["機種名"] == machine
    ]

    machine_numbers.sort(
        key=lambda x: (
            x["平均差枚"],
            x["プラス率"]
        ),
        reverse=True
    )

    print()
    print(
        f"■ {machine}"
    )

    for rank, result in enumerate(
        machine_numbers[:5],
        1
    ):

        print(
            f"  {rank}. "
            f"台{result['台番号']} / "
            f"{result['日数']}日 / "
            f"平均差枚 "
            f"{result['平均差枚']:+.1f}枚 / "
            f"プラス率 "
            f"{result['プラス率']:.1f}%"
        )


# ============================================================
# 隣接台分析
# ============================================================

print()
print("=" * 70)
print("【隣接台分析】")
print("=" * 70)


# 日付ごとに
# 台番号 → 機種・差枚
# を作る

daily_numbers = defaultdict(dict)


for row in rows:

    try:

        number = int(row["台番号"])

    except ValueError:

        continue

    daily_numbers[
        row["日付"]
    ][number] = row


neighbor_results = []


for date, number_data_daily in daily_numbers.items():

    sorted_numbers = sorted(
        number_data_daily.keys()
    )

    for i in range(
        len(sorted_numbers) - 1
    ):

        number1 = sorted_numbers[i]
        number2 = sorted_numbers[i + 1]

        # 台番号が連番の場合だけ
        if number2 != number1 + 1:
            continue

        row1 = number_data_daily[number1]
        row2 = number_data_daily[number2]

        diff1 = to_int(
            row1["差枚"]
        )

        diff2 = to_int(
            row2["差枚"]
        )

        positive1 = diff1 > 0
        positive2 = diff2 > 0

        if positive1 and positive2:

            neighbor_results.append({

                "日付": date,

                "台番号1": number1,

                "機種1": row1["機種名"],

                "差枚1": diff1,

                "台番号2": number2,

                "機種2": row2["機種名"],

                "差枚2": diff2,

                "両方プラス": "はい",
            })


print()
print(
    f"連番で両方プラスになった組み合わせ: "
    f"{len(neighbor_results)}件"
)


# ============================================================
# 同一機種の連番分析
# ============================================================

same_machine_neighbors = []


for date, number_data_daily in daily_numbers.items():

    sorted_numbers = sorted(
        number_data_daily.keys()
    )

    for i in range(
        len(sorted_numbers) - 1
    ):

        number1 = sorted_numbers[i]
        number2 = sorted_numbers[i + 1]

        if number2 != number1 + 1:
            continue

        row1 = number_data_daily[number1]
        row2 = number_data_daily[number2]

        if row1["機種名"] != row2["機種名"]:
            continue

        diff1 = to_int(
            row1["差枚"]
        )

        diff2 = to_int(
            row2["差枚"]
        )

        if diff1 > 0 and diff2 > 0:

            same_machine_neighbors.append({

                "日付": date,

                "台番号1": number1,

                "機種名": row1["機種名"],

                "差枚1": diff1,

                "台番号2": number2,

                "差枚2": diff2,
            })


print(
    f"同一機種かつ連番で両方プラス: "
    f"{len(same_machine_neighbors)}件"
)


# ============================================================
# CSV保存関数
# ============================================================

def save_csv(
    path,
    fieldnames,
    data
):

    try:

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

            writer.writerows(data)

        print()
        print("★ 保存成功")
        print(path)

    except Exception as e:

        print()
        print("[エラー]")
        print(
            f"{path} の保存に失敗: {e}"
        )


# ============================================================
# 機種別CSV
# ============================================================

save_csv(

    OUTPUT_MACHINE,

    [
        "機種名",
        "台数",
        "データ行数",
        "稼働日数",
        "平均G数",
        "総差枚",
        "平均差枚",
        "プラス日数",
        "マイナス日数",
        "差枚0",
        "プラス率",
    ],

    machine_results
)


# ============================================================
# 機種 × 台番号CSV
# ============================================================

machine_number_results.sort(

    key=lambda x: (
        x["機種名"],
        -x["平均差枚"]
    )
)


save_csv(

    OUTPUT_MACHINE_NUMBER,

    [
        "機種名",
        "台番号",
        "日数",
        "平均G数",
        "総差枚",
        "平均差枚",
        "プラス日数",
        "プラス率",
    ],

    machine_number_results
)


# ============================================================
# 隣接台CSV
# ============================================================

neighbor_output = []


for result in neighbor_results:

    result2 = dict(result)

    result2["同一機種"] = (
        "はい"
        if result["機種1"] == result["機種2"]
        else "いいえ"
    )

    neighbor_output.append(
        result2
    )


save_csv(

    OUTPUT_NEIGHBOR,

    [
        "日付",
        "台番号1",
        "機種1",
        "差枚1",
        "台番号2",
        "機種2",
        "差枚2",
        "両方プラス",
        "同一機種",
    ],

    neighbor_output
)


# ============================================================
# 終了
# ============================================================

print()
print("=" * 70)
print("機種別・台番号別 強弱分析終了")
print("=" * 70)

print()
print("出力ファイル:")

print(OUTPUT_MACHINE)
print(OUTPUT_MACHINE_NUMBER)
print(OUTPUT_NEIGHBOR)

print()
print("all_data.csv は変更していません。")

input("Enterキーで終了...")