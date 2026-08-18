import csv
from pathlib import Path
from collections import defaultdict


# ============================================================
# 基本設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "maruhan_maebashi"

INPUT_FILE = DATA_DIR / "all_data.csv"

OUTPUT_MACHINE = DATA_DIR / "machine_analysis.csv"
OUTPUT_NUMBER = DATA_DIR / "machine_number_analysis.csv"
OUTPUT_DAILY = DATA_DIR / "daily_analysis.csv"


# ============================================================
# 数値変換
# ============================================================

def to_int(value):
    """
    カンマ、+、-などを除去して整数に変換する。
    空欄や変換不能な場合は0。
    """

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
print("基礎解析プログラム")
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
# 必要な列を確認
# ============================================================

required_columns = [
    "日付",
    "機種名",
    "台番号",
    "G数",
    "差枚",
    "BB",
    "RB",
    "合成確率",
    "BB確率",
    "RB確率",
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
# 1. 全体サマリー
# ============================================================

print()
print("=" * 70)
print("【1】全体サマリー")
print("=" * 70)


dates = sorted(
    set(row["日付"] for row in rows)
)


total_games = sum(
    to_int(row["G数"])
    for row in rows
)


total_difference = sum(
    to_int(row["差枚"])
    for row in rows
)


positive_count = sum(
    1
    for row in rows
    if to_int(row["差枚"]) > 0
)


negative_count = sum(
    1
    for row in rows
    if to_int(row["差枚"]) < 0
)


zero_count = sum(
    1
    for row in rows
    if to_int(row["差枚"]) == 0
)


total_machines = len(rows)


average_difference = (
    total_difference / total_machines
    if total_machines
    else 0
)


positive_rate = (
    positive_count / total_machines * 100
    if total_machines
    else 0
)


average_games = (
    total_games / total_machines
    if total_machines
    else 0
)


print(f"収録日数       : {len(dates)}日")
print(f"総データ行数   : {total_machines:,}台分")
print(f"総G数          : {total_games:,}G")
print(f"総差枚         : {total_difference:+,}枚")
print(f"1台平均差枚    : {average_difference:+,.1f}枚")
print(f"平均G数        : {average_games:,.1f}G")
print(f"プラス台       : {positive_count:,}台分")
print(f"マイナス台     : {negative_count:,}台分")
print(f"差枚0          : {zero_count:,}台分")
print(f"プラス台率     : {positive_rate:.2f}%")


# ============================================================
# 2. 機種別集計
# ============================================================

print()
print("=" * 70)
print("【2】機種別集計")
print("=" * 70)


machine_data = defaultdict(
    lambda: {
        "count": 0,
        "games": 0,
        "difference": 0,
        "positive": 0,
        "max_difference": None,
        "min_difference": None,
    }
)


for row in rows:

    machine = row["機種名"]

    games = to_int(row["G数"])
    difference = to_int(row["差枚"])

    data = machine_data[machine]

    data["count"] += 1
    data["games"] += games
    data["difference"] += difference

    if difference > 0:
        data["positive"] += 1

    if (
        data["max_difference"] is None
        or difference > data["max_difference"]
    ):
        data["max_difference"] = difference

    if (
        data["min_difference"] is None
        or difference < data["min_difference"]
    ):
        data["min_difference"] = difference


machine_results = []


for machine, data in machine_data.items():

    count = data["count"]

    average_games_machine = (
        data["games"] / count
        if count
        else 0
    )

    average_difference_machine = (
        data["difference"] / count
        if count
        else 0
    )

    positive_rate_machine = (
        data["positive"] / count * 100
        if count
        else 0
    )

    machine_results.append({
        "機種名": machine,
        "台数": count,
        "総G数": data["games"],
        "平均G数": round(
            average_games_machine,
            1
        ),
        "総差枚": data["difference"],
        "平均差枚": round(
            average_difference_machine,
            1
        ),
        "プラス台数": data["positive"],
        "プラス率": round(
            positive_rate_machine,
            2
        ),
        "最大差枚": data["max_difference"],
        "最小差枚": data["min_difference"],
    })


# 平均差枚の高い順
machine_results.sort(
    key=lambda x: x["平均差枚"],
    reverse=True
)


print()
print("平均差枚 TOP20")

for rank, result in enumerate(
    machine_results[:20],
    1
):

    print(
        f"{rank:2d}. "
        f"{result['機種名']} / "
        f"台数 {result['台数']} / "
        f"平均差枚 {result['平均差枚']:+.1f}枚 / "
        f"プラス率 {result['プラス率']:.1f}%"
    )


# ============================================================
# 3. 台番号別集計
# ============================================================

print()
print("=" * 70)
print("【3】台番号別集計")
print("=" * 70)


number_data = defaultdict(
    lambda: {
        "count": 0,
        "games": 0,
        "difference": 0,
        "positive": 0,
        "machine_names": set(),
    }
)


for row in rows:

    number = row["台番号"]

    games = to_int(row["G数"])
    difference = to_int(row["差枚"])

    data = number_data[number]

    data["count"] += 1
    data["games"] += games
    data["difference"] += difference

    if difference > 0:
        data["positive"] += 1

    data["machine_names"].add(
        row["機種名"]
    )


number_results = []


for number, data in number_data.items():

    count = data["count"]

    average_games_number = (
        data["games"] / count
        if count
        else 0
    )

    average_difference_number = (
        data["difference"] / count
        if count
        else 0
    )

    positive_rate_number = (
        data["positive"] / count * 100
        if count
        else 0
    )

    machine_name = " / ".join(
        sorted(data["machine_names"])
    )

    number_results.append({
        "台番号": number,
        "機種名": machine_name,
        "日数": count,
        "平均G数": round(
            average_games_number,
            1
        ),
        "総差枚": data["difference"],
        "平均差枚": round(
            average_difference_number,
            1
        ),
        "プラス日数": data["positive"],
        "プラス率": round(
            positive_rate_number,
            2
        ),
    })


number_results.sort(
    key=lambda x: x["平均差枚"],
    reverse=True
)


print()
print("平均差枚 TOP20")

for rank, result in enumerate(
    number_results[:20],
    1
):

    print(
        f"{rank:2d}. "
        f"台{result['台番号']} / "
        f"{result['機種名']} / "
        f"平均差枚 {result['平均差枚']:+.1f}枚 / "
        f"プラス率 {result['プラス率']:.1f}%"
    )


# ============================================================
# 4. 日別集計
# ============================================================

print()
print("=" * 70)
print("【4】日別集計")
print("=" * 70)


daily_data = defaultdict(
    lambda: {
        "count": 0,
        "games": 0,
        "difference": 0,
        "positive": 0,
    }
)


for row in rows:

    date = row["日付"]

    games = to_int(row["G数"])
    difference = to_int(row["差枚"])

    data = daily_data[date]

    data["count"] += 1
    data["games"] += games
    data["difference"] += difference

    if difference > 0:
        data["positive"] += 1


daily_results = []


for date, data in daily_data.items():

    count = data["count"]

    average_games_daily = (
        data["games"] / count
        if count
        else 0
    )

    average_difference_daily = (
        data["difference"] / count
        if count
        else 0
    )

    positive_rate_daily = (
        data["positive"] / count * 100
        if count
        else 0
    )

    daily_results.append({
        "日付": date,
        "台数": count,
        "総G数": data["games"],
        "平均G数": round(
            average_games_daily,
            1
        ),
        "総差枚": data["difference"],
        "平均差枚": round(
            average_difference_daily,
            1
        ),
        "プラス台数": data["positive"],
        "プラス率": round(
            positive_rate_daily,
            2
        ),
    })


daily_results.sort(
    key=lambda x: x["日付"]
)


for result in daily_results:

    print(
        f"{result['日付']} / "
        f"{result['台数']}台 / "
        f"総差枚 {result['総差枚']:+,}枚 / "
        f"平均差枚 {result['平均差枚']:+.1f}枚 / "
        f"プラス率 {result['プラス率']:.1f}%"
    )


# ============================================================
# CSV保存
# ============================================================

print()
print("=" * 70)
print("【5】解析結果をCSV保存")
print("=" * 70)


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
        print(f"★ 保存成功")
        print(path)

    except Exception as e:

        print()
        print("[エラー]")
        print(f"保存失敗: {e}")


# 機種別
save_csv(
    OUTPUT_MACHINE,
    [
        "機種名",
        "台数",
        "総G数",
        "平均G数",
        "総差枚",
        "平均差枚",
        "プラス台数",
        "プラス率",
        "最大差枚",
        "最小差枚",
    ],
    machine_results
)


# 台番号別
save_csv(
    OUTPUT_NUMBER,
    [
        "台番号",
        "機種名",
        "日数",
        "平均G数",
        "総差枚",
        "平均差枚",
        "プラス日数",
        "プラス率",
    ],
    number_results
)


# 日別
save_csv(
    OUTPUT_DAILY,
    [
        "日付",
        "台数",
        "総G数",
        "平均G数",
        "総差枚",
        "平均差枚",
        "プラス台数",
        "プラス率",
    ],
    daily_results
)


# ============================================================
# 終了
# ============================================================

print()
print("=" * 70)
print("基礎解析終了")
print("=" * 70)

print()
print("出力ファイル:")

print(OUTPUT_MACHINE)
print(OUTPUT_NUMBER)
print(OUTPUT_DAILY)

print()
print("all_data.csv は変更していません。")

input("Enterキーで終了...")