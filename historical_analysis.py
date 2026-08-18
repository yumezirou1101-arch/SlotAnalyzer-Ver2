import csv
import os
from collections import defaultdict


# ==========================================
# 設定
# ==========================================

INPUT_FILE = "all_data.csv"


# ==========================================
# CSV読み込み
# ==========================================

if not os.path.exists(INPUT_FILE):

    print()
    print("all_data.csv が見つかりません。")
    print()
    print("先に merge_all_data.py を実行してください。")
    exit()


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)
    rows = list(reader)


if not rows:

    print()
    print("all_data.csv にデータがありません。")
    exit()


# ==========================================
# 数値変換
# ==========================================

def to_int(value):

    try:
        return int(
            str(value).replace(",", "").strip()
        )
    except:
        return 0


# ==========================================
# 差枚・G数を数値化
# ==========================================

for row in rows:

    row["_G数"] = to_int(
        row.get("G数", 0)
    )

    row["_差枚"] = to_int(
        row.get("差枚", 0)
    )


# ==========================================
# 基本統計計算
# ==========================================

def calc_stats(data):

    if not data:

        return {
            "台数": 0,
            "平均G": 0,
            "平均差枚": 0,
            "プラス台率": 0,
            "1000枚以上率": 0,
            "3000枚以上率": 0
        }

    count = len(data)

    total_g = sum(
        x["_G数"]
        for x in data
    )

    total_diff = sum(
        x["_差枚"]
        for x in data
    )

    plus_count = sum(
        x["_差枚"] > 0
        for x in data
    )

    plus1000_count = sum(
        x["_差枚"] >= 1000
        for x in data
    )

    plus3000_count = sum(
        x["_差枚"] >= 3000
        for x in data
    )

    return {
        "台数": count,
        "平均G": total_g / count,
        "平均差枚": total_diff / count,
        "プラス台率": plus_count / count * 100,
        "1000枚以上率": plus1000_count / count * 100,
        "3000枚以上率": plus3000_count / count * 100
    }


# ==========================================
# 表示
# ==========================================

def print_stats(stats):

    print(
        f"台数: {stats['台数']}"
    )

    print(
        f"平均G数: {stats['平均G']:.1f}"
    )

    print(
        f"平均差枚: {stats['平均差枚']:.1f}"
    )

    print(
        f"プラス台率: {stats['プラス台率']:.1f}%"
    )

    print(
        f"1000枚以上率: {stats['1000枚以上率']:.1f}%"
    )

    print(
        f"3000枚以上率: {stats['3000枚以上率']:.1f}%"
    )


# ==========================================
# 開始
# ==========================================

print()
print("========================================")
print(" SlotAnalyzer 過去データ基礎分析")
print("========================================")
print()

print(
    "読み込みデータ:",
    INPUT_FILE
)

print(
    "総台データ:",
    len(rows)
)

dates = sorted(
    set(
        row.get("日付", "")
        for row in rows
    )
)

print(
    "収録日数:",
    len(dates)
)

print(
    "収録期間:",
    dates[0],
    "～",
    dates[-1]
)


# ==========================================
# 全体
# ==========================================

print()
print("========================================")
print(" 【全体】")
print("========================================")
print()

stats = calc_stats(rows)

print_stats(stats)


# ==========================================
# 日付別
# ==========================================

by_date = defaultdict(list)

for row in rows:

    by_date[
        row.get("日付", "")
    ].append(row)


print()
print("========================================")
print(" 【日付別】")
print("========================================")
print()

for date in sorted(by_date):

    stats = calc_stats(
        by_date[date]
    )

    weekday = ""

    if by_date[date]:

        weekday = by_date[date][0].get(
            "曜日",
            ""
        )

    print(
        f"{date} ({weekday})"
    )

    print(
        f"  台数: {stats['台数']} "
        f"平均G: {stats['平均G']:.1f} "
        f"平均差枚: {stats['平均差枚']:.1f} "
        f"プラス率: {stats['プラス台率']:.1f}%"
    )


# ==========================================
# 曜日別
# ==========================================

by_weekday = defaultdict(list)

for row in rows:

    by_weekday[
        row.get("曜日", "")
    ].append(row)


weekday_order = [
    "月",
    "火",
    "水",
    "木",
    "金",
    "土",
    "日"
]


print()
print("========================================")
print(" 【曜日別】")
print("========================================")
print()

for weekday in weekday_order:

    if weekday not in by_weekday:
        continue

    stats = calc_stats(
        by_weekday[weekday]
    )

    print(
        f"{weekday}曜日"
    )

    print(
        f"  台数: {stats['台数']} "
        f"平均G: {stats['平均G']:.1f} "
        f"平均差枚: {stats['平均差枚']:.1f} "
        f"プラス率: {stats['プラス台率']:.1f}% "
        f"3000枚以上率: {stats['3000枚以上率']:.1f}%"
    )


# ==========================================
# 機種別
# ==========================================

by_machine = defaultdict(list)

for row in rows:

    by_machine[
        row.get("機種名", "")
    ].append(row)


machine_results = []

for machine, data in by_machine.items():

    stats = calc_stats(data)

    machine_results.append(
        (
            machine,
            stats
        )
    )


# 平均差枚の高い順
machine_results.sort(
    key=lambda x: x[1]["平均差枚"],
    reverse=True
)


print()
print("========================================")
print(" 【機種別】")
print("========================================")
print()

for machine, stats in machine_results:

    print(
        f"{machine}"
    )

    print(
        f"  台数: {stats['台数']} "
        f"平均G: {stats['平均G']:.1f} "
        f"平均差枚: {stats['平均差枚']:.1f} "
        f"プラス率: {stats['プラス台率']:.1f}% "
        f"1000枚以上率: {stats['1000枚以上率']:.1f}% "
        f"3000枚以上率: {stats['3000枚以上率']:.1f}%"
    )


# ==========================================
# 台番号別
# ==========================================

by_machine_number = defaultdict(list)

for row in rows:

    key = (
        row.get("台番号", ""),
        row.get("機種名", "")
    )

    by_machine_number[key].append(
        row
    )


machine_number_results = []

for key, data in by_machine_number.items():

    stats = calc_stats(data)

    machine_number_results.append(
        (
            key,
            stats
        )
    )


# 平均差枚の高い順
machine_number_results.sort(
    key=lambda x: x[1]["平均差枚"],
    reverse=True
)


print()
print("========================================")
print(" 【台番号別 平均差枚ランキング】")
print("========================================")
print()

for rank, (key, stats) in enumerate(
    machine_number_results[:30],
    1
):

    machine_number, machine = key

    print(
        f"{rank:2d}. "
        f"{machine_number} "
        f"{machine} "
        f"データ数:{stats['台数']} "
        f"平均差枚:{stats['平均差枚']:.1f} "
        f"プラス率:{stats['プラス台率']:.1f}%"
    )


# ==========================================
# 3000枚以上ランキング
# ==========================================

big_win_rows = sorted(
    rows,
    key=lambda x: x["_差枚"],
    reverse=True
)


print()
print("========================================")
print(" 【差枚ランキング TOP30】")
print("========================================")
print()

for rank, row in enumerate(
    big_win_rows[:30],
    1
):

    print(
        f"{rank:2d}. "
        f"{row.get('日付', '')} "
        f"{row.get('台番号', '')} "
        f"{row.get('機種名', '')} "
        f"G:{row['_G数']} "
        f"差枚:{row['_差枚']}"
    )


# ==========================================
# 完了
# ==========================================

print()
print("========================================")
print(" 分析完了")
print("========================================")
print()