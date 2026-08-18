import csv
from pathlib import Path
from collections import defaultdict


# ============================================================
# 投入パターン解析
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "maruhan_maebashi"

INPUT_FILE = DATA_DIR / "all_data.csv"

OUTPUT_MACHINE = DATA_DIR / "machine_number" / "investment_machine.csv"
OUTPUT_NUMBER = DATA_DIR / "machine_number" / "investment_number.csv"
OUTPUT_PATTERN = DATA_DIR / "machine_number" / "investment_pattern.csv"
OUTPUT_DAILY = DATA_DIR / "machine_number" / "investment_daily.csv"


# ============================================================
# 数値変換
# ============================================================

def to_int(value):
    if value is None:
        return 0

    value = str(value).strip()
    value = value.replace(",", "")
    value = value.replace("+", "")

    if value == "":
        return 0

    try:
        return int(value)
    except ValueError:
        return 0


# ============================================================
# CSV保存
# ============================================================

def save_csv(path, fieldnames, rows):

    path.parent.mkdir(parents=True, exist_ok=True)

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
    print("★ 保存成功")
    print(path)


# ============================================================
# 開始
# ============================================================

print("=" * 70)
print("投入パターン解析")
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
# 読み込み
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


print(f"読み込みデータ: {len(rows):,}行")


required = [
    "日付",
    "機種名",
    "台番号",
    "G数",
    "差枚"
]


missing = [
    column
    for column in required
    if column not in reader.fieldnames
]


if missing:

    print()
    print("[エラー]")
    print("必要な列がありません:")

    for column in missing:
        print(column)

    input("Enterキーで終了...")
    raise SystemExit


print("必要な列: OK")


# ============================================================
# データ整理
# ============================================================

daily = defaultdict(dict)

for row in rows:

    try:
        number = int(row["台番号"])
    except ValueError:
        continue

    difference = to_int(row["差枚"])
    games = to_int(row["G数"])

    row["_number"] = number
    row["_difference"] = difference
    row["_games"] = games

    daily[row["日付"]][number] = row


dates = sorted(daily.keys())

print()
print(f"解析日数: {len(dates)}日")


# ============================================================
# ① 台番号別投入傾向
# ============================================================

number_stats = defaultdict(list)

for date in dates:

    for number, row in daily[date].items():

        number_stats[number].append(row)


number_results = []


for number, records in number_stats.items():

    differences = [
        row["_difference"]
        for row in records
    ]

    positive_count = sum(
        1
        for value in differences
        if value > 0
    )

    high_count = sum(
        1
        for value in differences
        if value >= 1000
    )

    strong_count = sum(
        1
        for value in differences
        if value >= 2000
    )

    total_difference = sum(differences)

    average_difference = (
        total_difference / len(records)
    )

    positive_rate = (
        positive_count
        / len(records)
        * 100
    )

    high_rate = (
        high_count
        / len(records)
        * 100
    )

    strong_rate = (
        strong_count
        / len(records)
        * 100
    )


    # 最近3日
    recent = records[-3:]

    recent_difference = sum(
        row["_difference"]
        for row in recent
    ) / len(recent)


    recent_positive = sum(
        1
        for row in recent
        if row["_difference"] > 0
    )

    recent_positive_rate = (
        recent_positive
        / len(recent)
        * 100
    )


    # 総合スコア
    score = (
        average_difference / 50
        + positive_rate * 0.30
        + high_rate * 0.20
        + strong_rate * 0.15
        + recent_positive_rate * 0.15
    )


    # 上限・下限
    score = max(0, min(100, score))


    current_machine = records[-1]["機種名"]


    number_results.append({

        "台番号": number,

        "現在機種": current_machine,

        "分析日数": len(records),

        "平均差枚": round(
            average_difference,
            1
        ),

        "プラス率": round(
            positive_rate,
            1
        ),

        "1000枚以上率": round(
            high_rate,
            1
        ),

        "2000枚以上率": round(
            strong_rate,
            1
        ),

        "最近3日平均差枚": round(
            recent_difference,
            1
        ),

        "最近3日プラス率": round(
            recent_positive_rate,
            1
        ),

        "総合スコア": round(
            score,
            1
        )

    })


number_results.sort(
    key=lambda x: x["総合スコア"],
    reverse=True
)


print()
print("=" * 70)
print("【台番号別 投入傾向】")
print("=" * 70)


for rank, row in enumerate(
    number_results[:30],
    1
):

    print(
        f"{rank:2d}. "
        f"台{row['台番号']} / "
        f"{row['現在機種']} / "
        f"平均差枚 "
        f"{row['平均差枚']:+.1f}枚 / "
        f"プラス率 "
        f"{row['プラス率']:.1f}% / "
        f"最近3日 "
        f"{row['最近3日平均差枚']:+.1f}枚 / "
        f"スコア "
        f"{row['総合スコア']:.1f}"
    )


# ============================================================
# ② 機種別投入傾向
# ============================================================

machine_stats = defaultdict(list)

for row in rows:

    machine_stats[
        row["機種名"]
    ].append(row)


machine_results = []


for machine, records in machine_stats.items():

    differences = [
        row["_difference"]
        for row in records
    ]

    positive_count = sum(
        1
        for value in differences
        if value > 0
    )

    high_count = sum(
        1
        for value in differences
        if value >= 1000
    )

    strong_count = sum(
        1
        for value in differences
        if value >= 2000
    )

    average_difference = (
        sum(differences)
        / len(differences)
    )

    positive_rate = (
        positive_count
        / len(records)
        * 100
    )

    high_rate = (
        high_count
        / len(records)
        * 100
    )

    strong_rate = (
        strong_count
        / len(records)
        * 100
    )


    score = (
        average_difference / 60
        + positive_rate * 0.35
        + high_rate * 0.15
        + strong_rate * 0.10
    )

    score = max(0, min(100, score))


    machine_results.append({

        "機種名": machine,

        "データ数": len(records),

        "平均差枚": round(
            average_difference,
            1
        ),

        "プラス率": round(
            positive_rate,
            1
        ),

        "1000枚以上率": round(
            high_rate,
            1
        ),

        "2000枚以上率": round(
            strong_rate,
            1
        ),

        "総合スコア": round(
            score,
            1
        )

    })


machine_results.sort(
    key=lambda x: x["総合スコア"],
    reverse=True
)


print()
print("=" * 70)
print("【機種別 投入傾向】")
print("=" * 70)


for rank, row in enumerate(
    machine_results[:30],
    1
):

    print(
        f"{rank:2d}. "
        f"{row['機種名']} / "
        f"{row['データ数']}台日 / "
        f"平均差枚 "
        f"{row['平均差枚']:+.1f}枚 / "
        f"プラス率 "
        f"{row['プラス率']:.1f}% / "
        f"スコア "
        f"{row['総合スコア']:.1f}"
    )


# ============================================================
# ③ 日別投入傾向
# ============================================================

daily_results = []


for date in dates:

    records = list(
        daily[date].values()
    )

    differences = [
        row["_difference"]
        for row in records
    ]

    positive_count = sum(
        1
        for value in differences
        if value > 0
    )

    strong_count = sum(
        1
        for value in differences
        if value >= 1000
    )

    total_difference = sum(
        differences
    )

    average_difference = (
        total_difference
        / len(records)
    )

    positive_rate = (
        positive_count
        / len(records)
        * 100
    )


    daily_results.append({

        "日付": date,

        "台数": len(records),

        "総差枚": total_difference,

        "平均差枚": round(
            average_difference,
            1
        ),

        "プラス台数": positive_count,

        "プラス率": round(
            positive_rate,
            1
        ),

        "1000枚以上台数": strong_count

    })


print()
print("=" * 70)
print("【日別投入傾向】")
print("=" * 70)


for row in daily_results:

    print(
        f"{row['日付']} / "
        f"総差枚 "
        f"{row['総差枚']:+,}枚 / "
        f"平均 "
        f"{row['平均差枚']:+.1f}枚 / "
        f"プラス率 "
        f"{row['プラス率']:.1f}% / "
        f"+1000枚以上 "
        f"{row['1000枚以上台数']}台"
    )


# ============================================================
# ④ 台番号 × 機種の組み合わせ
# ============================================================

combination_stats = defaultdict(list)

for row in rows:

    key = (
        row["_number"],
        row["機種名"]
    )

    combination_stats[key].append(row)


combination_results = []


for (number, machine), records in combination_stats.items():

    differences = [
        row["_difference"]
        for row in records
    ]

    positive_count = sum(
        1
        for value in differences
        if value > 0
    )

    average_difference = (
        sum(differences)
        / len(differences)
    )

    positive_rate = (
        positive_count
        / len(records)
        * 100
    )


    # 同じ台・同じ機種が長期間続いているほど信頼度を高くする
    continuity = min(
        len(records) / len(dates),
        1
    )


    score = (
        average_difference / 60
        + positive_rate * 0.40
        + continuity * 20
    )

    score = max(0, min(100, score))


    combination_results.append({

        "台番号": number,

        "機種名": machine,

        "日数": len(records),

        "平均差枚": round(
            average_difference,
            1
        ),

        "プラス率": round(
            positive_rate,
            1
        ),

        "継続率": round(
            continuity * 100,
            1
        ),

        "総合スコア": round(
            score,
            1
        )

    })


combination_results.sort(
    key=lambda x: x["総合スコア"],
    reverse=True
)


print()
print("=" * 70)
print("【台番号 × 機種 投入傾向】")
print("=" * 70)


for rank, row in enumerate(
    combination_results[:30],
    1
):

    print(
        f"{rank:2d}. "
        f"台{row['台番号']} / "
        f"{row['機種名']} / "
        f"{row['日数']}日 / "
        f"平均差枚 "
        f"{row['平均差枚']:+.1f}枚 / "
        f"プラス率 "
        f"{row['プラス率']:.1f}% / "
        f"スコア "
        f"{row['総合スコア']:.1f}"
    )


# ============================================================
# CSV保存
# ============================================================

save_csv(
    OUTPUT_NUMBER,

    [
        "台番号",
        "現在機種",
        "分析日数",
        "平均差枚",
        "プラス率",
        "1000枚以上率",
        "2000枚以上率",
        "最近3日平均差枚",
        "最近3日プラス率",
        "総合スコア"
    ],

    number_results
)


save_csv(
    OUTPUT_MACHINE,

    [
        "機種名",
        "データ数",
        "平均差枚",
        "プラス率",
        "1000枚以上率",
        "2000枚以上率",
        "総合スコア"
    ],

    machine_results
)


save_csv(
    OUTPUT_DAILY,

    [
        "日付",
        "台数",
        "総差枚",
        "平均差枚",
        "プラス台数",
        "プラス率",
        "1000枚以上台数"
    ],

    daily_results
)


save_csv(
    OUTPUT_PATTERN,

    [
        "台番号",
        "機種名",
        "日数",
        "平均差枚",
        "プラス率",
        "継続率",
        "総合スコア"
    ],

    combination_results
)


# ============================================================
# 終了
# ============================================================

print()
print("=" * 70)
print("★★★★★ 投入パターン解析終了 ★★★★★")
print("=" * 70)

print()
print("出力ファイル:")

print(OUTPUT_NUMBER)
print(OUTPUT_MACHINE)
print(OUTPUT_DAILY)
print(OUTPUT_PATTERN)

print()
print("all_data.csv は変更していません。")

input("Enterキーで終了...")