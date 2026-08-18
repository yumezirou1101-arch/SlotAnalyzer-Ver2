import csv
from pathlib import Path
from collections import defaultdict


# ============================================================
# 並び投入パターン解析
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "maruhan_maebashi"

INPUT_FILE = DATA_DIR / "all_data.csv"

OUTPUT_2 = DATA_DIR / "machine_number" / "neighbor_2.csv"
OUTPUT_3 = DATA_DIR / "machine_number" / "neighbor_3.csv"
OUTPUT_4 = DATA_DIR / "machine_number" / "neighbor_4.csv"
OUTPUT_SUMMARY = DATA_DIR / "machine_number" / "neighbor_summary.csv"


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
# CSV保存
# ============================================================

def save_csv(path, fieldnames, rows):

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
    print("★ 保存成功")
    print(path)


# ============================================================
# 開始
# ============================================================

print("=" * 70)
print("並び投入パターン解析")
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


print(
    f"読み込みデータ: {len(rows):,}行"
)


required_columns = [
    "日付",
    "機種名",
    "台番号",
    "G数",
    "差枚",
]


missing = [
    column
    for column in required_columns
    if column not in rows[0]
]


if missing:

    print()
    print("[エラー]")
    print("必要な列がありません。")

    for column in missing:
        print(column)

    input("Enterキーで終了...")
    raise SystemExit


print("必要な列: OK")


# ============================================================
# 日付 → 台番号 → データ
# ============================================================

daily_data = defaultdict(dict)

for row in rows:

    try:
        number = int(row["台番号"])
    except ValueError:
        continue

    daily_data[
        row["日付"]
    ][number] = row


dates = sorted(daily_data.keys())


print()
print(f"解析日数: {len(dates)}日")


# ============================================================
# 並びを抽出する関数
# ============================================================

def find_runs(number_data, run_length):

    numbers = sorted(
        number_data.keys()
    )

    results = []

    if len(numbers) < run_length:
        return results


    for i in range(
        len(numbers) - run_length + 1
    ):

        selected = numbers[
            i:i + run_length
        ]

        # 台番号が完全な連番か確認
        is_consecutive = all(
            selected[j] + 1 == selected[j + 1]
            for j in range(
                len(selected) - 1
            )
        )

        if not is_consecutive:
            continue


        selected_rows = [
            number_data[number]
            for number in selected
        ]


        differences = [
            to_int(row["差枚"])
            for row in selected_rows
        ]


        games = [
            to_int(row["G数"])
            for row in selected_rows
        ]


        machines = [
            row["機種名"]
            for row in selected_rows
        ]


        positive_count = sum(
            1
            for value in differences
            if value > 0
        )


        total_difference = sum(
            differences
        )


        average_difference = (
            total_difference / run_length
        )


        positive_rate = (
            positive_count
            / run_length
            * 100
        )


        same_machine = (
            len(set(machines)) == 1
        )


        # 全台プラス
        all_positive = (
            positive_count == run_length
        )


        # 半分以上プラス
        majority_positive = (
            positive_count
            >= (run_length + 1) // 2
        )


        results.append({

            "台番号開始":
                selected[0],

            "台番号終了":
                selected[-1],

            "台番号並び":
                "-".join(
                    str(number)
                    for number in selected
                ),

            "機種並び":
                " / ".join(machines),

            "同一機種":
                "はい"
                if same_machine
                else "いいえ",

            "台数":
                run_length,

            "総G数":
                sum(games),

            "平均G数":
                round(
                    sum(games)
                    / run_length,
                    1
                ),

            "総差枚":
                total_difference,

            "平均差枚":
                round(
                    average_difference,
                    1
                ),

            "プラス台数":
                positive_count,

            "プラス率":
                round(
                    positive_rate,
                    1
                ),

            "全台プラス":
                "はい"
                if all_positive
                else "いいえ",

            "半分以上プラス":
                "はい"
                if majority_positive
                else "いいえ",

        })


    return results


# ============================================================
# 全日程の並びを収集
# ============================================================

all_runs = {
    2: [],
    3: [],
    4: []
}


for date in dates:

    number_data = daily_data[date]


    for run_length in [2, 3, 4]:

        runs = find_runs(
            number_data,
            run_length
        )


        for result in runs:

            result["日付"] = date

            all_runs[
                run_length
            ].append(result)


# ============================================================
# 2台並び
# ============================================================

runs_2 = all_runs[2]


print()
print("=" * 70)
print("【2台並び】")
print("=" * 70)

print(
    f"2台連番組み合わせ数: "
    f"{len(runs_2):,}"
)


# 全台プラスを表示
all_positive_2 = [
    row
    for row in runs_2
    if row["全台プラス"] == "はい"
]


print(
    f"2台ともプラス: "
    f"{len(all_positive_2):,}件"
)


for row in sorted(
    all_positive_2,
    key=lambda x: x["平均差枚"],
    reverse=True
)[:20]:

    print(
        f"{row['日付']} / "
        f"台{row['台番号並び']} / "
        f"{row['機種並び']} / "
        f"平均差枚 "
        f"{row['平均差枚']:+.1f}枚"
    )


# ============================================================
# 3台並び
# ============================================================

runs_3 = all_runs[3]


print()
print("=" * 70)
print("【3台並び】")
print("=" * 70)

print(
    f"3台連番組み合わせ数: "
    f"{len(runs_3):,}"
)


all_positive_3 = [
    row
    for row in runs_3
    if row["全台プラス"] == "はい"
]


print(
    f"3台ともプラス: "
    f"{len(all_positive_3):,}件"
)


for row in sorted(
    all_positive_3,
    key=lambda x: x["平均差枚"],
    reverse=True
)[:20]:

    print(
        f"{row['日付']} / "
        f"台{row['台番号並び']} / "
        f"{row['機種並び']} / "
        f"平均差枚 "
        f"{row['平均差枚']:+.1f}枚"
    )


# ============================================================
# 4台並び
# ============================================================

runs_4 = all_runs[4]


print()
print("=" * 70)
print("【4台並び】")
print("=" * 70)

print(
    f"4台連番組み合わせ数: "
    f"{len(runs_4):,}"
)


all_positive_4 = [
    row
    for row in runs_4
    if row["全台プラス"] == "はい"
]


print(
    f"4台ともプラス: "
    f"{len(all_positive_4):,}件"
)


for row in sorted(
    all_positive_4,
    key=lambda x: x["平均差枚"],
    reverse=True
)[:20]:

    print(
        f"{row['日付']} / "
        f"台{row['台番号並び']} / "
        f"{row['機種並び']} / "
        f"平均差枚 "
        f"{row['平均差枚']:+.1f}枚"
    )


# ============================================================
# 繰り返し出現する並び
# ============================================================

def repeated_runs(runs):

    grouped = defaultdict(list)

    for row in runs:

        grouped[
            row["台番号並び"]
        ].append(row)


    results = []


    for numbers, records in grouped.items():

        days = len(records)


        total_difference = sum(
            row["総差枚"]
            for row in records
        )


        total_units = sum(
            row["台数"]
            for row in records
        )


        positive_rate = (
            sum(
                row["プラス率"]
                for row in records
            )
            / days
        )


        all_positive_count = sum(
            1
            for row in records
            if row["全台プラス"] == "はい"
        )


        same_machine_count = sum(
            1
            for row in records
            if row["同一機種"] == "はい"
        )


        results.append({

            "台番号並び":
                numbers,

            "出現日数":
                days,

            "平均差枚":
                round(
                    total_difference
                    / total_units,
                    1
                ),

            "平均プラス率":
                round(
                    positive_rate,
                    1
                ),

            "全台プラス回数":
                all_positive_count,

            "同一機種回数":
                same_machine_count,

            "初回日":
                min(
                    row["日付"]
                    for row in records
                ),

            "最終日":
                max(
                    row["日付"]
                    for row in records
                ),
        })


    return results


repeat_2 = repeated_runs(runs_2)
repeat_3 = repeated_runs(runs_3)
repeat_4 = repeated_runs(runs_4)


# ============================================================
# 繰り返し並びランキング
# ============================================================

print()
print("=" * 70)
print("【繰り返し出現する2台並び】")
print("=" * 70)


repeat_2.sort(
    key=lambda x: (
        x["出現日数"],
        x["平均差枚"],
        x["平均プラス率"]
    ),
    reverse=True
)


for rank, row in enumerate(
    repeat_2[:20],
    1
):

    print(
        f"{rank:2d}. "
        f"台{row['台番号並び']} / "
        f"{row['出現日数']}日 / "
        f"平均差枚 "
        f"{row['平均差枚']:+.1f}枚 / "
        f"平均プラス率 "
        f"{row['平均プラス率']:.1f}% / "
        f"全台プラス "
        f"{row['全台プラス回数']}回"
    )


print()
print("=" * 70)
print("【繰り返し出現する3台並び】")
print("=" * 70)


repeat_3.sort(
    key=lambda x: (
        x["出現日数"],
        x["平均差枚"],
        x["平均プラス率"]
    ),
    reverse=True
)


for rank, row in enumerate(
    repeat_3[:20],
    1
):

    print(
        f"{rank:2d}. "
        f"台{row['台番号並び']} / "
        f"{row['出現日数']}日 / "
        f"平均差枚 "
        f"{row['平均差枚']:+.1f}枚 / "
        f"平均プラス率 "
        f"{row['平均プラス率']:.1f}% / "
        f"全台プラス "
        f"{row['全台プラス回数']}回"
    )


print()
print("=" * 70)
print("【繰り返し出現する4台並び】")
print("=" * 70)


repeat_4.sort(
    key=lambda x: (
        x["出現日数"],
        x["平均差枚"],
        x["平均プラス率"]
    ),
    reverse=True
)


for rank, row in enumerate(
    repeat_4[:20],
    1
):

    print(
        f"{rank:2d}. "
        f"台{row['台番号並び']} / "
        f"{row['出現日数']}日 / "
        f"平均差枚 "
        f"{row['平均差枚']:+.1f}枚 / "
        f"平均プラス率 "
        f"{row['平均プラス率']:.1f}% / "
        f"全台プラス "
        f"{row['全台プラス回数']}回"
    )


# ============================================================
# 保存
# ============================================================

fields = [
    "日付",
    "台番号開始",
    "台番号終了",
    "台番号並び",
    "機種並び",
    "同一機種",
    "台数",
    "総G数",
    "平均G数",
    "総差枚",
    "平均差枚",
    "プラス台数",
    "プラス率",
    "全台プラス",
    "半分以上プラス",
]


save_csv(
    OUTPUT_2,
    fields,
    runs_2
)


save_csv(
    OUTPUT_3,
    fields,
    runs_3
)


save_csv(
    OUTPUT_4,
    fields,
    runs_4
)


# ============================================================
# サマリー
# ============================================================

summary_rows = []


for run_length, runs in [
    (2, runs_2),
    (3, runs_3),
    (4, runs_4),
]:

    if not runs:
        continue


    all_positive_count = sum(
        1
        for row in runs
        if row["全台プラス"] == "はい"
    )


    majority_positive_count = sum(
        1
        for row in runs
        if row["半分以上プラス"] == "はい"
    )


    same_machine_count = sum(
        1
        for row in runs
        if row["同一機種"] == "はい"
    )


    average_difference = (
        sum(
            row["平均差枚"]
            for row in runs
        )
        / len(runs)
    )


    average_positive_rate = (
        sum(
            row["プラス率"]
            for row in runs
        )
        / len(runs)
    )


    summary_rows.append({

        "並び台数":
            run_length,

        "総組み合わせ数":
            len(runs),

        "全台プラス件数":
            all_positive_count,

        "全台プラス率":
            round(
                all_positive_count
                / len(runs)
                * 100,
                2
            ),

        "半分以上プラス件数":
            majority_positive_count,

        "同一機種件数":
            same_machine_count,

        "同一機種率":
            round(
                same_machine_count
                / len(runs)
                * 100,
                2
            ),

        "平均差枚":
            round(
                average_difference,
                1
            ),

        "平均プラス率":
            round(
                average_positive_rate,
                1
            ),
    })


save_csv(

    OUTPUT_SUMMARY,

    [
        "並び台数",
        "総組み合わせ数",
        "全台プラス件数",
        "全台プラス率",
        "半分以上プラス件数",
        "同一機種件数",
        "同一機種率",
        "平均差枚",
        "平均プラス率",
    ],

    summary_rows
)


# ============================================================
# 終了
# ============================================================

print()
print("=" * 70)
print("★★★★★ 並び投入パターン解析終了 ★★★★★")
print("=" * 70)

print()
print("出力ファイル:")

print(OUTPUT_2)
print(OUTPUT_3)
print(OUTPUT_4)
print(OUTPUT_SUMMARY)

print()
print("all_data.csv は変更していません。")

input("Enterキーで終了...")