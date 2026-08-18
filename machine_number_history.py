import csv
from pathlib import Path
from collections import defaultdict


# ============================================================
# 基本設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "maruhan_maebashi"

INPUT_FILE = DATA_DIR / "all_data.csv"

OUTPUT_HISTORY = DATA_DIR / "machine_number_history.csv"
OUTPUT_MACHINE_CHANGE = DATA_DIR / "machine_change_history.csv"
OUTPUT_NUMBER_SUMMARY = DATA_DIR / "machine_number_summary.csv"


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
print("台番号・機種変更履歴解析")
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
# 日付順に並べ替え
# ============================================================

rows.sort(
    key=lambda x: (
        x["台番号"],
        x["日付"]
    )
)


# ============================================================
# 1. 台番号ごとの履歴作成
# ============================================================

print()
print("=" * 70)
print("【1】台番号ごとの時系列履歴")
print("=" * 70)


number_history = defaultdict(list)


for row in rows:

    number_history[row["台番号"]].append(row)


history_results = []


for number in sorted(
    number_history.keys(),
    key=lambda x: int(x) if str(x).isdigit() else 999999
):

    records = number_history[number]

    for index, row in enumerate(records):

        previous_machine = ""

        machine_changed = "いいえ"

        if index > 0:

            previous_machine = records[
                index - 1
            ]["機種名"]

            if previous_machine != row["機種名"]:

                machine_changed = "はい"

        history_results.append({

            "台番号": number,

            "日付": row["日付"],

            "機種名": row["機種名"],

            "前日の機種": previous_machine,

            "機種変更": machine_changed,

            "G数": to_int(row["G数"]),

            "差枚": to_int(row["差枚"]),

            "BB": to_int(row["BB"]),

            "RB": to_int(row["RB"]),

            "合成確率": row["合成確率"],

            "BB確率": row["BB確率"],

            "RB確率": row["RB確率"],
        })


print(
    f"履歴データ: {len(history_results):,}行"
)


# ============================================================
# 2. 機種変更履歴
# ============================================================

print()
print("=" * 70)
print("【2】機種変更履歴")
print("=" * 70)


machine_change_results = []


for number in sorted(
    number_history.keys(),
    key=lambda x: int(x) if str(x).isdigit() else 999999
):

    records = number_history[number]

    previous_machine = None

    for row in records:

        current_machine = row["機種名"]

        if (
            previous_machine is not None
            and current_machine != previous_machine
        ):

            machine_change_results.append({

                "台番号": number,

                "日付": row["日付"],

                "変更前機種": previous_machine,

                "変更後機種": current_machine,
            })

        previous_machine = current_machine


print(
    f"機種変更件数: {len(machine_change_results)}件"
)


if machine_change_results:

    print()
    print("機種変更の例:")

    for result in machine_change_results[:20]:

        print(
            f"台{result['台番号']} / "
            f"{result['日付']} / "
            f"{result['変更前機種']} "
            f"→ "
            f"{result['変更後機種']}"
        )


# ============================================================
# 3. 台番号別サマリー
# ============================================================

print()
print("=" * 70)
print("【3】台番号別サマリー")
print("=" * 70)


number_summary_data = []


for number in sorted(
    number_history.keys(),
    key=lambda x: int(x) if str(x).isdigit() else 999999
):

    records = number_history[number]

    total_days = len(records)

    total_games = sum(
        to_int(row["G数"])
        for row in records
    )

    total_difference = sum(
        to_int(row["差枚"])
        for row in records
    )

    positive_days = sum(
        1
        for row in records
        if to_int(row["差枚"]) > 0
    )

    negative_days = sum(
        1
        for row in records
        if to_int(row["差枚"]) < 0
    )

    zero_days = sum(
        1
        for row in records
        if to_int(row["差枚"]) == 0
    )

    average_games = (
        total_games / total_days
        if total_days
        else 0
    )

    average_difference = (
        total_difference / total_days
        if total_days
        else 0
    )

    positive_rate = (
        positive_days / total_days * 100
        if total_days
        else 0
    )

    machines = []

    for row in records:

        machine = row["機種名"]

        if machine not in machines:
            machines.append(machine)

    machine_change_count = max(
        0,
        len(machines) - 1
    )

    max_difference = max(
        to_int(row["差枚"])
        for row in records
    )

    min_difference = min(
        to_int(row["差枚"])
        for row in records
    )

    current_machine = records[-1]["機種名"]

    number_summary_data.append({

        "台番号": number,

        "現在の機種": current_machine,

        "稼働日数": total_days,

        "機種種類数": len(machines),

        "機種変更回数": machine_change_count,

        "平均G数": round(
            average_games,
            1
        ),

        "総差枚": total_difference,

        "平均差枚": round(
            average_difference,
            1
        ),

        "プラス日数": positive_days,

        "マイナス日数": negative_days,

        "差枚0日数": zero_days,

        "プラス率": round(
            positive_rate,
            2
        ),

        "最大差枚": max_difference,

        "最小差枚": min_difference,

        "使用機種": " / ".join(machines),
    })


# ============================================================
# 平均差枚順
# ============================================================

number_summary_data.sort(
    key=lambda x: x["平均差枚"],
    reverse=True
)


print()
print("平均差枚 TOP20")


for rank, result in enumerate(
    number_summary_data[:20],
    1
):

    print(
        f"{rank:2d}. "
        f"台{result['台番号']} / "
        f"{result['現在の機種']} / "
        f"{result['稼働日数']}日 / "
        f"平均差枚 "
        f"{result['平均差枚']:+.1f}枚 / "
        f"プラス率 "
        f"{result['プラス率']:.1f}%"
    )


# ============================================================
# 4. 機種ごとの台番号履歴
# ============================================================

print()
print("=" * 70)
print("【4】台番号 × 機種別集計")
print("=" * 70)


number_machine_data = defaultdict(
    lambda: {
        "days": 0,
        "games": 0,
        "difference": 0,
        "positive": 0,
        "max_difference": None,
        "min_difference": None,
    }
)


for row in rows:

    key = (
        row["台番号"],
        row["機種名"]
    )

    data = number_machine_data[key]

    difference = to_int(row["差枚"])
    games = to_int(row["G数"])

    data["days"] += 1
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


number_machine_results = []


for (
    number,
    machine
), data in number_machine_data.items():

    days = data["days"]

    average_games = (
        data["games"] / days
        if days
        else 0
    )

    average_difference = (
        data["difference"] / days
        if days
        else 0
    )

    positive_rate = (
        data["positive"] / days * 100
        if days
        else 0
    )

    number_machine_results.append({

        "台番号": number,

        "機種名": machine,

        "日数": days,

        "平均G数": round(
            average_games,
            1
        ),

        "総差枚": data["difference"],

        "平均差枚": round(
            average_difference,
            1
        ),

        "プラス日数": data["positive"],

        "プラス率": round(
            positive_rate,
            2
        ),

        "最大差枚": data["max_difference"],

        "最小差枚": data["min_difference"],
    })


number_machine_results.sort(
    key=lambda x: x["平均差枚"],
    reverse=True
)


print()
print("台番号 × 機種別 平均差枚 TOP20")


for rank, result in enumerate(
    number_machine_results[:20],
    1
):

    print(
        f"{rank:2d}. "
        f"台{result['台番号']} / "
        f"{result['機種名']} / "
        f"{result['日数']}日 / "
        f"平均差枚 "
        f"{result['平均差枚']:+.1f}枚"
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
        print("★ 保存成功")
        print(path)

    except Exception as e:

        print()
        print("[エラー]")
        print(f"保存失敗: {e}")


# ------------------------------------------------------------
# 履歴CSV
# ------------------------------------------------------------

save_csv(

    OUTPUT_HISTORY,

    [
        "台番号",
        "日付",
        "機種名",
        "前日の機種",
        "機種変更",
        "G数",
        "差枚",
        "BB",
        "RB",
        "合成確率",
        "BB確率",
        "RB確率",
    ],

    history_results
)


# ------------------------------------------------------------
# 機種変更CSV
# ------------------------------------------------------------

save_csv(

    OUTPUT_MACHINE_CHANGE,

    [
        "台番号",
        "日付",
        "変更前機種",
        "変更後機種",
    ],

    machine_change_results
)


# ------------------------------------------------------------
# 台番号サマリーCSV
# ------------------------------------------------------------

save_csv(

    OUTPUT_NUMBER_SUMMARY,

    [
        "台番号",
        "現在の機種",
        "稼働日数",
        "機種種類数",
        "機種変更回数",
        "平均G数",
        "総差枚",
        "平均差枚",
        "プラス日数",
        "マイナス日数",
        "差枚0日数",
        "プラス率",
        "最大差枚",
        "最小差枚",
        "使用機種",
    ],

    number_summary_data
)


# ============================================================
# 終了
# ============================================================

print()
print("=" * 70)
print("台番号・機種変更履歴解析終了")
print("=" * 70)

print()
print("出力ファイル:")

print(OUTPUT_HISTORY)
print(OUTPUT_MACHINE_CHANGE)
print(OUTPUT_NUMBER_SUMMARY)

print()
print("all_data.csv は変更していません。")

input("Enterキーで終了...")