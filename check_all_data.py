import csv
from pathlib import Path
from collections import Counter
from datetime import datetime


# ============================================================
# 設定
# ============================================================

# このファイルは test フォルダにあるので、
# test フォルダの1つ上が SlotAnalyzer
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "maruhan_maebashi"

CSV_FILE = DATA_DIR / "all_data.csv"

EXPECTED_HEADERS = [
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


# ============================================================
# 開始
# ============================================================

print("=" * 70)
print("all_data.csv 総合検証")
print("=" * 70)

print()
print("確認ファイル:")
print(CSV_FILE)


# ============================================================
# ファイル存在確認
# ============================================================

if not CSV_FILE.exists():
    print()
    print("[エラー]")
    print("all_data.csv が見つかりません。")
    print()
    input("Enterキーで終了...")
    raise SystemExit


# ============================================================
# CSV読み込み
# ============================================================

print()
print("CSVを読み込みます...")

try:

    with open(
        CSV_FILE,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.reader(f)
        rows = list(reader)

except Exception as e:

    print()
    print("[エラー]")
    print(f"CSV読み込み失敗: {e}")

    input("Enterキーで終了...")
    raise SystemExit


print(f"読み込み行数: {len(rows):,}")


# ============================================================
# 空ファイル確認
# ============================================================

if not rows:

    print()
    print("[エラー]")
    print("CSVが空です。")

    input("Enterキーで終了...")
    raise SystemExit


# ============================================================
# ヘッダー確認
# ============================================================

header = rows[0]

print()
print("列数:", len(header))

print()
print("ヘッダー:")

for i, name in enumerate(header, 1):
    print(f"{i}. {name}")


header_ok = header == EXPECTED_HEADERS

print()

if header_ok:
    print("★ ヘッダー: OK")
else:

    print("★ ヘッダー: NG")

    print()
    print("想定:")
    print(EXPECTED_HEADERS)

    print()
    print("実際:")
    print(header)


# ============================================================
# データ行
# ============================================================

data_rows = rows[1:]

print()
print(f"データ行数: {len(data_rows):,}")


# ============================================================
# 列数チェック
# ============================================================

print()
print("各行の列数を確認します...")

wrong_column_rows = []

for line_no, row in enumerate(data_rows, 2):

    if len(row) != 10:

        wrong_column_rows.append(
            (line_no, len(row))
        )


if not wrong_column_rows:

    print("★ 全データ行: 10列で正常")
else:

    print(
        f"★ 列数異常: "
        f"{len(wrong_column_rows)}行"
    )

    for line_no, count in wrong_column_rows[:10]:

        print(
            f"  CSV行 {line_no}: {count}列"
        )


# ============================================================
# 日付確認
# ============================================================

print()
print("収録日を確認します...")

date_counter = Counter()

invalid_dates = []

for line_no, row in enumerate(data_rows, 2):

    if len(row) < 10:
        continue

    date_text = row[0]

    try:

        datetime.strptime(
            date_text,
            "%Y-%m-%d"
        )

        date_counter[date_text] += 1

    except ValueError:

        invalid_dates.append(
            (line_no, date_text)
        )


print(
    f"収録日数: {len(date_counter)}日"
)

print()

for date_text in sorted(date_counter):

    print(
        f"{date_text}: "
        f"{date_counter[date_text]}台"
    )


if invalid_dates:

    print()
    print(
        f"★ 日付異常: "
        f"{len(invalid_dates)}件"
    )

    for line_no, date_text in invalid_dates[:10]:

        print(
            f"  CSV行 {line_no}: {date_text}"
        )
else:

    print()
    print("★ 日付形式: OK")


# ============================================================
# 日付＋台番号 重複チェック
# ============================================================

print()
print("日付＋台番号の重複を確認します...")

machine_keys = {}
duplicates = []

for line_no, row in enumerate(data_rows, 2):

    if len(row) < 10:
        continue

    date_text = row[0]
    machine_number = row[2]

    key = (
        date_text,
        machine_number
    )

    if key in machine_keys:

        duplicates.append(
            (
                line_no,
                machine_keys[key],
                date_text,
                machine_number
            )
        )

    else:

        machine_keys[key] = line_no


if not duplicates:

    print("★ 重複: なし")
else:

    print(
        f"★ 重複: "
        f"{len(duplicates)}件"
    )

    for item in duplicates[:10]:

        line_no = item[0]
        first_line = item[1]
        date_text = item[2]
        machine_number = item[3]

        print(
            f"  {date_text} / "
            f"台番号 {machine_number} / "
            f"CSV行 {first_line} と {line_no}"
        )


# ============================================================
# 空欄チェック
# ============================================================

print()
print("空欄を確認します...")

empty_cells = []

for line_no, row in enumerate(data_rows, 2):

    for column_no, value in enumerate(row, 1):

        if value.strip() == "":

            empty_cells.append(
                (
                    line_no,
                    column_no
                )
            )


if not empty_cells:

    print("★ 空欄: なし")
else:

    print(
        f"★ 空欄: "
        f"{len(empty_cells)}セル"
    )

    for line_no, column_no in empty_cells[:10]:

        print(
            f"  CSV行 {line_no} / "
            f"列 {column_no}"
        )


# ============================================================
# 台数チェック
# ============================================================

print()
print("1日あたりの台数を確認します...")

expected_machine_count = 514

machine_count_ok = True

for date_text in sorted(date_counter):

    count = date_counter[date_text]

    if count == expected_machine_count:

        print(
            f"  ○ {date_text}: "
            f"{count}台"
        )

    else:

        machine_count_ok = False

        print(
            f"  × {date_text}: "
            f"{count}台 "
            f"(想定 {expected_machine_count}台)"
        )


# ============================================================
# 機種名チェック
# ============================================================

print()
print("機種名を確認します...")

machine_name_counter = Counter()

for row in data_rows:

    if len(row) >= 10:

        machine_name = row[1]

        machine_name_counter[
            machine_name
        ] += 1


print(
    f"機種種類数: "
    f"{len(machine_name_counter)}機種"
)

print()

for machine_name, count in (
    machine_name_counter
    .most_common(10)
):

    print(
        f"  {machine_name}: "
        f"{count}台分"
    )


# ============================================================
# 台番号チェック
# ============================================================

print()
print("台番号を確認します...")

machine_numbers = set()

for row in data_rows:

    if len(row) >= 10:

        machine_numbers.add(
            row[2]
        )


print(
    f"ユニーク台番号数: "
    f"{len(machine_numbers)}台"
)

print()
print("最初の20台番号:")

for number in sorted(
    machine_numbers,
    key=lambda x: int(x)
)[:20]:

    print(number)


# ============================================================
# 総合判定
# ============================================================

print()
print("=" * 70)
print("総合判定")
print("=" * 70)

total_ok = True


if not header_ok:
    total_ok = False


if wrong_column_rows:
    total_ok = False


if invalid_dates:
    total_ok = False


if duplicates:
    total_ok = False


if empty_cells:
    total_ok = False


if not machine_count_ok:
    total_ok = False


if len(data_rows) != 4626:
    print()
    print(
        f"[注意] 総データ行数は "
        f"4,626行ではありません: "
        f"{len(data_rows):,}行"
    )


print()

if total_ok:

    print("★★★★★ 検証成功 ★★★★★")
    print()
    print("all_data.csv は正常です。")

else:

    print("★★★ 検証結果に問題があります ★★★")


print()
print("=" * 70)
print("検証終了")
print("=" * 70)

input("Enterキーで終了...")