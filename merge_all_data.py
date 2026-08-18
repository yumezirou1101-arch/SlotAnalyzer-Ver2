import csv
import glob
import os
from datetime import datetime


# ==========================================
# 設定
# ==========================================

DATA_DIR = "data"
OUTPUT_FILE = "all_data.csv"

WEEKDAYS = [
    "月",
    "火",
    "水",
    "木",
    "金",
    "土",
    "日"
]


# ==========================================
# 日付から曜日を取得
# ==========================================

def get_weekday(date_text):

    try:
        date = datetime.strptime(
            date_text,
            "%Y-%m-%d"
        )

        return WEEKDAYS[date.weekday()]

    except Exception:
        return ""


# ==========================================
# CSVファイルを取得
# ==========================================

files = sorted(
    glob.glob(
        os.path.join(
            DATA_DIR,
            "*.csv"
        )
    )
)

# all_data.csv は対象外
files = [
    f
    for f in files
    if os.path.basename(f) != OUTPUT_FILE
]


# ==========================================
# 開始
# ==========================================

print()
print("================================")
print(" SlotAnalyzer 複数日データ統合")
print("================================")
print()

print(
    "対象CSV:",
    len(files),
    "ファイル"
)

print()


if not files:

    print(
        "dataフォルダにCSVがありません。"
    )

    exit()


# ==========================================
# データ読み込み
# ==========================================

all_rows = []

fieldnames = None


for filename in files:

    print(
        "読み込み:",
        filename
    )

    try:

        with open(
            filename,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            rows = list(reader)

            if not rows:

                print(
                    "  → データなし"
                )

                continue


            # ----------------------------------
            # 最初のCSVから列構成を取得
            # ----------------------------------

            original_fields = reader.fieldnames

            if fieldnames is None:

                fieldnames = [
                    "日付",
                    "曜日"
                ] + [
                    x
                    for x in original_fields
                    if x != "日付"
                ]


            # ----------------------------------
            # 列構成チェック
            # ----------------------------------

            expected_original_fields = [
                x
                for x in fieldnames
                if x != "曜日"
            ]

            if original_fields != expected_original_fields:

                print(
                    "  → 列構成が違うためスキップ"
                )

                continue


            # ----------------------------------
            # 曜日を追加
            # ----------------------------------

            for row in rows:

                date = row.get(
                    "日付",
                    ""
                )

                weekday = get_weekday(
                    date
                )

                new_row = {
                    "日付": date,
                    "曜日": weekday
                }

                for key, value in row.items():

                    if key != "日付":

                        new_row[key] = value

                all_rows.append(
                    new_row
                )


            print(
                "  →",
                len(rows),
                "台"
            )


    except Exception as e:

        print(
            "  → 読み込みエラー:",
            e
        )


# ==========================================
# データ確認
# ==========================================

print()

print(
    "統合データ:",
    len(all_rows),
    "行"
)


if not all_rows:

    print(
        "統合できるデータがありません。"
    )

    exit()


# ==========================================
# 日付順・台番号順に並べ替え
# ==========================================

all_rows.sort(
    key=lambda x: (
        x.get(
            "日付",
            ""
        ),
        int(
            x.get(
                "台番号",
                0
            )
        )
        if x.get(
            "台番号",
            ""
        ).isdigit()
        else 0
    )
)


# ==========================================
# all_data.csv 保存
# ==========================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        all_rows
    )


# ==========================================
# 完了
# ==========================================

dates = sorted(
    set(
        row.get(
            "日付",
            ""
        )
        for row in all_rows
    )
)

print()
print("================================")
print(" 統合完了")
print("================================")
print()

print(
    "日数:",
    len(dates)
)

print(
    "総台データ:",
    len(all_rows)
)

print(
    "保存ファイル:",
    OUTPUT_FILE
)

print()

print(
    "収録日:"
)

for date in dates:

    weekday = get_weekday(
        date
    )

    print(
        f"  {date} ({weekday})"
    )

print()