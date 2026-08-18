from bs4 import BeautifulSoup
import csv
import sys


INPUT_FILE = "browser_html.html"
OUTPUT_FILE = "slot_data.csv"


print()
print("================================")
print(" SlotAnalyzer 台データ抽出")
print("================================")
print()


# ========================================
# HTML読み込み
# ========================================

try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        html = f.read()

except FileNotFoundError:
    print("browser_html.html が見つかりません。")
    sys.exit(1)


print("HTML文字数:", len(html))


# ========================================
# HTML解析
# ========================================

soup = BeautifulSoup(html, "html.parser")


# ========================================
# all_data_tableを探す
# ========================================

table = soup.find("table", id="all_data_table")


if table is None:
    print("all_data_table が見つかりません。")
    sys.exit(1)


print("データテーブル発見: all_data_table")


# ========================================
# 行を取得
# ========================================

rows = table.find_all("tr")


print("テーブル行数:", len(rows))


if len(rows) < 2:
    print("台データがありません。")
    sys.exit(1)


# ========================================
# ヘッダー取得
# ========================================

headers = [
    cell.get_text(" ", strip=True)
    for cell in rows[0].find_all(["th", "td"])
]


print()
print("ヘッダー:")
print(headers)


# ========================================
# 列位置を取得
# ========================================

try:
    machine_index = headers.index("機種名")
    number_index = headers.index("台番号")
    game_index = headers.index("G数")
    diff_index = headers.index("差枚")
    bb_index = headers.index("BB")
    rb_index = headers.index("RB")
    art_index = headers.index("ART")

except ValueError as e:
    print()
    print("必要な列が見つかりません。")
    print("エラー:", e)
    sys.exit(1)


if "合成" in headers:
    combined_index = headers.index("合成")
elif "合成確率" in headers:
    combined_index = headers.index("合成確率")
else:
    combined_index = None


print()
print("列位置:")
print("機種名:", machine_index)
print("台番号:", number_index)
print("G数:", game_index)
print("差枚:", diff_index)
print("BB:", bb_index)
print("RB:", rb_index)
print("ART:", art_index)
print("合成:", combined_index)


# ========================================
# 台データ抽出
# ========================================

machines = []


for tr in rows[1:]:

    cells = tr.find_all(["td", "th"])

    if len(cells) < 8:
        continue

    values = [
        cell.get_text(" ", strip=True)
        for cell in cells
    ]

    try:
        machine = values[machine_index].strip()
        machine_number = values[number_index].strip()
        game = values[game_index].strip()
        diff = values[diff_index].strip()
        bb = values[bb_index].strip()
        rb = values[rb_index].strip()
        art = values[art_index].strip()

        if combined_index is not None:
            combined = values[combined_index].strip()
        else:
            combined = ""

    except IndexError:
        continue

    # 空行を除外
    if machine == "":
        continue

    if machine_number == "":
        continue

    # 数値のカンマを除去
    game = game.replace(",", "")
    diff = diff.replace(",", "")
    bb = bb.replace(",", "")
    rb = rb.replace(",", "")
    art = art.replace(",", "")

    machines.append({
        "機種名": machine,
        "台番号": machine_number,
        "G数": game,
        "BB": bb,
        "RB": rb,
        "ART": art,
        "合成": combined,
        "差枚": diff
    })


# ========================================
# 抽出結果
# ========================================

print()
print("================================")
print(" 抽出結果")
print("================================")
print()

print("取得台数:", len(machines))


if len(machines) == 0:
    print()
    print("台データを取得できませんでした。")
    sys.exit(1)


# ========================================
# CSV保存
# ========================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "機種名",
            "台番号",
            "G数",
            "BB",
            "RB",
            "ART",
            "合成",
            "差枚"
        ]
    )

    writer.writeheader()
    writer.writerows(machines)


# ========================================
# 完了
# ========================================

print()
print("================================")
print(" 保存完了")
print("================================")
print()

print("取得台数:", len(machines))
print("ファイル名:", OUTPUT_FILE)


# ========================================
# 先頭5台確認
# ========================================

print()
print("先頭5台:")

for row in machines[:5]:

    print(
        row["台番号"],
        row["機種名"],
        "G:", row["G数"],
        "BB:", row["BB"],
        "RB:", row["RB"],
        "ART:", row["ART"],
        "合成:", row["合成"],
        "差枚:", row["差枚"]
    )

print()