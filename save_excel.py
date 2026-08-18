from bs4 import BeautifulSoup
import re
from openpyxl import Workbook
from openpyxl.styles import Font

# HTML読み込み
with open("browser_html.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

machines = []

for li in soup.find_all("li"):

    script = li.find("script")
    table = li.find("table")

    if script is None or table is None:
        continue

    js = script.get_text()

    # 台番号・機種名
    m = re.search(
        r'text:\s*\[\s*"(\d+)"\s*,\s*"([^"]+)"\s*\]',
        js
    )

    if not m:
        continue

    dai = m.group(1)
    machine = m.group(2)

    # 差枚
    m2 = re.search(
        r'var\s+y_value\s*=\s*\[(.*?)\];',
        js,
        re.S
    )

    last_diff = 0

    if m2:
        values = [v.strip() for v in m2.group(1).split(",")]

        if values:
            try:
                last_diff = int(float(values[-1]))
            except:
                last_diff = 0

    # テーブル
    rows = table.find_all("tr")

    if len(rows) < 2:
        continue

    tds = rows[1].find_all("td")

    if len(tds) < 4:
        continue

    g = int(tds[0].get_text(strip=True))
    bb = int(tds[1].get_text(strip=True))
    rb = int(tds[2].get_text(strip=True))
    total = tds[3].get_text(strip=True)

    machines.append([
        int(dai),
        machine,
        g,
        bb,
        rb,
        total,
        last_diff
    ])

# ==========================
# Excel作成
# ==========================

wb = Workbook()
ws = wb.active
ws.title = "全台データ"

headers = [
    "台番号",
    "機種名",
    "G数",
    "BB",
    "RB",
    "合成",
    "差枚"
]

ws.append(headers)

# 見出しを太字
for cell in ws[1]:
    cell.font = Font(bold=True)

# データ
for row in machines:
    ws.append(row)

# 列幅
widths = {
    "A": 10,
    "B": 25,
    "C": 10,
    "D": 8,
    "E": 8,
    "F": 12,
    "G": 10
}

for col, w in widths.items():
    ws.column_dimensions[col].width = w

# 保存
wb.save("slot_data.xlsx")

print("保存完了")
print("件数:", len(machines))
print("ファイル名: slot_data.xlsx")