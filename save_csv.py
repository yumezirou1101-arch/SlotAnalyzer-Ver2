from bs4 import BeautifulSoup
import re
import csv

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

    g = tds[0].get_text(strip=True)
    bb = tds[1].get_text(strip=True)
    rb = tds[2].get_text(strip=True)
    total = tds[3].get_text(strip=True)

    machines.append([
        dai,
        machine,
        g,
        bb,
        rb,
        total,
        last_diff
    ])

# CSV保存
with open("slot_data.csv", "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)

    writer.writerow([
        "台番号",
        "機種名",
        "G数",
        "BB",
        "RB",
        "合成",
        "差枚"
    ])

    writer.writerows(machines)

print("保存完了")
print("件数:", len(machines))
print("ファイル名: slot_data.csv")