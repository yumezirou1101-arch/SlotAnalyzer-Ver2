from bs4 import BeautifulSoup
import re
from collections import defaultdict

with open("browser_html.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

machines = defaultdict(list)

for li in soup.find_all("li"):

    script = li.find("script")
    table = li.find("table")

    if script is None or table is None:
        continue

    js = script.get_text()

    m = re.search(
        r'text:\s*\[\s*"(\d+)"\s*,\s*"([^"]+)"\s*\]',
        js
    )

    if not m:
        continue

    dai = m.group(1)
    machine = m.group(2)

    rows = table.find_all("tr")

    if len(rows) < 2:
        continue

    headers = [
        cell.get_text(strip=True)
        for cell in rows[0].find_all(["th", "td"])
    ]

    values = [
        cell.get_text(strip=True)
        for cell in rows[1].find_all(["th", "td"])
    ]

    machines[dai].append({
        "machine": machine,
        "headers": headers,
        "values": values
    })


# ==========================================
# 集計
# ==========================================

print("=" * 70)
print("台番号ごとのテーブル数")
print("=" * 70)

table_count = defaultdict(int)

for dai, records in machines.items():
    table_count[len(records)] += 1

for count, num in sorted(table_count.items()):
    print(f"{count}テーブル: {num}台")


# ==========================================
# ヘッダー形式の集計
# ==========================================

print()
print("=" * 70)
print("テーブル形式の種類")
print("=" * 70)

header_count = defaultdict(int)

for records in machines.values():
    for record in records:
        header = tuple(record["headers"])
        header_count[header] += 1

for header, count in sorted(
    header_count.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(count, "件:", list(header))


# ==========================================
# 3件以上ある台
# ==========================================

print()
print("=" * 70)
print("3テーブル以上ある台")
print("=" * 70)

for dai, records in machines.items():

    if len(records) >= 3:

        print()
        print("台番号:", dai)
        print("機種名:", records[0]["machine"])

        for i, record in enumerate(records, 1):

            print(
                i,
                record["headers"],
                record["values"]
            )