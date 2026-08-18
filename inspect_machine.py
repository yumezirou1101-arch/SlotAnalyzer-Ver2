from bs4 import BeautifulSoup
import re

with open("browser_html.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

print("=" * 70)
print("台番号961を検索")
print("=" * 70)

count = 0

for li in soup.find_all("li"):

    script = li.find("script")

    if script is None:
        continue

    js = script.get_text()

    m = re.search(
        r'text:\s*\[\s*"961"\s*,\s*"([^"]+)"\s*\]',
        js
    )

    if not m:
        continue

    count += 1

    print()
    print("=" * 70)
    print("961番台 -", count, "件目")
    print("機種名:", m.group(1))
    print("=" * 70)

    table = li.find("table")

    if table:
        for row in table.find_all("tr"):
            print([
                cell.get_text(strip=True)
                for cell in row.find_all(["th", "td"])
            ])

    print()
    print("HTMLの先頭部分:")
    print(li.get_text(" ", strip=True)[:500])

print()
print("=" * 70)
print("961番台の件数:", count)
print("=" * 70)