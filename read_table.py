from bs4 import BeautifulSoup

with open("page_new.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

tables = soup.find_all("table")

print("テーブル数:", len(tables))
print()

for i, table in enumerate(tables):
    rows = table.find_all("tr")
    print(f"===== Table {i} =====")
    print("行数:", len(rows))

    for row in rows[:3]:
        cols = [c.get_text(strip=True) for c in row.find_all(["th", "td"])]
        print(cols)

    print()