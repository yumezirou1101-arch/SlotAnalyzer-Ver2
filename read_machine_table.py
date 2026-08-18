from bs4 import BeautifulSoup

# browser_html.html を開く
with open("browser_html.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

# 全テーブル取得
tables = soup.find_all("table")

print("テーブル数:", len(tables))

for i, table in enumerate(tables):

    print("\n" + "=" * 80)
    print(f"===== Table {i} =====")
    print("=" * 80)

    # テーブル全体を表示
    print(table.get_text("\n", strip=True))