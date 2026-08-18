from bs4 import BeautifulSoup
import re

# HTML読み込み
with open("browser_html.html", "r", encoding="utf-8") as f:
    html = f.read()

print("HTML文字数:", len(html))

# 調べたいキーワード
keywords = [
    "BB",
    "RB",
    "ART",
    "AT",
    "初当たり",
    "機械割",
    "出率",
    "設定",
    "小役",
    "CZ",
    "合成",
    "G数",
    "差枚",
    "平均",
]

print()
print("=" * 70)
print("キーワード検索")
print("=" * 70)

for keyword in keywords:
    count = html.count(keyword)
    print(f"{keyword:10s}: {count}")

# --------------------------------------------------
# 周辺のHTMLを表示する
# --------------------------------------------------

print()
print("=" * 70)
print("重要キーワード周辺のHTML")
print("=" * 70)

targets = [
    "初当たり",
    "機械割",
    "出率",
    "設定",
    "CZ",
]

for keyword in targets:

    print()
    print("-" * 70)
    print(f"【{keyword}】")
    print("-" * 70)

    positions = [m.start() for m in re.finditer(re.escape(keyword), html)]

    # 最大5か所まで
    for pos in positions[:5]:

        start = max(0, pos - 300)
        end = min(len(html), pos + 500)

        text = html[start:end]

        # 改行を整理
        text = re.sub(r"\s+", " ", text)

        print(text)
        print()