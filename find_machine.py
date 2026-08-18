from bs4 import BeautifulSoup

with open("browser_html.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

# 「マイV」を含む script を探す
for script in soup.find_all("script"):
    text = script.get_text()

    if "マイV" in text:
        print("=" * 80)
        print(text[:3000])   # 最初の3000文字だけ表示
        break