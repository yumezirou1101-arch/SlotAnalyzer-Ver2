from bs4 import BeautifulSoup

with open("browser_html.html", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

# 「マイV」を含む script を探す
for script in soup.find_all("script"):
    if "マイV" in script.get_text():
        li = script.find_parent("li")

        print(li.prettify()[:8000])   # 最初の8000文字だけ表示
        break