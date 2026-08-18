from bs4 import BeautifulSoup

# 保存したHTMLを開く
with open("page.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

results = []

for a in soup.find_all("a", href=True):
    text = a.get_text(strip=True)
    href = a["href"]

    # 日付リンクだけ取得
    if text.startswith("202"):
        if href.startswith("/"):
            href = "https://ana-slo.com" + href

        results.append(f"{text}|{href}")

# 保存
with open("urls.txt", "w", encoding="utf-8") as f:
    for line in results:
        f.write(line + "\n")

print("保存件数:", len(results))
print("urls.txt を作成しました。")