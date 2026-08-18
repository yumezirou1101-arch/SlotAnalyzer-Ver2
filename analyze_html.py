from bs4 import BeautifulSoup

with open("page.html", "r", encoding="utf-8") as f:
    html = f.read()

print("HTML文字数:", len(html))

soup = BeautifulSoup(html, "html.parser")

links = soup.find_all("a")

print("aタグ数:", len(links))

print("-" * 50)

count = 0

for a in links:
    text = a.get_text(strip=True)
    href = a.get("href")

    if text or href:
        print("TEXT :", repr(text))
        print("HREF :", href)
        print()

        count += 1

    if count >= 30:
        break