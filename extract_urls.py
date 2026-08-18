from bs4 import BeautifulSoup

with open("page_new.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

urls = []

for a in soup.find_all("a"):
    text = a.get_text(strip=True)
    href = a.get("href")

    if text.startswith("2026/"):
        urls.append((text, href))

print("取得件数:", len(urls))
print()

for date, url in urls[:20]:
    print(date)
    print(url)
    print()