import requests
from bs4 import BeautifulSoup

url = "https://ana-slo.com/2026-07-31-%e3%83%9e%e3%83%ab%e3%83%8f%e3%83%b3%e3%83%a1%e3%82%ac%e3%82%b7%e3%83%86%e3%82%a3%e5%89%8d%e6%a9%8b%e3%82%a4%e3%83%b3%e3%82%bf%e3%83%bc-data/"

headers = {
    "User-Agent": "Mozilla/5.0"
}

html = requests.get(url, headers=headers, timeout=30)

print("ステータス:", html.status_code)

soup = BeautifulSoup(html.text, "html.parser")

print("HTML文字数:", len(html.text))

links = soup.find_all("a")

print("リンク数:", len(links))

for a in links:
    href = a.get("href")
    text = a.get_text(strip=True)

    if href and ("pdf" in href.lower() or "PDF" in text):
        print()
        print(text)
        print(href)