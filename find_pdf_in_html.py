from bs4 import BeautifulSoup

with open("page_new.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

count = 0

for a in soup.find_all("a"):
    href = a.get("href")

    if href and ".pdf" in href.lower():
        count += 1
        print("=" * 60)
        print("PDF", count)
        print(href)

print()
print("PDF件数:", count)