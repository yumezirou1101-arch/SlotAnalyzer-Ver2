from bs4 import BeautifulSoup

with open("browser_html.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

tables = soup.find_all("table")

n = 1187

li = tables[n].find_parent("li")

text = li.get_text("\n", strip=True)

print(text)