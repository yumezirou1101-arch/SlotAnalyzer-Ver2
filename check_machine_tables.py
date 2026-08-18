from bs4 import BeautifulSoup
import re

with open("browser_html.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

art_column = 0
art_nonzero = 0
examples = []

for li in soup.find_all("li"):

    script = li.find("script")
    table = li.find("table")

    if script is None or table is None:
        continue

    js = script.get_text()

    m = re.search(
        r'text:\s*\[\s*"(\d+)"\s*,\s*"([^"]+)"\s*\]',
        js
    )

    if not m:
        continue

    dai = m.group(1)
    machine = m.group(2)

    rows = table.find_all("tr")

    if len(rows) < 2:
        continue

    headers = [
        x.get_text(strip=True)
        for x in rows[0].find_all(["th", "td"])
    ]

    values = [
        x.get_text(strip=True)
        for x in rows[1].find_all(["th", "td"])
    ]

    if "ART" not in headers:
        continue

    art_column += 1

    art_index = headers.index("ART")

    if art_index < len(values):
        art = values[art_index]

        try:
            if int(art) > 0:
                art_nonzero += 1

                if len(examples) < 20:
                    examples.append(
                        [dai, machine, art, values]
                    )
        except:
            pass

print("ART列あり:", art_column)
print("ARTが1以上:", art_nonzero)

print()
print("ARTが1以上の例")

for x in examples:
    print(x)