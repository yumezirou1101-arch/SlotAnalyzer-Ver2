from pathlib import Path

html = Path("page_new.html").read_text(
    encoding="utf-8",
    errors="ignore"
)

keywords = [
    "差枚",
    "G数",
    "BB",
    "RB",
    "ART",
    "機種",
    "台番号",
    "slot",
    "graph",
    "machine",
    "__NEXT_DATA__",
    "__INITIAL_STATE__"
]

for key in keywords:
    print("=" * 60)
    print("検索:", key)

    count = html.count(key)
    print("件数:", count)

    if count > 0:
        pos = html.find(key)
        start = max(0, pos - 300)
        end = min(len(html), pos + 500)

        print(html[start:end])
        print()