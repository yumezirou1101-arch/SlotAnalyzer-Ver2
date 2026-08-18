with open("page_new.html", "r", encoding="utf-8") as f:
    html = f.read().lower()

print("pdf の出現回数:", html.count("pdf"))

pos = html.find("pdf")

if pos != -1:
    print()
    print(html[pos-500:pos+500])
else:
    print("pdfという文字はありません。")