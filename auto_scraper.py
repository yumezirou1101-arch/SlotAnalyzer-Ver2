from playwright.sync_api import sync_playwright

date = input("日付を入力してください(YYYY-MM-DD): ")

url = f"https://ana-slo.com/{date}-マルハンメガシティ前橋インター-data/"

with sync_playwright() as p:

    browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")

    page = None

    # アナスロのタブを探す
    for context in browser.contexts:
        for pg in context.pages:
            if "ana-slo.com" in pg.url:
                page = pg
                break

    # 無ければ新しく作る
    if page is None:
        context = browser.contexts[0]
        page = context.new_page()

    print("アクセス:", url)

    page.goto(url, wait_until="domcontentloaded")

    # JavaScriptの描画待ち
    page.wait_for_timeout(8000)

    print("タイトル:", page.title())

    html = page.content()

    print("HTML文字数:", len(html))
    print("機種名:", html.count("機種名"))

    with open("browser_html.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("保存しました")

    input("Enterで終了")