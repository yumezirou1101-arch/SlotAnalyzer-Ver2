from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")

    target = None

    for context in browser.contexts:
        for page in context.pages:
            print(page.url)

            if (
                "ana-slo.com" in page.url
                and "data" in page.url
            ):
                target = page

    if target is None:
        print("見つかりません")
        exit()

    print()
    print("見つけたページ")
    print(target.url)

    html = target.content()

    print("HTML文字数:", len(html))
    print("機種名:", html.count("機種名"))

    with open("page_from_playwright.html", "w", encoding="utf-8") as f:
        f.write(html)
