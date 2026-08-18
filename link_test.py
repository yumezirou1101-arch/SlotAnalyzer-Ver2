from playwright.sync_api import sync_playwright

with sync_playwright() as p:

    browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")

    page = browser.contexts[0].pages[1]

    page.wait_for_timeout(5000)

    html = page.content()

    with open("page2.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("保存しました")

    input()
