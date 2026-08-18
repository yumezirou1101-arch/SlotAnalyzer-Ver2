from playwright.sync_api import sync_playwright

with sync_playwright() as p:

    browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")

    context = browser.contexts[0]
    page = context.pages[0]

    # 読み込み完了を待つ
    page.wait_for_load_state("networkidle")

    print("タイトル:", page.title())
    print("URL:", page.url)

    print("リンク数:", page.locator("a").count())

    input("Enterで終了")