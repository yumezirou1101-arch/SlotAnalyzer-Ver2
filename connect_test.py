from playwright.sync_api import sync_playwright

with sync_playwright() as p:

    browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")

    print("接続成功")

    for context in browser.contexts:

        print(f"タブ数: {len(context.pages)}")

        for i, page in enumerate(context.pages):

            print("-" * 50)
            print(f"タブ{i}")
            print("タイトル:", page.title())
            print("URL:", page.url)

    input("Enterで終了")