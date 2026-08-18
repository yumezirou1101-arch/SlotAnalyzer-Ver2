from playwright.sync_api import sync_playwright

with sync_playwright() as p:

    browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")

    page = browser.contexts[0].pages[0]

    print("フレーム数:", len(page.frames))

    for i, frame in enumerate(page.frames):
        print("-" * 40)
        print("Frame", i)
        print("URL:", frame.url)
        print("タイトル:", frame.title())

        try:
            print("aタグ:", frame.evaluate("document.querySelectorAll('a').length"))
        except Exception as e:
            print("取得失敗:", e)

    input("Enter")