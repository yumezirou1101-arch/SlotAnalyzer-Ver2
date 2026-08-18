from playwright.sync_api import sync_playwright
import time

URL = "https://ana-slo.com/ホールデータ/群馬県/マルハンメガシティ前橋インター-データ一覧"

print("★★★★★ PDF保存テスト ★★★★★")

with sync_playwright() as p:

    browser = p.chromium.launch(
        channel="chrome",
        headless=False
    )

    page = browser.new_page()

    page.goto(URL)

    print("10秒待機...")
    time.sleep(10)

    links = page.locator("a").all()

    target_url = None
    target_date = None

    for link in links:

        text = link.inner_text().strip()
        href = link.get_attribute("href")

        if text.startswith("2026/07/31"):

            target_date = text
            target_url = href
            break

    print(target_date)
    print(target_url)

    print("ページを開きます")

    page.goto(target_url)

    time.sleep(5)

    print("PDF保存します")

    page.pdf(
        path="2026-07-31.pdf",
        format="A4",
        print_background=True
    )

    print("保存完了")

    input("Enterで終了")

    browser.close()