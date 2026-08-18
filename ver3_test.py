from playwright.sync_api import sync_playwright

from playwright.sync_api import sync_playwright

URL = "https://ana-slo.com/ホールデータ/群馬県/マルハンメガシティ前橋インター-データ一覧/"

with sync_playwright() as p:

    context = p.chromium.launch_persistent_context(
        user_data_dir="C:/ChromeDebug",
        headless=False,
    )

    page = context.new_page()

    page.goto(URL, wait_until="domcontentloaded")

    page.wait_for_timeout(5000)

    print("タイトル:", page.title())

    links = page.locator("a").all()

    print("aタグ数:", len(links))

    for link in links[:20]:
        try:
            print(
                repr(link.inner_text()),
                "=>",
                link.get_attribute("href")
            )
        except:
            pass

    input("Enterで終了")

    context.close()