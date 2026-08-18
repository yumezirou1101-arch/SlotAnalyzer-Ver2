from playwright.sync_api import sync_playwright
import time

LIST_URL = "https://ana-slo.com/ホールデータ/群馬県/マルハンメガシティ前橋インター-データ一覧"

print("★★★★★ PDF Downloader ★★★★★")

with sync_playwright() as p:

    context = p.chromium.launch_persistent_context(
    user_data_dir=r"C:\Users\user\AppData\Local\Google\Chrome\User Data",
    channel="chrome",
    headless=False,
    args=["--profile-directory=Profile 1"],
)

    page = context.new_page()

    print("一覧ページを開きます...")
    page.goto(LIST_URL)

    print("10秒待機します...")
    time.sleep(10)

    links = page.locator("a").all()

    target_url = None

    for link in links:

        text = link.inner_text().strip()
        href = link.get_attribute("href")

        if text.startswith("2026/07/31"):

            target_url = href
            break

    print("取得したURL")
    print(target_url)

    print("31日のページへ移動します")

    page.goto(target_url, wait_until="domcontentloaded")

    time.sleep(10)

    print("現在のURL")
    print(page.url)

    print("ページタイトル")
    print(page.title())

    input("31日のページが表示されたら Enter を押してください")

    context.close()