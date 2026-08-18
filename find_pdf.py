from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")

    context = browser.contexts[0]
    page = context.pages[0]

    print("現在のページ")
    print(page.url)

    # 一覧ページが表示されていることを確認
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_timeout(3000)

    print("2026/07/31(金) を探します...")

    # 日付リンクをクリック
    page.get_by_text("2026/07/31(金)", exact=True).click()

    print("クリックしました")
    page.wait_for_timeout(5000)

    print("現在のURL")
    print(page.url)

    print("\nPDFリンク検索中...\n")

    links = page.locator("a").evaluate_all("""
els => els.map(e => ({
    text: e.innerText,
    href: e.href
}))
""")

    found = False

    for link in links:
        href = link["href"]
        text = link["text"]

        if href and ".pdf" in href.lower():
            found = True
            print("PDF発見！")
            print(text)
            print(href)
            print("-" * 60)

    if not found:
        print("PDFリンクは見つかりませんでした。")

input("Enterで終了")