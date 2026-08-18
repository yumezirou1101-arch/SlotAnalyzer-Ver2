import asyncio
from pathlib import Path
from urllib.parse import unquote

from playwright.async_api import async_playwright


TARGET_URL = (
    "https://ana-slo.com/"
    "2026-08-02-マルハンメガシティ前橋インター-data/"
)


async def main():

    print("=" * 60)
    print("Chrome 通常ページ遷移テスト")
    print("=" * 60)

    async with async_playwright() as p:

        print()
        print("9222 Chromeへ接続中...")

        browser = await p.chromium.connect_over_cdp(
            "http://127.0.0.1:9222"
        )

        print("Chrome接続成功。")

        page = None

        # --------------------------------------------------
        # 既存アナスロページを探す
        # --------------------------------------------------

        for context in browser.contexts:

            for existing_page in context.pages:

                try:

                    if existing_page.is_closed():
                        continue

                    if "ana-slo.com" in existing_page.url:

                        page = existing_page
                        break

                except Exception:
                    pass

            if page is not None:
                break

        if page is None:

            print()
            print("[エラー]")
            print("アナスロのタブが見つかりません。")

            return

        print()
        print("現在のページ:")
        print(page.url)

        # --------------------------------------------------
        # 8/2 URLへ通常遷移
        # --------------------------------------------------

        print()
        print("8/2ページへ通常遷移します。")

        print()
        print(TARGET_URL)

        try:

            response = await page.goto(
                TARGET_URL,
                wait_until="domcontentloaded",
                timeout=60000
            )

            if response:

                print()
                print(
                    f"Navigation HTTP Status: {response.status}"
                )

            else:

                print()
                print(
                    "Navigation Response: None"
                )

        except Exception as e:

            print()
            print(
                "[注意] page.gotoで例外"
            )

            print(e)

        # --------------------------------------------------
        # 少し待機
        # --------------------------------------------------

        print()
        print("ページを60秒確認します。")

        for i in range(60):

            await asyncio.sleep(1)

            if (i + 1) % 5 == 0:

                print(
                    f"{i + 1}秒経過..."
                )

        # --------------------------------------------------
        # 現在のURL
        # --------------------------------------------------

        current_url = page.url

        print()
        print("=" * 60)
        print("現在のChromeページ")
        print("=" * 60)

        print()
        print(
            f"URL: {current_url}"
        )

        print()
        print(
            f"URL(デコード): {unquote(current_url)}"
        )

        # --------------------------------------------------
        # タイトル
        # --------------------------------------------------

        try:

            title = await page.title()

        except Exception:

            title = ""

        print()
        print(
            f"TITLE: {title}"
        )

        # --------------------------------------------------
        # DOM取得
        # --------------------------------------------------

        try:

            html = await page.content()

        except Exception as e:

            print()
            print(
                "[エラー] DOM取得失敗"
            )

            print(e)

            return

        print()
        print(
            f"HTML文字数: {len(html):,}"
        )

        # --------------------------------------------------
        # all_data_table
        # --------------------------------------------------

        if "all_data_table" in html:

            print()
            print(
                "all_data_table: 発見"
            )

        else:

            print()
            print(
                "all_data_table: 見つかりません"
            )

        # --------------------------------------------------
        # 2026-08-02
        # --------------------------------------------------

        if "2026/08/02" in html:

            print()
            print(
                "ページ内に 2026/08/02 を確認"
            )

        elif "2026-08-02" in html:

            print()
            print(
                "ページ内に 2026-08-02 を確認"
            )

        else:

            print()
            print(
                "ページ内に8/2の日付表記が見つかりません"
            )

        # --------------------------------------------------
        # 保存
        # --------------------------------------------------

        output = Path(
            "test_2026-08-02_navigation.html"
        )

        output.write_text(
            html,
            encoding="utf-8"
        )

        print()
        print(
            f"HTML保存: {output.resolve()}"
        )

        print()
        print("=" * 60)
        print("テスト終了")
        print("=" * 60)


if __name__ == "__main__":

    asyncio.run(main())