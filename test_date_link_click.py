import asyncio
from urllib.parse import unquote

from playwright.async_api import async_playwright


CDP_URL = "http://127.0.0.1:9222"

LIST_URL = (
    "https://ana-slo.com/"
    "ホールデータ/群馬県/"
    "マルハンメガシティ前橋インター-データ一覧/"
)

TARGET_DATE = "2026-08-02"


async def main():

    print()
    print("=" * 60)
    print(" アナスロ 8/2 強制クリックテスト")
    print("=" * 60)

    async with async_playwright() as p:

        print()
        print("9222 Chromeへ接続中...")

        browser = await p.chromium.connect_over_cdp(
            CDP_URL
        )

        print("Chrome接続成功。")

        context = browser.contexts[0]

        # ----------------------------------------------------
        # アナスロページを探す
        # ----------------------------------------------------

        page = None

        for existing_page in context.pages:

            try:

                if "ana-slo.com" in existing_page.url:

                    page = existing_page
                    break

            except Exception:
                pass

        if page is None:

            page = await context.new_page()

        # ----------------------------------------------------
        # 一覧ページ
        # ----------------------------------------------------

        print()
        print("一覧ページを開きます。")

        try:

            await page.goto(
                LIST_URL,
                wait_until="domcontentloaded",
                timeout=60000
            )

        except Exception as e:

            print()
            print("[注意]")
            print(e)

        await asyncio.sleep(5)

        print()
        print("現在URL:")
        print(page.url)

        # ----------------------------------------------------
        # 8/2リンクを検索
        # ----------------------------------------------------

        links = page.locator("a")

        count = await links.count()

        target_indexes = []

        for i in range(count):

            try:

                href = await links.nth(i).get_attribute(
                    "href"
                )

                if not href:
                    continue

                decoded = unquote(href)

                if (
                    TARGET_DATE in decoded
                    and "-data" in decoded
                ):

                    target_indexes.append(i)

            except Exception:
                continue

        print()
        print(
            f"8/2リンク数: {len(target_indexes)}"
        )

        if not target_indexes:

            print()
            print(
                "[エラー] 8/2リンクがありません。"
            )

            return

        # ----------------------------------------------------
        # 8/2リンクをJavaScriptクリック
        # ----------------------------------------------------

        index = target_indexes[0]

        locator = links.nth(index)

        href = await locator.get_attribute(
            "href"
        )

        print()
        print("対象リンク:")
        print(href)

        print()
        print(
            "通常クリックではなく"
        )

        print(
            "JavaScriptクリックを実行します。"
        )

        # ----------------------------------------------------
        # URL変更を監視
        # ----------------------------------------------------

        try:

            async with page.expect_navigation(
                wait_until="domcontentloaded",
                timeout=60000
            ):

                await locator.evaluate(
                    """
                    el => {
                        el.click();
                    }
                    """
                )

        except Exception as e:

            print()
            print(
                "[Navigation待機結果]"
            )

            print(e)

            await asyncio.sleep(10)

        # ----------------------------------------------------
        # 結果確認
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("クリック後の状態")
        print("=" * 60)

        print()
        print("URL:")
        print(page.url)

        try:

            title = await page.title()

        except Exception:

            title = ""

        print()
        print("TITLE:")
        print(title)

        # ----------------------------------------------------
        # HTML
        # ----------------------------------------------------

        try:

            html = await page.content()

        except Exception as e:

            print()
            print(
                "[エラー] HTML取得失敗"
            )

            print(e)

            return

        print()
        print(
            f"HTML文字数: {len(html):,}"
        )

        # ----------------------------------------------------
        # 判定
        # ----------------------------------------------------

        if "all_data_table" in html:

            print()
            print(
                "★★★★★ 成功 ★★★★★"
            )

            print()
            print(
                "all_data_table を発見しました。"
            )

            # 台数だけ簡易確認
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(
                html,
                "html.parser"
            )

            table = soup.find(
                "table",
                id="all_data_table"
            )

            if table:

                rows = table.find_all("tr")

                print()
                print(
                    f"all_data_table行数: {len(rows)}"
                )

        else:

            print()
            print(
                "[失敗]"
            )

            print(
                "all_data_table はありません。"
            )

            # エラーHTML保存
            from pathlib import Path

            output = (
                Path(__file__).resolve().parent
                / "test_date_link_click_error.html"
            )

            output.write_text(
                html,
                encoding="utf-8"
            )

            print()
            print(
                f"HTML保存: {output}"
            )

        print()
        print(
            "Chromeはそのままで構いません。"
        )


if __name__ == "__main__":

    asyncio.run(main())