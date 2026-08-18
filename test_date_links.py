import asyncio
from urllib.parse import unquote

from playwright.async_api import async_playwright


async def main():

    print("=" * 60)
    print("アナスロ 日付リンク確認テスト")
    print("=" * 60)

    async with async_playwright() as p:

        print()
        print("9222 Chromeへ接続中...")

        browser = await p.chromium.connect_over_cdp(
            "http://127.0.0.1:9222"
        )

        print("Chrome接続成功。")

        page = None

        # ----------------------------------------------------
        # 既存のアナスロページを探す
        # ----------------------------------------------------

        for context in browser.contexts:

            for existing_page in context.pages:

                try:

                    if existing_page.is_closed():
                        continue

                    url = existing_page.url

                    if (
                        "ana-slo.com" in url
                        and "2026-08-01" in url
                    ):

                        page = existing_page
                        break

                except Exception:
                    pass

            if page is not None:
                break

        if page is None:

            print()
            print("[エラー]")
            print("8/1のアナスロページが見つかりません。")

            return

        print()
        print("対象ページ:")
        print(page.url)

        print()
        print("ページHTMLを確認します...")

        html = await page.content()

        print(
            f"HTML文字数: {len(html):,}"
        )

        # ----------------------------------------------------
        # 全リンク取得
        # ----------------------------------------------------

        links = await page.locator("a").evaluate_all(
            """
            elements => elements.map(a => ({
                text: (a.innerText || a.textContent || '').trim(),
                href: a.href
            }))
            """
        )

        print()
        print(
            f"リンク総数: {len(links)}"
        )

        print()
        print("=" * 60)
        print("8/2関連リンク")
        print("=" * 60)

        found = 0

        for link in links:

            text = link.get("text", "")
            href = link.get("href", "")

            decoded_href = unquote(href)

            if (
                "2026-08-02" in decoded_href
                or "8/2" in text
                or "08/02" in text
                or "8月2日" in text
            ):

                found += 1

                print()
                print(f"[{found}]")
                print(
                    f"TEXT: {text}"
                )
                print(
                    f"HREF: {href}"
                )
                print(
                    f"DECODED: {decoded_href}"
                )

        if found == 0:

            print()
            print(
                "8/2に直接つながるリンクは見つかりませんでした。"
            )

        # ----------------------------------------------------
        # 日付らしいリンクを広く確認
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("日付らしいリンク一覧")
        print("=" * 60)

        date_found = 0

        for link in links:

            text = link.get("text", "")
            href = link.get("href", "")

            decoded_href = unquote(href)

            if re_match_date(decoded_href):

                date_found += 1

                print()
                print(
                    f"[{date_found}] TEXT: {text}"
                )

                print(
                    f"URL: {decoded_href}"
                )

                if date_found >= 30:
                    break

        print()
        print(
            f"確認した日付リンク: {date_found}"
        )

        print()
        print("テスト終了。")

        await browser.close()


def re_match_date(text):

    import re

    return re.search(
        r"20\d{2}-\d{2}-\d{2}",
        text
    ) is not None


if __name__ == "__main__":

    asyncio.run(main())