import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


CDP_URL = "http://127.0.0.1:9222"


async def main():

    print()
    print("=" * 60)
    print("9222 Chrome 実表示DOM取得テスト")
    print("=" * 60)

    async with async_playwright() as p:

        print()
        print("9222 Chromeへ接続中...")

        browser = await p.chromium.connect_over_cdp(
            CDP_URL
        )

        print("Chrome接続成功。")

        context = browser.contexts[0]

        pages = context.pages

        print()
        print(
            f"Chromeタブ数: {len(pages)}"
        )

        # ==================================================
        # 全タブを調査
        # ==================================================

        for i, page in enumerate(pages):

            print()
            print("=" * 60)
            print(f"タブ [{i}]")
            print("=" * 60)

            try:

                print("URL:")
                print(page.url)

                print()
                print("TITLE:")

                try:
                    print(await page.title())
                except Exception as e:
                    print("[取得失敗]", e)

                # ------------------------------------------
                # 少し待つ
                # ------------------------------------------

                await asyncio.sleep(3)

                # ------------------------------------------
                # document.documentElement.outerHTML
                # ------------------------------------------

                try:

                    outer_html = await page.evaluate(
                        """
                        () => document.documentElement
                            ? document.documentElement.outerHTML
                            : ""
                        """
                    )

                except Exception as e:

                    print()
                    print(
                        "outerHTML取得失敗:",
                        e
                    )

                    outer_html = ""

                print()
                print(
                    f"outerHTML文字数: {len(outer_html):,}"
                )

                # ------------------------------------------
                # body.innerText
                # ------------------------------------------

                try:

                    body_text = await page.evaluate(
                        """
                        () => document.body
                            ? document.body.innerText
                            : ""
                        """
                    )

                except Exception as e:

                    print()
                    print(
                        "body.innerText取得失敗:",
                        e
                    )

                    body_text = ""

                print()
                print(
                    f"body文字数: {len(body_text):,}"
                )

                # ------------------------------------------
                # アナスロ本体判定
                # ------------------------------------------

                if "all_data_table" in outer_html:

                    print()
                    print(
                        "★ all_data_table 発見"
                    )

                else:

                    print()
                    print(
                        "all_data_table なし"
                    )

                # ------------------------------------------
                # 8/2判定
                # ------------------------------------------

                for word in [
                    "2026-08-02",
                    "2026/08/02",
                    "マルハンメガシティ前橋インター",
                    "台番号",
                    "差枚",
                ]:

                    count = outer_html.count(word)

                    print(
                        f"{word}: {count}件"
                    )

                # ------------------------------------------
                # 本文の先頭を表示
                # ------------------------------------------

                print()
                print("--- body先頭500文字 ---")

                if body_text:

                    print(
                        body_text[:500]
                    )

                else:

                    print(
                        "(body文字なし)"
                    )

                # ------------------------------------------
                # 8/2ページなら保存
                # ------------------------------------------

                if (
                    "2026-08-02" in page.url
                    and "ana-slo.com" in page.url
                ):

                    output_file = (
                        Path(__file__).resolve().parent
                        / "chrome_8_2_outerhtml.html"
                    )

                    output_file.write_text(
                        outer_html,
                        encoding="utf-8"
                    )

                    print()
                    print(
                        f"★ 8/2 HTML保存:"
                        f" {output_file}"
                    )

            except Exception as e:

                print()
                print(
                    "[タブ処理エラー]"
                )
                print(e)

        print()
        print("=" * 60)
        print("テスト終了")
        print("=" * 60)

        print()
        print(
            "Chromeはそのままで構いません。"
        )


if __name__ == "__main__":

    asyncio.run(main())