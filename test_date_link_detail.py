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
    print(" アナスロ 一覧ページ内データ調査")
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
        # 新しいタブを使用
        # ----------------------------------------------------

        page = await context.new_page()

        print()
        print("新しいタブを作成しました。")

        # ----------------------------------------------------
        # 一覧ページを開く
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

        # ページが安定するまで待つ
        await asyncio.sleep(8)

        print()
        print("現在URL:")
        print(page.url)

        # ----------------------------------------------------
        # URL確認
        # ----------------------------------------------------

        if "ana-slo.com" not in page.url:

            print()
            print("[エラー]")
            print("アナスロ一覧ページではありません。")

            return

        # ----------------------------------------------------
        # HTML取得
        # ----------------------------------------------------

        try:

            html = await page.content()

        except Exception as e:

            print()
            print("[エラー] HTML取得失敗")
            print(e)

            return

        print()
        print(
            f"HTML文字数: {len(html):,}"
        )

        # ----------------------------------------------------
        # all_data_table確認
        # ----------------------------------------------------

        if "all_data_table" in html:

            print()
            print(
                "all_data_table: 発見"
            )

        else:

            print()
            print(
                "all_data_table: なし"
            )

        # ----------------------------------------------------
        # 8/2関連文字列を検索
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("8/2関連データを検索")
        print("=" * 60)

        search_words = [
            "2026-08-02",
            "2026/08/02",
            "08/02",
            "8/2",
            "2026年8月2日",
        ]

        for word in search_words:

            count = html.count(word)

            print(
                f"{word}: {count}件"
            )

        # ----------------------------------------------------
        # 8/2 URLを探す
        # ----------------------------------------------------

        target_url_text = (
            "2026-08-02-マルハンメガシティ前橋インター-data"
        )

        encoded_target = (
            "2026-08-02-"
        )

        print()
        print("=" * 60)
        print("8/2 URL確認")
        print("=" * 60)

        if encoded_target in html:

            print()
            print(
                "8/2 URL文字列: 発見"
            )

        else:

            print()
            print(
                "8/2 URL文字列: なし"
            )

        # ----------------------------------------------------
        # 8/2周辺HTMLを抽出
        # ----------------------------------------------------

        positions = []

        search_position = 0

        while True:

            position = html.find(
                "2026-08-02",
                search_position
            )

            if position == -1:
                break

            positions.append(position)

            search_position = position + 1

        print()
        print(
            f"2026-08-02 出現位置: {len(positions)}件"
        )

        # 最初の5箇所だけ表示
        for number, position in enumerate(
            positions[:5],
            start=1
        ):

            print()
            print(
                f"--- 出現箇所 {number} ---"
            )

            start = max(
                0,
                position - 500
            )

            end = min(
                len(html),
                position + 1000
            )

            print(
                html[start:end]
            )

        # ----------------------------------------------------
        # リンク一覧
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("8/2リンク確認")
        print("=" * 60)

        links = page.locator("a")

        link_count = await links.count()

        print()
        print(
            f"ページ内リンク総数: {link_count}"
        )

        found = 0

        for i in range(link_count):

            try:

                href = await links.nth(i).get_attribute(
                    "href"
                )

                if not href:
                    continue

                decoded = unquote(href)

                if (
                    "2026-08-02" in decoded
                    and "-data" in decoded
                ):

                    found += 1

                    text = await links.nth(i).inner_text()

                    print()
                    print(
                        f"8/2リンク {found}"
                    )

                    print(
                        "TEXT:",
                        repr(text)
                    )

                    print(
                        "HREF:",
                        href
                    )

            except Exception:
                continue

        print()
        print(
            f"8/2リンク発見数: {found}"
        )

        # ----------------------------------------------------
        # HTML保存
        # ----------------------------------------------------

        from pathlib import Path

        output_file = (
            Path(__file__).resolve().parent
            / "date_list_full.html"
        )

        output_file.write_text(
            html,
            encoding="utf-8"
        )

        print()
        print(
            f"HTML保存: {output_file}"
        )

        print()
        print("=" * 60)
        print("調査終了")
        print("=" * 60)

        print()
        print(
            "今回は8/2ページへの遷移をしていません。"
        )

        print(
            "Chromeはそのままで構いません。"
        )


if __name__ == "__main__":

    asyncio.run(main())