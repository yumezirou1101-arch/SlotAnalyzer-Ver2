import asyncio
from pathlib import Path
from urllib.parse import unquote

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright


CDP_URL = "http://127.0.0.1:9222"

TARGET_DATE = "2026-08-02"

LIST_URL = (
    "https://ana-slo.com/"
    "ホールデータ/群馬県/"
    "マルハンメガシティ前橋インター-データ一覧/"
)

DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "maruhan_maebashi"
)


# ============================================================
# 台データ抽出
# ============================================================

def extract_machine_data(html):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    table = soup.find(
        "table",
        id="all_data_table"
    )

    if table is None:
        return []

    rows = table.find_all("tr")

    if not rows:
        return []

    # ヘッダー
    header_cells = rows[0].find_all(
        ["th", "td"]
    )

    headers = [
        x.get_text(
            " ",
            strip=True
        )
        for x in header_cells
    ]

    print()
    print("ヘッダー:")
    print(headers)

    # 列番号
    def find_col(name):

        for i, header in enumerate(headers):

            if name in header:
                return i

        return None

    machine_col = find_col("機種名")
    number_col = find_col("台番号")
    game_col = find_col("G数")
    diff_col = find_col("差枚")
    bb_col = find_col("BB")
    rb_col = find_col("RB")
    art_col = find_col("ART")
    combined_col = find_col("合成")

    print()
    print("列位置:")
    print("機種名:", machine_col)
    print("台番号:", number_col)
    print("G数:", game_col)
    print("差枚:", diff_col)
    print("BB:", bb_col)
    print("RB:", rb_col)
    print("ART:", art_col)
    print("合成:", combined_col)

    if (
        machine_col is None
        or number_col is None
        or game_col is None
        or diff_col is None
    ):
        return []

    data = []

    for row in rows[1:]:

        cells = row.find_all(
            ["td", "th"]
        )

        if len(cells) <= number_col:
            continue

        values = [
            cell.get_text(
                " ",
                strip=True
            )
            for cell in cells
        ]

        machine_number = values[number_col]

        # 台番号だけを対象
        if not machine_number.isdigit():
            continue

        def get_value(index):

            if index is None:
                return ""

            if index >= len(values):
                return ""

            return values[index]

        data.append(
            {
                "日付": TARGET_DATE,
                "機種名": get_value(machine_col),
                "台番号": machine_number,
                "G数": get_value(game_col),
                "差枚": get_value(diff_col),
                "BB": get_value(bb_col),
                "RB": get_value(rb_col),
                "ART": get_value(art_col),
                "合成確率": get_value(combined_col),
            }
        )

    return data


# ============================================================
# CSV保存
# ============================================================

def save_csv(data):

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        DATA_DIR /
        f"{TARGET_DATE}.csv"
    )

    headers = [
        "日付",
        "機種名",
        "台番号",
        "G数",
        "差枚",
        "BB",
        "RB",
        "ART",
        "合成確率",
    ]

    import csv

    with open(
        output_file,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=headers
        )

        writer.writeheader()
        writer.writerows(data)

    return output_file


# ============================================================
# メイン
# ============================================================

async def main():

    print()
    print("=" * 60)
    print(" アナスロ過去データ取得テスト")
    print(" リンククリック方式")
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
        # 既存アナスロページを探す
        # ----------------------------------------------------

        page = None

        for p2 in context.pages:

            try:

                if "ana-slo.com" in p2.url:

                    page = p2
                    break

            except Exception:
                pass

        if page is None:

            page = await context.new_page()

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
        # 一覧ページHTML
        # ----------------------------------------------------

        html = await page.content()

        print()
        print(
            f"一覧ページHTML: {len(html):,}文字"
        )

        # ----------------------------------------------------
        # 8/2リンクを探す
        # ----------------------------------------------------

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        target_href = None

        for a in soup.find_all("a"):

            href = a.get("href")

            if not href:
                continue

            href_decoded = unquote(href)

            text = a.get_text(
                " ",
                strip=True
            )

            # 8/2の日付を探す
            if (
                "2026-08-02" in href_decoded
                or "2026/08/02" in text
                or "2026-08-02" in text
            ):

                if "-data" in href_decoded:

                    target_href = href

                    print()
                    print("8/2リンク発見:")
                    print(href)

                    print()
                    print("リンク文字:")
                    print(text)

                    break

        if target_href is None:

            print()
            print("[エラー]")
            print("8/2リンクが見つかりません。")

            return

        # ----------------------------------------------------
        # Playwrightでリンク要素を探す
        # ----------------------------------------------------

        print()
        print("8/2リンク要素を確認します。")

        locator = page.locator(
            f'a[href="{target_href}"]'
        )

        count = await locator.count()

        print()
        print(
            f"該当リンク数: {count}"
        )

        if count == 0:

            # URLエンコード等の違いを考慮して
            # hrefの部分一致で再検索

            links = page.locator(
                "a"
            )

            count_all = await links.count()

            found = False

            for i in range(count_all):

                try:

                    href = await links.nth(i).get_attribute(
                        "href"
                    )

                    if not href:
                        continue

                    if unquote(href) == unquote(
                        target_href
                    ):

                        locator = links.nth(i)
                        found = True
                        break

                except Exception:
                    continue

            if not found:

                print()
                print(
                    "[エラー] 8/2リンク要素を特定できません。"
                )

                return

        # ----------------------------------------------------
        # クリック
        # ----------------------------------------------------

        print()
        print("8/2リンクをクリックします。")

        try:

            async with page.expect_navigation(
                wait_until="domcontentloaded",
                timeout=60000
            ):

                await locator.first.click()

        except Exception as e:

            print()
            print(
                "通常のNavigation待機:"
            )

            print(e)

            # クリック自体は成功している可能性があるので
            # 少し待つ

            await asyncio.sleep(10)

        # ----------------------------------------------------
        # ページ確認
        # ----------------------------------------------------

        await asyncio.sleep(5)

        print()
        print("=" * 60)
        print("クリック後")
        print("=" * 60)

        print()
        print("現在URL:")
        print(page.url)

        try:

            title = await page.title()

        except Exception:

            title = ""

        print()
        print("タイトル:")
        print(title)

        # ----------------------------------------------------
        # DOM取得
        # ----------------------------------------------------

        html = await page.content()

        print()
        print(
            f"クリック後HTML文字数: {len(html):,}"
        )

        # ----------------------------------------------------
        # エラーページ判定
        # ----------------------------------------------------

        if page.url.startswith(
            "chrome-error://"
        ):

            print()
            print(
                "[失敗] Chromeエラーページです。"
            )

            return

        # ----------------------------------------------------
        # all_data_table
        # ----------------------------------------------------

        if "all_data_table" not in html:

            print()
            print(
                "[失敗]"
            )

            print(
                "all_data_table がありません。"
            )

            error_file = (
                Path(__file__).resolve().parent
                / "click_test_error.html"
            )

            error_file.write_text(
                html,
                encoding="utf-8"
            )

            print()
            print(
                f"HTML保存: {error_file}"
            )

            return

        print()
        print(
            "all_data_table: 発見"
        )

        # ----------------------------------------------------
        # 台データ抽出
        # ----------------------------------------------------

        data = extract_machine_data(
            html
        )

        print()
        print(
            f"取得台数: {len(data)}"
        )

        if not data:

            print()
            print(
                "[失敗] 台データがありません。"
            )

            return

        # ----------------------------------------------------
        # 先頭5台
        # ----------------------------------------------------

        print()
        print("先頭5台:")

        for row in data[:5]:

            print(
                row["台番号"],
                row["機種名"],
                "G:",
                row["G数"],
                "BB:",
                row["BB"],
                "RB:",
                row["RB"],
                "ART:",
                row["ART"],
                "合成:",
                row["合成確率"],
                "差枚:",
                row["差枚"]
            )

        # ----------------------------------------------------
        # CSV保存
        # ----------------------------------------------------

        output_file = save_csv(
            data
        )

        print()
        print("=" * 60)
        print("取得テスト成功")
        print("=" * 60)

        print()
        print(
            f"取得台数: {len(data)}"
        )

        print()
        print(
            f"保存ファイル: {output_file}"
        )

        print()
        print(
            "Chromeはそのままで構いません。"
        )


if __name__ == "__main__":

    asyncio.run(main())