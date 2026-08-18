import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


# ============================================================
# 設定
# ============================================================

TARGET_DATE = "2026-07-11"

TARGET_URL_KEYWORD = (
    "2026-07-11-"
    "マルハンメガシティ前橋インター"
    "-data"
)

CDP_URL = "http://127.0.0.1:9222"

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEBUG_HTML = (
    OUTPUT_DIR
    / "ana_slo_test_20260711_debug.html"
)


# ============================================================
# メイン処理
# ============================================================

async def main():

    print("=" * 70)
    print("アナスロ 7月11日 データ取得テスト")
    print("=" * 70)

    print()
    print("対象日:")
    print(TARGET_DATE)

    print()
    print("方式:")
    print("現在Chromeで開いているページを9222経由で読み取ります。")
    print("PythonからアナスロのURLは開きません。")

    print()
    print("Chrome 9222へ接続します...")

    async with async_playwright() as p:

        try:
            browser = await p.chromium.connect_over_cdp(CDP_URL)
        except Exception as e:

            print()
            print("[ERROR] Chromeへの接続に失敗しました。")
            print()
            print("Chromeが9222ポートで起動しているか確認してください。")
            print()
            print(f"詳細: {e}")

            return

        print("★ Chrome接続成功")

        contexts = browser.contexts

        print()
        print(f"ブラウザコンテキスト数: {len(contexts)}")

        if not contexts:

            print("[ERROR] ブラウザコンテキストがありません。")
            return

        context = contexts[0]

        pages = context.pages

        print(f"現在のタブ数: {len(pages)}")

        # ----------------------------------------------------
        # 現在のタブ一覧
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("【現在のChromeタブ】")
        print("=" * 70)

        for i, page in enumerate(pages):

            try:
                title = await page.title()
            except Exception:
                title = ""

            try:
                url = page.url
            except Exception:
                url = ""

            print()
            print(f"[{i}]")
            print(f"タイトル: {title}")
            print(f"URL: {url}")

        # ----------------------------------------------------
        # アナスロタブを探す
        #
        # 同じURLのタブが複数ある場合があるため、
        # URLだけでは決めない。
        #
        # 本文・タイトルを調べて、
        # 実際に内容が入っているタブを選択する。
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("【アナスロタブ検索】")
        print("=" * 70)

        target_page = None
        best_score = -1

        for i, page in enumerate(pages):

            try:

                url = page.url

                if "ana-slo.com" not in url:
                    continue

                title = ""

                try:
                    title = await page.title()
                except Exception:
                    pass

                body_text = ""

                try:
                    body_text = await page.locator(
                        "body"
                    ).inner_text(timeout=5000)
                except Exception:
                    pass

                text_length = len(body_text.strip())

                score = 0

                # アナスロURL
                if "ana-slo.com" in url:
                    score += 10

                # 対象日のURL
                if TARGET_DATE in url:
                    score += 10

                # タイトル
                if "アナスロ" in title:
                    score += 20

                # データまとめ
                if "データまとめ" in title:
                    score += 30

                # 本文が存在
                if text_length > 100:
                    score += 20

                if text_length > 1000:
                    score += 20

                print()
                print(
                    f"タブ[{i}] "
                    f"score={score} "
                    f"本文={text_length}文字"
                )

                print(f"タイトル: {title}")

                print(f"URL: {url}")

                if score > best_score:

                    best_score = score
                    target_page = page

            except Exception as e:

                print()
                print(
                    f"タブ[{i}]確認エラー: {e}"
                )

        # ----------------------------------------------------
        # タブが見つからない場合
        # ----------------------------------------------------

        if target_page is None:

            print()
            print("=" * 70)
            print("[ERROR] アナスロタブが見つかりません。")
            print("=" * 70)

            print()
            print("Chromeで対象ページを開いてから再実行してください。")

            return

        print()
        print(
            "★ 内容のあるアナスロタブを選択しました"
        )

        print(f"score: {best_score}")

        # ----------------------------------------------------
        # 選択したページ
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("【選択したページ】")
        print("=" * 70)

        try:
            selected_title = await target_page.title()
        except Exception:
            selected_title = ""

        print()
        print("タイトル:")
        print(selected_title)

        print()
        print("URL:")
        print(target_page.url)

        # ----------------------------------------------------
        # ページを前面にする
        # ----------------------------------------------------

        try:
            await target_page.bring_to_front()
        except Exception:
            pass

        # ----------------------------------------------------
        # 本文取得
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("【ページ本文取得】")
        print("=" * 70)

        try:

            body_text = await target_page.locator(
                "body"
            ).inner_text(timeout=10000)

            print()
            print(
                f"★ 本文取得成功"
            )

            print(
                f"本文文字数: {len(body_text)}"
            )

        except Exception as e:

            print()
            print(
                "[ERROR] 本文取得に失敗しました。"
            )

            print(e)

            body_text = ""

        # ----------------------------------------------------
        # 本文表示
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("【本文先頭5000文字】")
        print("=" * 70)

        print()

        if body_text:

            print(
                body_text[:5000]
            )

        else:

            print(
                "[本文なし]"
            )

        # ----------------------------------------------------
        # HTML保存
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("【HTML保存】")
        print("=" * 70)

        try:

            html = await target_page.content()

            DEBUG_HTML.write_text(
                html,
                encoding="utf-8"
            )

            print()
            print(
                "★ HTML保存成功"
            )

            print(
                DEBUG_HTML
            )

        except Exception as e:

            print()
            print(
                "[WARNING] HTML保存失敗"
            )

            print(e)

        # ----------------------------------------------------
        # テーブル確認
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("【テーブル解析】")
        print("=" * 70)

        try:

            tables = await target_page.locator(
                "table"
            ).all()

            print()
            print(
                f"ページ内table数: {len(tables)}"
            )

            for i, table in enumerate(tables[:20]):

                try:

                    table_text = await table.inner_text()

                    print()
                    print(
                        f"--- table[{i}] ---"
                    )

                    print(
                        table_text[:2000]
                    )

                except Exception:
                    pass

        except Exception as e:

            print()
            print(
                "[WARNING] table解析エラー"
            )

            print(e)

        # ----------------------------------------------------
        # 台番号候補検索
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("【台番号候補検索】")
        print("=" * 70)

        import re

        machine_numbers = []

        for match in re.findall(
            r"\b[5-9]\d{2}\b|\b10\d{2}\b|\b11\d{2}\b",
            body_text
        ):

            if match not in machine_numbers:

                machine_numbers.append(match)

        print()
        print(
            f"台番号候補数: {len(machine_numbers)}"
        )

        print()
        print(
            machine_numbers[:100]
        )

        # ----------------------------------------------------
        # データ関連キーワード
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("【データ関連キーワード】")
        print("=" * 70)

        keywords = [
            "全データ",
            "台番号",
            "機種名",
            "差枚",
            "G数",
            "BB",
            "RB",
            "合成確率",
            "BB確率",
            "RB確率",
            "データ表示",
        ]

        for keyword in keywords:

            count = body_text.count(keyword)

            print(
                f"{keyword:<15}: {count}件"
            )

        # ----------------------------------------------------
        # 終了
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("★★★★★ 7月11日 データ取得テスト完了 ★★★★★")
        print("=" * 70)

        print()
        print(
            "PythonからURLを開かず、"
        )

        print(
            "Chromeで既に表示されているページを読み取りました。"
        )

        print()
        print("デバッグHTML:")

        print(
            DEBUG_HTML
        )


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )