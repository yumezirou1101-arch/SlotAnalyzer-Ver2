import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


CDP_URL = "http://127.0.0.1:9222"


async def main():

    print()
    print("=" * 60)
    print(" 8/2ページ Network Response 監視テスト")
    print("=" * 60)

    async with async_playwright() as p:

        print()
        print("9222 Chromeへ接続中...")

        browser = await p.chromium.connect_over_cdp(
            CDP_URL
        )

        print("Chrome接続成功。")

        context = browser.contexts[0]

        # --------------------------------------------------
        # 8/2ページを探す
        # --------------------------------------------------

        target_page = None

        for page in context.pages:

            try:

                if (
                    "ana-slo.com" in page.url
                    and "2026-08-02" in page.url
                ):

                    target_page = page
                    break

            except Exception:
                continue

        if target_page is None:

            print()
            print("[エラー]")
            print("8/2のアナスロページが見つかりません。")

            return

        print()
        print("対象ページ:")
        print(target_page.url)

        try:
            print("TITLE:")
            print(await target_page.title())
        except:
            pass

        # --------------------------------------------------
        # 重要
        #
        # 既にページが読み込まれているので、
        # ここからページを移動しない。
        #
        # これから発生するResponseを監視するため、
        # まず監視をセットする。
        # --------------------------------------------------

        print()
        print("Network Response監視を開始します。")

        responses = []

        async def on_response(response):

            try:

                url = response.url

                # ana-slo.com のレスポンスだけ記録
                if "ana-slo.com" not in url:
                    return

                status = response.status

                content_type = response.headers.get(
                    "content-type",
                    ""
                )

                # HTML / JSON / CSVなどを対象
                if any(
                    x in content_type.lower()
                    for x in [
                        "text/html",
                        "application/json",
                        "text/plain",
                        "text/csv",
                    ]
                ):

                    print()
                    print(
                        "[Response]"
                    )

                    print(
                        "Status:",
                        status
                    )

                    print(
                        "Content-Type:",
                        content_type
                    )

                    print(
                        "URL:",
                        url
                    )

                    responses.append(
                        response
                    )

            except Exception:
                pass

        target_page.on(
            "response",
            on_response
        )

        # --------------------------------------------------
        # ここがポイント
        #
        # 現在の8/2ページをリロードする。
        #
        # PythonでURLへgotoするのではなく、
        # 現在Chromeで表示されているページを
        # Chrome自身にリロードさせる。
        # --------------------------------------------------

        print()
        print("=" * 60)
        print("現在の8/2ページをChromeでリロードします")
        print("=" * 60)

        try:

            await target_page.reload(
                wait_until="domcontentloaded",
                timeout=60000
            )

        except Exception as e:

            print()
            print("[注意]")
            print(
                "リロード中にエラーが発生しました。"
            )
            print(e)

        # --------------------------------------------------
        # レスポンスが発生するまで待つ
        # --------------------------------------------------

        print()
        print("Network Responseを確認しています...")

        await asyncio.sleep(15)

        # --------------------------------------------------
        # 結果
        # --------------------------------------------------

        print()
        print("=" * 60)
        print("Response確認結果")
        print("=" * 60)

        print()
        print(
            f"検出レスポンス数: {len(responses)}"
        )

        # --------------------------------------------------
        # 各Responseの本文を調べる
        # --------------------------------------------------

        saved_count = 0

        for i, response in enumerate(
            responses,
            start=1
        ):

            try:

                print()
                print(
                    f"--- Response {i} ---"
                )

                print(
                    "Status:",
                    response.status
                )

                print(
                    "URL:",
                    response.url
                )

                content_type = response.headers.get(
                    "content-type",
                    ""
                )

                print(
                    "Content-Type:",
                    content_type
                )

                # ------------------------------------------
                # 本文取得
                # ------------------------------------------

                try:

                    body = await response.body()

                    size = len(body)

                    print(
                        f"本文サイズ: {size:,} bytes"
                    )

                except Exception as e:

                    print(
                        "本文取得失敗:",
                        e
                    )

                    continue

                # ------------------------------------------
                # テキストとして解析
                # ------------------------------------------

                try:

                    text = body.decode(
                        "utf-8",
                        errors="ignore"
                    )

                except:

                    text = ""

                print(
                    f"テキストサイズ: {len(text):,}文字"
                )

                # ------------------------------------------
                # アナスロの台データ判定
                # ------------------------------------------

                keywords = [
                    "all_data_table",
                    "台番号",
                    "差枚",
                    "G数",
                    "2026-08-02",
                    "2026/08/02",
                ]

                print()
                print("キーワード:")

                for keyword in keywords:

                    count = text.count(
                        keyword
                    )

                    print(
                        f"  {keyword}: {count}件"
                    )

                # ------------------------------------------
                # all_data_table があれば保存
                # ------------------------------------------

                if "all_data_table" in text:

                    print()
                    print(
                        "★ all_data_table を発見しました！"
                    )

                    output_file = (
                        Path(__file__).resolve().parent
                        / f"network_response_{saved_count + 1}.html"
                    )

                    output_file.write_text(
                        text,
                        encoding="utf-8"
                    )

                    print(
                        "保存:",
                        output_file
                    )

                    saved_count += 1

            except Exception as e:

                print()
                print(
                    f"Response {i} 処理エラー:"
                )

                print(e)

        # --------------------------------------------------
        # 最終確認
        # --------------------------------------------------

        print()
        print("=" * 60)
        print("最終確認")
        print("=" * 60)

        if saved_count > 0:

            print()
            print(
                "★★★ 本体HTMLの取得に成功しました ★★★"
            )

            print()
            print(
                "Network Responseから"
                "all_data_tableを取得できています。"
            )

        else:

            print()
            print(
                "all_data_tableを含むResponseは"
                "見つかりませんでした。"
            )

        print()
        print(
            "Chromeはそのままで構いません。"
        )


if __name__ == "__main__":

    asyncio.run(main())