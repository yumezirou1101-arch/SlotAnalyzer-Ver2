import asyncio
import json
import urllib.request
from pathlib import Path

from playwright.async_api import async_playwright


# ============================================================
# 設定
# ============================================================

CDP_URL = "http://127.0.0.1:9222"

TARGET_DATE = "2026-07-15"

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_HTML = OUTPUT_DIR / "ana_slo_20260715_diagnose.html"


# ============================================================
# Chrome 9222 のターゲット一覧
# ============================================================

def get_chrome_targets():
    url = f"{CDP_URL}/json/list"

    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = response.read().decode("utf-8")
            return json.loads(data)

    except Exception as e:
        print()
        print("【ERROR】Chrome 9222のターゲット一覧を取得できませんでした。")
        print(f"詳細: {e}")
        return []


# ============================================================
# ページ情報取得
# ============================================================

async def inspect_page(page, index):

    print()
    print("=" * 70)
    print(f"【タブ {index} 詳細診断】")
    print("=" * 70)

    try:
        url = page.url
    except Exception as e:
        url = f"取得失敗: {e}"

    print()
    print("URL:")
    print(url)

    # --------------------------------------------------------
    # title
    # --------------------------------------------------------

    try:
        title = await page.title()
    except Exception as e:
        title = f"取得失敗: {e}"

    print()
    print("タイトル:")
    print(repr(title))

    # --------------------------------------------------------
    # readyState
    # --------------------------------------------------------

    try:
        ready_state = await page.evaluate(
            "() => document.readyState"
        )
    except Exception as e:
        ready_state = f"取得失敗: {e}"

    print()
    print("document.readyState:")
    print(repr(ready_state))

    # --------------------------------------------------------
    # documentElement
    # --------------------------------------------------------

    try:
        html_length = await page.evaluate(
            "() => document.documentElement ? document.documentElement.outerHTML.length : -1"
        )
    except Exception as e:
        html_length = f"取得失敗: {e}"

    print()
    print("document.documentElement.outerHTML 長さ:")
    print(html_length)

    # --------------------------------------------------------
    # body
    # --------------------------------------------------------

    try:
        body_length = await page.evaluate(
            "() => document.body ? document.body.innerText.length : -1"
        )
    except Exception as e:
        body_length = f"取得失敗: {e}"

    print()
    print("document.body.innerText 長さ:")
    print(body_length)

    # --------------------------------------------------------
    # body HTML
    # --------------------------------------------------------

    try:
        body_html_length = await page.evaluate(
            "() => document.body ? document.body.outerHTML.length : -1"
        )
    except Exception as e:
        body_html_length = f"取得失敗: {e}"

    print()
    print("document.body.outerHTML 長さ:")
    print(body_html_length)

    # --------------------------------------------------------
    # HTMLタグ数
    # --------------------------------------------------------

    try:
        html_count = await page.locator("html").count()
    except Exception as e:
        html_count = f"取得失敗: {e}"

    print()
    print("<html> 要素数:")
    print(html_count)

    # --------------------------------------------------------
    # bodyタグ数
    # --------------------------------------------------------

    try:
        body_count = await page.locator("body").count()
    except Exception as e:
        body_count = f"取得失敗: {e}"

    print()
    print("<body> 要素数:")
    print(body_count)

    # --------------------------------------------------------
    # table数
    # --------------------------------------------------------

    try:
        table_count = await page.locator("table").count()
    except Exception as e:
        table_count = f"取得失敗: {e}"

    print()
    print("<table> 要素数:")
    print(table_count)

    # --------------------------------------------------------
    # aタグ数
    # --------------------------------------------------------

    try:
        a_count = await page.locator("a").count()
    except Exception as e:
        a_count = f"取得失敗: {e}"

    print()
    print("<a> 要素数:")
    print(a_count)

    # --------------------------------------------------------
    # iframe数
    # --------------------------------------------------------

    try:
        iframe_count = await page.locator("iframe").count()
    except Exception as e:
        iframe_count = f"取得失敗: {e}"

    print()
    print("<iframe> 要素数:")
    print(iframe_count)

    # --------------------------------------------------------
    # frame一覧
    # --------------------------------------------------------

    print()
    print("【Frame一覧】")

    try:
        frames = page.frames

        print(f"Frame数: {len(frames)}")

        for frame_index, frame in enumerate(frames):

            try:
                frame_url = frame.url
            except Exception as e:
                frame_url = f"取得失敗: {e}"

            try:
                frame_title = await frame.title()
            except Exception as e:
                frame_title = f"取得失敗: {e}"

            try:
                frame_text = await frame.locator("body").inner_text(
                    timeout=3000
                )
                frame_text_length = len(frame_text)
            except Exception as e:
                frame_text_length = f"取得失敗: {e}"

            print()
            print(f"Frame[{frame_index}]")
            print(f"  URL   : {frame_url}")
            print(f"  title : {repr(frame_title)}")
            print(f"  本文  : {frame_text_length}文字")

    except Exception as e:
        print(f"Frame取得失敗: {e}")

    # --------------------------------------------------------
    # 本文
    # --------------------------------------------------------

    try:
        body_text = await page.locator("body").inner_text(
            timeout=5000
        )
    except Exception as e:
        body_text = f"取得失敗: {e}"

    print()
    print("【本文先頭1000文字】")
    print("-" * 70)

    if isinstance(body_text, str):
        print(body_text[:1000])
    else:
        print(body_text)

    print("-" * 70)

    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

    try:
        html = await page.content()

        print()
        print("page.content() 長さ:")
        print(len(html))

        OUTPUT_HTML.write_text(
            html,
            encoding="utf-8"
        )

        print()
        print("★ HTML保存成功")
        print(OUTPUT_HTML)

    except Exception as e:
        print()
        print("【ERROR】HTML保存失敗")
        print(e)


# ============================================================
# メイン
# ============================================================

async def main():

    print("=" * 70)
    print("アナスロ 7月15日 Chrome 9222 詳細診断")
    print("=" * 70)

    print()
    print("対象日:")
    print(TARGET_DATE)

    print()
    print("方式:")
    print("現在Chromeで開いているページを9222経由で診断します。")
    print("PythonからURLは開きません。")
    print("ページの再読み込みも行いません。")

    # --------------------------------------------------------
    # Chromeターゲット
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("【Chrome 9222 ターゲット一覧】")
    print("=" * 70)

    targets = get_chrome_targets()

    print()
    print(f"ターゲット数: {len(targets)}")

    for index, target in enumerate(targets):

        print()
        print(f"[Target {index}]")
        print(f"  type : {target.get('type')}")
        print(f"  title: {target.get('title')}")
        print(f"  url  : {target.get('url')}")
        print(f"  id   : {target.get('id')}")

    # --------------------------------------------------------
    # Playwright接続
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("【Playwright接続】")
    print("=" * 70)

    async with async_playwright() as p:

        try:
            browser = await p.chromium.connect_over_cdp(
                CDP_URL
            )

        except Exception as e:

            print()
            print("【ERROR】Chrome 9222への接続に失敗しました。")
            print(e)
            return

        print()
        print("★ Chrome接続成功")

        contexts = browser.contexts

        print()
        print(f"ブラウザコンテキスト数: {len(contexts)}")

        # ----------------------------------------------------
        # 全ページ診断
        # ----------------------------------------------------

        page_index = 0

        for context_index, context in enumerate(contexts):

            print()
            print("=" * 70)
            print(f"【コンテキスト {context_index}】")
            print("=" * 70)

            print(
                f"タブ数: {len(context.pages)}"
            )

            for page in context.pages:

                await inspect_page(
                    page,
                    page_index
                )

                page_index += 1

        print()
        print("=" * 70)
        print("★★★★★ 診断完了 ★★★★★")
        print("=" * 70)

        print()
        print("診断HTML:")
        print(OUTPUT_HTML)

        print()
        print("この診断では")
        print("・URL")
        print("・title")
        print("・readyState")
        print("・document HTML")
        print("・body")
        print("・table")
        print("・iframe")
        print("・frame")
        print("を確認しました。")


if __name__ == "__main__":
    asyncio.run(main())