from __future__ import annotations

import asyncio
from playwright.async_api import async_playwright


CDP_URL = "http://127.0.0.1:9222"
TARGET_HOST = "min-repo.com"


async def main() -> None:
    print("=" * 88)
    print("Min-Repo Browser Probe")
    print("=" * 88)

    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(CDP_URL)

        pages = []
        for context in browser.contexts:
            pages.extend(context.pages)

        print(f"pages: {len(pages)}")
        print()

        targets = [
            page
            for page in pages
            if TARGET_HOST in (page.url or "")
        ]

        if not targets:
            print("[NOT FOUND] min-repo.com page is not open in the 9222 Chrome.")
            print()
            print("Open this page manually in the 9222 Chrome, then run this script again:")
            print("https://min-repo.com/3278114/?kishu=all")
            return

        for i, page in enumerate(targets, start=1):
            print("-" * 88)
            print(f"TARGET {i}/{len(targets)}")
            print(f"URL   : {page.url}")

            try:
                title = await page.title()
            except Exception as e:
                title = f"<title error: {e}>"
            print(f"TITLE : {title}")

            try:
                body_text = await page.locator("body").inner_text(timeout=5000)
            except Exception as e:
                body_text = ""
                print(f"BODY TEXT ERROR: {e}")

            try:
                html = await page.content()
            except Exception as e:
                html = ""
                print(f"PAGE CONTENT ERROR: {e}")

            try:
                tables = await page.locator("table").count()
            except Exception:
                tables = -1

            try:
                trs = await page.locator("tr").count()
            except Exception:
                trs = -1

            print(f"HTML chars : {len(html):,}")
            print(f"BODY chars : {len(body_text):,}")
            print(f"TABLES     : {tables}")
            print(f"TR rows    : {trs}")

            print()
            print("BODY SAMPLE:")
            print(body_text[:1500])

            if tables > 0:
                print()
                print("TABLE SHAPES / HEADERS:")
                for table_no in range(min(tables, 20)):
                    table = page.locator("table").nth(table_no)

                    try:
                        rows = await table.locator("tr").count()
                    except Exception:
                        rows = -1

                    try:
                        headers = await table.locator("th").all_inner_texts()
                    except Exception:
                        headers = []

                    print(
                        f"table={table_no} rows={rows} "
                        f"headers={headers[:20]}"
                    )

        print()
        print("=" * 88)
        print("Probe complete.")
        print("=" * 88)


if __name__ == "__main__":
    asyncio.run(main())
