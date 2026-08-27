from __future__ import annotations

from pathlib import Path
from datetime import date
import asyncio
import re

from playwright.async_api import async_playwright


# ============================================================
# Big March Takasaki Oyagi - Link-Click 3 Day Fetch Test
# ============================================================
#
# Target dates:
#   2026-08-22
#   2026-08-21
#   2026-08-20
#
# Strategy:
# - Connect to the already-running Chrome at 127.0.0.1:9222
# - Find the currently open Big March Oyagi Ana-Slo tab
# - DO NOT construct and navigate directly to date URLs
# - Instead, use the date links that actually exist in the page DOM
# - Click one day at a time like a normal browser user
# - Validate page title, store name, date, headers and 276 machine rows
# - Save page.content() only after validation passes
#
# Safety:
# - No Maruhan files are touched
# - Existing source tab is not intentionally closed
# - Stops on unexpected page structure
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

CDP_URL = "http://127.0.0.1:9222"

TARGET_DATES = [
    date(2026, 8, 22),
    date(2026, 8, 21),
    date(2026, 8, 20),
]

EXPECTED_MACHINES = 276
STORE_TEXT = "ビックマーチ高崎おおやぎ店"


def header(title: str) -> None:
    print()
    print("=" * 104)
    print(title)
    print("=" * 104)


async def find_source_page(context):
    for page in context.pages:
        try:
            title = await page.title()
        except Exception:
            continue

        if (
            "ana-slo.com" in page.url
            and STORE_TEXT in title
        ):
            return page

    return None


async def validate_page(page, target: date):
    title = await page.title()
    body_text = await page.locator("body").inner_text()

    date_text = target.strftime("%Y/%m/%d")

    tables = page.locator("table")
    table_count = await tables.count()

    main_rows = None
    header_ok = False

    for i in range(table_count):
        table = tables.nth(i)

        try:
            headers = [
                x.strip()
                for x in await table.locator("th").all_inner_texts()
            ]
        except Exception:
            continue

        required = {
            "機種名",
            "台番号",
            "G数",
            "差枚",
        }

        if required.issubset(set(headers)):
            rows = await table.locator("tbody tr").count()

            if rows == 0:
                rows = max(
                    0,
                    (await table.locator("tr").count()) - 1,
                )

            main_rows = rows
            header_ok = True
            break

    result = {
        "title": title,
        "date_ok": (
            date_text in title
            or date_text in body_text
        ),
        "store_ok": (
            STORE_TEXT in title
            or STORE_TEXT in body_text
        ),
        "header_ok": header_ok,
        "main_rows": main_rows,
        "row_count_ok": (
            main_rows == EXPECTED_MACHINES
        ),
    }

    ok = all(
        [
            result["date_ok"],
            result["store_ok"],
            result["header_ok"],
            result["row_count_ok"],
        ]
    )

    return ok, result


async def find_date_link(page, target: date):
    label = target.strftime("%Y/%m/%d")

    # Prefer exact date-prefix anchors, but allow weekday suffix.
    anchors = page.locator("a")

    count = await anchors.count()

    candidates = []

    for i in range(count):
        a = anchors.nth(i)

        try:
            text = (
                await a.inner_text()
            ).strip()
        except Exception:
            continue

        if text.startswith(label):
            candidates.append(a)

    if not candidates:
        return None, label

    # Prefer a link whose href contains the target date and this store slug.
    date_slug = target.strftime("%Y-%m-%d")

    for a in candidates:
        try:
            href = await a.get_attribute("href")
        except Exception:
            href = None

        if (
            href
            and date_slug in href
            and "ana-slo.com" in href
            and "%e3%83%93%e3%83%83%e3%82%af" in href.lower()
        ):
            return a, label

    return candidates[0], label


async def main():
    header(
        "Big March Takasaki Oyagi - Link-Click 3 Day Fetch Test"
    )

    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(
            CDP_URL
        )

        if not browser.contexts:
            raise RuntimeError(
                "No Chrome context found."
            )

        context = browser.contexts[0]

        source_page = await find_source_page(
            context
        )

        if source_page is None:
            raise RuntimeError(
                "Big March Oyagi Ana-Slo tab was not found."
            )

        print(
            f"source tab title      : {await source_page.title()}"
        )
        print(
            f"source tab URL        : {source_page.url}"
        )
        print(
            f"expected machines     : {EXPECTED_MACHINES}"
        )

        saved = []
        failed = []

        # Work in the existing page, like a real user clicking links.
        page = source_page

        for target in TARGET_DATES:
            header(
                f"CLICK FETCH {target}"
            )

            try:
                link, label = await find_date_link(
                    page,
                    target,
                )

                if link is None:
                    print(
                        f"date link             : NOT FOUND ({label})"
                    )
                    failed.append(
                        (
                            target,
                            "date link not found",
                        )
                    )
                    continue

                href = await link.get_attribute("href")

                print(
                    f"date link             : {label}"
                )
                print(
                    f"href                  : {href}"
                )

                # Normal browser-like navigation via click.
                async with page.expect_navigation(
                    wait_until="domcontentloaded",
                    timeout=30000,
                ):
                    await link.click()

                await page.wait_for_timeout(
                    2500
                )

                print(
                    f"current URL           : {page.url}"
                )

                ok, result = await validate_page(
                    page,
                    target,
                )

                print(
                    f"title                 : {result['title']}"
                )
                print(
                    f"date check            : {result['date_ok']}"
                )
                print(
                    f"store check           : {result['store_ok']}"
                )
                print(
                    f"header check          : {result['header_ok']}"
                )
                print(
                    f"main table rows       : {result['main_rows']}"
                )
                print(
                    f"row count check       : {result['row_count_ok']}"
                )

                if not ok:
                    failed.append(
                        (
                            target,
                            "validation failed",
                        )
                    )
                    print(
                        "RESULT                : NOT SAVED"
                    )
                    continue

                html = await page.content()

                out = (
                    PROJECT_ROOT
                    / (
                        "ana_slo_bigmarch_oyagi_"
                        f"{target.strftime('%Y%m%d')}"
                        "_source.html"
                    )
                )

                out.write_text(
                    html,
                    encoding="utf-8",
                )

                saved.append(
                    out
                )

                print(
                    f"html chars            : {len(html):,}"
                )
                print(
                    f"saved                 : {out}"
                )
                print(
                    "RESULT                : OK"
                )

            except Exception as exc:
                failed.append(
                    (
                        target,
                        repr(exc),
                    )
                )

                print(
                    f"ERROR                 : {exc}"
                )

        header(
            "SUMMARY"
        )

        print(
            f"saved files           : {len(saved)}"
        )

        for path in saved:
            print(
                f"  OK   {path.name}"
            )

        print(
            f"failed                : {len(failed)}"
        )

        for target, reason in failed:
            print(
                f"  FAIL {target}: {reason}"
            )

        print()
        print(
            "Note: The Big March tab may now be showing the last clicked date."
        )
        print(
            "No Maruhan data files were modified."
        )

        # Detach without intentionally closing the user's Chrome.
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
