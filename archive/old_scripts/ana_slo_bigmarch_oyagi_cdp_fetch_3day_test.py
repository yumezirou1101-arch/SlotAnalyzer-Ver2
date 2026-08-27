from __future__ import annotations

from pathlib import Path
from datetime import date
import asyncio
import re

from playwright.async_api import async_playwright


# ============================================================
# Big March Takasaki Oyagi - 3 Day CDP Fetch Test
# ============================================================
#
# Target dates:
#   2026-08-20
#   2026-08-21
#   2026-08-22
#
# Strategy:
# - Connect to the already-running Chrome at 127.0.0.1:9222
# - Find the currently open Big March Oyagi Ana-Slo tab
# - Reuse its URL slug and replace only the date prefix
# - Open a NEW tab for fetching, so the user's existing tab is preserved
# - Validate title / store name / date / table header / row count
# - Save page.content() HTML only when validation passes
#
# This script does NOT touch Maruhan Maebashi files.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

CDP_URL = "http://127.0.0.1:9222"

TARGET_DATES = [
    date(2026, 8, 20),
    date(2026, 8, 21),
    date(2026, 8, 22),
]

EXPECTED_MACHINES = 276

STORE_TEXT = "ビックマーチ高崎おおやぎ店"

URL_RE = re.compile(
    r"^(https://ana-slo\.com/)"
    r"\d{4}-\d{2}-\d{2}"
    r"(-%e3%83%93%e3%83%83%e3%82%af%e3%83%9e%e3%83%bc%e3%83%81"
    r"%e9%ab%98%e5%b4%8e%e3%81%8a%e3%81%8a%e3%82%84%e3%81%8e"
    r"%e5%ba%97-data/?)$",
    re.IGNORECASE,
)


def header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


async def find_source_page(context):
    for page in context.pages:
        try:
            title = await page.title()
        except Exception:
            continue

        url = page.url

        if (
            "ana-slo.com" in url
            and "ビックマーチ高崎おおやぎ店" in title
        ):
            return page

    return None


def build_url(base_url: str, target: date) -> str:
    m = URL_RE.match(base_url)

    if not m:
        raise RuntimeError(
            "Could not recognize the current Big March Oyagi URL format:\n"
            f"{base_url}"
        )

    return (
        f"{m.group(1)}"
        f"{target.strftime('%Y-%m-%d')}"
        f"{m.group(2)}"
    )


async def validate_page(page, target: date) -> tuple[bool, dict]:
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
            th_texts = await table.locator("th").all_inner_texts()
        except Exception:
            continue

        required = {"機種名", "台番号", "G数", "差枚"}

        if required.issubset(
            {x.strip() for x in th_texts}
        ):
            tr_count = await table.locator("tbody tr").count()

            if tr_count == 0:
                # Some tables may not have tbody in DOM.
                tr_count = await table.locator("tr").count() - 1

            main_rows = tr_count
            header_ok = True
            break

    result = {
        "title": title,
        "date_ok": date_text in title or date_text in body_text,
        "store_ok": STORE_TEXT in title or STORE_TEXT in body_text,
        "header_ok": header_ok,
        "main_rows": main_rows,
        "row_count_ok": main_rows == EXPECTED_MACHINES,
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


async def main():
    header(
        "Big March Takasaki Oyagi - 3 Day CDP Fetch Test"
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

        source_url = source_page.url

        print(
            f"source tab title      : {await source_page.title()}"
        )
        print(
            f"source tab URL        : {source_url}"
        )
        print(
            f"expected machines     : {EXPECTED_MACHINES}"
        )

        fetch_page = await context.new_page()

        saved = []
        failed = []

        try:
            for target in TARGET_DATES:
                header(
                    f"FETCH {target}"
                )

                url = build_url(
                    source_url,
                    target,
                )

                print(
                    f"URL                   : {url}"
                )

                try:
                    response = await fetch_page.goto(
                        url,
                        wait_until="domcontentloaded",
                        timeout=30000,
                    )

                    # Give client-side table rendering some time.
                    await fetch_page.wait_for_timeout(
                        2500
                    )

                    status = (
                        response.status
                        if response is not None
                        else None
                    )

                    print(
                        f"HTTP status           : {status}"
                    )

                    ok, result = await validate_page(
                        fetch_page,
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

                    html = await fetch_page.content()

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

        finally:
            await fetch_page.close()

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
            "Existing user tabs were not intentionally modified."
        )

        # Do not close the user's Chrome browser.
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
