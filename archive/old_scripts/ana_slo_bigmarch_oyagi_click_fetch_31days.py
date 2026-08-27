from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import asyncio
import csv
import re

from playwright.async_api import async_playwright


PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
CDP_URL = "http://127.0.0.1:9222"

STORE_TEXTS = (
    "ビックマーチ高崎おおやぎ店",
    "ビッグマーチ高崎おおやぎ店",
)
LIST_TITLE_TEXT = "データ一覧"
EXPECTED_MACHINES = 276
DEFAULT_MAX_DAYS = 31

LOG_DIR = (
    PROJECT_ROOT
    / "data"
    / "bigmarch_takasaki_oyagi"
    / "fetch_logs"
)

DATE_TEXT_RE = re.compile(r"^(20\d{2})/(\d{2})/(\d{2})")
DATE_URL_RE = re.compile(r"/(20\d{2})-(\d{2})-(\d{2})-")


def header(title: str) -> None:
    print()
    print("=" * 108)
    print(title)
    print("=" * 108)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Fetch up to 31 Big March Takasaki Oyagi Ana-Slo daily pages "
            "by clicking date links from the store data-list page."
        )
    )
    p.add_argument(
        "--max-days",
        type=int,
        default=DEFAULT_MAX_DAYS,
        help="Maximum number of date links to process. Default: 31.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing source HTML files. Default: skip existing files.",
    )
    return p.parse_args()


def has_store_text(text: str) -> bool:
    return any(name in text for name in STORE_TEXTS)


async def find_store_page(context):
    fallback = None

    for page in context.pages:
        try:
            title = await page.title()
        except Exception:
            continue

        if "ana-slo.com" not in page.url:
            continue
        if not has_store_text(title):
            continue

        if LIST_TITLE_TEXT in title:
            return page

        if fallback is None:
            fallback = page

    return fallback


async def ensure_list_page(page):
    title = await page.title()

    if LIST_TITLE_TEXT in title and has_store_text(title):
        return

    print("returning to list page : browser back")

    await page.go_back(
        wait_until="domcontentloaded",
        timeout=30000,
    )
    await page.wait_for_timeout(1200)

    title = await page.title()
    print(f"list page title       : {title}")

    if LIST_TITLE_TEXT not in title or not has_store_text(title):
        raise RuntimeError(
            "Could not return to the Big March Oyagi data-list page."
        )


async def collect_date_links(page):
    """Read all usable store date links currently present on the list page."""
    anchors = page.locator("a")
    count = await anchors.count()

    found = {}

    for i in range(count):
        a = anchors.nth(i)

        try:
            text = (await a.inner_text()).strip()
            href = await a.get_attribute("href")
        except Exception:
            continue

        if not href:
            continue

        tm = DATE_TEXT_RE.match(text)
        um = DATE_URL_RE.search(href)

        if not tm or not um:
            continue

        text_date = "-".join(tm.groups())
        url_date = "-".join(um.groups())

        if text_date != url_date:
            continue

        # Keep only this store's daily data URLs.
        href_lower = href.lower()
        if "ana-slo.com" not in href_lower:
            continue
        if "-data" not in href_lower:
            continue

        found[text_date] = {
            "date": text_date,
            "label": text,
            "href": href,
        }

    return sorted(
        found.values(),
        key=lambda x: x["date"],
        reverse=True,
    )


async def find_link_by_date(page, date_iso: str):
    anchors = page.locator("a")
    count = await anchors.count()
    label_prefix = date_iso.replace("-", "/")

    for i in range(count):
        a = anchors.nth(i)

        try:
            text = (await a.inner_text()).strip()
            href = await a.get_attribute("href")
        except Exception:
            continue

        if not href:
            continue

        if (
            text.startswith(label_prefix)
            and date_iso in href
            and "ana-slo.com" in href
        ):
            return a

    return None


async def validate_daily_page(page, date_iso: str):
    title = await page.title()
    body_text = await page.locator("body").inner_text()

    date_text = date_iso.replace("-", "/")

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

        if {"機種名", "台番号", "G数", "差枚"}.issubset(set(headers)):
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
        "date_ok": date_text in title or date_text in body_text,
        "store_ok": has_store_text(title) or has_store_text(body_text),
        "header_ok": header_ok,
        "main_rows": main_rows,
        "row_count_ok": main_rows == EXPECTED_MACHINES,
    }

    ok = all(
        (
            result["date_ok"],
            result["store_ok"],
            result["header_ok"],
            result["row_count_ok"],
        )
    )

    return ok, result


def output_path(date_iso: str) -> Path:
    compact = date_iso.replace("-", "")
    return (
        PROJECT_ROOT
        / f"ana_slo_bigmarch_oyagi_{compact}_source.html"
    )


def save_log(rows):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"bigmarch_oyagi_fetch_{stamp}.csv"

    fields = [
        "date",
        "status",
        "reason",
        "rows",
        "title",
        "file",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return path


async def main():
    args = parse_args()

    if args.max_days < 1:
        raise ValueError("--max-days must be at least 1.")

    header("Big March Takasaki Oyagi - Multi-Day Link-Click Fetch")

    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(CDP_URL)

        if not browser.contexts:
            raise RuntimeError("No Chrome context found.")

        context = browser.contexts[0]
        page = await find_store_page(context)

        if page is None:
            raise RuntimeError(
                "Big March Oyagi Ana-Slo tab was not found."
            )

        print(f"source tab title      : {await page.title()}")
        print(f"source tab URL        : {page.url}")
        print(f"expected machines     : {EXPECTED_MACHINES}")
        print(f"max days              : {args.max_days}")
        print(f"overwrite             : {args.overwrite}")

        await ensure_list_page(page)

        date_links = await collect_date_links(page)

        if not date_links:
            raise RuntimeError(
                "No usable daily date links were found on the list page."
            )

        selected = date_links[: args.max_days]

        print(f"date links found      : {len(date_links)}")
        print(f"dates selected        : {len(selected)}")
        print(
            f"selected range        : "
            f"{selected[-1]['date']} to {selected[0]['date']}"
        )

        logs = []

        for pos, item in enumerate(selected, start=1):
            date_iso = item["date"]
            out = output_path(date_iso)

            header(
                f"[{pos}/{len(selected)}] {date_iso}"
            )

            if out.exists() and not args.overwrite:
                print(f"existing file         : {out}")
                print("RESULT                : SKIPPED_EXISTING")

                logs.append(
                    {
                        "date": date_iso,
                        "status": "SKIPPED_EXISTING",
                        "reason": "",
                        "rows": "",
                        "title": "",
                        "file": str(out),
                    }
                )
                continue

            try:
                await ensure_list_page(page)

                link = await find_link_by_date(page, date_iso)

                if link is None:
                    print("RESULT                : LINK_NOT_FOUND")
                    logs.append(
                        {
                            "date": date_iso,
                            "status": "FAILED",
                            "reason": "date link not found",
                            "rows": "",
                            "title": "",
                            "file": "",
                        }
                    )
                    continue

                href = await link.get_attribute("href")
                print(f"href                  : {href}")

                async with page.expect_navigation(
                    wait_until="domcontentloaded",
                    timeout=30000,
                ):
                    await link.click()

                await page.wait_for_timeout(2200)

                ok, result = await validate_daily_page(
                    page,
                    date_iso,
                )

                print(f"title                 : {result['title']}")
                print(f"date check            : {result['date_ok']}")
                print(f"store check           : {result['store_ok']}")
                print(f"header check          : {result['header_ok']}")
                print(f"main table rows       : {result['main_rows']}")
                print(f"row count check       : {result['row_count_ok']}")

                if not ok:
                    print("RESULT                : NOT SAVED")
                    logs.append(
                        {
                            "date": date_iso,
                            "status": "FAILED_VALIDATION",
                            "reason": "daily page validation failed",
                            "rows": result["main_rows"],
                            "title": result["title"],
                            "file": "",
                        }
                    )
                    continue

                html = await page.content()
                out.write_text(html, encoding="utf-8")

                print(f"html chars            : {len(html):,}")
                print(f"saved                 : {out}")
                print("RESULT                : OK")

                logs.append(
                    {
                        "date": date_iso,
                        "status": "OK",
                        "reason": "",
                        "rows": result["main_rows"],
                        "title": result["title"],
                        "file": str(out),
                    }
                )

            except Exception as exc:
                print(f"ERROR                 : {exc}")

                logs.append(
                    {
                        "date": date_iso,
                        "status": "ERROR",
                        "reason": repr(exc),
                        "rows": "",
                        "title": "",
                        "file": "",
                    }
                )

        # Leave the tab on the store list page when possible.
        try:
            await ensure_list_page(page)
        except Exception as exc:
            print(f"final return warning  : {exc}")

        log_path = save_log(logs)

        counts = {}
        for row in logs:
            counts[row["status"]] = counts.get(row["status"], 0) + 1

        header("SUMMARY")

        print(f"processed             : {len(logs)}")
        for status, count in sorted(counts.items()):
            print(f"{status:<22}: {count}")

        print(f"log                   : {log_path}")
        print()
        print("Existing source HTML is skipped unless --overwrite is specified.")
        print("Only pages passing the 276-machine validation are newly saved.")
        print("No Maruhan data files were modified.")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
