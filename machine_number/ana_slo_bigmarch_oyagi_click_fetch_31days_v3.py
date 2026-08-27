from __future__ import annotations

from pathlib import Path
from datetime import datetime
from io import StringIO
import argparse
import asyncio
import csv
import re

import pandas as pd
from playwright.async_api import async_playwright


PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
CDP_URL = "http://127.0.0.1:9222"

STORE_TEXTS = (
    "ビックマーチ高崎おおやぎ店",
    "ビッグマーチ高崎おおやぎ店",
)

LIST_TITLE_TEXT = "データ一覧"
DEFAULT_MAX_DAYS = 31
DEFAULT_MIN_MACHINES = 200

STORE_LIST_URL = (
    "https://ana-slo.com/"
    "%e3%83%9b%e3%83%bc%e3%83%ab%e3%83%87%e3%83%bc%e3%82%bf/"
    "%e7%be%a4%e9%a6%ac%e7%9c%8c/"
    "%e3%83%93%e3%83%83%e3%82%af%e3%83%9e%e3%83%bc%e3%83%81"
    "%e9%ab%98%e5%b4%8e%e3%81%8a%e3%81%8a%e3%82%84%e3%81%8e"
    "%e5%ba%97-%e3%83%87%e3%83%bc%e3%82%bf%e4%b8%80%e8%a6%a7/"
)

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
    print("=" * 112)
    print(title)
    print("=" * 112)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Big March Takasaki Oyagi Ana-Slo daily pages "
            "by clicking date links from the store data-list page."
        )
    )
    parser.add_argument("--max-days", type=int, default=DEFAULT_MAX_DAYS)
    parser.add_argument("--min-machines", type=int, default=DEFAULT_MIN_MACHINES)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


async def open_store_list_page(context):
    header("AUTO OPEN STORE LIST PAGE")
    print(f"URL                   : {STORE_LIST_URL}")

    page = await context.new_page()

    response = await page.goto(
        STORE_LIST_URL,
        wait_until="domcontentloaded",
        timeout=30000,
    )

    await page.wait_for_timeout(1800)

    status = response.status if response is not None else None
    title = await page.title()

    print(f"HTTP status           : {status}")
    print(f"title                 : {title}")

    if LIST_TITLE_TEXT not in title or not has_store_text(title):
        raise RuntimeError(
            "Could not auto-open the Big March Oyagi data-list page."
        )

    print("auto-open             : OK")
    return page


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

        text_match = DATE_TEXT_RE.match(text)
        url_match = DATE_URL_RE.search(href)

        if not text_match or not url_match:
            continue

        text_date = "-".join(text_match.groups())
        url_date = "-".join(url_match.groups())

        if text_date != url_date:
            continue

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


def find_main_table(html: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))
    required = {"機種名", "台番号", "G数", "差枚"}
    candidates = []

    for index, table in enumerate(tables):
        columns = {str(c).strip() for c in table.columns}
        if required.issubset(columns):
            candidates.append((len(table), index, table.copy()))

    if not candidates:
        raise RuntimeError("Main machine table not found.")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][2]


def clean_for_quality(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip() for c in x.columns]

    x = x.rename(
        columns={
            "機種名": "machine_name",
            "台番号": "machine_no",
            "G数": "G",
            "差枚": "diff",
        }
    )

    x["machine_name"] = x["machine_name"].astype(str).str.strip()
    x["machine_no"] = pd.to_numeric(x["machine_no"], errors="coerce")

    for col in ("G", "diff"):
        x[col] = (
            x[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("+", "", regex=False)
            .str.strip()
        )
        x[col] = pd.to_numeric(x[col], errors="coerce")

    return x


def validate_daily_html(
    html: str,
    title: str,
    body_text: str,
    date_iso: str,
    min_machines: int,
):
    date_text = date_iso.replace("-", "/")

    date_ok = date_text in title or date_text in body_text
    store_ok = has_store_text(title) or has_store_text(body_text)

    table = find_main_table(html)
    x = clean_for_quality(table)

    records = len(x)
    unique_machines = x["machine_no"].nunique(dropna=True)
    duplicate_rows = int(
        x["machine_no"].duplicated(keep=False).sum()
    )
    missing_machine = int(x["machine_no"].isna().sum())

    missing_name = int(
        x["machine_name"]
        .astype(str)
        .str.strip()
        .isin(["", "nan", "None"])
        .sum()
    )

    invalid_diff = int(x["diff"].isna().sum())
    invalid_g = int(x["G"].isna().sum())
    negative_g = int(((x["G"] < 0).fillna(False)).sum())

    machine_count_ok = records >= min_machines
    unique_ok = (
        unique_machines == records
        and duplicate_rows == 0
        and missing_machine == 0
    )
    data_ok = (
        missing_name == 0
        and invalid_diff == 0
        and invalid_g == 0
        and negative_g == 0
    )

    result = {
        "title": title,
        "date_ok": date_ok,
        "store_ok": store_ok,
        "records": records,
        "unique_machines": unique_machines,
        "duplicate_rows": duplicate_rows,
        "missing_machine": missing_machine,
        "missing_name": missing_name,
        "invalid_diff": invalid_diff,
        "invalid_G": invalid_g,
        "negative_G": negative_g,
        "machine_count_ok": machine_count_ok,
    }

    ok = all(
        (
            date_ok,
            store_ok,
            machine_count_ok,
            unique_ok,
            data_ok,
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
    path = LOG_DIR / f"bigmarch_oyagi_fetch_v3_{stamp}.csv"

    fields = [
        "date",
        "status",
        "reason",
        "records",
        "unique_machines",
        "duplicate_rows",
        "missing_machine",
        "missing_name",
        "invalid_diff",
        "invalid_G",
        "negative_G",
        "title",
        "file",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return path


def blank_log_row(
    date_iso: str,
    status: str,
    reason: str = "",
    file: str = "",
):
    return {
        "date": date_iso,
        "status": status,
        "reason": reason,
        "records": "",
        "unique_machines": "",
        "duplicate_rows": "",
        "missing_machine": "",
        "missing_name": "",
        "invalid_diff": "",
        "invalid_G": "",
        "negative_G": "",
        "title": "",
        "file": file,
    }


async def main():
    args = parse_args()

    if args.max_days < 1:
        raise ValueError("--max-days must be at least 1.")
    if args.min_machines < 1:
        raise ValueError("--min-machines must be at least 1.")

    header(
        "Big March Takasaki Oyagi - Multi-Day Link-Click Fetch V3"
    )

    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(CDP_URL)

        if not browser.contexts:
            raise RuntimeError("No Chrome context found.")

        context = browser.contexts[0]

        page = await find_store_page(context)

        if page is None:
            print("store tab             : NOT FOUND")
            print("action                : auto-open list page")
            page = await open_store_list_page(context)
        else:
            print("store tab             : FOUND")

        print(f"source tab title      : {await page.title()}")
        print(f"source tab URL        : {page.url}")
        print(f"max days              : {args.max_days}")
        print(f"min machines          : {args.min_machines}")
        print(f"overwrite             : {args.overwrite}")

        await ensure_list_page(page)

        date_links = await collect_date_links(page)

        if not date_links:
            raise RuntimeError("No usable daily date links were found.")

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

            header(f"[{pos}/{len(selected)}] {date_iso}")

            if out.exists() and not args.overwrite:
                print(f"existing file         : {out}")
                print("RESULT                : SKIPPED_EXISTING")
                logs.append(
                    blank_log_row(
                        date_iso,
                        "SKIPPED_EXISTING",
                        file=str(out),
                    )
                )
                continue

            try:
                await ensure_list_page(page)

                link = await find_link_by_date(page, date_iso)

                if link is None:
                    print("RESULT                : LINK_NOT_FOUND")
                    logs.append(
                        blank_log_row(
                            date_iso,
                            "FAILED",
                            "date link not found",
                        )
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

                title = await page.title()
                body_text = await page.locator("body").inner_text()
                html = await page.content()

                ok, result = validate_daily_html(
                    html=html,
                    title=title,
                    body_text=body_text,
                    date_iso=date_iso,
                    min_machines=args.min_machines,
                )

                print(f"title                 : {result['title']}")
                print(f"date check            : {result['date_ok']}")
                print(f"store check           : {result['store_ok']}")
                print(f"records               : {result['records']}")
                print(f"unique machines       : {result['unique_machines']}")
                print(f"duplicate rows        : {result['duplicate_rows']}")
                print(f"missing machine       : {result['missing_machine']}")
                print(f"missing name          : {result['missing_name']}")
                print(f"invalid diff          : {result['invalid_diff']}")
                print(f"invalid G             : {result['invalid_G']}")
                print(f"negative G            : {result['negative_G']}")
                print(f"machine count check   : {result['machine_count_ok']}")

                if not ok:
                    print("RESULT                : NOT SAVED")
                    logs.append(
                        {
                            "date": date_iso,
                            "status": "FAILED_VALIDATION",
                            "reason": "daily page validation failed",
                            "records": result["records"],
                            "unique_machines": result["unique_machines"],
                            "duplicate_rows": result["duplicate_rows"],
                            "missing_machine": result["missing_machine"],
                            "missing_name": result["missing_name"],
                            "invalid_diff": result["invalid_diff"],
                            "invalid_G": result["invalid_G"],
                            "negative_G": result["negative_G"],
                            "title": result["title"],
                            "file": "",
                        }
                    )
                    continue

                out.write_text(html, encoding="utf-8")

                print(f"html chars            : {len(html):,}")
                print(f"saved                 : {out}")
                print("RESULT                : OK")

                logs.append(
                    {
                        "date": date_iso,
                        "status": "OK",
                        "reason": "",
                        "records": result["records"],
                        "unique_machines": result["unique_machines"],
                        "duplicate_rows": result["duplicate_rows"],
                        "missing_machine": result["missing_machine"],
                        "missing_name": result["missing_name"],
                        "invalid_diff": result["invalid_diff"],
                        "invalid_G": result["invalid_G"],
                        "negative_G": result["negative_G"],
                        "title": result["title"],
                        "file": str(out),
                    }
                )

            except Exception as exc:
                print(f"ERROR                 : {exc}")
                logs.append(
                    blank_log_row(
                        date_iso,
                        "ERROR",
                        repr(exc),
                    )
                )

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
        print("If the store tab was missing, V3 opened it automatically.")
        print("Historical machine count changes are allowed.")
        print(f"Pages with fewer than {args.min_machines} machines are rejected.")
        print("No Maruhan data files were modified.")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
