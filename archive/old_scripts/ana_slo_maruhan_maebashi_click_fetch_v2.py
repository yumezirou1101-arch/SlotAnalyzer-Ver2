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


# ============================================================
# Maruhan Mega City Maebashi Inter - Ana-Slo Link Fetch V2
# ============================================================
#
# Main change from V1
# -------------------
# V1 required both visible link text and URL date format to match.
# V2 treats the visible date text such as:
#     2026/08/26(水)
# as the primary source of truth.
#
# The href is only used as a safety check that:
# - it points to ana-slo.com
# - it looks like a daily "-data" page
#
# This is intended to handle the actual Maebashi store list page
# where the visible date links are present even if the title is blank
# or the href format differs from the stricter V1 assumption.
#
# Safety
# ------
# - Existing HTML is skipped unless --overwrite is used
# - Date/store/table/basic quality are checked before saving
# - Strict 514-machine CSV validation remains delegated to:
#     ana_slo_source_html_to_daily_csv_auto.py
# - No Big March files are modified
# ============================================================


PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
CDP_URL = "http://127.0.0.1:9222"

STORE_NAME = "マルハンメガシティ前橋インター"
LIST_TITLE_TEXT = "データ一覧"

STORE_LIST_URL = (
    "https://ana-slo.com/"
    "%e3%83%9b%e3%83%bc%e3%83%ab%e3%83%87%e3%83%bc%e3%82%bf/"
    "%e7%be%a4%e9%a6%ac%e7%9c%8c/"
    "%e3%83%9e%e3%83%ab%e3%83%8f%e3%83%b3%e3%83%a1%e3%82%ac%e3%82%b7%e3%83%86%e3%82%a3"
    "%e5%89%8d%e6%a9%8b%e3%82%a4%e3%83%b3%e3%82%bf%e3%83%bc-"
    "%e3%83%87%e3%83%bc%e3%82%bf%e4%b8%80%e8%a6%a7/"
)

DEFAULT_MAX_DAYS = 1
MIN_REASONABLE_MACHINES = 450
EXPECTED_CURRENT_MACHINES = 514

LOG_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "fetch_logs"
)

# Visible text example:
#   2026/08/26(水)
DATE_TEXT_RE = re.compile(
    r"^\s*(20\d{2})/(\d{1,2})/(\d{1,2})"
)


def header(title: str) -> None:
    print()
    print("=" * 116)
    print(title)
    print("=" * 116)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Maruhan Mega City Maebashi Inter Ana-Slo daily pages "
            "by clicking visible date links from the store list page."
        )
    )

    parser.add_argument(
        "--max-days",
        type=int,
        default=DEFAULT_MAX_DAYS,
        help="Newest N visible date links to inspect. Default: 1.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ana_slo_YYYYMMDD_source.html files.",
    )

    return parser.parse_args()


async def page_title(page) -> str:
    try:
        return await page.title()
    except Exception:
        return ""


def parse_visible_date(text: str) -> str | None:
    match = DATE_TEXT_RE.match(
        str(text).strip()
    )

    if not match:
        return None

    y, m, d = match.groups()

    return (
        f"{int(y):04d}-"
        f"{int(m):02d}-"
        f"{int(d):02d}"
    )


def href_looks_daily(href: str | None) -> bool:
    if not href:
        return False

    s = href.lower()

    return (
        "ana-slo.com" in s
        and "-data" in s
    )


async def find_store_page(context):
    fallback = None

    for page in context.pages:
        url = page.url or ""

        if "ana-slo.com" not in url:
            continue

        title = await page_title(page)

        if STORE_NAME in title:
            if LIST_TITLE_TEXT in title:
                return page

            if fallback is None:
                fallback = page

        # URL fallback because Maebashi list page sometimes has blank title.
        lower_url = url.lower()

        if (
            fallback is None
            and "%e3%83%9e%e3%83%ab%e3%83%8f%e3%83%b3" in lower_url
            and "%e5%89%8d%e6%a9%8b" in lower_url
            and "%e3%83%87%e3%83%bc%e3%82%bf%e4%b8%80%e8%a6%a7" in lower_url
        ):
            fallback = page

    return fallback


async def open_store_list_page(context):
    header("AUTO OPEN STORE LIST PAGE")
    print(f"URL                   : {STORE_LIST_URL}")

    page = await context.new_page()

    response = await page.goto(
        STORE_LIST_URL,
        wait_until="domcontentloaded",
        timeout=60000,
    )

    await page.wait_for_timeout(2500)

    status = response.status if response is not None else None
    title = await page_title(page)

    print(f"HTTP status           : {status}")
    print(f"title                 : {title}")
    print(f"current URL           : {page.url}")

    if "ana-slo.com" not in page.url:
        raise RuntimeError(
            "Could not auto-open the Maruhan Maebashi Ana-Slo list page."
        )

    print("auto-open             : OK")
    return page


async def ensure_list_page(page):
    lower_url = (page.url or "").lower()
    title = await page_title(page)

    if (
        STORE_NAME in title
        and LIST_TITLE_TEXT in title
    ):
        return

    if (
        "ana-slo.com" in lower_url
        and "%e3%83%87%e3%83%bc%e3%82%bf%e4%b8%80%e8%a6%a7" in lower_url
        and "%e5%89%8d%e6%a9%8b" in lower_url
    ):
        return

    print("returning to list page : browser back")

    await page.go_back(
        wait_until="domcontentloaded",
        timeout=60000,
    )

    await page.wait_for_timeout(1800)

    title = await page_title(page)
    lower_url = (page.url or "").lower()

    print(f"list page title       : {title}")
    print(f"list page URL         : {page.url}")

    if not (
        "ana-slo.com" in lower_url
        and "%e3%83%87%e3%83%bc%e3%82%bf%e4%b8%80%e8%a6%a7" in lower_url
        and "%e5%89%8d%e6%a9%8b" in lower_url
    ):
        raise RuntimeError(
            "Could not return to the Maruhan Maebashi data-list page."
        )


async def collect_date_links(page):
    anchors = page.locator("a")
    count = await anchors.count()

    found: dict[str, dict[str, str]] = {}

    for i in range(count):
        a = anchors.nth(i)

        try:
            text = (await a.inner_text()).strip()
            href = await a.get_attribute("href")
        except Exception:
            continue

        date_iso = parse_visible_date(
            text
        )

        if date_iso is None:
            continue

        if not href_looks_daily(
            href
        ):
            continue

        found[date_iso] = {
            "date": date_iso,
            "label": text,
            "href": href,
        }

    return sorted(
        found.values(),
        key=lambda x: x["date"],
        reverse=True,
    )


async def find_link_by_date(
    page,
    date_iso: str,
):
    label_prefix = date_iso.replace(
        "-",
        "/",
    )

    anchors = page.locator("a")
    count = await anchors.count()

    for i in range(count):
        a = anchors.nth(i)

        try:
            text = (await a.inner_text()).strip()
            href = await a.get_attribute("href")
        except Exception:
            continue

        if not href_looks_daily(
            href
        ):
            continue

        parsed = parse_visible_date(
            text
        )

        if parsed == date_iso:
            return a

        # Secondary fallback.
        if text.startswith(
            label_prefix
        ):
            return a

    return None


def find_main_table(
    html: str,
) -> tuple[pd.DataFrame, int]:

    tables = pd.read_html(
        StringIO(html)
    )

    required = {
        "機種名",
        "台番号",
        "G数",
        "差枚",
    }

    candidates = []

    for idx, table in enumerate(
        tables
    ):
        cols = {
            str(c).strip()
            for c in table.columns
        }

        if required.issubset(
            cols
        ):
            candidates.append(
                (
                    len(table),
                    idx,
                    table.copy(),
                )
            )

    if not candidates:
        raise RuntimeError(
            "Main machine table not found."
        )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    _, idx, table = candidates[0]

    return table, idx


def clean_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    x = df.copy()

    x.columns = [
        str(c).strip()
        for c in x.columns
    ]

    x = x.rename(
        columns={
            "機種名": "machine_name",
            "台番号": "machine_no",
            "G数": "G",
            "差枚": "diff",
        }
    )

    x["machine_name"] = (
        x["machine_name"]
        .astype(str)
        .str.strip()
    )

    x["machine_no"] = pd.to_numeric(
        x["machine_no"],
        errors="coerce",
    )

    for col in (
        "G",
        "diff",
    ):
        x[col] = (
            x[col]
            .astype(str)
            .str.replace(
                ",",
                "",
                regex=False,
            )
            .str.replace(
                "+",
                "",
                regex=False,
            )
            .str.strip()
        )

        x[col] = pd.to_numeric(
            x[col],
            errors="coerce",
        )

    return x


def validate_daily_html(
    html: str,
    title: str,
    body_text: str,
    date_iso: str,
):

    date_text = date_iso.replace(
        "-",
        "/",
    )

    date_ok = (
        date_text in title
        or date_text in body_text
        or date_iso in html
    )

    store_ok = (
        STORE_NAME in title
        or STORE_NAME in body_text
        or STORE_NAME in html
    )

    table, table_index = find_main_table(
        html
    )

    x = clean_table(
        table
    )

    records = len(
        x
    )

    unique_machines = int(
        x["machine_no"]
        .nunique(
            dropna=True
        )
    )

    duplicate_rows = int(
        x["machine_no"]
        .duplicated(
            keep=False
        )
        .sum()
    )

    missing_machine = int(
        x["machine_no"]
        .isna()
        .sum()
    )

    missing_name = int(
        x["machine_name"]
        .astype(str)
        .str.strip()
        .isin(
            [
                "",
                "nan",
                "None",
            ]
        )
        .sum()
    )

    invalid_diff = int(
        x["diff"]
        .isna()
        .sum()
    )

    invalid_g = int(
        x["G"]
        .isna()
        .sum()
    )

    negative_g = int(
        (
            (x["G"] < 0)
            .fillna(False)
        )
        .sum()
    )

    minimum_count_ok = (
        records
        >= MIN_REASONABLE_MACHINES
    )

    current_count_match = (
        records
        == EXPECTED_CURRENT_MACHINES
    )

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

    ok = all(
        (
            date_ok,
            store_ok,
            minimum_count_ok,
            unique_ok,
            data_ok,
        )
    )

    result = {
        "title": title,
        "date_ok": date_ok,
        "store_ok": store_ok,
        "table_index": table_index,
        "records": records,
        "unique_machines": unique_machines,
        "duplicate_rows": duplicate_rows,
        "missing_machine": missing_machine,
        "missing_name": missing_name,
        "invalid_diff": invalid_diff,
        "invalid_G": invalid_g,
        "negative_G": negative_g,
        "minimum_count_ok": minimum_count_ok,
        "current_514_match": current_count_match,
    }

    return (
        ok,
        result,
    )


def output_path(
    date_iso: str,
) -> Path:

    compact = date_iso.replace(
        "-",
        "",
    )

    return (
        PROJECT_ROOT
        / f"ana_slo_{compact}_source.html"
    )


def save_log(
    rows,
):
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    path = (
        LOG_DIR
        / f"maruhan_maebashi_fetch_v2_{stamp}.csv"
    )

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
        "current_514_match",
        "title",
        "file",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

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
        "current_514_match": "",
        "title": "",
        "file": file,
    }


async def main():
    args = parse_args()

    if args.max_days < 1:
        raise ValueError(
            "--max-days must be >= 1"
        )

    header(
        "Maruhan Mega City Maebashi Inter - Ana-Slo Link Fetch V2"
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

        page = await find_store_page(
            context
        )

        if page is None:
            print(
                "store tab             : NOT FOUND"
            )
            print(
                "action                : auto-open list page"
            )

            page = await open_store_list_page(
                context
            )

        else:
            print(
                "store tab             : FOUND"
            )

        print(
            f"source tab title      : {await page_title(page)}"
        )
        print(
            f"source tab URL        : {page.url}"
        )
        print(
            f"max days              : {args.max_days}"
        )
        print(
            f"overwrite             : {args.overwrite}"
        )
        print(
            f"expected current count: {EXPECTED_CURRENT_MACHINES}"
        )

        await ensure_list_page(
            page
        )

        # Give the list page a little extra time to finish rendering.
        await page.wait_for_timeout(
            1500
        )

        date_links = await collect_date_links(
            page
        )

        print(
            f"visible daily links   : {len(date_links)}"
        )

        if not date_links:
            # Diagnostic output before failing.
            anchors = page.locator("a")
            count = await anchors.count()

            print(
                f"all anchor count      : {count}"
            )

            preview = []

            for i in range(
                min(
                    count,
                    30,
                )
            ):
                a = anchors.nth(i)

                try:
                    text = (
                        await a.inner_text()
                    ).strip()

                    href = await a.get_attribute(
                        "href"
                    )

                    if text:
                        preview.append(
                            (
                                text,
                                href,
                            )
                        )
                except Exception:
                    continue

            print()
            print(
                "ANCHOR PREVIEW"
            )

            for text, href in preview:
                print(
                    f"TEXT={text!r}  HREF={href!r}"
                )

            raise RuntimeError(
                "No usable visible daily date links were found."
            )

        selected = date_links[
            :args.max_days
        ]

        print(
            f"dates selected        : {len(selected)}"
        )

        print(
            f"selected range        : "
            f"{selected[-1]['date']} to "
            f"{selected[0]['date']}"
        )

        logs = []

        for pos, item in enumerate(
            selected,
            start=1,
        ):
            date_iso = item[
                "date"
            ]

            out = output_path(
                date_iso
            )

            header(
                f"[{pos}/{len(selected)}] {date_iso}"
            )

            print(
                f"visible label         : {item['label']}"
            )

            print(
                f"href                  : {item['href']}"
            )

            if (
                out.exists()
                and not args.overwrite
            ):
                print(
                    f"existing file         : {out}"
                )

                print(
                    "RESULT                : SKIPPED_EXISTING"
                )

                logs.append(
                    blank_log_row(
                        date_iso,
                        "SKIPPED_EXISTING",
                        file=str(out),
                    )
                )

                continue

            try:
                await ensure_list_page(
                    page
                )

                link = await find_link_by_date(
                    page,
                    date_iso,
                )

                if link is None:
                    print(
                        "RESULT                : LINK_NOT_FOUND"
                    )

                    logs.append(
                        blank_log_row(
                            date_iso,
                            "FAILED",
                            "visible date link not found",
                        )
                    )

                    continue

                async with page.expect_navigation(
                    wait_until="domcontentloaded",
                    timeout=60000,
                ):
                    await link.click()

                await page.wait_for_timeout(
                    2500
                )

                title = await page_title(
                    page
                )

                body_text = await page.locator(
                    "body"
                ).inner_text()

                html = await page.content()

                ok, result = validate_daily_html(
                    html=html,
                    title=title,
                    body_text=body_text,
                    date_iso=date_iso,
                )

                print(
                    f"title                 : {result['title']}"
                )
                print(
                    f"current URL           : {page.url}"
                )
                print(
                    f"date check            : {result['date_ok']}"
                )
                print(
                    f"store check           : {result['store_ok']}"
                )
                print(
                    f"main table index      : {result['table_index']}"
                )
                print(
                    f"records               : {result['records']}"
                )
                print(
                    f"unique machines       : {result['unique_machines']}"
                )
                print(
                    f"duplicate rows        : {result['duplicate_rows']}"
                )
                print(
                    f"missing machine       : {result['missing_machine']}"
                )
                print(
                    f"missing name          : {result['missing_name']}"
                )
                print(
                    f"invalid diff          : {result['invalid_diff']}"
                )
                print(
                    f"invalid G             : {result['invalid_G']}"
                )
                print(
                    f"negative G            : {result['negative_G']}"
                )
                print(
                    f"minimum count check   : {result['minimum_count_ok']}"
                )
                print(
                    f"current 514 match     : {result['current_514_match']}"
                )

                if not ok:
                    print(
                        "RESULT                : NOT SAVED"
                    )

                    logs.append(
                        {
                            "date": date_iso,
                            "status": "FAILED_VALIDATION",
                            "reason": "HTML validation failed",
                            "records": result["records"],
                            "unique_machines": result["unique_machines"],
                            "duplicate_rows": result["duplicate_rows"],
                            "missing_machine": result["missing_machine"],
                            "missing_name": result["missing_name"],
                            "invalid_diff": result["invalid_diff"],
                            "invalid_G": result["invalid_G"],
                            "negative_G": result["negative_G"],
                            "current_514_match": result["current_514_match"],
                            "title": result["title"],
                            "file": "",
                        }
                    )

                    continue

                out.write_text(
                    html,
                    encoding="utf-8",
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
                        "current_514_match": result["current_514_match"],
                        "title": result["title"],
                        "file": str(out),
                    }
                )

            except Exception as exc:
                print(
                    f"ERROR                 : {exc}"
                )

                logs.append(
                    blank_log_row(
                        date_iso,
                        "ERROR",
                        repr(exc),
                    )
                )

        try:
            await ensure_list_page(
                page
            )
        except Exception as exc:
            print(
                f"final return warning  : {exc}"
            )

        log_path = save_log(
            logs
        )

        counts = {}

        for row in logs:
            counts[
                row["status"]
            ] = counts.get(
                row["status"],
                0,
            ) + 1

        header(
            "SUMMARY"
        )

        print(
            f"processed             : {len(logs)}"
        )

        for status, count in sorted(
            counts.items()
        ):
            print(
                f"{status:<22}: {count}"
            )

        print(
            f"log                   : {log_path}"
        )

        print()
        print(
            "V2 uses visible date text as the primary date-link detector."
        )
        print(
            "Strict 514-machine CSV validation remains delegated to "
            "ana_slo_source_html_to_daily_csv_auto.py."
        )
        print(
            "No Big March files were modified."
        )

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
