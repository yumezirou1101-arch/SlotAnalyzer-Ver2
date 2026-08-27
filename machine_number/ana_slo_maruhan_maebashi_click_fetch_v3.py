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
# Maruhan Mega City Maebashi Inter - Ana-Slo Link Fetch V3
# ============================================================
#
# V3 change
# ---------
# On some Chrome/CDP sessions the existing Maebashi tab reports:
#   - correct URL
#   - blank title
#   - locator("a").count() == 0
# even though the page is visibly rendered in Chrome.
#
# V3 therefore does NOT trust an existing tab only from its URL.
# It verifies that the DOM is actually accessible.
#
# If the existing tab has no usable DOM:
#   1) reload/navigate it explicitly to STORE_LIST_URL
#   2) wait for the DOM
#   3) if still unusable, open a fresh tab and navigate there
#
# After the list DOM is accessible:
#   - visible date text is the primary date detector
#   - target date link is clicked normally
#   - date/store/table/basic quality are checked
#   - source is saved as ana_slo_YYYYMMDD_source.html
#
# Strict 514-machine CSV validation remains delegated to:
#   ana_slo_source_html_to_daily_csv_auto.py
# ============================================================


PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
CDP_URL = "http://127.0.0.1:9222"

STORE_NAME = "マルハンメガシティ前橋インター"

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

DATE_TEXT_RE = re.compile(
    r"^\s*(20\d{2})/(\d{1,2})/(\d{1,2})"
)


def header(title: str) -> None:
    print()
    print("=" * 118)
    print(title)
    print("=" * 118)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Maruhan Mega City Maebashi Inter Ana-Slo daily pages "
            "with robust CDP DOM recovery."
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


async def safe_title(page) -> str:
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

    value = href.lower()

    return (
        "ana-slo.com" in value
        and "-data" in value
    )


def url_looks_maebashi_list(url: str) -> bool:
    value = (url or "").lower()

    return (
        "ana-slo.com" in value
        and "%e3%83%87%e3%83%bc%e3%82%bf%e4%b8%80%e8%a6%a7" in value
        and "%e5%89%8d%e6%a9%8b" in value
        and "%e3%83%9e%e3%83%ab%e3%83%8f%e3%83%b3" in value
    )


async def dom_anchor_count(page) -> int:
    try:
        return await page.locator("a").count()
    except Exception:
        return 0


async def dom_body_length(page) -> int:
    try:
        text = await page.locator("body").inner_text()
        return len(text)
    except Exception:
        return 0


async def dom_usable(page) -> bool:
    anchors = await dom_anchor_count(page)
    body_len = await dom_body_length(page)

    return (
        anchors > 0
        and body_len > 100
    )


async def find_candidate_page(context):
    for page in context.pages:
        if url_looks_maebashi_list(
            page.url
        ):
            return page

    for page in context.pages:
        if "ana-slo.com" in (page.url or ""):
            return page

    return None


async def navigate_to_list(page):
    header("NAVIGATE TO STORE LIST PAGE")
    print(f"URL                   : {STORE_LIST_URL}")

    response = await page.goto(
        STORE_LIST_URL,
        wait_until="domcontentloaded",
        timeout=60000,
    )

    await page.wait_for_timeout(
        3000
    )

    status = (
        response.status
        if response is not None
        else None
    )

    print(
        f"HTTP status           : {status}"
    )
    print(
        f"title                 : {await safe_title(page)}"
    )
    print(
        f"current URL           : {page.url}"
    )
    print(
        f"anchor count          : {await dom_anchor_count(page)}"
    )
    print(
        f"body chars            : {await dom_body_length(page)}"
    )


async def ensure_accessible_list_page(
    context,
    page,
):
    # Existing tab is accepted only if its DOM is accessible.
    if (
        page is not None
        and url_looks_maebashi_list(page.url)
    ):
        anchors = await dom_anchor_count(page)
        body_len = await dom_body_length(page)

        print(
            f"existing anchor count : {anchors}"
        )
        print(
            f"existing body chars   : {body_len}"
        )

        if await dom_usable(page):
            print(
                "existing DOM          : OK"
            )
            return page

        print(
            "existing DOM          : NOT ACCESSIBLE"
        )
        print(
            "action                : explicit reload/navigation"
        )

        try:
            await navigate_to_list(
                page
            )

            if await dom_usable(page):
                print(
                    "reloaded DOM          : OK"
                )
                return page
        except Exception as exc:
            print(
                f"reload warning        : {exc}"
            )

    # Existing page absent or reload still unusable.
    print(
        "action                : open fresh Ana-Slo tab"
    )

    fresh = await context.new_page()

    await navigate_to_list(
        fresh
    )

    if not await dom_usable(
        fresh
    ):
        raise RuntimeError(
            "Maebashi list page was opened, but its DOM is still not accessible."
        )

    print(
        "fresh DOM             : OK"
    )

    return fresh


async def collect_date_links(page):
    anchors = page.locator(
        "a"
    )

    count = await anchors.count()

    found = {}

    for i in range(
        count
    ):
        a = anchors.nth(
            i
        )

        try:
            text = (
                await a.inner_text()
            ).strip()

            href = await a.get_attribute(
                "href"
            )

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

        found[
            date_iso
        ] = {
            "date": date_iso,
            "label": text,
            "href": href,
        }

    return sorted(
        found.values(),
        key=lambda item: item["date"],
        reverse=True,
    )


async def find_link_by_date(
    page,
    date_iso: str,
):
    anchors = page.locator(
        "a"
    )

    count = await anchors.count()

    for i in range(
        count
    ):
        a = anchors.nth(
            i
        )

        try:
            text = (
                await a.inner_text()
            ).strip()

            href = await a.get_attribute(
                "href"
            )

        except Exception:
            continue

        if not href_looks_daily(
            href
        ):
            continue

        if parse_visible_date(
            text
        ) == date_iso:
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
            str(col).strip()
            for col in table.columns
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
        key=lambda row: row[0],
        reverse=True,
    )

    _, idx, table = candidates[0]

    return (
        table,
        idx,
    )


def clean_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    x = df.copy()

    x.columns = [
        str(col).strip()
        for col in x.columns
    ]

    x = x.rename(
        columns={
            "機種名": "machine_name",
            "台番号": "machine_no",
            "G数": "G",
            "差枚": "diff",
        }
    )

    x[
        "machine_name"
    ] = (
        x[
            "machine_name"
        ]
        .astype(str)
        .str.strip()
    )

    x[
        "machine_no"
    ] = pd.to_numeric(
        x[
            "machine_no"
        ],
        errors="coerce",
    )

    for col in (
        "G",
        "diff",
    ):
        x[
            col
        ] = (
            x[
                col
            ]
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

        x[
            col
        ] = pd.to_numeric(
            x[
                col
            ],
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
        x[
            "machine_no"
        ].nunique(
            dropna=True
        )
    )

    duplicate_rows = int(
        x[
            "machine_no"
        ]
        .duplicated(
            keep=False
        )
        .sum()
    )

    missing_machine = int(
        x[
            "machine_no"
        ]
        .isna()
        .sum()
    )

    missing_name = int(
        x[
            "machine_name"
        ]
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
        x[
            "diff"
        ]
        .isna()
        .sum()
    )

    invalid_g = int(
        x[
            "G"
        ]
        .isna()
        .sum()
    )

    negative_g = int(
        (
            (
                x[
                    "G"
                ]
                < 0
            )
            .fillna(
                False
            )
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
        unique_machines
        == records
        and duplicate_rows
        == 0
        and missing_machine
        == 0
    )

    data_ok = (
        missing_name
        == 0
        and invalid_diff
        == 0
        and invalid_g
        == 0
        and negative_g
        == 0
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

    return (
        ok,
        {
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
        },
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
        / f"maruhan_maebashi_fetch_v3_{stamp}.csv"
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
    ) as file:

        writer = csv.DictWriter(
            file,
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
        "Maruhan Mega City Maebashi Inter - Ana-Slo Link Fetch V3"
    )

    async with async_playwright() as playwright:

        browser = await playwright.chromium.connect_over_cdp(
            CDP_URL
        )

        if not browser.contexts:
            raise RuntimeError(
                "No Chrome context found."
            )

        context = browser.contexts[
            0
        ]

        candidate = await find_candidate_page(
            context
        )

        if candidate is None:
            print(
                "candidate tab          : NOT FOUND"
            )
        else:
            print(
                "candidate tab          : FOUND"
            )
            print(
                f"candidate title        : {await safe_title(candidate)}"
            )
            print(
                f"candidate URL          : {candidate.url}"
            )

        page = await ensure_accessible_list_page(
            context,
            candidate,
        )

        print()
        print(
            f"source tab title      : {await safe_title(page)}"
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
        print(
            f"anchor count          : {await dom_anchor_count(page)}"
        )
        print(
            f"body chars            : {await dom_body_length(page)}"
        )

        date_links = await collect_date_links(
            page
        )

        print(
            f"visible daily links   : {len(date_links)}"
        )

        if not date_links:
            raise RuntimeError(
                "DOM is accessible, but no visible daily date links were detected."
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
                # If previous iteration navigated away,
                # return explicitly to list page if needed.
                if not url_looks_maebashi_list(
                    page.url
                ):
                    await navigate_to_list(
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
                    2800
                )

                title = await safe_title(
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

        log_path = save_log(
            logs
        )

        counts = {}

        for row in logs:
            status = row[
                "status"
            ]

            counts[
                status
            ] = counts.get(
                status,
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
            "V3 verifies that the Playwright DOM is actually accessible."
        )
        print(
            "If an existing tab has URL-only/stale DOM state, it reloads "
            "or opens a fresh list tab."
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
