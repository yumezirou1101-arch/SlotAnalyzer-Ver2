from __future__ import annotations

from pathlib import Path
from datetime import date, datetime, timedelta
from io import StringIO
import argparse
import asyncio
import csv
import re
import sys

import pandas as pd
from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
)


# ============================================================
# Fixed configuration
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
CDP_URL = "http://127.0.0.1:9222"

STORE_TEXTS = (
    "ビックつばめ高崎店",
    "ビックつばめ高崎",
)

LIST_TITLE_TEXT = "データ一覧"

STORE_LIST_URL = (
    "https://ana-slo.com/"
    "%E3%83%9B%E3%83%BC%E3%83%AB%E3%83%87%E3%83%BC%E3%82%BF/"
    "%E7%BE%A4%E9%A6%AC%E7%9C%8C/"
    "%E3%83%93%E3%83%83%E3%82%AF%E3%81%A4%E3%81%B0%E3%82%81"
    "%E9%AB%98%E5%B4%8E%E5%BA%97-%E3%83%87%E3%83%BC%E3%82%BF"
    "%E4%B8%80%E8%A6%A7/"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "bic_tsubame_takasaki"
)

LOG_DIR = DATA_DIR / "fetch_logs"

DEFAULT_START_DATE = date(2026, 8, 8)
DEFAULT_END_DATE = date(2026, 9, 15)
DEFAULT_MIN_MACHINES = 200
DEFAULT_DELAY_SEC = 8.0

LIST_TIMEOUT_MS = 30000
DETAIL_RESPONSE_TIMEOUT_MS = 25000
DETAIL_LOAD_TIMEOUT_MS = 25000
CONTENT_TIMEOUT_SEC = 15.0

# Known store closure.
# These days are not treated as missing data, fetch failures, or freshness failures.
KNOWN_CLOSURES = (
    (date(2026, 7, 21), date(2026, 8, 7), "STORE_RENOVATION"),
)

DATE_TEXT_RE = re.compile(r"^(20\d{2})/(\d{2})/(\d{2})")
DATE_URL_RE = re.compile(r"/(20\d{2})-(\d{2})-(\d{2})-")

BLOCK_TEXTS = (
    "403 forbidden",
    "forbidden",
    "access denied",
    "アクセスが拒否",
    "アクセスできません",
    "temporarily unavailable",
    "service unavailable",
)


# ============================================================
# Generic helpers
# ============================================================

def header(title: str) -> None:
    print()
    print("=" * 112)
    print(title)
    print("=" * 112)


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}. Use YYYY-MM-DD."
        ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Bic Tsubame Takasaki Ana-Slo daily pages safely "
            "for an explicit date range. Each open-store day uses a "
            "fresh browser tab. The run stops immediately on the first "
            "failure and writes progress to the CSV log after every day."
        )
    )
    parser.add_argument(
        "--start-date",
        type=parse_iso_date,
        default=DEFAULT_START_DATE,
        help="Inclusive start date. Default: 2026-08-08",
    )
    parser.add_argument(
        "--end-date",
        type=parse_iso_date,
        default=DEFAULT_END_DATE,
        help="Inclusive end date. Default: 2026-09-15",
    )
    parser.add_argument(
        "--min-machines",
        type=int,
        default=DEFAULT_MIN_MACHINES,
    )
    parser.add_argument(
        "--delay-sec",
        type=float,
        default=DEFAULT_DELAY_SEC,
        help=(
            "Wait between newly fetched days. "
            f"Default: {DEFAULT_DELAY_SEC:.0f} seconds."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already saved source HTML files.",
    )
    return parser.parse_args()


def daterange(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def closure_reason(target: date) -> str | None:
    for start_date, end_date, reason in KNOWN_CLOSURES:
        if start_date <= target <= end_date:
            return reason
    return None


def has_store_text(text: str) -> bool:
    return any(name in text for name in STORE_TEXTS)


def looks_blocked(title: str, body_text: str) -> bool:
    haystack = f"{title}\n{body_text}".lower()
    return any(marker.lower() in haystack for marker in BLOCK_TEXTS)


def output_path(date_iso: str) -> Path:
    compact = date_iso.replace("-", "")
    return (
        PROJECT_ROOT
        / f"ana_slo_bic_tsubame_takasaki_{compact}_source.html"
    )


# ============================================================
# Logging
# ============================================================

LOG_FIELDS = [
    "date",
    "status",
    "reason",
    "list_http_status",
    "detail_http_status",
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
    "updated_at",
]


def make_log_path() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"bic_tsubame_takasaki_fetch_v4_{stamp}.csv"


def base_log_row(
    date_iso: str,
    status: str,
    reason: str = "",
    *,
    list_http_status: str | int = "",
    detail_http_status: str | int = "",
    file: str = "",
) -> dict:
    return {
        "date": date_iso,
        "status": status,
        "reason": reason,
        "list_http_status": list_http_status,
        "detail_http_status": detail_http_status,
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
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_log(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    temp_path.replace(path)


def upsert_log_row(rows: list[dict], new_row: dict) -> None:
    for index, row in enumerate(rows):
        if row["date"] == new_row["date"]:
            rows[index] = new_row
            return
    rows.append(new_row)


# ============================================================
# Browser helpers
# ============================================================

async def safe_title(page) -> str:
    return await asyncio.wait_for(
        page.title(),
        timeout=CONTENT_TIMEOUT_SEC,
    )


async def safe_body_text(page) -> str:
    return await asyncio.wait_for(
        page.locator("body").inner_text(),
        timeout=CONTENT_TIMEOUT_SEC,
    )


async def safe_content(page) -> str:
    return await asyncio.wait_for(
        page.content(),
        timeout=CONTENT_TIMEOUT_SEC,
    )


async def open_list_page(page) -> tuple[int | None, str]:
    response = await page.goto(
        STORE_LIST_URL,
        wait_until="domcontentloaded",
        timeout=LIST_TIMEOUT_MS,
    )

    await page.wait_for_timeout(1500)

    status = response.status if response is not None else None
    title = await safe_title(page)

    if status == 403:
        raise RuntimeError("LIST_HTTP_403")

    if status is not None and status >= 400:
        raise RuntimeError(f"LIST_HTTP_{status}")

    if LIST_TITLE_TEXT not in title or not has_store_text(title):
        raise RuntimeError(
            "LIST_PAGE_IDENTITY_MISMATCH: "
            f"title={title!r}"
        )

    return status, title


async def collect_date_links(page):
    anchor_rows = await page.locator("a").evaluate_all(
        """elements => elements.map((element, index) => ({
            index,
            text: (element.innerText || element.textContent || "").trim(),
            href: element.href || element.getAttribute("href") || ""
        }))"""
    )

    found = {}

    for row in anchor_rows:
        text = str(row.get("text", "")).strip()
        href = str(row.get("href", "")).strip()

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

    anchor_rows = await anchors.evaluate_all(
        """elements => elements.map((element, index) => ({
            index,
            text: (element.innerText || element.textContent || "").trim(),
            href: element.href || element.getAttribute("href") || ""
        }))"""
    )

    label_prefix = date_iso.replace("-", "/")

    for row in anchor_rows:
        text = str(row.get("text", "")).strip()
        href = str(row.get("href", "")).strip()

        if (
            text.startswith(label_prefix)
            and date_iso in href
            and "ana-slo.com" in href.lower()
            and "-data" in href.lower()
        ):
            return anchors.nth(int(row["index"]))

    return None


async def click_daily_link(page, date_iso: str):
    link = await find_link_by_date(page, date_iso)

    if link is None:
        raise RuntimeError("DATE_LINK_NOT_FOUND_AFTER_FRESH_LIST_LOAD")

    href = str(await link.get_attribute("href") or "").strip()
    href_lower = href.lower()

    if (
        "ana-slo.com" not in href_lower
        or date_iso not in href_lower
        or "-data" not in href_lower
    ):
        raise RuntimeError(
            f"INVALID_DAILY_HREF: {href!r}"
        )

    print(f"href                  : {href}")
    print(f"DOM click href        : {href}")

    response_status = None

    try:
        async with page.expect_response(
            lambda response: (
                date_iso in response.url
                and "-data" in response.url.lower()
            ),
            timeout=DETAIL_RESPONSE_TIMEOUT_MS,
        ) as response_info:
            await link.evaluate(
                "(element) => element.click()"
            )

        response = await response_info.value
        response_status = response.status

    except PlaywrightTimeoutError:
        raise RuntimeError(
            "DETAIL_RESPONSE_TIMEOUT"
        )

    if response_status == 403:
        raise RuntimeError("DETAIL_HTTP_403")

    if response_status is not None and response_status >= 400:
        raise RuntimeError(
            f"DETAIL_HTTP_{response_status}"
        )

    try:
        await page.wait_for_load_state(
            "domcontentloaded",
            timeout=DETAIL_LOAD_TIMEOUT_MS,
        )
    except PlaywrightTimeoutError:
        raise RuntimeError(
            "DETAIL_DOMCONTENTLOADED_TIMEOUT"
        )

    await page.wait_for_timeout(1800)

    print(f"detail HTTP status    : {response_status}")
    print(f"current URL after click: {page.url}")

    return href, response_status


# ============================================================
# HTML quality validation
# ============================================================

def find_main_table(html: str) -> pd.DataFrame:
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError as exc:
        raise RuntimeError(
            "NO_HTML_TABLES_FOUND"
        ) from exc

    required_variants = (
        {"機種名", "台番号", "G数", "差枚"},
        {"機種名", "台番号", "総回転数", "差枚"},
    )

    candidates = []

    for index, table in enumerate(tables):
        columns = {str(c).strip() for c in table.columns}

        if any(
            required.issubset(columns)
            for required in required_variants
        ):
            candidates.append(
                (len(table), index, table.copy())
            )

    if not candidates:
        raise RuntimeError(
            "MAIN_MACHINE_TABLE_NOT_FOUND"
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return candidates[0][2]


def clean_for_quality(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [
        str(column).strip()
        for column in x.columns
    ]

    x = x.rename(
        columns={
            "機種名": "machine_name",
            "台番号": "machine_no",
            "G数": "G",
            "総回転数": "G",
            "差枚": "diff",
        }
    )

    required = {
        "machine_name",
        "machine_no",
        "G",
        "diff",
    }

    missing = required - set(x.columns)

    if missing:
        raise RuntimeError(
            "REQUIRED_COLUMNS_MISSING: "
            f"{sorted(missing)}"
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

    for column in ("G", "diff"):
        x[column] = (
            x[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("+", "", regex=False)
            .str.strip()
        )

        x[column] = pd.to_numeric(
            x[column],
            errors="coerce",
        )

    return x


def validate_daily_html(
    *,
    html: str,
    title: str,
    body_text: str,
    current_url: str,
    date_iso: str,
    min_machines: int,
):
    date_text = date_iso.replace("-", "/")

    date_ok = (
        date_text in title
        or date_text in body_text
        or date_iso in current_url
    )

    store_ok = (
        has_store_text(title)
        or has_store_text(body_text)
    )

    url_ok = (
        "ana-slo.com" in current_url.lower()
        and date_iso in current_url
        and "-data" in current_url.lower()
    )

    blocked = looks_blocked(
        title,
        body_text,
    )

    if blocked:
        raise RuntimeError(
            "DETAIL_PAGE_BLOCK_OR_ACCESS_DENIED_TEXT"
        )

    table = find_main_table(html)
    x = clean_for_quality(table)

    records = len(x)

    unique_machines = int(
        x["machine_no"]
        .nunique(dropna=True)
    )

    duplicate_rows = int(
        x["machine_no"]
        .duplicated(keep=False)
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
        .isin(["", "nan", "None"])
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

    machine_count_ok = (
        records >= min_machines
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

    result = {
        "title": title,
        "date_ok": date_ok,
        "store_ok": store_ok,
        "url_ok": url_ok,
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
            url_ok,
            machine_count_ok,
            unique_ok,
            data_ok,
        )
    )

    return ok, result


# ============================================================
# One-day isolated fetch
# ============================================================

async def fetch_one_day(
    *,
    context,
    target: date,
    min_machines: int,
    overwrite: bool,
):
    date_iso = target.isoformat()
    out = output_path(date_iso)

    if out.exists() and not overwrite:
        return base_log_row(
            date_iso,
            "SKIPPED_EXISTING",
            file=str(out),
        )

    page = await context.new_page()

    list_http_status = ""
    detail_http_status = ""

    try:
        print("opening fresh tab      : YES")
        print("opening list page      : explicit URL")

        list_http_status, list_title = (
            await open_list_page(page)
        )

        print(
            f"list HTTP status      : "
            f"{list_http_status}"
        )
        print(
            f"list page title       : "
            f"{list_title}"
        )

        _, detail_http_status = (
            await click_daily_link(
                page,
                date_iso,
            )
        )

        title = await safe_title(page)
        body_text = await safe_body_text(page)
        html = await safe_content(page)

        ok, result = validate_daily_html(
            html=html,
            title=title,
            body_text=body_text,
            current_url=page.url,
            date_iso=date_iso,
            min_machines=min_machines,
        )

        print(
            f"title                 : "
            f"{result['title']}"
        )
        print(
            f"date check            : "
            f"{result['date_ok']}"
        )
        print(
            f"store check           : "
            f"{result['store_ok']}"
        )
        print(
            f"url check             : "
            f"{result['url_ok']}"
        )
        print(
            f"records               : "
            f"{result['records']}"
        )
        print(
            f"unique machines       : "
            f"{result['unique_machines']}"
        )
        print(
            f"duplicate rows        : "
            f"{result['duplicate_rows']}"
        )
        print(
            f"missing machine       : "
            f"{result['missing_machine']}"
        )
        print(
            f"missing name          : "
            f"{result['missing_name']}"
        )
        print(
            f"invalid diff          : "
            f"{result['invalid_diff']}"
        )
        print(
            f"invalid G             : "
            f"{result['invalid_G']}"
        )
        print(
            f"negative G            : "
            f"{result['negative_G']}"
        )
        print(
            f"machine count check   : "
            f"{result['machine_count_ok']}"
        )

        if not ok:
            row = base_log_row(
                date_iso,
                "FAILED_VALIDATION",
                "daily page validation failed",
                list_http_status=list_http_status,
                detail_http_status=detail_http_status,
            )

            for key in (
                "records",
                "unique_machines",
                "duplicate_rows",
                "missing_machine",
                "missing_name",
                "invalid_diff",
                "invalid_G",
                "negative_G",
                "title",
            ):
                row[key] = result[key]

            return row

        out.write_text(
            html,
            encoding="utf-8",
        )

        print(
            f"html chars            : "
            f"{len(html):,}"
        )
        print(
            f"saved                 : "
            f"{out}"
        )
        print(
            "RESULT                : OK"
        )

        row = base_log_row(
            date_iso,
            "OK",
            list_http_status=list_http_status,
            detail_http_status=detail_http_status,
            file=str(out),
        )

        for key in (
            "records",
            "unique_machines",
            "duplicate_rows",
            "missing_machine",
            "missing_name",
            "invalid_diff",
            "invalid_G",
            "negative_G",
            "title",
        ):
            row[key] = result[key]

        return row

    except asyncio.TimeoutError as exc:
        return base_log_row(
            date_iso,
            "ERROR",
            f"ASYNC_TIMEOUT: {exc!r}",
            list_http_status=list_http_status,
            detail_http_status=detail_http_status,
        )

    except Exception as exc:
        return base_log_row(
            date_iso,
            "ERROR",
            repr(exc),
            list_http_status=list_http_status,
            detail_http_status=detail_http_status,
        )

    finally:
        try:
            await asyncio.wait_for(
                page.close(),
                timeout=5.0,
            )
        except Exception:
            pass


# ============================================================
# Main
# ============================================================

async def main():
    args = parse_args()

    if args.start_date > args.end_date:
        raise ValueError(
            "--start-date must be on or before "
            "--end-date."
        )

    if args.min_machines < 1:
        raise ValueError(
            "--min-machines must be at least 1."
        )

    if args.delay_sec < 5.0:
        raise ValueError(
            "--delay-sec must be at least 5 seconds "
            "to keep access load low."
        )

    requested_dates = list(
        daterange(
            args.start_date,
            args.end_date,
        )
    )

    log_path = make_log_path()
    logs: list[dict] = []

    header(
        "Bic Tsubame Takasaki - "
        "Isolated Daily Fetch V4"
    )

    print(
        f"requested start       : "
        f"{args.start_date}"
    )
    print(
        f"requested end         : "
        f"{args.end_date}"
    )
    print(
        f"requested calendar days: "
        f"{len(requested_dates)}"
    )
    print(
        f"min machines          : "
        f"{args.min_machines}"
    )
    print(
        f"delay sec             : "
        f"{args.delay_sec}"
    )
    print(
        f"overwrite             : "
        f"{args.overwrite}"
    )
    print(
        "known closure         : "
        "2026-07-21 to 2026-08-07 "
        "(STORE_RENOVATION)"
    )
    print(
        f"progress log          : "
        f"{log_path}"
    )
    print(
        "failure policy        : "
        "STOP ON FIRST FAILURE"
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

        # One lightweight list-page probe at startup.
        probe = await context.new_page()

        try:
            probe_status, probe_title = (
                await open_list_page(probe)
            )

            date_links = await collect_date_links(
                probe
            )

            if not date_links:
                raise RuntimeError(
                    "No usable daily date links "
                    "were found."
                )

            available_dates = {
                item["date"]
                for item in date_links
            }

            print(
                f"probe HTTP status     : "
                f"{probe_status}"
            )
            print(
                f"probe title           : "
                f"{probe_title}"
            )
            print(
                f"date links found      : "
                f"{len(date_links)}"
            )
            print(
                f"latest link date      : "
                f"{date_links[0]['date']}"
            )
            print(
                f"oldest link date      : "
                f"{date_links[-1]['date']}"
            )

        finally:
            try:
                await asyncio.wait_for(
                    probe.close(),
                    timeout=5.0,
                )
            except Exception:
                pass

        failure_found = False

        for position, target in enumerate(
            requested_dates,
            start=1,
        ):
            date_iso = target.isoformat()

            header(
                f"[{position}/"
                f"{len(requested_dates)}] "
                f"{date_iso}"
            )

            closure = closure_reason(target)

            if closure is not None:
                row = base_log_row(
                    date_iso,
                    "KNOWN_CLOSURE",
                    closure,
                )

                upsert_log_row(
                    logs,
                    row,
                )
                save_log(
                    log_path,
                    logs,
                )

                print(
                    f"known closure         : "
                    f"{closure}"
                )
                print(
                    "RESULT                : "
                    "KNOWN_CLOSURE"
                )

                continue

            out = output_path(date_iso)

            if out.exists() and not args.overwrite:
                row = base_log_row(
                    date_iso,
                    "SKIPPED_EXISTING",
                    file=str(out),
                )

                upsert_log_row(
                    logs,
                    row,
                )
                save_log(
                    log_path,
                    logs,
                )

                print(
                    f"existing file         : "
                    f"{out}"
                )
                print(
                    "RESULT                : "
                    "SKIPPED_EXISTING"
                )

                continue

            if date_iso not in available_dates:
                row = base_log_row(
                    date_iso,
                    "FAILED",
                    "REQUESTED_OPEN_STORE_DATE_LINK_NOT_FOUND",
                )

                upsert_log_row(
                    logs,
                    row,
                )
                save_log(
                    log_path,
                    logs,
                )

                print(
                    "RESULT                : "
                    "FAILED"
                )
                print(
                    "reason                : "
                    "requested date link not found"
                )

                failure_found = True
                break

            # Write IN_PROGRESS before any browser action for this date.
            row = base_log_row(
                date_iso,
                "IN_PROGRESS",
                "fetch started",
            )

            upsert_log_row(
                logs,
                row,
            )
            save_log(
                log_path,
                logs,
            )

            row = await fetch_one_day(
                context=context,
                target=target,
                min_machines=args.min_machines,
                overwrite=args.overwrite,
            )

            upsert_log_row(
                logs,
                row,
            )
            save_log(
                log_path,
                logs,
            )

            if row["status"] not in {
                "OK",
                "SKIPPED_EXISTING",
                "KNOWN_CLOSURE",
            }:
                print(
                    "RESULT                : "
                    f"{row['status']}"
                )
                print(
                    "reason                : "
                    f"{row['reason']}"
                )
                print()
                print(
                    "STOP                  : "
                    "first failure reached; "
                    "no later dates were requested."
                )

                failure_found = True
                break

            if position < len(requested_dates):
                print(
                    f"low-load wait         : "
                    f"{args.delay_sec:.1f} sec"
                )
                await asyncio.sleep(
                    args.delay_sec
                )

        counts = {}

        for row in logs:
            counts[row["status"]] = (
                counts.get(
                    row["status"],
                    0,
                )
                + 1
            )

        header("SUMMARY")
        print(
            f"logged dates          : "
            f"{len(logs)}"
        )

        for status, count in sorted(
            counts.items()
        ):
            print(
                f"{status:<22}: "
                f"{count}"
            )

        print(
            f"log                   : "
            f"{log_path}"
        )

        if failure_found:
            failed_rows = [
                row
                for row in logs
                if row["status"] not in {
                    "OK",
                    "SKIPPED_EXISTING",
                    "KNOWN_CLOSURE",
                }
            ]

            print()
            print(
                "FINAL RESULT          : "
                "FAILED_SAFE_STOP"
            )

            for row in failed_rows:
                print(
                    f"  {row['date']} | "
                    f"{row['status']} | "
                    f"{row['reason']}"
                )

            raise RuntimeError(
                "Bic Tsubame Takasaki V4 stopped "
                "safely on the first failed date. "
                "Progress was saved. Re-run later "
                "to resume; existing HTML files "
                "will be skipped automatically."
            )

        print()
        print(
            "FINAL RESULT          : OK"
        )
        print()
        print(
            "Each fetched day used a fresh browser tab."
        )
        print(
            "Progress was written after every processed day."
        )
        print(
            "Known closure dates are not failures."
        )
        print(
            "The run stops on the first fetch/validation failure."
        )
        print(
            "Existing saved HTML files are skipped on resume."
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print()
        print(
            "INTERRUPTED            : "
            "Ctrl+C received."
        )
        print(
            "The latest V4 progress log remains on disk."
        )
        sys.exit(130)

    except Exception as exc:
        print()
        print(
            f"FATAL ERROR           : "
            f"{exc}"
        )
        sys.exit(1)
