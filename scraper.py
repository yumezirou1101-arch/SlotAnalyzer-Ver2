# scraper.py
"""アナスロから全台データを取得する処理。"""

from __future__ import annotations

import asyncio
import re
from datetime import date
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

from playwright.async_api import Browser, Page, async_playwright

from config import PAGE_TIMEOUT_MS, StoreConfig


class ScrapingError(RuntimeError):
    """必要なデータを取得できない場合の例外。"""


DATE_PATTERN = re.compile(
    r"(20\d{2})\s*[年/.\-]\s*(\d{1,2})\s*[月/.\-]\s*(\d{1,2})"
)


def normalize_text(value: str) -> str:
    """改行や連続した空白を統一する。"""
    return " ".join(value.replace("\u3000", " ").split())


def parse_date(value: str) -> date | None:
    """文字列から日付を取得する。"""
    match = DATE_PATTERN.search(value)

    if not match:
        return None

    try:
        year, month, day = map(int, match.groups())
        return date(year, month, day)
    except ValueError:
        return None


def parse_number(value: str) -> int | float | None:
    """数値を含む文字列を int または float に変換する。"""
    cleaned = normalize_text(value)
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("枚", "")

    match = re.search(r"[+-]?\d+(?:\.\d+)?", cleaned)

    if not match:
        return None

    number = float(match.group())

    if number.is_integer():
        return int(number)

    return number


def canonical_header(value: str) -> str | None:
    """サイト上の表ヘッダーを保存用の列名へ変換する。"""
    text = normalize_text(value).lower()

    rules = {
        "台番号": ("台番号", "台番", "台号"),
        "ゲーム数": ("ゲーム", "g数", "総回転", "回転数", "累計スタート"),
        "BB": ("bb", "big", "ビッグ"),
        "RB": ("rb", "reg", "レギュラー"),
        "合成確率": ("合成", "合算", "確率"),
        "差枚": ("差枚", "差メダル", "差玉", "出玉"),
    }

    for output_name, keywords in rules.items():
        if any(keyword in text for keyword in keywords):
            return output_name

    return None


async def goto(page: Page, url: str) -> None:
    """指定URLへ移動する。"""

    await page.goto(
        url,
        wait_until="domcontentloaded",
        timeout=PAGE_TIMEOUT_MS,
    )

    # 少し長めに待機
    await page.wait_for_timeout(3000)

    # ネットワーク通信が落ち着くまで待つ
    try:
        await page.wait_for_load_state("networkidle", timeout=10000)
    except:
        pass

    print(f"[DEBUG] URL = {page.url}")
    print(f"[DEBUG] TITLE = {await page.title()}")

    html = await page.content()
    print("[DEBUG] HTML先頭")
    print(html[:1000])

    with open("debug.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("[DEBUG] debug.html を保存しました")




async def collect_links(
    page: Page,
    selectors: tuple[str, ...],
) -> list[tuple[str, str]]:
    """セレクタに一致するリンクを取得する。"""

    links: list[tuple[str, str]] = []
    seen_urls: set[str] = set()

    print("[DEBUG] collect_links を開始します。")

    for selector in selectors:
        print(f"[DEBUG] selector = {selector}")

        anchors = await page.locator(selector).all()

        print(f"[DEBUG] anchors = {len(anchors)}")

        for anchor in anchors:
            href = await anchor.get_attribute("href")
            text = normalize_text(await anchor.inner_text())

            print(f"[DEBUG] text = {text}")
            print(f"[DEBUG] href = {href}")

            if not href or not text:
                continue

            absolute_url = urljoin(page.url, href)

            if absolute_url in seen_urls:
                continue

            links.append((text, absolute_url))
            seen_urls.add(absolute_url)

    print(f"[DEBUG] 合計リンク数 = {len(links)}")

    return links


async def find_latest_business_day(
    page: Page,
    store: StoreConfig,
) -> tuple[date, str]:
    """店舗ページから最新営業日とそのURLを取得する。"""
    candidates: list[tuple[date, str]] = []

    links = await collect_links(page, store.selectors["date_links"])

    for text, url in links:
        business_date = parse_date(text) or parse_date(url)

        if business_date is not None:
            candidates.append((business_date, url))

    if not candidates:
        raise ScrapingError(
            "最新営業日へのリンクを見つけられませんでした。"
        )

    return max(candidates, key=lambda item: item[0])


def is_same_site(url: str, base_url: str) -> bool:
    """外部リンクを除外する。"""
    target = urlparse(url)
    base = urlparse(base_url)

    return (
        target.netloc == base.netloc
        and target.scheme in {"http", "https"}
    )


def is_store_related(url: str, store_url: str) -> bool:
    """対象店舗のデータ一覧URL配下にあるリンクだけを許可する。"""
    candidate_path = unquote(urlparse(url).path).rstrip("/")
    store_path = unquote(urlparse(store_url).path).rstrip("/")

    return (
        candidate_path == store_path
        or candidate_path.startswith(f"{store_path}/")
    )


async def find_machine_links(
    page: Page,
    store: StoreConfig,
) -> list[tuple[str, str]]:
    """営業日ページから機種ページへのリンク一覧を取得する。"""
    machine_links: list[tuple[str, str]] = []

    ignored_names = {
        "トップ",
        "ホーム",
        "前へ",
        "次へ",
        "一覧",
        "日付",
        "ページ",
    }

    links = await collect_links(page, store.selectors["machine_links"])

    for machine_name, url in links:
        if parse_date(machine_name):
            continue

        if not is_same_site(url, store.url):
            continue

        if not is_store_related(url, store.url):
            continue

        if len(machine_name) < 2:
            continue

        if machine_name in ignored_names:
            continue

        if url.split("#")[0] == page.url.split("#")[0]:
            continue

        machine_links.append((machine_name, url))

    unique_links: dict[str, tuple[str, str]] = {}

    for machine_name, url in machine_links:
        unique_links.setdefault(url, (machine_name, url))

    if not unique_links:
        raise ScrapingError(
            "機種ページへのリンクを見つけられませんでした。"
        )

    return list(unique_links.values())


async def extract_machine_rows(
    page: Page,
    machine_name: str,
    store: StoreConfig,
) -> list[dict[str, Any]]:
    """機種ページの表から全台データを抽出する。"""
    rows: list[dict[str, Any]] = []

    for selector in store.selectors["data_tables"]:
        tables = await page.locator(selector).all()

        for table in tables:
            header_cells = await table.locator(
                "thead th, tr:first-child th, tr:first-child td"
            ).all_inner_texts()

            headers = [
                canonical_header(header)
                for header in header_cells
            ]

            if "台番号" not in headers or "差枚" not in headers:
                continue

            table_rows = await table.locator("tbody tr").all()

            if not table_rows:
                table_rows = await table.locator("tr").all()
                table_rows = table_rows[1:]

            for table_row in table_rows:
                cells = await table_row.locator("th, td").all_inner_texts()
                cells = [normalize_text(cell) for cell in cells]

                if len(cells) != len(headers):
                    continue

                raw_data = {
                    header: cell
                    for header, cell in zip(headers, cells)
                    if header is not None
                }

                if not raw_data.get("台番号"):
                    continue

                rows.append(
                    {
                        "機種名": machine_name,
                        "台番号": raw_data.get("台番号", ""),
                        "ゲーム数": parse_number(
                            raw_data.get("ゲーム数", "")
                        ),
                        "BB": parse_number(raw_data.get("BB", "")),
                        "RB": parse_number(raw_data.get("RB", "")),
                        "合成確率": raw_data.get("合成確率", ""),
                        "差枚": parse_number(raw_data.get("差枚", "")),
                    }
                )

    return rows


async def scrape_store_async(
    store: StoreConfig,
    headless: bool = True,
) -> list[dict[str, Any]]:
    """店舗の最新営業日における全機種・全台データを取得する。"""

    async with async_playwright() as playwright:

        context = await playwright.chromium.launch_persistent_context(
    user_data_dir=r"C:\Users\user\AppData\Local\Google\Chrome\User Data\Playwright",
    channel="chrome",
    headless=False,
        viewport={"width": 1400, "height": 900},
    locale="ja-JP",
        user_agent=(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
    ),

)

           
        

        page = context.pages[0] if context.pages else await context.new_page()

        try:
            await goto(page, store.url)

            business_date, business_day_url = await find_latest_business_day(
                page,
                store,
            )

            await goto(page, business_day_url)

            machine_links = await find_machine_links(page, store)

            all_rows: list[dict[str, Any]] = []

            for machine_name, machine_url in machine_links:
                await goto(page, machine_url)

                machine_rows = await extract_machine_rows(
                    page,
                    machine_name,
                    store,
                )

                for row in machine_rows:
                    row["日付"] = business_date
                    row["店舗"] = store.name
                    all_rows.append(row)

            if not all_rows:
                raise ScrapingError(
                    "全台データ表を取得できませんでした。"
                )

            return all_rows

        finally:
            print("Enterキーを押すとブラウザを閉じます")
            input()
            await context.close()


def scrape_store(
    store: StoreConfig,
    headless: bool = True,
) -> list[dict[str, Any]]:
    """main.py から呼び出す同期版の入口。"""
    return asyncio.run(
        scrape_store_async(
            store=store,
            headless=headless,
        )
    )