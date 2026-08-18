import asyncio
import csv
import re
from datetime import date, timedelta
from pathlib import Path

from playwright.async_api import async_playwright


# ============================================================
# 設定
# ============================================================

# 7/11はすでに取得済みなので、追加収集は7/12～8/10
START_DATE = "2026-07-12"
END_DATE = "2026-08-10"

CDP_URL = "http://127.0.0.1:9222"

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 取得済み日を保存する個別CSV
# 既存の ana_slo_20260711.csv はそのまま残します。

COMBINED_CSV = (
    OUTPUT_DIR
    / "ana_slo_20260712_20260810.csv"
)

# 正常ページ判定
MIN_BODY_LENGTH = 1000
MIN_TABLE_COUNT = 1

# ページ読み込み待機
PAGE_CHECK_ATTEMPTS = 15
PAGE_CHECK_INTERVAL = 2


# ============================================================
# 共通関数
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    return text.strip()


def normalize_machine_number(value):
    if value is None:
        return None

    text = str(value).strip()
    text = re.sub(r"[^\d]", "", text)

    if not text:
        return None

    try:
        number = int(text)
    except ValueError:
        return None

    if 500 <= number <= 1500:
        return str(number)

    return None


def extract_numbers(text):
    if not text:
        return []

    values = re.findall(r"\b\d{3,4}\b", text)
    result = []

    for value in values:
        normalized = normalize_machine_number(value)

        if normalized is not None:
            result.append(normalized)

    return result


def normalize_header(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace(" ", "")
    text = text.replace("\u3000", "")
    text = text.replace("\n", "")
    text = text.replace("\r", "")

    return text


# ============================================================
# 日付一覧
# ============================================================

def make_date_list(start_date, end_date):
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    result = []
    current = start

    while current <= end:
        result.append(current.isoformat())
        current += timedelta(days=1)

    return result


# ============================================================
# Chrome接続
# ============================================================

async def connect_to_chrome(playwright):
    print()
    print("Chrome 9222へ接続します...")

    try:
        browser = await playwright.chromium.connect_over_cdp(
            CDP_URL
        )
    except Exception as e:
        print()
        print("【ERROR】Chromeへの接続に失敗しました。")
        print("Chromeを --remote-debugging-port=9222 で起動してください。")
        print()
        print(f"詳細: {e}")
        return None

    print("★ Chrome接続成功")
    return browser


# ============================================================
# 現在のChromeタブ情報取得
# ============================================================

async def inspect_page(page, context_index, page_index):
    """
    同じURLの空タブと正常タブを区別するための詳細情報を取得する。

    正常ページの実測値:
      本文 約70,930文字
      table 1191個
      titleあり
      「台番号」「差枚」「機種」あり

    空タブの実測値:
      本文 7文字
      table 0個
      title空
    """

    try:
        url = page.url
    except Exception:
        url = ""

    try:
        title = await page.title()
    except Exception:
        title = ""

    body_text = ""
    try:
        body_text = await page.locator("body").inner_text(
            timeout=3000
        )
        body_text = clean_text(body_text)
    except Exception:
        pass

    table_count = 0
    try:
        table_count = await page.locator("table").count()
    except Exception:
        pass

    ready_state = ""
    try:
        ready_state = await page.evaluate(
            "() => document.readyState"
        )
    except Exception:
        pass

    html_length = 0
    try:
        html_length = await page.evaluate(
            "() => document.documentElement ? document.documentElement.outerHTML.length : 0"
        )
    except Exception:
        pass

    has_machine_number = "台番号" in body_text
    has_diff_medals = "差枚" in body_text
    has_machine_name = "機種" in body_text or "機種名" in body_text
    has_data_summary = "データまとめ" in title

    # --------------------------------------------------------
    # ページ正常度スコア
    # --------------------------------------------------------

    score = 0

    if "ana-slo.com" in url.lower():
        score += 50

    if "アナスロ" in title:
        score += 30

    if has_data_summary:
        score += 20

    if ready_state == "complete":
        score += 10

    if len(body_text) >= MIN_BODY_LENGTH:
        score += 20

    if table_count >= MIN_TABLE_COUNT:
        score += 20

    if table_count >= 10:
        score += 10

    if table_count >= 100:
        score += 10

    if has_machine_number:
        score += 30

    if has_diff_medals:
        score += 30

    if has_machine_name:
        score += 10

    if has_machine_number and has_diff_medals:
        score += 30

    # --------------------------------------------------------
    # 正常ページ判定
    # --------------------------------------------------------

    is_valid = (
        "ana-slo.com" in url.lower()
        and len(body_text) >= MIN_BODY_LENGTH
        and table_count >= MIN_TABLE_COUNT
        and has_machine_number
        and has_diff_medals
    )

    return {
        "page": page,
        "score": score,
        "url": url,
        "title": title,
        "body_length": len(body_text),
        "table_count": table_count,
        "ready_state": ready_state,
        "html_length": html_length,
        "has_machine_number": has_machine_number,
        "has_diff_medals": has_diff_medals,
        "has_machine_name": has_machine_name,
        "is_valid": is_valid,
        "context_index": context_index,
        "page_index": page_index,
    }


async def list_pages(browser):
    candidates = []

    for context_index, context in enumerate(browser.contexts):
        pages = context.pages

        for page_index, page in enumerate(pages):
            try:
                item = await inspect_page(
                    page,
                    context_index,
                    page_index
                )

                candidates.append(item)

            except Exception:
                continue

    print()
    print("======================================================================")
    print("【現在のChromeタブ】")
    print("======================================================================")

    for index, item in enumerate(candidates):
        print()
        print(
            f"タブ[{index}] "
            f"score={item['score']} "
            f"本文={item['body_length']}文字 "
            f"table={item['table_count']}"
        )
        print(f"タイトル: {item['title']}")
        print(f"URL: {item['url']}")
        print(f"readyState: {item['ready_state']}")
        print(f"HTML: {item['html_length']}文字")
        print(
            "判定: "
            f"{'正常候補' if item['is_valid'] else '除外'}"
        )

    return candidates


# ============================================================
# 対象日ページ検索
# ============================================================

async def find_target_page(browser, target_date):
    """
    対象日のアナスロページを探す。

    重要:
      同じURLのタブが複数存在しても、
      本文・table・必要キーワードを確認して正常ページだけを選択する。

    最大30秒待機:
      15回 × 2秒
    """

    date_compact = target_date.replace("-", "")
    date_slash = target_date.replace("-", "/")

    print()
    print("======================================================================")
    print("【対象ページ読み込み待機・正常ページ判定】")
    print("======================================================================")
    print(f"対象日: {target_date}")
    print(
        f"最大待機: {PAGE_CHECK_ATTEMPTS * PAGE_CHECK_INTERVAL}秒"
    )

    best = None

    for attempt in range(1, PAGE_CHECK_ATTEMPTS + 1):

        candidates = await list_pages(browser)

        target_candidates = []

        for item in candidates:

            title = item["title"]
            url = item["url"]

            # 対象日URLまたは対象日タイトル
            if date_compact not in url and date_slash not in title:
                continue

            target_candidates.append(item)

        print()
        print(
            f"対象日候補: {len(target_candidates)}件"
        )

        # ----------------------------------------------------
        # 正常候補だけを抽出
        # ----------------------------------------------------

        valid_candidates = [
            item
            for item in target_candidates
            if item["is_valid"]
        ]

        if valid_candidates:

            valid_candidates.sort(
                key=lambda x: (
                    x["score"],
                    x["table_count"],
                    x["body_length"],
                    x["html_length"],
                ),
                reverse=True,
            )

            candidate = valid_candidates[0]

            print()
            print(
                f"確認 {attempt}/{PAGE_CHECK_ATTEMPTS}: "
                f"本文={candidate['body_length']}文字 "
                f"table={candidate['table_count']} "
                f"score={candidate['score']}"
            )

            best = candidate
            break

        # ----------------------------------------------------
        # まだ正常ページがない場合の状況表示
        # ----------------------------------------------------

        if target_candidates:

            target_candidates.sort(
                key=lambda x: (
                    x["score"],
                    x["body_length"],
                    x["table_count"],
                ),
                reverse=True,
            )

            candidate = target_candidates[0]

            print()
            print(
                f"確認 {attempt}/{PAGE_CHECK_ATTEMPTS}: "
                f"正常条件未達 "
                f"本文={candidate['body_length']}文字 "
                f"table={candidate['table_count']} "
                f"score={candidate['score']}"
            )

        else:
            print()
            print(
                f"確認 {attempt}/{PAGE_CHECK_ATTEMPTS}: "
                "対象日ページ候補なし"
            )

        if attempt < PAGE_CHECK_ATTEMPTS:
            await asyncio.sleep(
                PAGE_CHECK_INTERVAL
            )

    if best is None:

        print()
        print(
            f"【ERROR】{target_date} の正常なアナスロデータページを"
            "確認できませんでした。"
        )
        print()
        print("必要条件:")
        print(
            f"  本文 >= {MIN_BODY_LENGTH}文字"
        )
        print(
            f"  table >= {MIN_TABLE_COUNT}"
        )
        print("  「台番号」を含む")
        print("  「差枚」を含む")
        return None

    print()
    print("★ 正常な対象日のアナスロページを選択しました")
    print(f"対象日: {target_date}")
    print(f"score: {best['score']}")
    print(f"本文: {best['body_length']}文字")
    print(f"table: {best['table_count']}個")
    print(f"HTML: {best['html_length']}文字")
    print(f"タイトル: {best['title']}")
    print(f"URL: {best['url']}")

    return best["page"]


# ============================================================
# 最終ページ検証
# ============================================================

async def validate_selected_page(page, target_date):
    """
    find_target_page()後にもう一度ページを検証する。
    空タブを掴んでしまった場合は保存処理へ進ませない。
    """

    print()
    print("======================================================================")
    print("【選択ページ最終検証】")
    print("======================================================================")

    try:
        url = page.url
    except Exception:
        url = ""

    try:
        title = await page.title()
    except Exception:
        title = ""

    try:
        body_text = await page.locator("body").inner_text(
            timeout=10000
        )
        body_text = clean_text(body_text)
    except Exception as e:
        print("【ERROR】ページ本文取得失敗")
        print(e)
        return False

    try:
        table_count = await page.locator("table").count()
    except Exception:
        table_count = 0

    try:
        html_length = await page.evaluate(
            "() => document.documentElement ? document.documentElement.outerHTML.length : 0"
        )
    except Exception:
        html_length = 0

    date_compact = target_date.replace("-", "")
    date_slash = target_date.replace("-", "/")

    date_ok = (
        date_compact in url
        or date_slash in title
    )

    data_ok = (
        len(body_text) >= MIN_BODY_LENGTH
        and table_count >= MIN_TABLE_COUNT
        and "台番号" in body_text
        and "差枚" in body_text
    )

    print(f"対象日: {target_date}")
    print(f"タイトル: {title}")
    print(f"URL: {url}")
    print(f"本文: {len(body_text)}文字")
    print(f"table: {table_count}個")
    print(f"HTML: {html_length}文字")
    print(f"日付一致: {'OK' if date_ok else 'NG'}")
    print(f"データページ: {'OK' if data_ok else 'NG'}")

    if not date_ok:
        print("【ERROR】対象日とページ日付が一致しません。")
        return False

    if not data_ok:
        print("【ERROR】データページとして不十分です。")
        return False

    print("★ 最終検証OK")
    return True


# ============================================================
# ページ確認
# ============================================================

async def get_page_information(page, target_date):
    try:
        title = await page.title()
    except Exception:
        title = ""

    try:
        url = page.url
    except Exception:
        url = ""

    try:
        body_text = await page.locator("body").inner_text(
            timeout=10000
        )
        body_text = clean_text(body_text)
    except Exception as e:
        print()
        print("【ERROR】ページ本文取得失敗")
        print(e)
        return "", title, url

    print()
    print("======================================================================")
    print("【ページ確認】")
    print("======================================================================")
    print()
    print(f"対象日: {target_date}")
    print(f"タイトル: {title}")
    print(f"URL: {url}")
    print(f"本文文字数: {len(body_text)}")

    return body_text, title, url


# ============================================================
# 詳細データtable検索
# ============================================================

async def find_detail_table(tables):
    table_count = await tables.count()

    print()
    print(f"ページ内table数: {table_count}")
    print()
    print(f"table解析開始: {table_count}個")

    best_table = None
    best_score = -1
    best_index = -1
    best_text = ""

    for index in range(table_count):
        try:
            table = tables.nth(index)

            text = await table.inner_text(
                timeout=3000
            )
            text = clean_text(text)

            if not text:
                continue

            normalized = normalize_header(text)

            score = 0

            if "台番号" in normalized:
                score += 30

            if "機種名" in normalized:
                score += 30

            if "機種" in normalized:
                score += 10

            if "差枚" in normalized:
                score += 30

            if "G数" in normalized:
                score += 20

            if "ゲーム数" in normalized:
                score += 20

            if "BB" in normalized:
                score += 10

            if "RB" in normalized:
                score += 10

            if "合成確率" in normalized:
                score += 10

            if "BB確率" in normalized:
                score += 5

            if "RB確率" in normalized:
                score += 5

            machine_numbers = extract_numbers(text)

            if machine_numbers:
                unique_numbers = list(
                    dict.fromkeys(machine_numbers)
                )
                score += min(
                    len(unique_numbers) * 2,
                    40
                )

            rows = table.locator("tr")
            row_count = await rows.count()

            if row_count >= 5:
                score += 5

            if row_count >= 10:
                score += 5

            if row_count >= 20:
                score += 10

            if row_count >= 50:
                score += 10

            if "台番号" in normalized and "差枚" in normalized:
                score += 40

            if (
                "台番号" in normalized
                and (
                    "G数" in normalized
                    or "ゲーム数" in normalized
                )
            ):
                score += 20

            if "機種名" in normalized and "差枚" in normalized:
                score += 20

            if score > best_score:
                best_score = score
                best_table = table
                best_index = index
                best_text = text

                print(
                    f"候補更新: table[{index}] "
                    f"score={score} "
                    f"rows={row_count} "
                    f"text={len(text)}文字"
                )

        except Exception:
            continue

    if best_table is None:
        print()
        print("【ERROR】詳細データtableを特定できませんでした。")
        return None, 0

    print()
    print(f"★ 詳細データtable候補: table[{best_index}]")
    print(f"★ スコア: {best_score}")
    print(f"★ 本文文字数: {len(best_text)}")

    return best_table, best_score


# ============================================================
# table行取得
# ============================================================

async def get_table_rows(table):
    rows = table.locator("tr")
    row_count = await rows.count()

    print()
    print(f"対象table行数: {row_count}")

    result = []

    for row_index in range(row_count):
        try:
            row = rows.nth(row_index)
            cells = row.locator("th, td")
            cell_count = await cells.count()

            values = []

            for cell_index in range(cell_count):
                cell = cells.nth(cell_index)

                value = await cell.inner_text(
                    timeout=3000
                )

                values.append(
                    clean_text(value)
                )

            if values:
                result.append(values)

        except Exception:
            continue

    return result


# ============================================================
# ヘッダー検索
# ============================================================

def find_header_row(rows):
    best_index = None
    best_score = -1

    for index, row in enumerate(rows):
        joined = "".join(row)
        score = 0

        if "台番号" in joined:
            score += 30

        if "機種名" in joined:
            score += 30

        if "機種" in joined:
            score += 10

        if "差枚" in joined:
            score += 30

        if "G数" in joined:
            score += 20

        if "ゲーム数" in joined:
            score += 20

        if "BB" in joined:
            score += 10

        if "RB" in joined:
            score += 10

        if score > best_score:
            best_score = score
            best_index = index

    return best_index, best_score


# ============================================================
# 列位置推定
# ============================================================

def detect_columns(header):
    columns = {}

    for index, value in enumerate(header):
        normalized = normalize_header(value)

        if (
            "台番号" in normalized
            or normalized == "台番"
        ):
            columns["台番号"] = index

        elif "機種名" in normalized:
            columns["機種名"] = index

        elif normalized == "機種":
            columns["機種名"] = index

        elif "差枚" in normalized:
            columns["差枚"] = index

        elif (
            "G数" in normalized
            or "ゲーム数" in normalized
        ):
            columns["G数"] = index

        elif normalized == "BB":
            columns["BB"] = index

        elif normalized == "RB":
            columns["RB"] = index

        elif "合成確率" in normalized:
            columns["合成確率"] = index

        elif "BB確率" in normalized:
            columns["BB確率"] = index

        elif "RB確率" in normalized:
            columns["RB確率"] = index

    return columns


# ============================================================
# 行から台番号検索
# ============================================================

def find_machine_number_in_row(row):
    for value in row:
        normalized = normalize_machine_number(value)

        if normalized is not None:
            return normalized

    joined = " ".join(row)
    numbers = extract_numbers(joined)

    if numbers:
        return numbers[0]

    return None


# ============================================================
# データ行解析
# ============================================================

def parse_data_rows(rows, header_index, target_date):
    if header_index is None:
        print(
            "【WARNING】ヘッダー行を特定できませんでした。"
        )
        return []

    header = rows[header_index]
    columns = detect_columns(header)

    print()
    print("======================================================================")
    print("【列解析】")
    print("======================================================================")
    print()
    print("ヘッダー:")
    print(header)

    print()
    print("検出列:")

    for key, value in columns.items():
        print(f"{key}: {value}")

    data_rows = []

    for row_index in range(
        header_index + 1,
        len(rows)
    ):
        row = rows[row_index]

        if not row:
            continue

        machine_number = None

        if "台番号" in columns:
            index = columns["台番号"]

            if index < len(row):
                machine_number = normalize_machine_number(
                    row[index]
                )

        if machine_number is None:
            machine_number = find_machine_number_in_row(
                row
            )

        if machine_number is None:
            continue

        record = {
            "日付": target_date,
            "台番号": machine_number,
            "機種名": "",
            "差枚": "",
            "G数": "",
            "BB": "",
            "RB": "",
            "合成確率": "",
            "BB確率": "",
            "RB確率": "",
        }

        for key in record.keys():

            if key == "日付":
                continue

            if key not in columns:
                continue

            index = columns[key]

            if index < len(row):
                record[key] = row[index]

        # 機種名と差枚の両方が空なら採用しない
        if (
            not record["機種名"]
            and not record["差枚"]
        ):
            continue

        data_rows.append(record)

    return data_rows


# ============================================================
# 重複除去
# ============================================================

def remove_duplicates(records):
    unique = {}

    for record in records:
        target_date = record.get("日付", "")
        machine_number = record.get("台番号", "")

        if not target_date or not machine_number:
            continue

        key = (
            target_date,
            machine_number
        )

        unique[key] = record

    return list(unique.values())


# ============================================================
# 日別CSV保存
# ============================================================

def save_daily_csv(records, target_date):
    output_csv = (
        OUTPUT_DIR
        / f"ana_slo_{target_date.replace('-', '')}.csv"
    )

    fieldnames = [
        "日付",
        "台番号",
        "機種名",
        "差枚",
        "G数",
        "BB",
        "RB",
        "合成確率",
        "BB確率",
        "RB確率",
    ]

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for record in records:
            writer.writerow(record)

    print()
    print("★ CSV保存成功")
    print(output_csv)

    return output_csv


# ============================================================
# デバッグHTML保存
# ============================================================

async def save_debug_html(page, target_date):
    debug_html = (
        OUTPUT_DIR
        / f"ana_slo_{target_date.replace('-', '')}_debug.html"
    )

    try:
        html = await page.content()

        debug_html.write_text(
            html,
            encoding="utf-8"
        )

        print()
        print("★ HTML保存成功")
        print(debug_html)

    except Exception as e:
        print()
        print("【WARNING】HTML保存失敗")
        print(e)


# ============================================================
# 結果表示
# ============================================================

def print_records(records):
    print()
    print("======================================================================")
    print("【取得結果】")
    print("======================================================================")
    print()
    print(f"取得レコード数: {len(records)}")

    for index, record in enumerate(
        records[:30],
        start=1
    ):
        print(
            f"{index:3d}. "
            f"台番号={record.get('台番号', '')} "
            f"機種={record.get('機種名', '')} "
            f"差枚={record.get('差枚', '')} "
            f"G数={record.get('G数', '')}"
        )

    if len(records) > 30:
        print()
        print(
            f"... 以下 {len(records) - 30}台"
        )


# ============================================================
# Enter待ち
# ============================================================

async def wait_for_enter(target_date):
    print()
    print("重要:")
    print(
        "Chromeで対象日のアナスロ「データまとめ」ページを"
        "開いてください。"
    )
    print()
    print(f"対象日: {target_date}")
    print()
    print(
        "ページを完全に表示したら、Enterを押してください..."
    )

    await asyncio.to_thread(input)


# ============================================================
# 1日分の収集
# ============================================================

async def collect_one_day(browser, target_date):
    print()
    print("#" * 70)
    print(f"【収集開始】{target_date}")
    print("#" * 70)

    # --------------------------------------------------------
    # 既存CSVがあれば自動スキップ
    # --------------------------------------------------------

    daily_csv = (
        OUTPUT_DIR
        / f"ana_slo_{target_date.replace('-', '')}.csv"
    )

    if daily_csv.exists():
        print()
        print("【既存データあり】")
        print(f"ファイル: {daily_csv}")
        print("★ この日は自動スキップします。")
        print(
            "  → Chromeでページを開く必要はありません。"
        )
        return True, daily_csv

    # --------------------------------------------------------
    # Chromeページ確認
    # --------------------------------------------------------

    await wait_for_enter(target_date)

    page = await find_target_page(
        browser,
        target_date
    )

    if page is None:
        return False, None

    # --------------------------------------------------------
    # 最終検証
    # --------------------------------------------------------

    if not await validate_selected_page(
        page,
        target_date
    ):
        print()
        print(
            "【ERROR】選択ページの最終検証に失敗しました。"
        )
        await save_debug_html(
            page,
            target_date
        )
        return False, None

    body_text, title, url = await get_page_information(
        page,
        target_date
    )

    if len(body_text) < MIN_BODY_LENGTH:
        print()
        print(
            f"【ERROR】{target_date} のページ本文が不足しています。"
        )
        await save_debug_html(
            page,
            target_date
        )
        return False, None

    # --------------------------------------------------------
    # 日付確認
    # --------------------------------------------------------

    date_compact = target_date.replace("-", "")
    date_slash = target_date.replace("-", "/")

    if (
        date_compact not in url
        and date_slash not in title
    ):
        print()
        print(
            "【ERROR】対象日と現在ページの日付が一致していません。"
        )
        print(f"対象日: {target_date}")
        print(f"現在URL: {url}")
        print(f"現在タイトル: {title}")
        return False, None

    # --------------------------------------------------------
    # デバッグHTML
    # --------------------------------------------------------

    await save_debug_html(
        page,
        target_date
    )

    # --------------------------------------------------------
    # 詳細データテーブル解析
    # --------------------------------------------------------

    print()
    print("======================================================================")
    print("【詳細データテーブル解析】")
    print("======================================================================")

    tables = page.locator("table")
    table_count = await tables.count()

    if table_count == 0:
        print()
        print("【ERROR】tableがありません。")
        return False, None

    detail_table, score = await find_detail_table(
        tables
    )

    if detail_table is None:
        return False, None

    rows = await get_table_rows(
        detail_table
    )

    print()
    print(f"取得した行数: {len(rows)}")

    if not rows:
        print()
        print(
            "【ERROR】tableから行を取得できませんでした。"
        )
        return False, None

    # --------------------------------------------------------
    # ヘッダー
    # --------------------------------------------------------

    header_index, header_score = find_header_row(
        rows
    )

    print()
    print(f"ヘッダー候補行: {header_index}")
    print(f"ヘッダースコア: {header_score}")

    if header_index is not None:
        print()
        print("ヘッダー候補:")
        print(rows[header_index])

    # --------------------------------------------------------
    # データ解析
    # --------------------------------------------------------

    records = parse_data_rows(
        rows,
        header_index,
        target_date
    )

    print()
    print(f"解析レコード数: {len(records)}")

    records = remove_duplicates(
        records
    )

    print()
    print(
        f"重複除去後: {len(records)}台"
    )

    # --------------------------------------------------------
    # 台数検証
    # --------------------------------------------------------

    if len(records) < 400:
        print()
        print(
            "【WARNING】取得台数が400台未満です。"
        )
        print(
            "ページ状態または解析結果を確認してください。"
        )

        answer = input(
            "この結果を保存しますか？ [y/N]: "
        ).strip().lower()

        if answer != "y":
            print(
                "★ この日の保存を中止しました。"
            )
            return False, None

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    print_records(
        records
    )

    output_csv = save_daily_csv(
        records,
        target_date
    )

    print()
    print(
        f"★ {target_date} の収集完了"
    )

    return True, output_csv


# ============================================================
# 取得済み日別CSVを統合
# ============================================================

def combine_daily_csvs(dates):
    print()
    print("======================================================================")
    print("【日別CSV統合】")
    print("======================================================================")

    all_records = []

    fieldnames = [
        "日付",
        "台番号",
        "機種名",
        "差枚",
        "G数",
        "BB",
        "RB",
        "合成確率",
        "BB確率",
        "RB確率",
    ]

    for target_date in dates:

        csv_path = (
            OUTPUT_DIR
            / f"ana_slo_{target_date.replace('-', '')}.csv"
        )

        if not csv_path.exists():
            continue

        try:
            with csv_path.open(
                "r",
                newline="",
                encoding="utf-8-sig"
            ) as f:

                reader = csv.DictReader(f)

                count = 0

                for row in reader:

                    record = {
                        key: row.get(
                            key,
                            ""
                        )
                        for key in fieldnames
                    }

                    all_records.append(record)
                    count += 1

            print(
                f"{target_date}: {count}台"
            )

        except Exception as e:
            print()
            print(
                f"【WARNING】{target_date} 読み込み失敗: {e}"
            )

    all_records = remove_duplicates(
        all_records
    )

    with COMBINED_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for record in all_records:
            writer.writerow(record)

    print()
    print("★ 統合CSV保存成功")
    print(COMBINED_CSV)
    print()
    print(
        f"統合レコード数: {len(all_records)}"
    )

    return len(all_records)


# ============================================================
# メイン
# ============================================================

async def main():
    dates = make_date_list(
        START_DATE,
        END_DATE
    )

    print(
        "======================================================================"
    )
    print(
        "アナスロ 30日分データ収集【正常ページ判定強化・途中再開対応版】"
    )
    print(
        "======================================================================"
    )

    print()
    print("方式:")
    print(
        "現在Chromeで開いているアナスロページを"
    )
    print(
        "9222経由で読み取ります。"
    )
    print(
        "PythonからURLは開きません。"
    )
    print(
        "ページの再読み込みも行いません。"
    )

    print()
    print("追加収集対象日:")

    for target_date in dates:
        print(
            f"  {target_date}"
        )

    print()
    print(
        f"対象日数: {len(dates)}日"
    )

    print()
    print("重要:")
    print(
        "2026-07-11 はすでに取得済みなので対象外です。"
    )
    print(
        "既存の daily CSV は自動スキップします。"
    )
    print(
        "既存の all_data.csv は変更しません。"
    )

    async with async_playwright() as playwright:

        browser = await connect_to_chrome(
            playwright
        )

        if browser is None:
            return

        success_dates = []
        failed_dates = []

        for target_date in dates:

            try:

                success, output_csv = await collect_one_day(
                    browser,
                    target_date
                )

                if success:
                    success_dates.append(
                        target_date
                    )
                else:
                    failed_dates.append(
                        target_date
                    )

            except Exception as e:

                print()
                print(
                    "【ERROR】予期しないエラー"
                )
                print(
                    f"対象日: {target_date}"
                )
                print(
                    f"詳細: {e}"
                )

                failed_dates.append(
                    target_date
                )

        # ----------------------------------------------------
        # 統合
        # ----------------------------------------------------

        combined_count = combine_daily_csvs(
            dates
        )

        # ----------------------------------------------------
        # 最終結果
        # ----------------------------------------------------

        print()
        print(
            "======================================================================"
        )
        print(
            "【30日分収集結果】"
        )
        print(
            "======================================================================"
        )

        print()
        print(
            f"成功: {len(success_dates)}日"
        )

        for target_date in success_dates:
            print(
                f"  ★ {target_date}"
            )

        print()
        print(
            f"失敗: {len(failed_dates)}日"
        )

        for target_date in failed_dates:
            print(
                f"  × {target_date}"
            )

        print()
        print(
            f"統合レコード数: {combined_count}"
        )

        print()
        print(
            "======================================================================"
        )
        print(
            "★★★★★ アナスロ追加データ収集完了 ★★★★★"
        )
        print(
            "======================================================================"
        )

        print()
        print("個別CSV:")
        print(
            f"{OUTPUT_DIR}\\ana_slo_YYYYMMDD.csv"
        )

        print()
        print("統合CSV:")
        print(
            COMBINED_CSV
        )

        print()
        print(
            "既存の all_data.csv は変更していません。"
        )


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())
