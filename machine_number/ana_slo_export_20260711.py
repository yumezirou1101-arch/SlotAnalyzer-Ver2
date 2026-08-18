import asyncio
import csv
import re
from pathlib import Path

from playwright.async_api import async_playwright


# ============================================================
# 設定
# ============================================================

TARGET_DATE = "2026-07-11"

CDP_URL = "http://127.0.0.1:9222"

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = (
    OUTPUT_DIR
    / f"ana_slo_{TARGET_DATE.replace('-', '')}.csv"
)

DEBUG_HTML = (
    OUTPUT_DIR
    / f"ana_slo_{TARGET_DATE.replace('-', '')}_debug.html"
)


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
    """
    台番号を整数文字列に正規化する。
    """

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

    # マルハン前橋インターの台番号として
    # 現実的な範囲だけを採用
    if 500 <= number <= 1500:
        return str(number)

    return None


def is_machine_number(value):
    """
    台番号らしい値か判定する。
    """

    return normalize_machine_number(value) is not None


def extract_numbers(text):
    """
    本文から3～4桁数字を抽出する。
    """

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
    """
    テーブル見出しを比較しやすくする。
    """

    if text is None:
        return ""

    text = str(text)

    text = text.replace(" ", "")
    text = text.replace("\u3000", "")
    text = text.replace("\n", "")
    text = text.replace("\r", "")

    return text


def contains_any(text, keywords):
    """
    text内にkeywordsのどれかが存在するか。
    """

    for keyword in keywords:
        if keyword in text:
            return True

    return False


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
        print()
        print("Chromeを --remote-debugging-port=9222 で")
        print("起動しているか確認してください。")
        print()
        print(f"詳細: {e}")

        return None

    print("★ Chrome接続成功")

    return browser


# ============================================================
# アナスロページ検索
# ============================================================

async def find_ana_slo_page(browser):

    print()
    print("ブラウザコンテキストを確認します...")

    contexts = browser.contexts

    print(
        f"ブラウザコンテキスト数: {len(contexts)}"
    )

    candidates = []

    for context_index, context in enumerate(contexts):

        pages = context.pages

        print(
            f"コンテキスト[{context_index}] "
            f"タブ数: {len(pages)}"
        )

        for page_index, page in enumerate(pages):

            try:

                url = page.url

                title = await page.title()

                body_text = ""

                try:
                    body_text = await page.locator(
                        "body"
                    ).inner_text(
                        timeout=3000
                    )
                except Exception:
                    pass

                score = 0

                if "ana-slo.com" in url.lower():
                    score += 50

                if "ana-slo.com" in title.lower():
                    score += 30

                if "アナスロ" in title:
                    score += 30

                if "データまとめ" in title:
                    score += 20

                if len(body_text) > 1000:
                    score += 20

                if "台番号" in body_text:
                    score += 30

                if "差枚" in body_text:
                    score += 20

                if "機種" in body_text:
                    score += 10

                candidates.append(
                    {
                        "page": page,
                        "score": score,
                        "url": url,
                        "title": title,
                        "body_length": len(body_text),
                    }
                )

            except Exception:
                continue

    if not candidates:

        print()
        print("【ERROR】Chrome内にタブがありません。")

        return None

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    print()
    print("======================================================================")
    print("【現在のChromeタブ】")
    print("======================================================================")

    for index, item in enumerate(candidates):

        print()
        print(
            f"[{index}] score={item['score']} "
            f"本文={item['body_length']}文字"
        )

        print(
            f"タイトル: {item['title']}"
        )

        print(
            f"URL: {item['url']}"
        )

    best = candidates[0]

    if best["score"] < 50:

        print()
        print(
            "【WARNING】アナスロページを確実に特定できませんでした。"
        )

        return None

    page = best["page"]

    print()
    print("★ 内容のあるアナスロページを選択しました")
    print(
        f"score: {best['score']}"
    )

    return page


# ============================================================
# ページ情報取得
# ============================================================

async def get_page_information(page):

    print()
    print("======================================================================")
    print("【ページ確認】")
    print("======================================================================")

    try:
        title = await page.title()
    except Exception:
        title = ""

    try:
        url = page.url
    except Exception:
        url = ""

    print()
    print("タイトル:")
    print(title)

    print()
    print("URL:")
    print(url)

    body_text = ""

    try:

        body = page.locator("body")

        body_text = await body.inner_text(
            timeout=10000
        )

        body_text = clean_text(body_text)

    except Exception as e:

        print()
        print("【ERROR】ページ本文取得失敗")
        print(e)

    print()
    print(
        f"本文文字数: {len(body_text)}"
    )

    return body_text


# ============================================================
# HTML保存
# ============================================================

async def save_debug_html(page):

    try:

        html = await page.content()

        DEBUG_HTML.write_text(
            html,
            encoding="utf-8"
        )

        print()
        print("★ HTML保存成功")
        print(DEBUG_HTML)

    except Exception as e:

        print()
        print("【WARNING】HTML保存失敗")
        print(e)


# ============================================================
# 詳細データtable検索
# ============================================================

async def find_detail_table(tables):

    table_count = await tables.count()

    print()
    print(
        f"table解析開始: {table_count}個"
    )

    best_table = None
    best_score = -1
    best_index = -1
    best_text = ""

    # --------------------------------------------------------
    # すべてのtableを順番に解析
    # --------------------------------------------------------

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

            # ------------------------------------------------
            # データ項目
            # ------------------------------------------------

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

            # ------------------------------------------------
            # 台番号
            # ------------------------------------------------

            machine_numbers = extract_numbers(text)

            if machine_numbers:

                unique_numbers = list(
                    dict.fromkeys(machine_numbers)
                )

                score += min(
                    len(unique_numbers) * 2,
                    40
                )

            # ------------------------------------------------
            # 行数
            # ------------------------------------------------

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

            # ------------------------------------------------
            # 強い組み合わせ
            # ------------------------------------------------

            if (
                "台番号" in normalized
                and "差枚" in normalized
            ):
                score += 40

            if (
                "台番号" in normalized
                and (
                    "G数" in normalized
                    or "ゲーム数" in normalized
                )
            ):
                score += 20

            if (
                "機種名" in normalized
                and "差枚" in normalized
            ):
                score += 20

            # ------------------------------------------------
            # 候補更新
            # ------------------------------------------------

            if score > best_score:

                best_score = score
                best_table = table
                best_index = index
                best_text = text

                print(
                    f"候補更新: "
                    f"table[{index}] "
                    f"score={score} "
                    f"rows={row_count} "
                    f"text={len(text)}文字"
                )

        except Exception:
            continue

    print()

    if best_table is None:

        print(
            "【ERROR】詳細データtableを特定できませんでした。"
        )

        return None, 0

    print(
        f"★ 詳細データtable候補: "
        f"table[{best_index}]"
    )

    print(
        f"★ スコア: {best_score}"
    )

    print(
        f"★ 本文文字数: {len(best_text)}"
    )

    return best_table, best_score


# ============================================================
# tableヘッダー解析
# ============================================================

async def get_table_rows(table):

    rows = table.locator("tr")

    row_count = await rows.count()

    print()
    print(
        f"対象table行数: {row_count}"
    )

    result = []

    for row_index in range(row_count):

        try:

            row = rows.nth(row_index)

            cells = row.locator(
                "th, td"
            )

            cell_count = await cells.count()

            values = []

            for cell_index in range(cell_count):

                cell = cells.nth(cell_index)

                value = await cell.inner_text(
                    timeout=3000
                )

                value = clean_text(value)

                values.append(value)

            if values:

                result.append(values)

        except Exception:
            continue

    return result


# ============================================================
# ヘッダー候補判定
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

        elif (
            "機種名" in normalized
        ):
            columns["機種名"] = index

        elif (
            normalized == "機種"
        ):
            columns["機種名"] = index

        elif (
            "差枚" in normalized
        ):
            columns["差枚"] = index

        elif (
            "G数" in normalized
            or "ゲーム数" in normalized
        ):
            columns["G数"] = index

        elif (
            normalized == "BB"
        ):
            columns["BB"] = index

        elif (
            normalized == "RB"
        ):
            columns["RB"] = index

        elif (
            "合成確率" in normalized
        ):
            columns["合成確率"] = index

        elif (
            "BB確率" in normalized
        ):
            columns["BB確率"] = index

        elif (
            "RB確率" in normalized
        ):
            columns["RB確率"] = index

    return columns


# ============================================================
# 行から台番号を探す
# ============================================================

def find_machine_number_in_row(row):

    # --------------------------------------------------------
    # まず各セルを調べる
    # --------------------------------------------------------

    for value in row:

        normalized = normalize_machine_number(
            value
        )

        if normalized is not None:
            return normalized

    # --------------------------------------------------------
    # セル内に複数文字がある場合
    # --------------------------------------------------------

    joined = " ".join(row)

    numbers = extract_numbers(joined)

    if numbers:

        return numbers[0]

    return None


# ============================================================
# データ行解析
# ============================================================

def parse_data_rows(rows, header_index):

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

        print(
            f"{key}: {value}"
        )

    data_rows = []

    # --------------------------------------------------------
    # データ行
    # --------------------------------------------------------

    for row_index in range(
        header_index + 1,
        len(rows)
    ):

        row = rows[row_index]

        if not row:
            continue

        machine_number = None

        # ----------------------------------------------------
        # 台番号列が分かっている場合
        # ----------------------------------------------------

        if "台番号" in columns:

            index = columns["台番号"]

            if index < len(row):

                machine_number = (
                    normalize_machine_number(
                        row[index]
                    )
                )

        # ----------------------------------------------------
        # 分からなければ行全体から探す
        # ----------------------------------------------------

        if machine_number is None:

            machine_number = (
                find_machine_number_in_row(
                    row
                )
            )

        if machine_number is None:
            continue

        record = {
            "日付": TARGET_DATE,
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

        # ----------------------------------------------------
        # 各列
        # ----------------------------------------------------

        for key in record.keys():

            if key == "日付":
                continue

            if key not in columns:
                continue

            index = columns[key]

            if index >= len(row):
                continue

            value = row[index]

            record[key] = value

        data_rows.append(record)

    return data_rows


# ============================================================
# 重複除去
# ============================================================

def remove_duplicates(records):

    unique = {}

    for record in records:

        machine_number = record.get(
            "台番号",
            ""
        )

        if not machine_number:
            continue

        key = (
            TARGET_DATE,
            machine_number
        )

        # 後のデータを優先
        unique[key] = record

    return list(unique.values())


# ============================================================
# CSV保存
# ============================================================

def save_csv(records):

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

    with OUTPUT_CSV.open(
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
    print(OUTPUT_CSV)


# ============================================================
# 結果表示
# ============================================================

def print_records(records):

    print()
    print("======================================================================")
    print("【取得結果】")
    print("======================================================================")

    print()
    print(
        f"取得レコード数: {len(records)}"
    )

    if not records:
        return

    print()

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
# メイン
# ============================================================

async def main():

    print("======================================================================")
    print("アナスロ 7月11日 全台データCSV化")
    print("======================================================================")

    print()
    print("対象日:")
    print(TARGET_DATE)

    print()
    print("方式:")
    print("現在Chromeで開いているアナスロページを")
    print("9222経由で読み取ります。")
    print("PythonからURLは開きません。")

    async with async_playwright() as playwright:

        # ----------------------------------------------------
        # Chrome接続
        # ----------------------------------------------------

        browser = await connect_to_chrome(
            playwright
        )

        if browser is None:
            return

        # ----------------------------------------------------
        # アナスロページ検索
        # ----------------------------------------------------

        page = await find_ana_slo_page(
            browser
        )

        if page is None:

            print()
            print(
                "【ERROR】アナスロページが見つかりません。"
            )

            return

        # ----------------------------------------------------
        # ページ確認
        # ----------------------------------------------------

        body_text = await get_page_information(
            page
        )

        if len(body_text) < 100:

            print()
            print(
                "【ERROR】ページ本文がほとんど取得できません。"
            )

            print()
            print(
                "Chromeでアナスロのデータページを"
            )

            print(
                "完全に表示した状態で再実行してください。"
            )

            await save_debug_html(page)

            return

        # ----------------------------------------------------
        # HTML保存
        # ----------------------------------------------------

        await save_debug_html(page)

        # ----------------------------------------------------
        # table取得
        # ----------------------------------------------------

        print()
        print("======================================================================")
        print("【詳細データテーブル解析】")
        print("======================================================================")

        tables = page.locator(
            "table"
        )

        table_count = await tables.count()

        print()
        print(
            f"ページ内table数: {table_count}"
        )

        if table_count == 0:

            print()
            print(
                "【ERROR】tableがありません。"
            )

            return

        # ----------------------------------------------------
        # 詳細table検索
        # ----------------------------------------------------

        detail_table, score = (
            await find_detail_table(
                tables
            )
        )

        if detail_table is None:

            print()
            print(
                "【ERROR】台データのtableを"
                "特定できませんでした。"
            )

            return

        # ----------------------------------------------------
        # table行取得
        # ----------------------------------------------------

        rows = await get_table_rows(
            detail_table
        )

        print()
        print(
            f"取得した行数: {len(rows)}"
        )

        if not rows:

            print()
            print(
                "【ERROR】tableから行を取得できませんでした。"
            )

            return

        # ----------------------------------------------------
        # ヘッダー検索
        # ----------------------------------------------------

        header_index, header_score = (
            find_header_row(rows)
        )

        print()
        print(
            f"ヘッダー候補行: {header_index}"
        )

        print(
            f"ヘッダースコア: {header_score}"
        )

        if header_index is not None:

            print()
            print(
                "ヘッダー候補:"
            )

            print(
                rows[header_index]
            )

        # ----------------------------------------------------
        # データ解析
        # ----------------------------------------------------

        records = parse_data_rows(
            rows,
            header_index
        )

        print()
        print(
            f"解析レコード数: {len(records)}"
        )

        # ----------------------------------------------------
        # 重複除去
        # ----------------------------------------------------

        records = remove_duplicates(
            records
        )

        print()
        print(
            f"重複除去後: {len(records)}台"
        )

        # ----------------------------------------------------
        # 結果表示
        # ----------------------------------------------------

        print_records(
            records
        )

        # ----------------------------------------------------
        # CSV保存
        # ----------------------------------------------------

        if records:

            save_csv(
                records
            )

        else:

            print()
            print(
                "【WARNING】CSVに保存できる"
                "データがありません。"
            )

        # ----------------------------------------------------
        # 完了
        # ----------------------------------------------------

        print()
        print("======================================================================")
        print("★★★★★ アナスロ 7月11日 データ取得完了 ★★★★★")
        print("======================================================================")

        print()
        print(
            "デバッグHTML:"
        )

        print(
            DEBUG_HTML
        )

        if records:

            print()
            print(
                "CSV:"
            )

            print(
                OUTPUT_CSV
            )


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )