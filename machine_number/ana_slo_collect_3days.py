import asyncio
import csv
import re
from pathlib import Path

from playwright.async_api import async_playwright


# ============================================================
# 設定
# ============================================================

CDP_URL = "http://127.0.0.1:9222"

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
)

# 今回収集する日付
TARGET_DATES = [
    "2026-07-11",
    
]


# ============================================================
# 共通関数
# ============================================================

def clean_text(value):
    if value is None:
        return ""

    value = str(value)

    value = value.replace("\xa0", " ")
    value = value.replace("\u3000", " ")

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_machine_number(value):
    text = clean_text(value)

    match = re.search(r"\d{3,5}", text)

    if not match:
        return ""

    return match.group(0)


def normalize_diff(value):
    text = clean_text(value)

    if not text:
        return ""

    text = text.replace(",", "")

    match = re.search(r"[+-]?\d+", text)

    if not match:
        return ""

    return match.group(0)


def normalize_number(value):
    text = clean_text(value)

    if not text:
        return ""

    text = text.replace(",", "")

    match = re.search(r"\d+", text)

    if not match:
        return ""

    return match.group(0)


# ============================================================
# ページスコア
# ============================================================

async def score_page(page):

    try:
        title = await page.title()
    except Exception:
        title = ""

    try:
        url = page.url
    except Exception:
        url = ""

    try:
        body = await page.locator("body").inner_text(
            timeout=5000
        )
    except Exception:
        body = ""

    score = 0

    if "アナスロ" in title:
        score += 50

    if "ana-slo.com" in url.lower():
        score += 50

    if "台番号" in body:
        score += 30

    if "機種名" in body:
        score += 20

    if "差枚" in body:
        score += 20

    if "G数" in body:
        score += 10

    if len(body) > 10000:
        score += 20

    return score, title, url, body


# ============================================================
# アナスロページ検索
# ============================================================

async def find_target_page(browser):

    best_page = None
    best_score = -1

    print()
    print("=" * 70)
    print("【現在のChromeタブ】")
    print("=" * 70)

    tab_index = 0

    for context in browser.contexts:

        for page in context.pages:

            try:

                score, title, url, body = (
                    await score_page(page)
                )

                print()
                print(
                    f"タブ[{tab_index}] "
                    f"score={score} "
                    f"本文={len(body)}文字"
                )

                print(f"タイトル: {title}")
                print(f"URL: {url}")

                if score > best_score:

                    best_score = score
                    best_page = page

                tab_index += 1

            except Exception as e:

                print(
                    f"タブ解析エラー: {e}"
                )

    if best_page is None:

        print()
        print(
            "【ERROR】利用可能なChromeタブがありません。"
        )

        return None

    if best_score < 80:

        print()
        print(
            "【ERROR】アナスロのデータページを"
            "確認できませんでした。"
        )

        return None

    print()
    print(
        "★ 内容のあるアナスロページを選択しました"
    )

    print(
        f"score: {best_score}"
    )

    return best_page


# ============================================================
# 詳細データtable検索
# ============================================================

async def find_detail_table(page):

    tables = page.locator("table")

    table_count = await tables.count()

    print()
    print("=" * 70)
    print("【詳細データテーブル解析】")
    print("=" * 70)

    print()
    print(
        f"ページ内table数: {table_count}"
    )

    print()
    print(
        f"table解析開始: {table_count}個"
    )

    best_index = None
    best_score = -1
    best_rows = 0
    best_text_length = 0

    for index in range(table_count):

        try:

            table = tables.nth(index)

            text = await table.inner_text(
                timeout=3000
            )

            text_clean = clean_text(text)

            rows = table.locator("tr")

            row_count = await rows.count()

            if row_count < 2:
                continue

            score = 0

            if "機種名" in text_clean:
                score += 80

            if "台番号" in text_clean:
                score += 80

            if "差枚" in text_clean:
                score += 60

            if "G数" in text_clean:
                score += 40

            if "BB" in text_clean:
                score += 20

            if "RB" in text_clean:
                score += 20

            if row_count >= 100:
                score += 30

            if row_count >= 300:
                score += 20

            if len(text_clean) > 10000:
                score += 20

            if score > best_score:

                best_score = score
                best_index = index
                best_rows = row_count
                best_text_length = len(text_clean)

                print(
                    f"候補更新: table[{index}] "
                    f"score={score} "
                    f"rows={row_count} "
                    f"text={len(text_clean)}文字"
                )

        except Exception:
            continue

    if best_index is None:

        print()
        print(
            "【ERROR】詳細データtableが"
            "見つかりませんでした。"
        )

        return None

    print()
    print(
        f"★ 詳細データtable候補: "
        f"table[{best_index}]"
    )

    print(
        f"★ スコア: {best_score}"
    )

    print(
        f"★ 本文文字数: {best_text_length}"
    )

    return tables.nth(best_index)


# ============================================================
# table解析
# ============================================================

async def parse_detail_table(
    table,
    target_date
):

    rows = table.locator("tr")

    row_count = await rows.count()

    print()
    print(
        f"対象table行数: {row_count}"
    )

    raw_rows = []

    # --------------------------------------------------------
    # ここが重要
    # row = rows.nth(row_index)
    # --------------------------------------------------------

    for row_index in range(row_count):

        try:

            row = rows.nth(row_index)

            cells = row.locator("th, td")

            cell_count = await cells.count()

            if cell_count == 0:
                continue

            values = []

            for cell_index in range(cell_count):

                try:

                    value = await cells.nth(
                        cell_index
                    ).inner_text(
                        timeout=2000
                    )

                except Exception:

                    value = ""

                values.append(
                    clean_text(value)
                )

            if values:
                raw_rows.append(values)

        except Exception:

            continue

    print()
    print(
        f"取得した行数: {len(raw_rows)}"
    )

    if not raw_rows:

        print()
        print(
            "【ERROR】tableから行データを"
            "取得できませんでした。"
        )

        return []

    # ========================================================
    # ヘッダー検索
    # ========================================================

    header_index = None
    header_score = -1

    header_keywords = [
        "機種名",
        "台番号",
        "G数",
        "差枚",
        "BB",
        "RB",
        "合成確率",
        "BB確率",
        "RB確率",
    ]

    for index, row in enumerate(
        raw_rows[:30]
    ):

        joined = " ".join(row)

        score = 0

        for keyword in header_keywords:

            if keyword in joined:
                score += 20

        if "機種名" in joined:
            score += 30

        if "台番号" in joined:
            score += 30

        if "差枚" in joined:
            score += 30

        if score > header_score:

            header_score = score
            header_index = index

    if header_index is None:

        print()
        print(
            "【ERROR】ヘッダーを"
            "特定できませんでした。"
        )

        return []

    header = raw_rows[header_index]

    print()
    print(
        f"ヘッダー候補行: {header_index}"
    )

    print(
        f"ヘッダースコア: {header_score}"
    )

    print()
    print("ヘッダー候補:")
    print(header)

    # ========================================================
    # 列位置
    # ========================================================

    def find_column(names):

        for name in names:

            for index, value in enumerate(
                header
            ):

                if value == name:
                    return index

        return None

    machine_col = find_column(
        ["機種名"]
    )

    machine_number_col = find_column(
        ["台番号"]
    )

    game_col = find_column(
        ["G数"]
    )

    diff_col = find_column(
        ["差枚"]
    )

    bb_col = find_column(
        ["BB"]
    )

    rb_col = find_column(
        ["RB"]
    )

    total_prob_col = find_column(
        ["合成確率"]
    )

    bb_prob_col = find_column(
        ["BB確率"]
    )

    rb_prob_col = find_column(
        ["RB確率"]
    )

    print()
    print("=" * 70)
    print("【列解析】")
    print("=" * 70)

    print()
    print("ヘッダー:")
    print(header)

    print()
    print("検出列:")

    print(
        f"機種名: {machine_col}"
    )

    print(
        f"台番号: {machine_number_col}"
    )

    print(
        f"G数: {game_col}"
    )

    print(
        f"差枚: {diff_col}"
    )

    print(
        f"BB: {bb_col}"
    )

    print(
        f"RB: {rb_col}"
    )

    print(
        f"合成確率: {total_prob_col}"
    )

    print(
        f"BB確率: {bb_prob_col}"
    )

    print(
        f"RB確率: {rb_prob_col}"
    )

    if machine_col is None:

        print()
        print(
            "【ERROR】機種名列を"
            "検出できませんでした。"
        )

        return []

    if machine_number_col is None:

        print()
        print(
            "【ERROR】台番号列を"
            "検出できませんでした。"
        )

        return []

    if diff_col is None:

        print()
        print(
            "【ERROR】差枚列を"
            "検出できませんでした。"
        )

        return []

    # ========================================================
    # レコード作成
    # ========================================================

    records = []

    for row in raw_rows[
        header_index + 1:
    ]:

        if not row:
            continue

        required_columns = [
            machine_col,
            machine_number_col,
            diff_col,
        ]

        max_index = max(
            required_columns
        )

        if len(row) <= max_index:
            continue

        machine_name = clean_text(
            row[machine_col]
        )

        machine_number = (
            normalize_machine_number(
                row[machine_number_col]
            )
        )

        diff = normalize_diff(
            row[diff_col]
        )

        if not machine_name:
            continue

        if not machine_number:
            continue

        if not diff:
            continue

        record = {
            "日付": target_date,
            "台番号": machine_number,
            "機種名": machine_name,
            "G数": "",
            "差枚": diff,
            "BB": "",
            "RB": "",
            "合成確率": "",
            "BB確率": "",
            "RB確率": "",
        }

        if (
            game_col is not None
            and len(row) > game_col
        ):

            record["G数"] = (
                normalize_number(
                    row[game_col]
                )
            )

        if (
            bb_col is not None
            and len(row) > bb_col
        ):

            record["BB"] = (
                normalize_number(
                    row[bb_col]
                )
            )

        if (
            rb_col is not None
            and len(row) > rb_col
        ):

            record["RB"] = (
                normalize_number(
                    row[rb_col]
                )
            )

        if (
            total_prob_col is not None
            and len(row) > total_prob_col
        ):

            record["合成確率"] = clean_text(
                row[total_prob_col]
            )

        if (
            bb_prob_col is not None
            and len(row) > bb_prob_col
        ):

            record["BB確率"] = clean_text(
                row[bb_prob_col]
            )

        if (
            rb_prob_col is not None
            and len(row) > rb_prob_col
        ):

            record["RB確率"] = clean_text(
                row[rb_prob_col]
            )

        records.append(record)

    print()
    print(
        f"解析レコード数: {len(records)}"
    )

    # ========================================================
    # 重複除去
    # ========================================================

    unique = {}

    for record in records:

        key = (
            record["日付"],
            record["台番号"],
        )

        unique[key] = record

    records = list(
        unique.values()
    )

    records.sort(
        key=lambda x: int(
            x["台番号"]
        )
    )

    print()
    print(
        f"重複除去後: {len(records)}台"
    )

    return records


# ============================================================
# CSV保存
# ============================================================

def save_csv(
    records,
    target_date
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    date_text = target_date.replace(
        "-",
        ""
    )

    output_path = (
        OUTPUT_DIR
        / f"ana_slo_{date_text}.csv"
    )

    fieldnames = [
        "日付",
        "台番号",
        "機種名",
        "G数",
        "差枚",
        "BB",
        "RB",
        "合成確率",
        "BB確率",
        "RB確率",
    ]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(records)

    return output_path


# ============================================================
# 1日分収集
# ============================================================

async def collect_one_day(
    browser,
    target_date
):

    print()
    print()
    print("#" * 70)
    print(
        f"【収集開始】{target_date}"
    )
    print("#" * 70)

    print()
    print(
        "Chromeで対象日のアナスロページを"
        "開いてください。"
    )

    print()
    print(
        f"対象日: {target_date}"
    )

    input(
        "\nChromeでこの日のページを表示したら、"
        "Enterを押してください..."
    )

    page = await find_target_page(
        browser
    )

    if page is None:
        return False

    print()
    print("=" * 70)
    print("【ページ確認】")
    print("=" * 70)

    try:

        title = await page.title()

    except Exception:

        title = ""

    print()
    print(
        f"タイトル: {title}"
    )

    print(
        f"URL: {page.url}"
    )

    try:

        body = await page.locator(
            "body"
        ).inner_text(
            timeout=5000
        )

    except Exception:

        body = ""

    print()
    print(
        f"本文文字数: {len(body)}"
    )

    if len(body) < 10000:

        print()
        print(
            "【WARNING】本文が短すぎます。"
        )

        print(
            "アナスロの詳細データが"
            "表示されているか確認してください。"
        )

        return False

    table = await find_detail_table(
        page
    )

    if table is None:
        return False

    records = await parse_detail_table(
        table,
        target_date
    )

    if not records:
        return False

    print()
    print("=" * 70)
    print("【取得結果】")
    print("=" * 70)

    print()
    print(
        f"取得レコード数: {len(records)}"
    )

    for index, record in enumerate(
        records[:30],
        start=1
    ):

        print(
            f"{index:3d}. "
            f"台番号={record['台番号']} "
            f"機種={record['機種名']} "
            f"差枚={record['差枚']} "
            f"G数={record['G数']}"
        )

    if len(records) > 30:

        print()
        print(
            f"... 以下 "
            f"{len(records) - 30}台"
        )

    output_path = save_csv(
        records,
        target_date
    )

    print()
    print(
        "★ CSV保存成功"
    )

    print(
        output_path
    )

    return True


# ============================================================
# メイン
# ============================================================

async def main():

    print("=" * 70)
    print("アナスロ 複数日データ収集")
    print("=" * 70)

    print()
    print(
        "方式:"
    )

    print(
        "現在Chromeで開いている"
        "アナスロページを"
    )

    print(
        "9222経由で読み取ります。"
    )

    print(
        "PythonからURLは開きません。"
    )

    print()
    print("対象日:")

    for date in TARGET_DATES:

        print(
            f"  {date}"
        )

    print()
    print(
        "Chrome 9222へ接続します..."
    )

    async with async_playwright() as p:

        try:

            browser = (
                await p.chromium.connect_over_cdp(
                    CDP_URL
                )
            )

        except Exception as e:

            print()
            print(
                "【ERROR】Chromeへの接続に"
                "失敗しました。"
            )

            print()
            print(
                str(e)
            )

            print()
            print(
                "Chromeが"
                "--remote-debugging-port=9222"
                "で起動しているか確認してください。"
            )

            return

        print(
            "★ Chrome接続成功"
        )

        print()
        print(
            f"ブラウザコンテキスト数: "
            f"{len(browser.contexts)}"
        )

        success_dates = []
        failed_dates = []

        for target_date in TARGET_DATES:

            try:

                success = (
                    await collect_one_day(
                        browser,
                        target_date
                    )
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
                    f"【ERROR】{target_date} "
                    "処理中にエラーが発生しました。"
                )

                print(
                    f"{type(e).__name__}: {e}"
                )

                failed_dates.append(
                    target_date
                )

        print()
        print("=" * 70)
        print("【収集結果】")
        print("=" * 70)

        print()
        print(
            f"成功: {len(success_dates)}日"
        )

        for date in success_dates:

            print(
                f"  ★ {date}"
            )

        print()
        print(
            f"失敗: {len(failed_dates)}日"
        )

        for date in failed_dates:

            print(
                f"  × {date}"
            )

        print()
        print("=" * 70)
        print(
            "★★★★★ 複数日収集完了 ★★★★★"
        )
        print("=" * 70)

        print()
        print(
            "保存先:"
        )

        print(
            OUTPUT_DIR
        )


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )