import asyncio
import re
from pathlib import Path
from datetime import datetime
import pandas as pd

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError


# ============================================================
# 設定
# ============================================================

BASE_URL = (
    "https://ana-slo.com/"
    "%E3%83%9B%E3%83%BC%E3%83%AB%E3%83%87%E3%83%BC%E3%82%BF/"
    "%E7%BE%A4%E9%A6%AC%E7%9C%8C/"
    "%E3%83%9E%E3%83%AB%E3%83%8F%E3%83%B3%E3%83%A1%E3%82%AC%E3%82%B7%E3%83%86%E3%82%A3%E5%89%8D%E6%A9%8B%E3%82%A4%E3%83%B3%E3%82%BF%E3%83%BC-"
    "%E3%83%87%E3%83%BC%E3%82%BF%E4%B8%80%E8%A6%A7/"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
)

OUTPUT_FILE = OUTPUT_DIR / "ana_slo_2026_07.csv"
LOG_FILE = OUTPUT_DIR / "ana_slo_2026_07_log.csv"

TARGET_START = datetime(2026, 7, 1)
TARGET_END = datetime(2026, 7, 31)

HEADLESS = False

WAIT_AFTER_PAGE = 1.0


# ============================================================
# 共通関数
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    value = str(value)
    value = value.replace("\u3000", " ")
    value = value.replace("\xa0", " ")
    return value.strip()


def normalize_machine_number(value):
    text = normalize_text(value)

    # カンマ・小数点などを除去
    text = text.replace(",", "")

    match = re.search(r"\d+", text)

    if not match:
        return None

    return int(match.group())


def normalize_medals(value):
    """
    差枚を整数化する。

    例:
    +1,200
    -500
    1,200枚
    0
    """
    text = normalize_text(value)

    if not text:
        return None

    text = text.replace(",", "")
    text = text.replace("枚", "")
    text = text.replace("+", "")

    # 数字を含むものだけ処理
    match = re.search(r"-?\d+", text)

    if not match:
        return None

    try:
        return int(match.group())
    except ValueError:
        return None


def extract_date_from_text(text):
    """
    ページ内から YYYY/MM/DD または YYYY-MM-DD を探す。
    """

    text = normalize_text(text)

    patterns = [
        r"(2026)[/年-](07)[/月-](\d{1,2})",
        r"(2026)[/年-](\d{1,2})[/月-](\d{1,2})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            try:
                year = int(match.group(1))
                month = int(match.group(2))
                day = int(match.group(3))

                return datetime(year, month, day)

            except ValueError:
                pass

    return None


def is_target_date(date_value):
    if date_value is None:
        return False

    return TARGET_START <= date_value <= TARGET_END


# ============================================================
# 日付一覧から7月リンクを取得
# ============================================================

async def collect_date_links(page):

    print()
    print("=" * 70)
    print("【日付一覧ページから7月データリンクを取得】")
    print("=" * 70)

    print(f"URL:")
    print(BASE_URL)

    await page.goto(
        BASE_URL,
        wait_until="domcontentloaded",
        timeout=60000,
    )

    await page.wait_for_timeout(2000)

    links = await page.locator("a").all()

    results = []

    for link in links:

        try:
            text = normalize_text(await link.inner_text())
            href = await link.get_attribute("href")

        except Exception:
            continue

        if not href:
            continue

        # 2026/07/xx または 2026-07-xx を検索
        date_value = extract_date_from_text(text)

        if date_value is None:
            date_value = extract_date_from_text(href)

        if not is_target_date(date_value):
            continue

        if href.startswith("/"):
            href = "https://ana-slo.com" + href

        elif href.startswith("./"):
            href = BASE_URL.rstrip("/") + "/" + href[2:]

        elif not href.startswith("http"):
            continue

        item = {
            "date": date_value.strftime("%Y-%m-%d"),
            "url": href,
        }

        if item not in results:
            results.append(item)

    # 日付順
    results.sort(key=lambda x: x["date"])

    print()
    print(f"7月対象リンク数: {len(results)}")

    for item in results:
        print(
            f"{item['date']}  {item['url']}"
        )

    return results


# ============================================================
# 個別ページから台データ取得
# ============================================================

async def collect_daily_data(context, item):

    target_date = item["date"]
    url = item["url"]

    print()
    print("-" * 70)
    print(f"【取得】{target_date}")
    print("-" * 70)
    print(url)

    page = await context.new_page()

    rows = []

    try:

        await page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=60000,
        )

        await page.wait_for_timeout(
            int(WAIT_AFTER_PAGE * 1000)
        )

        # ----------------------------------------------------
        # 「全データ一覧」周辺を探す
        # ----------------------------------------------------

        body_text = normalize_text(
            await page.locator("body").inner_text()
        )

        print(
            f"ページタイトル確認: "
            f"{body_text[:120].replace(chr(10), ' ')}"
        )

        # ----------------------------------------------------
        # ページ内のテーブルを調査
        # ----------------------------------------------------

        tables = page.locator("table")

        table_count = await tables.count()

        print(f"テーブル数: {table_count}")

        best_table = None
        best_score = -1

        for i in range(table_count):

            table = tables.nth(i)

            try:
                table_text = normalize_text(
                    await table.inner_text()
                )
            except Exception:
                continue

            if not table_text:
                continue

            score = 0

            # 台番号
            if "台番号" in table_text:
                score += 10

            # 機種
            if "機種" in table_text:
                score += 10

            # 差枚
            if "差枚" in table_text:
                score += 10

            # G数
            if "G数" in table_text:
                score += 3

            # 勝率
            if "勝率" in table_text:
                score += 3

            # 514台前後ある場合はかなり有力
            row_count = await table.locator("tr").count()

            if row_count >= 100:
                score += 20

            if score > best_score:
                best_score = score
                best_table = table

        if best_table is None:

            print("対象テーブルを発見できませんでした。")

            # HTML保存
            debug_file = (
                OUTPUT_DIR
                / f"debug_{target_date}.html"
            )

            try:
                html = await page.content()
                debug_file.write_text(
                    html,
                    encoding="utf-8"
                )

                print(
                    f"デバッグHTML保存: {debug_file}"
                )

            except Exception:
                pass

            return rows, "TABLE_NOT_FOUND"

        print(
            f"対象テーブル検出: score={best_score}"
        )

        # ----------------------------------------------------
        # 行取得
        # ----------------------------------------------------

        trs = best_table.locator("tr")

        tr_count = await trs.count()

        print(f"対象テーブル行数: {tr_count}")

        for i in range(tr_count):

            tr = trs.nth(i)

            try:
                cells = tr.locator("th, td")

                cell_count = await cells.count()

                if cell_count == 0:
                    continue

                values = []

                for j in range(cell_count):

                    value = normalize_text(
                        await cells.nth(j).inner_text()
                    )

                    values.append(value)

            except Exception:
                continue

            row_text = " ".join(values)

            # ------------------------------------------------
            # 台番号を探す
            # ------------------------------------------------

            machine_number = None

            for value in values:

                number = normalize_machine_number(value)

                if number is None:
                    continue

                # マルハン前橋インターの台番号として
                # 現実的な範囲を設定
                if 500 <= number <= 1200:
                    machine_number = number
                    break

            if machine_number is None:
                continue

            # ------------------------------------------------
            # 機種名を推定
            # ------------------------------------------------

            machine_name = ""

            for value in values:

                if value == "":
                    continue

                # 台番号そのものは除外
                if normalize_machine_number(value) == machine_number:
                    continue

                # 差枚だけのセルを除外
                if re.fullmatch(
                    r"[+\-]?\d[\d,]*枚?",
                    value
                ):
                    continue

                # G数だけのセルを除外
                if re.fullmatch(
                    r"\d[\d,]*G?",
                    value
                ):
                    continue

                # 勝率表記を除外
                if "%" in value:
                    continue

                # 明らかに短い数値だけなら除外
                if re.fullmatch(
                    r"[+\-]?\d[\d,]*",
                    value
                ):
                    continue

                # ある程度文字があるものを機種名候補に
                if len(value) >= 2:
                    machine_name = value
                    break

            if not machine_name:
                continue

            # ------------------------------------------------
            # 差枚を探す
            # ------------------------------------------------

            medals = None

            for value in values:

                if not re.search(r"\d", value):
                    continue

                # + / - を含むものを優先
                if value.startswith("+") or value.startswith("-"):

                    candidate = normalize_medals(value)

                    if candidate is not None:
                        medals = candidate
                        break

            # + / - が無い場合
            if medals is None:

                for value in values:

                    if value == str(machine_number):
                        continue

                    candidate = normalize_medals(value)

                    if candidate is None:
                        continue

                    # 機種名ではない
                    if any(
                        ch.isalpha()
                        for ch in value
                    ):
                        continue

                    medals = candidate
                    break

            if medals is None:
                continue

            rows.append(
                {
                    "日付": target_date,
                    "台番号": machine_number,
                    "機種名": machine_name,
                    "差枚": medals,
                }
            )

        # ----------------------------------------------------
        # 重複除去
        # ----------------------------------------------------

        unique = {}

        for row in rows:

            key = (
                row["日付"],
                row["台番号"],
            )

            unique[key] = row

        rows = list(unique.values())

        print(
            f"取得台数: {len(rows)}"
        )

        if len(rows) > 0:

            print(
                "サンプル:"
            )

            for sample in rows[:5]:

                print(
                    f"  "
                    f"{sample['台番号']} / "
                    f"{sample['機種名']} / "
                    f"{sample['差枚']:+d}"
                )

            return rows, "OK"

        return rows, "NO_DATA"

    except PlaywrightTimeoutError:

        print(
            "タイムアウトしました。"
        )

        return rows, "TIMEOUT"

    except Exception as e:

        print(
            f"エラー: {type(e).__name__}: {e}"
        )

        return rows, "ERROR"

    finally:

        await page.close()


# ============================================================
# CSV保存
# ============================================================

def save_rows(rows):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    columns = [
        "日付",
        "台番号",
        "機種名",
        "差枚",
    ]

    if rows:

        df = pd.DataFrame(rows)

        for column in columns:
            if column not in df.columns:
                df[column] = None

        df = df[columns]

        df["日付"] = pd.to_datetime(
            df["日付"],
            errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        df["台番号"] = pd.to_numeric(
            df["台番号"],
            errors="coerce"
        )

        df["差枚"] = pd.to_numeric(
            df["差枚"],
            errors="coerce"
        )

        df = df.dropna(
            subset=[
                "日付",
                "台番号",
                "機種名",
                "差枚",
            ]
        )

        df["台番号"] = df["台番号"].astype(int)
        df["差枚"] = df["差枚"].astype(int)

        df = df.drop_duplicates(
            subset=[
                "日付",
                "台番号",
            ]
        )

        df = df.sort_values(
            [
                "日付",
                "台番号",
            ]
        )

    else:

        df = pd.DataFrame(
            columns=columns
        )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        f"★ CSV保存: {OUTPUT_FILE}"
    )

    print(
        f"総行数: {len(df):,}"
    )

    if len(df) > 0:

        print(
            f"日数: {df['日付'].nunique()}日"
        )

        print(
            f"最古: {df['日付'].min()}"
        )

        print(
            f"最新: {df['日付'].max()}"
        )


def save_log(log_rows):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame(log_rows)

    df.to_csv(
        LOG_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"★ ログ保存: {LOG_FILE}"
    )


# ============================================================
# メイン
# ============================================================

async def main():

    print("=" * 70)
    print("アナスロ 2026年7月データ収集")
    print("=" * 70)

    print()
    print(
        "対象期間: "
        f"{TARGET_START.strftime('%Y-%m-%d')} ～ "
        f"{TARGET_END.strftime('%Y-%m-%d')}"
    )

    print()
    print(
        "既存 all_data.csv は変更しません。"
    )

    all_rows = []
    log_rows = []

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=HEADLESS
        )

        context = await browser.new_context(
            viewport={
                "width": 1440,
                "height": 1000,
            },
            locale="ja-JP",
        )

        page = await context.new_page()

        # ----------------------------------------------------
        # 日付リンク取得
        # ----------------------------------------------------

        date_links = await collect_date_links(page)

        if not date_links:

            print()
            print(
                "7月の日付リンクを取得できませんでした。"
            )

            await browser.close()
            return

        # ----------------------------------------------------
        # 各日取得
        # ----------------------------------------------------

        total = len(date_links)

        for index, item in enumerate(
            date_links,
            start=1
        ):

            print()
            print(
                f"[{index}/{total}] "
                f"{item['date']}"
            )

            rows, status = await collect_daily_data(
                context,
                item
            )

            all_rows.extend(rows)

            log_rows.append(
                {
                    "日付": item["date"],
                    "URL": item["url"],
                    "ステータス": status,
                    "取得台数": len(rows),
                }
            )

            # 毎回保存
            save_rows(all_rows)
            save_log(log_rows)

            await asyncio.sleep(1)

        await browser.close()

    # --------------------------------------------------------
    # 最終結果
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("【7月データ収集完了】")
    print("=" * 70)

    print(
        f"対象日数: {len(date_links)}"
    )

    print(
        f"取得総行数: {len(all_rows):,}"
    )

    ok_days = sum(
        1
        for x in log_rows
        if x["ステータス"] == "OK"
    )

    print(
        f"正常取得日数: {ok_days}"
    )

    print()
    print(
        f"保存先:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        LOG_FILE
    )

    print()
    print(
        "※ all_data.csv は変更していません。"
    )


if __name__ == "__main__":
    asyncio.run(main())