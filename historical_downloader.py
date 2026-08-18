import asyncio
import csv
from datetime import datetime, timedelta
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright


# ============================================================
# 基本設定
# ============================================================

CDP_URL = "http://127.0.0.1:9222"

STORE_NAME = "マルハンメガシティ前橋インター"
DATA_FOLDER = "maruhan_maebashi"

LIST_URL = (
    "https://ana-slo.com/"
    "%E3%83%9B%E3%83%BC%E3%83%AB%E3%83%87%E3%83%BC%E3%82%BF/"
    "%E7%BE%A4%E9%A6%AC%E7%9C%8C/"
    "%E3%83%9E%E3%83%AB%E3%83%8F%E3%83%B3%E3%83%A1%E3%82%AC%E3%82%B7%E3%83%86%E3%82%A3%E5%89%8D%E6%A9%8B%E3%82%A4%E3%83%B3%E3%82%BF%E3%83%BC-"
    "%E3%83%87%E3%83%BC%E3%82%BF%E4%B8%80%E8%A6%A7/"
)


# ============================================================
# CSV検証設定
# ============================================================

EXPECTED_HEADERS = [
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

# 現在のマルハンメガシティ前橋インターは514台。
# 多少の台数変動を考慮して400台以上を正常とする。
MIN_MACHINE_ROWS = 400


# ============================================================
# 台データ抽出
# ============================================================

def extract_machine_data(html):
    """
    all_data_tableから台データを抽出する。

    戻り値:
        headers, rows
    """

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    table = soup.find(
        "table",
        id="all_data_table"
    )

    if table is None:
        return None, []

    # --------------------------------------------------------
    # ヘッダー
    # --------------------------------------------------------

    thead = table.find("thead")

    headers = []

    if thead:

        headers = [
            cell.get_text(
                " ",
                strip=True
            )
            for cell in thead.find_all("th")
        ]

    # --------------------------------------------------------
    # 本体
    # --------------------------------------------------------

    tbody = table.find("tbody")

    if tbody is None:
        return headers, []

    rows = []

    for tr in tbody.find_all(
        "tr",
        recursive=False
    ):

        cells = tr.find_all(
            "td",
            recursive=False
        )

        values = [
            cell.get_text(
                " ",
                strip=True
            )
            for cell in cells
        ]

        if values:
            rows.append(values)

    return headers, rows


# ============================================================
# CSV保存
# ============================================================

def save_csv(
    output_file,
    headers,
    rows
):
    """
    CSVを保存する。
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(headers)

        writer.writerows(rows)


# ============================================================
# CSV検証
# ============================================================

def validate_csv(csv_file):
    """
    既存CSVが正常なデータか検証する。

    戻り値:
        (True, "正常: 514台")
        (False, "異常理由")
    """

    # --------------------------------------------------------
    # ファイル存在確認
    # --------------------------------------------------------

    if not csv_file.exists():

        return (
            False,
            "ファイルが存在しません"
        )

    # --------------------------------------------------------
    # ファイルサイズ確認
    # --------------------------------------------------------

    try:

        file_size = csv_file.stat().st_size

    except Exception as e:

        return (
            False,
            f"ファイル情報取得失敗: {e}"
        )

    if file_size == 0:

        return (
            False,
            "ファイルサイズが0です"
        )

    # --------------------------------------------------------
    # CSV読み込み
    # --------------------------------------------------------

    try:

        with open(
            csv_file,
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            reader = csv.reader(f)

            rows = list(reader)

    except Exception as e:

        return (
            False,
            f"CSV読み込み失敗: {e}"
        )

    # --------------------------------------------------------
    # 行数確認
    # --------------------------------------------------------

    if len(rows) < 2:

        return (
            False,
            "データ行がありません"
        )

    headers = rows[0]

    data_rows = rows[1:]

    # --------------------------------------------------------
    # 列数確認
    # --------------------------------------------------------

    if len(headers) != 9:

        return (
            False,
            f"ヘッダーの列数が9ではありません: "
            f"{len(headers)}列"
        )

    # --------------------------------------------------------
    # ヘッダー内容確認
    # --------------------------------------------------------

    if headers != EXPECTED_HEADERS:

        return (
            False,
            "ヘッダーが想定と一致しません"
        )

    # --------------------------------------------------------
    # データ行数確認
    # --------------------------------------------------------

    if len(data_rows) < MIN_MACHINE_ROWS:

        return (
            False,
            f"台数が少なすぎます: "
            f"{len(data_rows)}台"
        )

    # --------------------------------------------------------
    # 各行の列数確認
    # --------------------------------------------------------

    for index, row in enumerate(
        data_rows,
        start=2
    ):

        if len(row) != 9:

            return (
                False,
                f"{index}行目の列数が9ではありません: "
                f"{len(row)}列"
            )

    # --------------------------------------------------------
    # 台番号確認
    # --------------------------------------------------------

    machine_numbers = []

    for index, row in enumerate(
        data_rows,
        start=2
    ):

        machine_number = row[1].strip()

        if not machine_number:

            return (
                False,
                f"{index}行目の台番号が空です"
            )

        machine_numbers.append(
            machine_number
        )

    # --------------------------------------------------------
    # 台番号重複確認
    # --------------------------------------------------------

    if len(machine_numbers) != len(
        set(machine_numbers)
    ):

        duplicates = []

        seen = set()

        for number in machine_numbers:

            if number in seen:

                if number not in duplicates:

                    duplicates.append(
                        number
                    )

            else:

                seen.add(number)

        return (
            False,
            "台番号が重複しています: "
            + ", ".join(
                duplicates[:10]
            )
        )

    # --------------------------------------------------------
    # G数・差枚確認
    # --------------------------------------------------------

    for index, row in enumerate(
        data_rows,
        start=2
    ):

        games = row[2].strip()
        diff = row[3].strip()

        if not games:

            return (
                False,
                f"{index}行目のG数が空です"
            )

        if not diff:

            return (
                False,
                f"{index}行目の差枚が空です"
            )

    # --------------------------------------------------------
    # 正常
    # --------------------------------------------------------

    return (
        True,
        f"正常: {len(data_rows)}台"
    )


# ============================================================
# all_data.csv 自動統合
# ============================================================

def integrate_all_csvs(data_dir):
    """正常な日別CSVをすべて統合して all_data.csv を作成する。"""

    print()
    print("=" * 70)
    print("日別CSV → all_data.csv 自動統合")
    print("=" * 70)

    csv_files = sorted(data_dir.glob("????-??-??.csv"))

    if not csv_files:
        print("[統合] 統合対象のCSVがありません。")
        return False

    all_rows = []
    integrated_dates = []
    skipped_files = []

    for csv_file in csv_files:
        date_text = csv_file.stem

        try:
            datetime.strptime(date_text, "%Y-%m-%d")
        except ValueError:
            continue

        is_valid, reason = validate_csv(csv_file)

        if not is_valid:
            print(f"[統合除外] {date_text}: {reason}")
            skipped_files.append(date_text)
            continue

        try:
            with open(csv_file, "r", newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                rows = list(reader)
        except Exception as e:
            print(f"[統合除外] {date_text}: CSV読み込み失敗: {e}")
            skipped_files.append(date_text)
            continue

        if not rows or rows[0] != EXPECTED_HEADERS:
            print(f"[統合除外] {date_text}: ヘッダー不一致")
            skipped_files.append(date_text)
            continue

        data_rows = rows[1:]

        for row in data_rows:
            all_rows.append([date_text] + row)

        integrated_dates.append(date_text)
        print(f"  ○ {date_text}: {len(data_rows)}台")

    if not all_rows:
        print("[統合失敗] 統合できる正常CSVがありません。")
        return False

    # 日付＋台番号の重複確認
    seen = set()
    duplicates = []

    for row in all_rows:
        if len(row) >= 3:
            key = (row[0], row[2])
            if key in seen and key not in duplicates:
                duplicates.append(key)
            seen.add(key)

    if duplicates:
        print()
        print(f"[警告] 日付＋台番号の重複: {len(duplicates)}件")
        for date_text, machine_number in duplicates[:10]:
            print(f"  {date_text} / 台番号 {machine_number}")

    output_file = data_dir / "all_data.csv"
    output_headers = ["日付"] + EXPECTED_HEADERS

    try:
        with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(output_headers)
            writer.writerows(all_rows)
    except Exception as e:
        print(f"[統合失敗] all_data.csv 保存失敗: {e}")
        return False

    print()
    print("★ all_data.csv 統合成功")
    print(f"収録日数: {len(integrated_dates)}日")
    print(f"総データ行数: {len(all_rows):,}行")
    print(f"平均台数/日: {len(all_rows) / len(integrated_dates):.1f}台")
    print()
    print("保存ファイル:")
    print(output_file)

    return True



# ============================================================
# 日付一覧作成
# ============================================================

def make_date_list(
    start_date,
    end_date
):

    dates = []

    current = start_date

    while current <= end_date:

        dates.append(
            current
        )

        current += timedelta(
            days=1
        )

    return dates


# ============================================================
# メイン
# ============================================================

async def main():

    print()
    print("=" * 70)
    print("historical_downloader.py")
    print("過去データ期間取得版")
    print("CSV検証機能付き")
    print("=" * 70)

    # ========================================================
    # 日付入力
    # ========================================================

    print()

    start_text = input(
        "取得開始日 (YYYY-MM-DD): "
    ).strip()

    end_text = input(
        "取得終了日 (YYYY-MM-DD): "
    ).strip()

    try:

        start_date = datetime.strptime(
            start_text,
            "%Y-%m-%d"
        ).date()

        end_date = datetime.strptime(
            end_text,
            "%Y-%m-%d"
        ).date()

    except ValueError:

        print()
        print(
            "[エラー] 日付形式が正しくありません。"
        )

        return

    if start_date > end_date:

        print()
        print(
            "[エラー] 開始日が終了日より後になっています。"
        )

        return

    target_dates = make_date_list(
        start_date,
        end_date
    )

    print()
    print(
        f"店舗: {STORE_NAME}"
    )

    print(
        f"開始日: {start_date}"
    )

    print(
        f"終了日: {end_date}"
    )

    print(
        f"対象日数: {len(target_dates)}"
    )

    # ========================================================
    # 保存フォルダ
    # ========================================================

    base_dir = Path(
        __file__
    ).resolve().parent

    data_dir = (
        base_dir
        / "data"
        / DATA_FOLDER
    )

    data_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        f"保存フォルダ: {data_dir}"
    )

    # ========================================================
    # 結果
    # ========================================================

    success_dates = []
    skip_dates = []
    failed_dates = []

    # ========================================================
    # Chrome接続
    # ========================================================

    async with async_playwright() as p:

        print()
        print(
            "9222 Chromeへ接続中..."
        )

        browser = await p.chromium.connect_over_cdp(
            CDP_URL
        )

        print(
            "Chrome接続成功。"
        )

        context = browser.contexts[0]

        # ====================================================
        # 一覧ページ用タブ
        # ====================================================

        page = await context.new_page()

        print()
        print(
            "一覧ページ用の新しいタブを作成しました。"
        )

        print()
        print(
            "一覧ページを開きます。"
        )

        try:

            await page.goto(
                LIST_URL,
                wait_until="domcontentloaded",
                timeout=60000
            )

        except Exception as e:

            print()
            print(
                "[注意] 一覧ページ:"
            )

            print(e)

        await asyncio.sleep(5)

        print()
        print(
            "一覧ページURL:"
        )

        print(
            page.url
        )

        # ====================================================
        # 一覧ページHTML取得
        # ====================================================

        try:

            list_html = await page.content()

        except Exception as e:

            print()
            print(
                "[エラー] 一覧ページHTML取得:"
            )

            print(e)

            return

        print()
        print(
            f"一覧HTML文字数: {len(list_html):,}"
        )

        # ====================================================
        # 日付リンク取得
        # ====================================================

        soup = BeautifulSoup(
            list_html,
            "html.parser"
        )

        date_links = {}

        for a in soup.find_all("a"):

            text = a.get_text(
                " ",
                strip=True
            )

            href = a.get("href")

            if not href:
                continue

            for target_date in target_dates:

                date_str = target_date.strftime(
                    "%Y-%m-%d"
                )

                display_date = (
                    f"{target_date.year}/"
                    f"{target_date.month:02d}/"
                    f"{target_date.day:02d}"
                )

                if (
                    display_date in text
                    and f"{date_str}-" in href
                    and "-data" in href
                ):

                    if date_str not in date_links:

                        date_links[
                            date_str
                        ] = href

        print()
        print(
            f"対象リンク発見数: "
            f"{len(date_links)}"
        )

        # ====================================================
        # リンク状況表示
        # ====================================================

        for target_date in target_dates:

            date_str = target_date.strftime(
                "%Y-%m-%d"
            )

            if date_str in date_links:

                print(
                    f"{date_str} : リンクあり"
                )

            else:

                print(
                    f"{date_str} : リンクなし"
                )

        # ====================================================
        # 日付ごとの処理
        # ====================================================

        for index, target_date in enumerate(
            target_dates,
            start=1
        ):

            date_str = target_date.strftime(
                "%Y-%m-%d"
            )

            print()
            print()
            print("=" * 70)
            print(
                f"[{index}/{len(target_dates)}] "
                f"{date_str}"
            )
            print("=" * 70)

            # ------------------------------------------------
            # 保存ファイル
            # ------------------------------------------------

            output_file = (
                data_dir
                / f"{date_str}.csv"
            )

            # ------------------------------------------------
            # 既存CSV検証
            # ------------------------------------------------

            if output_file.exists():

                print()
                print(
                    "既存CSVがあります。"
                )

                print(
                    output_file
                )

                print()
                print(
                    "CSVを検証します..."
                )

                is_valid, reason = (
                    validate_csv(
                        output_file
                    )
                )

                if is_valid:

                    print(
                        f"★ CSV検証OK: "
                        f"{reason}"
                    )

                    print(
                        "→ 正常な既存データなので"
                        "スキップします。"
                    )

                    skip_dates.append(
                        date_str
                    )

                    continue

                else:

                    print(
                        f"★ CSV検証NG: "
                        f"{reason}"
                    )

                    print(
                        "→ 既存CSVを再取得します。"
                    )

            # ------------------------------------------------
            # 日付リンク確認
            # ------------------------------------------------

            target_href = date_links.get(
                date_str
            )

            if target_href is None:

                print()
                print(
                    "[失敗]"
                )

                print(
                    f"{date_str} のリンクがありません。"
                )

                failed_dates.append(
                    date_str
                )

                continue

            print()
            print(
                "対象URL:"
            )

            print(
                target_href
            )

            # ------------------------------------------------
            # Playwrightリンク
            # ------------------------------------------------

            locator = page.locator(
                f'a[href="{target_href}"]'
            )

            count = await locator.count()

            print()
            print(
                f"該当リンク数: {count}"
            )

            if count == 0:

                print()
                print(
                    "[失敗]"
                )

                print(
                    "Playwright側でリンクを"
                    "特定できませんでした。"
                )

                failed_dates.append(
                    date_str
                )

                continue

            link = locator.first

            # ------------------------------------------------
            # Network Response監視
            # ------------------------------------------------

            captured = []

            async def handle_response(
                response
            ):

                try:

                    url = response.url

                    if "ana-slo.com" not in url:
                        return

                    content_type = (
                        response.headers.get(
                            "content-type",
                            ""
                        )
                    )

                    if (
                        "text/html"
                        not in
                        content_type.lower()
                    ):
                        return

                    body = await response.body()

                    text = body.decode(
                        "utf-8",
                        errors="ignore"
                    )

                    if (
                        "all_data_table"
                        not in
                        text
                    ):
                        return

                    print()
                    print(
                        "★ 本体HTML Response検出"
                    )

                    print(
                        f"Status: "
                        f"{response.status}"
                    )

                    print(
                        f"HTML文字数: "
                        f"{len(text):,}"
                    )

                    captured.append(
                        text
                    )

                except Exception:
                    pass

            page.on(
                "response",
                handle_response
            )

            # ------------------------------------------------
            # JavaScriptクリック
            # ------------------------------------------------

            print()
            print(
                "JavaScriptクリック実行..."
            )

            try:

                await link.evaluate(
                    """
                    element => {
                        element.click();
                    }
                    """
                )

                print(
                    "クリック実行完了。"
                )

            except Exception as e:

                print()
                print(
                    "[失敗]"
                )

                print(e)

                page.remove_listener(
                    "response",
                    handle_response
                )

                failed_dates.append(
                    date_str
                )

                continue

            # ------------------------------------------------
            # 対象ページへの遷移待機
            # ------------------------------------------------

            print()
            print(
                "対象ページへの遷移を待ちます。"
            )

            reached_target = False

            for i in range(30):

                await asyncio.sleep(1)

                current_url = page.url

                print(
                    f"待機中... "
                    f"{i + 1}/30秒"
                )

                if (
                    f"{date_str}-"
                    in current_url
                    and "-data"
                    in current_url
                ):

                    reached_target = True

                    print()
                    print(
                        "★ 対象ページへ遷移しました。"
                    )

                    break

            # ------------------------------------------------
            # 遷移確認
            # ------------------------------------------------

            if not reached_target:

                print()
                print(
                    "[警告]"
                )

                print(
                    f"{date_str} の対象ページへの"
                    "遷移を確認できませんでした。"
                )

            # ------------------------------------------------
            # ページHTML確認
            # ------------------------------------------------

            await asyncio.sleep(2)

            try:

                current_html = (
                    await page.content()
                )

            except Exception:

                current_html = ""

            print()
            print(
                f"現在HTML文字数: "
                f"{len(current_html):,}"
            )

            if "all_data_table" in current_html:

                print(
                    "★ all_data_table確認"
                )

            else:

                print(
                    "現在HTMLには"
                    "all_data_tableがありません。"
                )

            # ------------------------------------------------
            # Network Response待機
            # ------------------------------------------------

            if not captured:

                print()
                print(
                    "クリック時の本体Responseを"
                    "まだ取得していません。"
                )

                print(
                    "リロードして再取得します。"
                )

                try:

                    await page.reload(
                        wait_until="domcontentloaded",
                        timeout=60000
                    )

                except Exception as e:

                    print()
                    print(
                        "[注意] リロード:"
                    )

                    print(e)

                for i in range(30):

                    await asyncio.sleep(1)

                    print(
                        f"Response待機中... "
                        f"{i + 1}/30秒"
                    )

                    if captured:
                        break

            # ------------------------------------------------
            # Response確認
            # ------------------------------------------------

            print()
            print(
                f"本体HTML Response数: "
                f"{len(captured)}"
            )

            if not captured:

                print()
                print(
                    "[失敗]"
                )

                print(
                    f"{date_str} のHTMLを"
                    "取得できませんでした。"
                )

                failed_dates.append(
                    date_str
                )

                page.remove_listener(
                    "response",
                    handle_response
                )

                # --------------------------------------------
                # 一覧ページへ戻る
                # --------------------------------------------

                try:

                    await page.goto(
                        LIST_URL,
                        wait_until="domcontentloaded",
                        timeout=60000
                    )

                    await asyncio.sleep(3)

                except Exception:
                    pass

                continue

            # ------------------------------------------------
            # 最大HTMLを使用
            # ------------------------------------------------

            best_html = max(
                captured,
                key=len
            )

            print()
            print(
                f"使用HTML文字数: "
                f"{len(best_html):,}"
            )

            # ------------------------------------------------
            # 台データ抽出
            # ------------------------------------------------

            headers, rows = (
                extract_machine_data(
                    best_html
                )
            )

            print()
            print(
                f"列数: "
                f"{len(headers) if headers else 0}"
            )

            print(
                f"取得台数: "
                f"{len(rows)}"
            )

            # ------------------------------------------------
            # 台データ取得失敗
            # ------------------------------------------------

            if not rows:

                print()
                print(
                    "[失敗]"
                )

                print(
                    "台データを抽出できませんでした。"
                )

                failed_dates.append(
                    date_str
                )

                page.remove_listener(
                    "response",
                    handle_response
                )

                try:

                    await page.goto(
                        LIST_URL,
                        wait_until="domcontentloaded",
                        timeout=60000
                    )

                    await asyncio.sleep(3)

                except Exception:
                    pass

                continue

            # ------------------------------------------------
            # 抽出データの簡易検証
            # ------------------------------------------------

            if len(headers) != 9:

                print()
                print(
                    "[失敗]"
                )

                print(
                    f"列数が9ではありません: "
                    f"{len(headers)}"
                )

                failed_dates.append(
                    date_str
                )

                page.remove_listener(
                    "response",
                    handle_response
                )

                try:

                    await page.goto(
                        LIST_URL,
                        wait_until="domcontentloaded",
                        timeout=60000
                    )

                    await asyncio.sleep(3)

                except Exception:
                    pass

                continue

            if len(rows) < MIN_MACHINE_ROWS:

                print()
                print(
                    "[失敗]"
                )

                print(
                    f"台数が少なすぎます: "
                    f"{len(rows)}台"
                )

                failed_dates.append(
                    date_str
                )

                page.remove_listener(
                    "response",
                    handle_response
                )

                try:

                    await page.goto(
                        LIST_URL,
                        wait_until="domcontentloaded",
                        timeout=60000
                    )

                    await asyncio.sleep(3)

                except Exception:
                    pass

                continue

            # ------------------------------------------------
            # CSV保存
            # ------------------------------------------------

            save_csv(
                output_file,
                headers,
                rows
            )

            print()
            print(
                "★ CSV保存成功"
            )

            print(
                output_file
            )

            print(
                f"取得台数: "
                f"{len(rows)}"
            )

            success_dates.append(
                date_str
            )

            # ------------------------------------------------
            # Response監視解除
            # ------------------------------------------------

            page.remove_listener(
                "response",
                handle_response
            )

            # ------------------------------------------------
            # 一覧ページへ戻る
            # ------------------------------------------------

            print()
            print(
                "次の日のため一覧ページへ戻ります。"
            )

            try:

                await page.goto(
                    LIST_URL,
                    wait_until="domcontentloaded",
                    timeout=60000
                )

                await asyncio.sleep(3)

            except Exception as e:

                print()
                print(
                    "[注意] 一覧ページ復帰:"
                )

                print(e)

        # ====================================================
        # 日別CSVの自動統合
        # ====================================================

        integrate_all_csvs(
            data_dir
        )

        # ====================================================
        # 最終結果
        # ====================================================

        print()
        print()
        print("=" * 70)
        print("期間取得完了")
        print("=" * 70)

        print()
        print(
            f"店舗: {STORE_NAME}"
        )

        print(
            f"期間: "
            f"{start_date} ～ {end_date}"
        )

        print()

        # ----------------------------------------------------
        # 成功
        # ----------------------------------------------------

        print(
            f"取得成功: "
            f"{len(success_dates)}日"
        )

        for date_str in success_dates:

            print(
                f"  ○ {date_str}"
            )

        print()

        # ----------------------------------------------------
        # スキップ
        # ----------------------------------------------------

        print(
            f"既存スキップ: "
            f"{len(skip_dates)}日"
        )

        for date_str in skip_dates:

            print(
                f"  → {date_str}"
            )

        print()

        # ----------------------------------------------------
        # 失敗
        # ----------------------------------------------------

        print(
            f"失敗: "
            f"{len(failed_dates)}日"
        )

        for date_str in failed_dates:

            print(
                f"  × {date_str}"
            )

        print()

        print(
            "保存フォルダ:"
        )

        print(
            data_dir
        )

        print()
        print(
            "Chromeはそのままで構いません。"
        )


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )