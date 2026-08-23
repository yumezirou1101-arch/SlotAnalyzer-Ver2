import asyncio
import csv
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import unquote

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright


# ============================================================
# 基本設定
# ============================================================

# multi_day_scraper.py があるフォルダ
# ＝ SlotAnalyzer フォルダ
BASE_DIR = Path(__file__).resolve().parent

HTML_FILE = BASE_DIR / "browser_html.html"
SLOT_DATA_FILE = BASE_DIR / "slot_data.csv"
EXTRACT_SCRIPT = BASE_DIR / "extract_machine_data.py"

CHROME_CDP_URL = "http://127.0.0.1:9222"


# ============================================================
# 店舗設定
# ============================================================

STORES = {
    1: {
        "name": "マルハンメガシティ前橋インター",
        "url_name": "マルハンメガシティ前橋インター",
        "folder": BASE_DIR / "data" / "maruhan_maebashi",
        "keywords": [
            "マルハンメガシティ前橋インター",
        ],
    },

    2: {
        "name": "ビックマーチ高崎大八木店",
        "url_name": "ビックマーチ高崎おおやぎ店",
        "folder": BASE_DIR / "data" / "bigmarch_takasaki_oyagi",
        "keywords": [
            "ビックマーチ高崎おおやぎ店",
            "ビックマーチ高崎大八木店",
        ],
    },

    3: {
        "name": "ビックつばめ高崎店",
        "url_name": "ビックつばめ高崎店",
        "folder": BASE_DIR / "data" / "bigtubame_takasaki",
        "keywords": [
            "ビックつばめ高崎店",
        ],
    },

    4: {
        "name": "やすだ前橋店",
        "url_name": "やすだ前橋店",
        "folder": BASE_DIR / "data" / "yasuda_maebashi",
        "keywords": [
            "やすだ前橋店",
        ],
    },
}


# ============================================================
# URL作成
# ============================================================

def make_url(store, date_str):

    return (
        f"https://ana-slo.com/"
        f"{date_str}-{store['url_name']}-data/"
    )


# ============================================================
# URL正規化
# ============================================================

def normalize_url(url):

    if not url:
        return ""

    # URLエンコードを戻す
    url = unquote(url)

    # 末尾スラッシュを統一
    url = url.rstrip("/")

    return url


# ============================================================
# URL比較
# ============================================================

def urls_match(url1, url2):

    return (
        normalize_url(url1)
        == normalize_url(url2)
    )


# ============================================================
# 日付確認
# ============================================================

def is_valid_date(date_str):

    try:

        datetime.strptime(
            date_str,
            "%Y-%m-%d"
        )

        return True

    except ValueError:

        return False


# ============================================================
# タイトル取得
# ============================================================

def get_title(html):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    title = soup.find("title")

    if title is None:
        return ""

    return title.get_text(
        " ",
        strip=True
    )


# ============================================================
# ページ日付取得
# ============================================================

def get_page_date(html):

    title = get_title(html)

    if not title:
        return None

    # 2026-08-19 と 2026/08/19 の両方に対応
    match = re.search(
        r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})",
        title
    )

    if match:

        y, m, d = match.groups()

        return (
            f"{int(y):04d}-"
            f"{int(m):02d}-"
            f"{int(d):02d}"
        )

    return None


# ============================================================
# 店舗確認
# ============================================================

def check_store(html, store):

    title = get_title(html)

    if not title:
        return False

    for keyword in store["keywords"]:

        if keyword in title:
            return True

    return False


# ============================================================
# all_data_table確認
# ============================================================

def get_all_data_table(html):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    table = soup.find(
        "table",
        id="all_data_table"
    )

    if table is None:

        return None, 0

    rows = table.find_all("tr")

    return table, len(rows)


# ============================================================
# HTML保存
# ============================================================

def save_html(html):

    with open(
        HTML_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(html)

    print()
    print("HTML保存完了:")
    print(HTML_FILE)


# ============================================================
# CSV台数確認
# ============================================================

def count_csv_rows(csv_file):

    if not csv_file.exists():

        return 0

    try:

        with open(
            csv_file,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            rows = list(
                csv.reader(f)
            )

        if len(rows) <= 1:

            return 0

        return len(rows) - 1

    except Exception as e:

        print()
        print("[警告] CSV読み込みエラー:")
        print(csv_file)
        print(e)

        return 0


# ============================================================
# extract_machine_data.py実行
# ============================================================

def run_extractor():

    if not EXTRACT_SCRIPT.exists():

        print()
        print(
            "[エラー] extract_machine_data.py がありません。"
        )

        print(EXTRACT_SCRIPT)

        return False

    print()
    print(
        "extract_machine_data.py を実行します..."
    )

    try:

        result = subprocess.run(
            [
                "python",
                str(EXTRACT_SCRIPT)
            ],
            cwd=str(BASE_DIR)
        )

    except Exception as e:

        print()
        print(
            "[エラー] extract_machine_data.py 実行失敗"
        )

        print(e)

        return False

    if result.returncode != 0:

        print()
        print(
            "[エラー] extract_machine_data.py が異常終了しました。"
        )

        return False

    if not SLOT_DATA_FILE.exists():

        print()
        print(
            "[エラー] slot_data.csv が作成されませんでした。"
        )

        return False

    return True


# ============================================================
# 日付CSV保存
# ============================================================

def save_date_csv(
    store,
    date_str
):

    folder = store["folder"]

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = (
        folder /
        f"{date_str}.csv"
    )

    source_count = count_csv_rows(
        SLOT_DATA_FILE
    )

    if source_count <= 0:

        print()
        print(
            "[エラー] slot_data.csv にデータがありません。"
        )

        return 0

    with open(
        SLOT_DATA_FILE,
        "rb"
    ) as src:

        data = src.read()

    with open(
        destination,
        "wb"
    ) as dst:

        dst.write(data)

    saved_count = count_csv_rows(
        destination
    )

    print()
    print(
        f"日付: {date_str}"
    )

    print(
        f"取得台数: {saved_count}"
    )

    print(
        f"保存ファイル: {destination}"
    )

    return saved_count


# ============================================================
# all_data.csv再構築
# ============================================================

def rebuild_all_data(store):

    folder = store["folder"]

    print()
    print("========================================")
    print("all_data.csv 再構築")
    print("========================================")

    csv_files = []

    for file in folder.glob("*.csv"):

        if file.name == "all_data.csv":
            continue

        if re.fullmatch(
            r"\d{4}-\d{2}-\d{2}\.csv",
            file.name
        ):

            csv_files.append(file)

    csv_files.sort()

    print()
    print(
        f"対象CSV: {len(csv_files)} ファイル"
    )

    if not csv_files:

        print()
        print(
            "対象CSVがありません。"
        )

        return

    header = None
    all_rows = []
    dates = []

    for csv_file in csv_files:

        print()
        print(
            f"読み込み: {csv_file.name}"
        )

        try:

            with open(
                csv_file,
                "r",
                encoding="utf-8-sig",
                newline=""
            ) as f:

                reader = csv.reader(f)

                rows = list(reader)

        except Exception as e:

            print()
            print(
                "[警告] 読み込み失敗:"
            )

            print(e)

            continue

        if not rows:
            continue

        if header is None:

            header = rows[0]

        data_rows = rows[1:]

        all_rows.extend(
            data_rows
        )

        date_str = csv_file.stem

        if is_valid_date(
            date_str
        ):

            dates.append(
                date_str
            )

        print(
            f"→ {len(data_rows)} 台"
        )

    if header is None:

        print()
        print(
            "ヘッダーを取得できませんでした。"
        )

        return

    all_data_file = (
        folder / "all_data.csv"
    )

    with open(
        all_data_file,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(header)

        writer.writerows(
            all_rows
        )

    print()
    print(
        f"統合データ: {len(all_rows)} 行"
    )

    print(
        f"収録日数: {len(dates)}"
    )

    print(
        f"保存ファイル: {all_data_file}"
    )

    if dates:

        print()
        print("収録日:")

        for date_str in dates:

            dt = datetime.strptime(
                date_str,
                "%Y-%m-%d"
            )

            weekday = [
                "月",
                "火",
                "水",
                "木",
                "金",
                "土",
                "日"
            ][dt.weekday()]

            print(
                f"{date_str} ({weekday})"
            )


# ============================================================
# Network Response取得
# ============================================================

async def capture_target_response(
    page,
    target_url,
    timeout_seconds=60
):

    captured = {
        "body": None,
        "status": None,
        "url": None
    }

    normalized_target = normalize_url(
        target_url
    )

    # --------------------------------------------------------
    # Response監視
    # --------------------------------------------------------

    async def handle_response(response):

        response_url = response.url

        if not urls_match(
            response_url,
            target_url
        ):

            return

        if response.request.resource_type != "document":

            return

        print()
        print(
            "【対象ページ DOCUMENT RESPONSE】"
        )

        print(
            f"URL: {response_url}"
        )

        print(
            f"HTTP Status: {response.status}"
        )

        if response.status != 200:

            print()
            print(
                "[警告] HTTP Status が200ではありません。"
            )

            captured["status"] = response.status
            captured["url"] = response_url

            return

        try:

            body = await response.body()

        except Exception as e:

            print()
            print(
                "[エラー] Response body取得失敗"
            )

            print(e)

            return

        print(
            f"BODY SIZE: {len(body):,}"
        )

        if len(body) < 10000:

            print()
            print(
                "[警告] HTMLが短すぎます。"
            )

            return

        captured["body"] = body
        captured["status"] = response.status
        captured["url"] = response_url

        print()
        print(
            "[成功] 対象ページHTMLを捕捉しました。"
        )

    # --------------------------------------------------------
    # イベント登録
    # --------------------------------------------------------

    page.on(
        "response",
        handle_response
    )

    print()
    print(
        "Chromeで対象URLを開きます。"
    )

    print()
    print(target_url)

    print()
    print(
        f"正規化URL: {normalized_target}"
    )

    # --------------------------------------------------------
    # Chromeでページ移動
    # --------------------------------------------------------

    try:

        await page.goto(
            target_url,
            wait_until="domcontentloaded",
            timeout=60000
        )

    except Exception as e:

        print()
        print(
            "[注意] page.gotoで例外が発生しました。"
        )

        print(e)

        print()
        print(
            "Network Responseを引き続き待機します。"
        )

    # --------------------------------------------------------
    # Network待機
    # --------------------------------------------------------

    print()
    print(
        "対象ページのNetwork Responseを待機します。"
    )

    for i in range(
        timeout_seconds
    ):

        if captured["body"] is not None:

            break

        if (
            captured["status"] is not None
            and captured["status"] != 200
        ):

            print()
            print(
                f"対象ページが HTTP {captured['status']} を返したため、"
            )
            print(
                "Network待機を終了して手動表示フォールバックへ進みます。"
            )

            break

        await asyncio.sleep(1)

        if (
            (i + 1) % 5 == 0
        ):

            print(
                f"待機中... {i + 1}/{timeout_seconds}秒"
            )

    # --------------------------------------------------------
    # イベント解除
    # --------------------------------------------------------

    page.remove_listener(
        "response",
        handle_response
    )

    # --------------------------------------------------------
    # 手動表示フォールバック
    # --------------------------------------------------------

    if captured["body"] is None:

        print()
        print(
            "自動取得では対象HTMLを取得できませんでした。"
        )
        print()
        print(
            "9222付きChromeで、次の対象URLを手動で開いてください。"
        )
        print()
        print(target_url)
        print()
        print(
            "ページの台データ表が正常に表示されたことを確認してください。"
        )
        print(
            "表示できたら、このPowerShell画面に戻って Enter を押してください。"
        )
        print()

        print(
            "重要: 9222付きChromeで対象ページを手動表示してください。"
        )
        print(
            "Chromeの読み込みマークが回り続けていても問題ありません。"
        )
        print(
            "台データ表が画面に見えた時点で、PowerShellに戻って Enter を押してください。"
        )
        print()

        input(
            "台データ表を確認後 Enter: "
        )

        # ----------------------------------------------------
        # all_data_table が存在する生きたタブだけを探索する。
        # ページ全体の load 完了は待たない。
        # ----------------------------------------------------

        candidate_pages = []

        try:

            context = page.context

            for p in list(
                context.pages
            ):

                try:

                    if p.is_closed():

                        continue

                    candidate_pages.append(
                        p
                    )

                except Exception:

                    continue

        except Exception as e:

            print()
            print(
                "[警告] Chromeタブ一覧の取得に失敗しました。"
            )
            print(e)

        print()
        print(
            f"確認可能タブ数: {len(candidate_pages)}"
        )

        best_html = None
        best_url = None

        for idx, candidate_page in enumerate(
            reversed(candidate_pages),
            start=1
        ):

            try:

                if candidate_page.is_closed():

                    continue

                current_url = candidate_page.url

                print()
                print(
                    f"タブ候補 {idx}/{len(candidate_pages)}: {current_url}"
                )

                # locator.count() は load 完了を待たず、現在DOMだけを確認する。
                table_count = await (
                    candidate_page.locator(
                        "#all_data_table"
                    ).count()
                )

                print(
                    f"#all_data_table: {table_count}"
                )

                if table_count < 1:

                    continue

                # テーブルがDOMに存在するタブだけHTML化する。
                # networkidle / load / domcontentloaded は待たない。
                dom_html = await (
                    candidate_page.locator(
                        "html"
                    ).evaluate(
                        "(el) => el.outerHTML"
                    )
                )

                print(
                    f"DOM HTML文字数: {len(dom_html):,}"
                )

                if (
                    best_html is None
                    or len(dom_html) > len(best_html)
                ):

                    best_html = dom_html
                    best_url = current_url

            except Exception as e:

                print()
                print(
                    "[注意] このタブは読み取れませんでした。次を試します。"
                )
                print(e)

                continue

        if (
            best_html is not None
            and len(best_html) >= 10000
        ):

            captured["body"] = (
                best_html.encode(
                    "utf-8"
                )
            )

            captured["status"] = (
                "MANUAL_TABLE_DOM_FALLBACK"
            )

            captured["url"] = (
                best_url
            )

            print()
            print(
                "[成功] #all_data_table が存在するChrome DOMを取得しました。"
            )
            print(
                f"採用HTML文字数: {len(best_html):,}"
            )
            print(
                f"採用URL: {best_url}"
            )

        else:

            print()
            print(
                "[警告] #all_data_table を含む正常DOMを取得できませんでした。"
            )
            print(
                "Chrome画面で台データ表が見えているか確認してください。"
            )

    return captured


# ============================================================
# 1日取得
# ============================================================

async def fetch_one_day(
    page,
    store,
    date_str
):

    print()
    print("========================================")
    print(
        f"{date_str} 取得開始"
    )
    print("========================================")

    destination = (
        store["folder"] /
        f"{date_str}.csv"
    )

    # --------------------------------------------------------
    # 既存CSV
    # --------------------------------------------------------

    if destination.exists():

        count = count_csv_rows(
            destination
        )

        if count > 0:

            print()
            print(
                "同日のCSVが既に存在します。"
            )

            print(
                f"既存台数: {count}"
            )

            print(
                "今回は取得をスキップします。"
            )

            return "skip"

    # --------------------------------------------------------
    # URL
    # --------------------------------------------------------

    target_url = make_url(
        store,
        date_str
    )

    print()
    print(
        f"対象店舗: {store['name']}"
    )

    print(
        f"対象日: {date_str}"
    )

    print()
    print(
        "対象URL:"
    )

    print(
        target_url
    )

    # --------------------------------------------------------
    # Network取得
    # --------------------------------------------------------

    result = await capture_target_response(
        page,
        target_url
    )

    body = result["body"]

    if body is None:

        print()
        print(
            "[エラー] 対象ページのHTMLを取得できませんでした。"
        )

        return "fail"

    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

    html = body.decode(
        "utf-8",
        errors="replace"
    )

    print()
    print(
        f"最終HTML文字数: {len(html):,}"
    )

    # --------------------------------------------------------
    # タイトル
    # --------------------------------------------------------

    title = get_title(
        html
    )

    print()
    print(
        f"ページタイトル: {title}"
    )

    if not title:

        print()
        print(
            "[エラー] ページタイトルがありません。"
        )

        return "fail"

    # --------------------------------------------------------
    # 日付
    # --------------------------------------------------------

    page_date = get_page_date(
        html
    )

    if page_date != date_str:

        print()
        print(
            "[エラー] 指定日とページ日付が一致しません。"
        )

        print(
            f"指定日: {date_str}"
        )

        print(
            f"ページ日: {page_date}"
        )

        print(
            f"タイトル: {title}"
        )

        return "fail"

    print()
    print(
        f"日付確認OK: {page_date}"
    )

    # --------------------------------------------------------
    # 店舗
    # --------------------------------------------------------

    if not check_store(
        html,
        store
    ):

        print()
        print(
            "[エラー] 店舗確認に失敗しました。"
        )

        print(
            f"対象店舗: {store['name']}"
        )

        print(
            f"ページタイトル: {title}"
        )

        return "fail"

    print()
    print(
        f"店舗確認OK: {store['name']}"
    )

    # --------------------------------------------------------
    # テーブル
    # --------------------------------------------------------

    table, row_count = get_all_data_table(
        html
    )

    if table is None:

        print()
        print(
            "[エラー] all_data_table が見つかりません。"
        )

        return "fail"

    print()
    print(
        "データテーブル発見: all_data_table"
    )

    print(
        f"テーブル行数: {row_count}"
    )

    if row_count <= 1:

        print()
        print(
            "[エラー] データ行がありません。"
        )

        return "fail"

    # --------------------------------------------------------
    # HTML保存
    # --------------------------------------------------------

    save_html(
        html
    )

    # --------------------------------------------------------
    # 抽出
    # --------------------------------------------------------

    if not run_extractor():

        return "fail"

    # --------------------------------------------------------
    # 台数
    # --------------------------------------------------------

    count = count_csv_rows(
        SLOT_DATA_FILE
    )

    print()
    print(
        f"取得台数確認: {count}"
    )

    if count <= 0:

        print()
        print(
            "[エラー] 台データが0台です。"
        )

        return "fail"

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    saved_count = save_date_csv(
        store,
        date_str
    )

    if saved_count <= 0:

        return "fail"

    print()
    print(
        f"{date_str} の取得に成功しました。"
    )

    return "success"


# ============================================================
# 日付リスト
# ============================================================

def make_date_list(
    start_date,
    end_date
):

    start = datetime.strptime(
        start_date,
        "%Y-%m-%d"
    )

    end = datetime.strptime(
        end_date,
        "%Y-%m-%d"
    )

    dates = []

    current = start

    while current <= end:

        dates.append(
            current.strftime(
                "%Y-%m-%d"
            )
        )

        current += timedelta(
            days=1
        )

    return dates


# ============================================================
# 店舗選択
# ============================================================

def select_store():

    print()
    print(
        "店舗を選択してください。"
    )

    print()

    for number, store in STORES.items():

        print(
            f"{number}. {store['name']}"
        )

    print()

    while True:

        try:

            number = int(
                input(
                    "店舗番号: "
                )
            )

        except ValueError:

            print(
                "数字を入力してください。"
            )

            continue

        if number in STORES:

            store = STORES[number]

            print()
            print(
                f"選択店舗: {store['name']}"
            )

            return store

        print(
            "正しい店舗番号を入力してください。"
        )


# ============================================================
# 取得方法
# ============================================================

def select_method():

    print()
    print(
        "取得方法を選択してください。"
    )

    print()
    print(
        "1. 1日だけ取得"
    )

    print(
        "2. 期間を指定して取得"
    )

    print()

    while True:

        try:

            number = int(
                input("番号: ")
            )

        except ValueError:

            print(
                "数字を入力してください。"
            )

            continue

        if number in (
            1,
            2
        ):

            return number

        print(
            "1 または 2 を入力してください。"
        )


# ============================================================
# 日付入力
# ============================================================

def input_date(message):

    while True:

        date_str = input(
            message
        ).strip()

        if is_valid_date(
            date_str
        ):

            return date_str

        print()
        print(
            "YYYY-MM-DD形式で入力してください。"
        )


# ============================================================
# メイン
# ============================================================

async def main():

    print()
    print("========================================")
    print(" SlotAnalyzer 複数店舗データ収集")
    print("========================================")

    store = select_store()

    print()
    print(
        f"店舗: {store['name']}"
    )

    print(
        f"データフォルダ: {store['folder']}"
    )

    store["folder"].mkdir(
        parents=True,
        exist_ok=True
    )

    method = select_method()

    if method == 1:

        date_str = input_date(
            "取得する日付 (YYYY-MM-DD): "
        )

        dates = [
            date_str
        ]

    else:

        start_date = input_date(
            "取得開始日 (YYYY-MM-DD): "
        )

        end_date = input_date(
            "取得終了日 (YYYY-MM-DD): "
        )

        start = datetime.strptime(
            start_date,
            "%Y-%m-%d"
        )

        end = datetime.strptime(
            end_date,
            "%Y-%m-%d"
        )

        if start > end:

            print()
            print(
                "[エラー] 開始日が終了日より後です。"
            )

            return

        dates = make_date_list(
            start_date,
            end_date
        )

    print()
    print(
        f"店舗: {store['name']}"
    )

    print(
        f"開始日: {dates[0]}"
    )

    print(
        f"終了日: {dates[-1]}"
    )

    print(
        f"対象日数: {len(dates)}"
    )

    # --------------------------------------------------------
    # Chrome
    # --------------------------------------------------------

    async with async_playwright() as p:

        print()
        print(
            "9222付きChromeに接続します。"
        )

        try:

            browser = await p.chromium.connect_over_cdp(
                CHROME_CDP_URL
            )

        except Exception as e:

            print()
            print(
                "【エラー】9222付きChromeに接続できません。"
            )

            print(e)

            return

        print()
        print(
            "Chrome接続成功。"
        )

        context = None
        page = None

        # ----------------------------------------------------
        # 既存アナスロタブ
        # ----------------------------------------------------

        for browser_context in browser.contexts:

            if context is None:

                context = browser_context

            for existing_page in browser_context.pages:

                try:

                    if existing_page.is_closed():

                        continue

                    if "ana-slo.com" in existing_page.url:

                        page = existing_page

                        break

                except Exception:

                    pass

            if page is not None:

                break

        # ----------------------------------------------------
        # なければ新規タブ
        # ----------------------------------------------------

        if page is None:

            print()
            print(
                "既存のアナスロページがありません。"
            )

            print(
                "新しいタブを作成します。"
            )

            page = await context.new_page()

        else:

            print()
            print(
                "既存のアナスロページを使用します。"
            )

            print(
                page.url
            )

        # ----------------------------------------------------
        # 日付処理
        # ----------------------------------------------------

        success_count = 0
        skip_count = 0
        fail_count = 0

        for index, date_str in enumerate(
            dates,
            start=1
        ):

            print()
            print("========================================")

            print(
                f"進捗: {index}/{len(dates)}"
            )

            print("========================================")

            result = await fetch_one_day(
                page,
                store,
                date_str
            )

            if result == "success":

                success_count += 1

            elif result == "skip":

                skip_count += 1

            else:

                fail_count += 1

            if index < len(dates):

                print()
                print(
                    "次の日付まで3秒待機します。"
                )

                await asyncio.sleep(3)

        # ----------------------------------------------------
        # 統合
        # ----------------------------------------------------

        rebuild_all_data(
            store
        )

        # ----------------------------------------------------
        # 結果
        # ----------------------------------------------------

        print()
        print("========================================")
        print("期間取得完了")
        print("========================================")

        print()
        print(
            f"店舗: {store['name']}"
        )

        print(
            f"期間: {dates[0]} ～ {dates[-1]}"
        )

        print()
        print(
            f"取得成功: {success_count}日"
        )

        print(
            f"既存スキップ: {skip_count}日"
        )

        print(
            f"失敗: {fail_count}日"
        )

        print()
        print(
            "データフォルダ:"
        )

        print(
            store["folder"]
        )

        print()
        print(
            "Chromeはそのままです。"
        )


if __name__ == "__main__":

    asyncio.run(
        main()
    )