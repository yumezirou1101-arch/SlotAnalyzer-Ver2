import csv
import os
import statistics
from collections import defaultdict
from datetime import datetime


# ============================================================
# SlotAnalyzer
# 過去データ基礎分析 v3
# 複数店舗対応版
# ============================================================


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 店舗設定
# ============================================================

STORES = {
    "1": {
        "name": "マルハンメガシティ前橋インター",
        "folder": os.path.join(
            BASE_DIR,
            "data",
            "maruhan_maebashi"
        )
    },

    "2": {
        "name": "ビックマーチ高崎大八木店",
        "folder": os.path.join(
            BASE_DIR,
            "data",
            "bigmarch_takasaki_oyagi"
        )
    },

    "3": {
        "name": "ビックつばめ高崎店",
        "folder": os.path.join(
            BASE_DIR,
            "data",
            "bigtsubame_takasaki"
        )
    },

    "4": {
        "name": "やすだ前橋店",
        "folder": os.path.join(
            BASE_DIR,
            "data",
            "yasuda_maebashi"
        )
    }
}


# ============================================================
# 共通関数
# ============================================================

def safe_int(value):
    """
    数値文字列を整数に変換
    """
    if value is None:
        return 0

    value = str(value)

    value = value.replace(",", "")
    value = value.replace("+", "")
    value = value.strip()

    if value == "":
        return 0

    try:
        return int(float(value))
    except ValueError:
        return 0


def safe_float(value):
    """
    数値文字列をfloatに変換
    """
    if value is None:
        return 0.0

    value = str(value)

    value = value.replace(",", "")
    value = value.replace("+", "")
    value = value.strip()

    if value == "":
        return 0.0

    try:
        return float(value)
    except ValueError:
        return 0.0


def percent(value, total):
    """
    パーセント計算
    """
    if total == 0:
        return 0.0

    return value / total * 100


def format_date(date_string):
    """
    YYYY-MM-DD → YYYY-MM-DD (曜日)
    """
    try:
        dt = datetime.strptime(date_string, "%Y-%m-%d")

        weekdays = [
            "月",
            "火",
            "水",
            "木",
            "金",
            "土",
            "日"
        ]

        return f"{date_string} ({weekdays[dt.weekday()]})"

    except Exception:
        return date_string


# ============================================================
# CSV読み込み
# ============================================================

def load_csv(file_path):

    rows = []

    if not os.path.exists(file_path):
        return rows

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp932"
    ]

    for encoding in encodings:

        try:

            with open(
                file_path,
                "r",
                encoding=encoding,
                newline=""
            ) as f:

                reader = csv.DictReader(f)

                for row in reader:

                    if not row:
                        continue

                    rows.append(row)

            return rows

        except UnicodeDecodeError:
            rows = []
            continue

        except Exception as e:
            print("CSV読み込みエラー:", e)
            return []

    return rows


# ============================================================
# 店舗選択
# ============================================================

def select_store():

    print()
    print("========================================")
    print(" SlotAnalyzer 過去データ分析")
    print("========================================")
    print()
    print("店舗を選択してください。")
    print()

    for key, store in STORES.items():
        print(f"{key}. {store['name']}")

    print()

    while True:

        choice = input("店舗番号: ").strip()

        if choice in STORES:
            return STORES[choice]

        print("店舗番号が正しくありません。")
        print("1～4を入力してください。")


# ============================================================
# データ読み込み
# ============================================================

def load_store_data(store):

    folder = store["folder"]

    all_data_path = os.path.join(
        folder,
        "all_data.csv"
    )

    print()
    print("選択店舗:", store["name"])
    print("データファイル:", all_data_path)
    print()

    if not os.path.exists(all_data_path):

        print("========================================")
        print(" エラー")
        print("========================================")
        print()
        print("all_data.csv が見つかりません。")
        print()
        print("確認してください:")
        print(all_data_path)
        print()

        return []

    rows = load_csv(all_data_path)

    if not rows:

        print("CSVにデータがありません。")
        return []

    print("読み込み完了")
    print("総台データ:", len(rows))

    return rows


# ============================================================
# 基本情報
# ============================================================

def show_basic_information(rows, store):

    dates = sorted(
        set(
            row.get("日付", "")
            for row in rows
            if row.get("日付", "")
        )
    )

    print()
    print("========================================")
    print(" 【基本情報】")
    print("========================================")
    print()

    print("店舗:", store["name"])
    print("総台データ:", len(rows))
    print("収録日数:", len(dates))

    if dates:

        print(
            "収録期間:",
            dates[0],
            "～",
            dates[-1]
        )

        print()
        print("収録日:")

        for date in dates:
            print(" ", format_date(date))

    print()


# ============================================================
# 全体分析
# ============================================================

def analyze_overall(rows):

    print("========================================")
    print(" 【全体】")
    print("========================================")
    print()

    if not rows:
        return

    g_values = [
        safe_int(row.get("G数"))
        for row in rows
    ]

    diff_values = [
        safe_int(row.get("差枚"))
        for row in rows
    ]

    positive = [
        x for x in diff_values
        if x > 0
    ]

    over_1000 = [
        x for x in diff_values
        if x >= 1000
    ]

    over_3000 = [
        x for x in diff_values
        if x >= 3000
    ]

    print("台数:", len(rows))

    print(
        "平均G数:",
        round(
            sum(g_values) / len(g_values),
            1
        )
    )

    print(
        "平均差枚:",
        round(
            sum(diff_values) / len(diff_values),
            1
        )
    )

    print(
        "プラス台率:",
        f"{percent(len(positive), len(rows)):.1f}%"
    )

    print(
        "1000枚以上率:",
        f"{percent(len(over_1000), len(rows)):.1f}%"
    )

    print(
        "3000枚以上率:",
        f"{percent(len(over_3000), len(rows)):.1f}%"
    )

    print()


# ============================================================
# 日付別分析
# ============================================================

def analyze_by_date(rows):

    print("========================================")
    print(" 【日付別】")
    print("========================================")
    print()

    grouped = defaultdict(list)

    for row in rows:

        date = row.get("日付", "")

        if date:
            grouped[date].append(row)

    for date in sorted(grouped):

        data = grouped[date]

        diffs = [
            safe_int(row.get("差枚"))
            for row in data
        ]

        gs = [
            safe_int(row.get("G数"))
            for row in data
        ]

        positive = [
            x for x in diffs
            if x > 0
        ]

        print(
            format_date(date)
        )

        print(
            "  台数:",
            len(data),
            "平均G:",
            round(sum(gs) / len(gs), 1),
            "平均差枚:",
            round(sum(diffs) / len(diffs), 1),
            "プラス率:",
            f"{percent(len(positive), len(data)):.1f}%"
        )

    print()


# ============================================================
# 曜日別分析
# ============================================================

def analyze_by_weekday(rows):

    print("========================================")
    print(" 【曜日別】")
    print("========================================")
    print()

    grouped = defaultdict(list)

    for row in rows:

        date = row.get("日付", "")

        try:

            dt = datetime.strptime(
                date,
                "%Y-%m-%d"
            )

            weekday = [
                "月曜日",
                "火曜日",
                "水曜日",
                "木曜日",
                "金曜日",
                "土曜日",
                "日曜日"
            ][dt.weekday()]

            grouped[weekday].append(row)

        except Exception:
            continue

    weekday_order = [
        "月曜日",
        "火曜日",
        "水曜日",
        "木曜日",
        "金曜日",
        "土曜日",
        "日曜日"
    ]

    for weekday in weekday_order:

        if weekday not in grouped:
            continue

        data = grouped[weekday]

        diffs = [
            safe_int(row.get("差枚"))
            for row in data
        ]

        positive = [
            x for x in diffs
            if x > 0
        ]

        over_3000 = [
            x for x in diffs
            if x >= 3000
        ]

        print(
            weekday
        )

        print(
            "  台数:",
            len(data),
            "平均差枚:",
            round(sum(diffs) / len(diffs), 1),
            "プラス率:",
            f"{percent(len(positive), len(data)):.1f}%",
            "3000枚以上率:",
            f"{percent(len(over_3000), len(data)):.1f}%"
        )

    print()


# ============================================================
# 前日 → 翌日分析
# ============================================================

def analyze_next_day(rows):

    print("========================================")
    print(" 【前日 → 翌日分析】")
    print("========================================")
    print()

    machine_map = defaultdict(dict)

    for row in rows:

        date = row.get("日付", "")
        dai = row.get("台番号", "")

        if date and dai:
            machine_map[dai][date] = row

    dates = sorted(
        set(
            row.get("日付", "")
            for row in rows
            if row.get("日付", "")
        )
    )

    if len(dates) < 2:

        print("比較できる日数がありません。")
        print()

        return

    categories = {
        "前日3000枚以上": [],
        "前日1000～2999枚": [],
        "前日0～999枚": [],
        "前日-1～-999枚": [],
        "前日-1000～-2999枚": [],
        "前日-3000枚以下": []
    }

    comparable = 0

    for i in range(len(dates) - 1):

        previous_date = dates[i]
        next_date = dates[i + 1]

        try:

            dt1 = datetime.strptime(
                previous_date,
                "%Y-%m-%d"
            )

            dt2 = datetime.strptime(
                next_date,
                "%Y-%m-%d"
            )

        except Exception:
            continue

        # 必ず翌日であることを確認
        if (dt2 - dt1).days != 1:
            continue

        for dai, date_data in machine_map.items():

            if previous_date not in date_data:
                continue

            if next_date not in date_data:
                continue

            previous_diff = safe_int(
                date_data[previous_date].get("差枚")
            )

            next_diff = safe_int(
                date_data[next_date].get("差枚")
            )

            if previous_diff >= 3000:

                categories[
                    "前日3000枚以上"
                ].append(next_diff)

            elif previous_diff >= 1000:

                categories[
                    "前日1000～2999枚"
                ].append(next_diff)

            elif previous_diff >= 0:

                categories[
                    "前日0～999枚"
                ].append(next_diff)

            elif previous_diff >= -999:

                categories[
                    "前日-1～-999枚"
                ].append(next_diff)

            elif previous_diff >= -2999:

                categories[
                    "前日-1000～-2999枚"
                ].append(next_diff)

            else:

                categories[
                    "前日-3000枚以下"
                ].append(next_diff)

            comparable += 1

    print(
        "比較できる台データ:",
        comparable
    )

    print()

    for category, values in categories.items():

        if not values:
            continue

        positive = [
            x for x in values
            if x > 0
        ]

        over_3000 = [
            x for x in values
            if x >= 3000
        ]

        print(category)

        print(
            "件数:",
            len(values),
            "翌日平均差枚:",
            round(sum(values) / len(values), 1),
            "翌日プラス率:",
            f"{percent(len(positive), len(values)):.1f}%",
            "翌日3000枚以上率:",
            f"{percent(len(over_3000), len(values)):.1f}%"
        )

    print()


# ============================================================
# 差枚変化分析
# ============================================================

def analyze_change(rows):

    print("========================================")
    print(" 【前日からの差枚変化】")
    print("========================================")
    print()

    machine_map = defaultdict(dict)

    for row in rows:

        date = row.get("日付", "")
        dai = row.get("台番号", "")

        if date and dai:
            machine_map[dai][date] = row

    dates = sorted(
        set(
            row.get("日付", "")
            for row in rows
        )
    )

    rising = []
    flat = []
    falling = []

    for i in range(len(dates) - 1):

        previous_date = dates[i]
        next_date = dates[i + 1]

        try:

            dt1 = datetime.strptime(
                previous_date,
                "%Y-%m-%d"
            )

            dt2 = datetime.strptime(
                next_date,
                "%Y-%m-%d"
            )

        except Exception:
            continue

        if (dt2 - dt1).days != 1:
            continue

        for dai, date_data in machine_map.items():

            if previous_date not in date_data:
                continue

            if next_date not in date_data:
                continue

            previous_diff = safe_int(
                date_data[previous_date].get("差枚")
            )

            next_diff = safe_int(
                date_data[next_date].get("差枚")
            )

            change = next_diff - previous_diff

            if change >= 1000:

                rising.append(
                    (change, next_diff)
                )

            elif change <= -1000:

                falling.append(
                    (change, next_diff)
                )

            else:

                flat.append(
                    (change, next_diff)
                )

    groups = [
        ("上昇1000枚以上", rising),
        ("ほぼ横ばい", flat),
        ("下降1000枚以上", falling)
    ]

    for name, values in groups:

        if not values:
            print(name)
            print("件数: 0")
            print()
            continue

        changes = [
            x[0]
            for x in values
        ]

        next_values = [
            x[1]
            for x in values
        ]

        positive = [
            x for x in next_values
            if x > 0
        ]

        print(name)

        print(
            "件数:",
            len(values),
            "平均差枚変化:",
            round(sum(changes) / len(changes), 1),
            "翌日プラス率:",
            f"{percent(len(positive), len(next_values)):.1f}%"
        )

    print()


# ============================================================
# 台番号別ランキング
# ============================================================

def analyze_machine_number(rows):

    print("========================================")
    print(" 【台番号別 平均差枚ランキング】")
    print("========================================")
    print()

    grouped = defaultdict(list)

    for row in rows:

        dai = row.get("台番号", "")

        if not dai:
            continue

        diff = safe_int(
            row.get("差枚")
        )

        grouped[dai].append(diff)

    results = []

    for dai, values in grouped.items():

        if not values:
            continue

        results.append(
            {
                "dai": dai,
                "count": len(values),
                "average": sum(values) / len(values),
                "median": statistics.median(values),
                "positive": percent(
                    len(
                        [
                            x for x in values
                            if x > 0
                        ]
                    ),
                    len(values)
                )
            }
        )

    results.sort(
        key=lambda x: x["average"],
        reverse=True
    )

    for i, item in enumerate(
        results[:30],
        start=1
    ):

        print(
            f"{i}.",
            item["dai"],
            "データ:",
            item["count"],
            "平均差枚:",
            f"{item['average']:.1f}",
            "中央値:",
            f"{item['median']:.1f}",
            "プラス率:",
            f"{item['positive']:.1f}%"
        )

    print()


# ============================================================
# 機種別ランキング
# ============================================================

def analyze_machine_name(rows):

    print("========================================")
    print(" 【機種別 平均差枚ランキング】")
    print("========================================")
    print()

    grouped = defaultdict(list)

    for row in rows:

        name = row.get("機種名", "")

        if not name:
            continue

        diff = safe_int(
            row.get("差枚")
        )

        grouped[name].append(diff)

    results = []

    for name, values in grouped.items():

        if not values:
            continue

        over_1000 = [
            x for x in values
            if x >= 1000
        ]

        over_3000 = [
            x for x in values
            if x >= 3000
        ]

        positive = [
            x for x in values
            if x > 0
        ]

        results.append(
            {
                "name": name,
                "count": len(values),
                "average": sum(values) / len(values),
                "positive": percent(
                    len(positive),
                    len(values)
                ),
                "over1000": percent(
                    len(over_1000),
                    len(values)
                ),
                "over3000": percent(
                    len(over_3000),
                    len(values)
                )
            }
        )

    results.sort(
        key=lambda x: x["average"],
        reverse=True
    )

    for i, item in enumerate(
        results[:50],
        start=1
    ):

        print(
            f"{i}.",
            item["name"]
        )

        print(
            "  台数:",
            item["count"],
            "平均差枚:",
            f"{item['average']:.1f}",
            "プラス率:",
            f"{item['positive']:.1f}%",
            "1000枚以上率:",
            f"{item['over1000']:.1f}%",
            "3000枚以上率:",
            f"{item['over3000']:.1f}%"
        )

    print()


# ============================================================
# 連続プラス分析
# ============================================================

def analyze_consecutive(rows):

    print("========================================")
    print(" 【台番号別 連続プラス分析】")
    print("========================================")
    print()

    machine_map = defaultdict(dict)

    machine_names = {}

    for row in rows:

        date = row.get("日付", "")
        dai = row.get("台番号", "")

        if not date or not dai:
            continue

        machine_map[dai][date] = row

        machine_names[dai] = row.get(
            "機種名",
            ""
        )

    results = []

    for dai, date_data in machine_map.items():

        dates = sorted(date_data)

        current = 0
        maximum = 0

        for date in dates:

            diff = safe_int(
                date_data[date].get("差枚")
            )

            if diff > 0:

                current += 1

                if current > maximum:
                    maximum = current

            else:

                current = 0

        if maximum >= 2:

            results.append(
                {
                    "dai": dai,
                    "name": machine_names.get(
                        dai,
                        ""
                    ),
                    "maximum": maximum,
                    "count": len(dates)
                }
            )

    results.sort(
        key=lambda x: (
            x["maximum"],
            x["count"]
        ),
        reverse=True
    )

    for i, item in enumerate(
        results[:30],
        start=1
    ):

        print(
            f"{i})",
            item["dai"],
            item["name"],
            "最大連続プラス:",
            f"{item['maximum']}回",
            "データ:",
            item["count"]
        )

    print()


# ============================================================
# 機種 × 曜日分析
# ============================================================

def analyze_machine_weekday(rows):

    print("========================================")
    print(" 【機種 × 曜日分析】")
    print("========================================")
    print()

    grouped = defaultdict(list)

    for row in rows:

        name = row.get("機種名", "")
        date = row.get("日付", "")

        if not name or not date:
            continue

        try:

            dt = datetime.strptime(
                date,
                "%Y-%m-%d"
            )

            weekday = [
                "月曜日",
                "火曜日",
                "水曜日",
                "木曜日",
                "金曜日",
                "土曜日",
                "日曜日"
            ][dt.weekday()]

        except Exception:
            continue

        key = (
            name,
            weekday
        )

        grouped[key].append(
            safe_int(
                row.get("差枚")
            )
        )

    results = []

    for (
        name,
        weekday
    ), values in grouped.items():

        if not values:
            continue

        positive = [
            x for x in values
            if x > 0
        ]

        results.append(
            {
                "name": name,
                "weekday": weekday,
                "count": len(values),
                "average": sum(values) / len(values),
                "positive": percent(
                    len(positive),
                    len(values)
                )
            }
        )

    results.sort(
        key=lambda x: x["average"],
        reverse=True
    )

    for i, item in enumerate(
        results[:50],
        start=1
    ):

        print(
            f"{i}.",
            item["name"],
            item["weekday"],
            "台数:",
            item["count"],
            "平均差枚:",
            f"{item['average']:.1f}",
            "プラス率:",
            f"{item['positive']:.1f}%"
        )

    print()


# ============================================================
# 最終注意表示
# ============================================================

def show_data_warning(rows):

    dates = sorted(
        set(
            row.get("日付", "")
            for row in rows
            if row.get("日付", "")
        )
    )

    print("========================================")
    print(" 【データ信頼度について】")
    print("========================================")
    print()

    print(
        "現在の収録日数:",
        len(dates)
    )

    if len(dates) < 7:

        print(
            "まだデータが少ないため、"
            "傾向判断は暫定です。"
        )

    elif len(dates) < 14:

        print(
            "データが蓄積されてきましたが、"
            "曜日・イベント日などの影響を考慮する必要があります。"
        )

    elif len(dates) < 30:

        print(
            "一定量のデータがあります。"
            "ただし、機種ごとのサンプル数には注意してください。"
        )

    else:

        print(
            "十分な期間のデータが蓄積されています。"
            "機種・曜日・台番号などを組み合わせた分析が可能です。"
        )

    print()

    print(
        "注意: 平均差枚やプラス率だけで"
        "設定投入を断定することはできません。"
    )

    print(
        "サンプル数、稼働G数、機種構成、"
        "イベント日などを合わせて判断してください。"
    )

    print()


# ============================================================
# メイン
# ============================================================

def main():

    store = select_store()

    rows = load_store_data(
        store
    )

    if not rows:

        input(
            "\nEnterキーで終了します..."
        )

        return

    show_basic_information(
        rows,
        store
    )

    analyze_overall(
        rows
    )

    analyze_by_date(
        rows
    )

    analyze_by_weekday(
        rows
    )

    analyze_next_day(
        rows
    )

    analyze_change(
        rows
    )

    analyze_machine_number(
        rows
    )

    analyze_machine_name(
        rows
    )

    analyze_consecutive(
        rows
    )

    analyze_machine_weekday(
        rows
    )

    show_data_warning(
        rows
    )

    print("========================================")
    print(" 分析完了")
    print("========================================")
    print()

    input(
        "Enterキーで終了します..."
    )


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    main()