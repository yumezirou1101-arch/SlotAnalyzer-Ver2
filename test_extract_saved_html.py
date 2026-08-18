from pathlib import Path
from bs4 import BeautifulSoup


def main():

    print()
    print("=" * 70)
    print("保存済みNetwork Response HTML 台データ抽出テスト")
    print("=" * 70)

    # --------------------------------------------------
    # HTMLファイル
    # --------------------------------------------------

    base_dir = Path(__file__).resolve().parent

    html_file = base_dir / "network_response_1.html"

    print()
    print("対象HTML:")
    print(html_file)

    if not html_file.exists():

        print()
        print("[エラー]")
        print("network_response_1.html が見つかりません。")

        return

    # --------------------------------------------------
    # 読み込み
    # --------------------------------------------------

    print()
    print("HTMLを読み込みます...")

    html = html_file.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    print(
        f"HTML文字数: {len(html):,}"
    )

    # --------------------------------------------------
    # BeautifulSoup
    # --------------------------------------------------

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    # --------------------------------------------------
    # all_data_table
    # --------------------------------------------------

    table = soup.find(
        "table",
        id="all_data_table"
    )

    if table is None:

        print()
        print(
            "[エラー]"
        )
        print(
            "all_data_table が見つかりません。"
        )

        return

    print()
    print(
        "★ all_data_table 発見"
    )

    # --------------------------------------------------
    # ヘッダー取得
    # --------------------------------------------------

    header_row = table.find(
        "thead"
    )

    if header_row:

        headers = [
            cell.get_text(
                " ",
                strip=True
            )
            for cell in header_row.find_all("th")
        ]

    else:

        first_row = table.find("tr")

        if first_row:

            headers = [
                cell.get_text(
                    " ",
                    strip=True
                )
                for cell in first_row.find_all(
                    ["th", "td"]
                )
            ]

        else:

            headers = []

    print()
    print(
        "列数:",
        len(headers)
    )

    print()
    print("列名:")

    for i, header in enumerate(
        headers,
        start=1
    ):

        print(
            f"  {i}. {header}"
        )

    # --------------------------------------------------
    # tbody
    # --------------------------------------------------

    tbody = table.find("tbody")

    if tbody is None:

        print()
        print(
            "[エラー]"
        )
        print(
            "tbody が見つかりません。"
        )

        return

    # --------------------------------------------------
    # データ行
    # --------------------------------------------------

    rows = tbody.find_all(
        "tr",
        recursive=False
    )

    print()
    print(
        "データ行数:",
        len(rows)
    )

    # --------------------------------------------------
    # 実際のデータを解析
    # --------------------------------------------------

    data_rows = []

    for row in rows:

        cells = row.find_all(
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

        if not values:
            continue

        data_rows.append(values)

    print()
    print(
        "実データ行数:",
        len(data_rows)
    )

    # --------------------------------------------------
    # 最初の10台を表示
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("最初の10台")
    print("=" * 70)

    for i, values in enumerate(
        data_rows[:10],
        start=1
    ):

        print()
        print(
            f"[{i}]"
        )

        for j, value in enumerate(
            values,
            start=1
        ):

            if j <= len(headers):

                name = headers[j - 1]

            else:

                name = f"列{j}"

            print(
                f"  {name}: {value}"
            )

    # --------------------------------------------------
    # 列数の確認
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("列数チェック")
    print("=" * 70)

    count_map = {}

    for values in data_rows:

        count = len(values)

        count_map[count] = (
            count_map.get(count, 0) + 1
        )

    for count in sorted(
        count_map
    ):

        print(
            f"{count}列: "
            f"{count_map[count]}行"
        )

    # --------------------------------------------------
    # 台番号らしきデータを確認
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("台番号確認")
    print("=" * 70)

    machine_numbers = []

    for values in data_rows:

        # 通常は2列目が台番号
        if len(values) >= 2:

            machine_number = values[1].strip()

            if machine_number:

                machine_numbers.append(
                    machine_number
                )

    print(
        "台番号取得件数:",
        len(machine_numbers)
    )

    print()

    print(
        "最初の20台番号:"
    )

    for number in machine_numbers[:20]:

        print(
            " ",
            number
        )

    # --------------------------------------------------
    # CSV化できる形式か確認
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("CSV化判定")
    print("=" * 70)

    if (
        len(headers) > 0
        and len(data_rows) > 0
    ):

        print()
        print(
            "★ 台データの表形式抽出に成功しました。"
        )

        print()
        print(
            "このHTMLからCSVを作成できる可能性が高いです。"
        )

    else:

        print()
        print(
            "[失敗]"
        )
        print(
            "表形式の台データを取得できませんでした。"
        )

    print()
    print("=" * 70)
    print("テスト終了")
    print("=" * 70)


if __name__ == "__main__":

    main()