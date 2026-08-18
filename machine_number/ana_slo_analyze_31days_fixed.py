# -*- coding: utf-8 -*-
"""
アナスロ 31日分 基礎解析【7/11を確実に含める修正版】

対象期間:
2026-07-11 ～ 2026-08-10

重要:
- 7/11の日別CSVを必ず追加
- 7/12～8/10の統合CSVも読み込む
- 既存のCSVは変更しない
- 31日 × 514台 = 15,934件を検証
- 日付＋台番号の重複を除去
"""

from pathlib import Path
import csv
import re
from collections import defaultdict
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "maruhan_maebashi" / "machine_number"
OUTPUT_DIR = DATA_DIR / "analysis_31days"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2026-07-11"
END_DATE = "2026-08-10"
EXPECTED_DAYS = 31
EXPECTED_MACHINES_PER_DAY = 514
EXPECTED_RECORDS = EXPECTED_DAYS * EXPECTED_MACHINES_PER_DAY

DAY11_FILE = DATA_DIR / "ana_slo_20260711.csv"
MERGED_FILE = DATA_DIR / "ana_slo_20260712_20260810.csv"


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_key(s):
    return re.sub(r"\s+", "", str(s or "")).lower()


def find_col(headers, candidates):
    normalized = {normalize_key(h): h for h in headers}

    for candidate in candidates:
        key = normalize_key(candidate)
        if key in normalized:
            return normalized[key]

    for h in headers:
        nh = normalize_key(h)
        for candidate in candidates:
            nc = normalize_key(candidate)
            if nc in nh or nh in nc:
                return h

    return None


def to_int(value):
    if value is None:
        return None

    s = str(value).strip()
    s = s.replace(",", "")
    s = s.replace("+", "")

    if s in ("", "-", "--", "None", "null"):
        return None

    try:
        return int(float(s))
    except Exception:
        return None


def filename_date(path):
    m = re.search(r"ana_slo_(\d{8})(?:_\d{8})?\.csv$", path.name)

    if not m:
        return ""

    return datetime.strptime(
        m.group(1), "%Y%m%d"
    ).strftime("%Y-%m-%d")


def load_file(path, forced_date=""):
    rows = read_csv(path)

    if not rows:
        return []

    headers = list(rows[0].keys())

    machine_col = find_col(
        headers,
        ["機種名", "機種", "machine", "machine_name"]
    )

    number_col = find_col(
        headers,
        ["台番号", "台番", "台No", "台NO", "machine_no", "machine_number"]
    )

    games_col = find_col(
        headers,
        ["G数", "ゲーム数", "games", "game"]
    )

    diff_col = find_col(
        headers,
        ["差枚", "差枚数", "diff"]
    )

    if not number_col or not diff_col:
        raise ValueError(
            f"必要列を認識できません。\n"
            f"ファイル: {path}\n"
            f"列: {headers}"
        )

    # 統合CSVに日付列がある場合にも対応
    date_col = find_col(
        headers,
        ["日付", "date", "対象日"]
    )

    default_date = forced_date or filename_date(path)

    result = []

    for row in rows:
        machine_no = to_int(row.get(number_col))
        diff = to_int(row.get(diff_col))
        games = to_int(row.get(games_col)) if games_col else 0

        if machine_no is None or diff is None:
            continue

        date = ""

        if date_col:
            raw_date = str(row.get(date_col, "")).strip()

            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
                try:
                    date = datetime.strptime(
                        raw_date, fmt
                    ).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    pass

        if not date:
            date = default_date

        machine = ""
        if machine_col:
            machine = str(row.get(machine_col, "")).strip()

        result.append({
            "date": date,
            "machine_no": machine_no,
            "machine": machine,
            "games": games if games is not None else 0,
            "diff": diff,
        })

    return result


def load_31_days():
    print("=" * 70)
    print("【入力データ確認】")
    print("=" * 70)

    if not DAY11_FILE.exists():
        raise FileNotFoundError(
            f"7/11のCSVがありません:\n{DAY11_FILE}"
        )

    if not MERGED_FILE.exists():
        raise FileNotFoundError(
            f"7/12～8/10の統合CSVがありません:\n{MERGED_FILE}"
        )

    print(f"7/11: {DAY11_FILE}")
    print(f"7/12～8/10: {MERGED_FILE}")
    print()

    rows_11 = load_file(
        DAY11_FILE,
        forced_date="2026-07-11"
    )

    rows_12_810 = load_file(MERGED_FILE)

    print(f"7/11 読込件数: {len(rows_11):,}")
    print(f"7/12～8/10 読込件数: {len(rows_12_810):,}")
    print()

    # 7/11 + 7/12～8/10
    records = rows_11 + rows_12_810

    # 日付＋台番号で重複除去
    unique = {}

    for row in records:
        key = (row["date"], row["machine_no"])

        if key not in unique:
            unique[key] = row

    records = list(unique.values())

    return records


def check_quality(records):
    print("=" * 70)
    print("【31日分データ品質チェック】")
    print("=" * 70)

    by_date = defaultdict(list)

    for row in records:
        by_date[row["date"]].append(row)

    expected_dates = []

    current = datetime.strptime(
        START_DATE, "%Y-%m-%d"
    )

    end = datetime.strptime(
        END_DATE, "%Y-%m-%d"
    )

    while current <= end:
        expected_dates.append(current.strftime("%Y-%m-%d"))
        current = current.fromordinal(
            current.toordinal() + 1
        )

    missing_dates = []
    abnormal_dates = []

    for date in expected_dates:
        count = len(by_date.get(date, []))

        if count == EXPECTED_MACHINES_PER_DAY:
            print(f"OK {date} {count}台")
        else:
            print(
                f"NG {date} {count}台 "
                f"(期待={EXPECTED_MACHINES_PER_DAY})"
            )
            abnormal_dates.append(date)

        if count == 0:
            missing_dates.append(date)

    duplicate_keys = len(records)

    # 必須項目チェック
    missing_required = 0
    number_errors = 0
    games_errors = 0
    diff_errors = 0

    for row in records:
        if not row["date"] or not row["machine"]:
            missing_required += 1

        if row["machine_no"] is None:
            number_errors += 1

        if row["games"] is None:
            games_errors += 1

        if row["diff"] is None:
            diff_errors += 1

    print()
    print(f"取得日数: {len(by_date)} / {EXPECTED_DAYS}")
    print(f"総レコード数: {len(records):,}")
    print(f"想定レコード数: {EXPECTED_RECORDS:,}")
    print(f"欠落日数: {len(missing_dates)}")
    print(f"異常日数: {len(abnormal_dates)}")
    print(f"必須項目異常: {missing_required}")
    print(f"台番号変換エラー: {number_errors}")
    print(f"G数変換エラー: {games_errors}")
    print(f"差枚変換エラー: {diff_errors}")

    if (
        len(records) == EXPECTED_RECORDS
        and len(by_date) == EXPECTED_DAYS
        and not missing_dates
        and not abnormal_dates
        and missing_required == 0
        and number_errors == 0
        and games_errors == 0
        and diff_errors == 0
    ):
        print()
        print("★★★★★ 31日分データ品質 PASS ★★★★★")
        return True

    print()
    print("[WARNING] 31日分の完全一致条件を満たしていません。")
    return False


def pct(a, b):
    return round(a / b * 100, 2) if b else 0.0


def analyze(records):
    by_number = defaultdict(list)
    by_machine = defaultdict(list)
    by_date = defaultdict(list)
    by_tail = defaultdict(list)
    by_parity = defaultdict(list)
    by_weekday = defaultdict(list)

    for row in records:
        by_number[row["machine_no"]].append(row)
        by_machine[row["machine"]].append(row)
        by_date[row["date"]].append(row)

        by_tail[row["machine_no"] % 10].append(row)

        parity = (
            "偶数"
            if row["machine_no"] % 2 == 0
            else "奇数"
        )

        by_parity[parity].append(row)

        if row["date"]:
            weekday = ["月", "火", "水", "木", "金", "土", "日"][
                datetime.strptime(
                    row["date"], "%Y-%m-%d"
                ).weekday()
            ]

            by_weekday[weekday].append(row)

    # --------------------------------------------------------
    # 台番号別
    # --------------------------------------------------------

    number_rows = []

    for number, rows in by_number.items():
        diffs = [r["diff"] for r in rows]
        games = [r["games"] for r in rows]

        wins = sum(d > 0 for d in diffs)
        losses = sum(d < 0 for d in diffs)

        total = sum(diffs)
        average = total / len(diffs)

        # 暫定スコア
        average_component = average / 100
        win_component = pct(wins, len(rows))
        game_component = min(
            (sum(games) / len(games)) / 7000,
            1.2
        ) * 20

        score = (
            average_component * 0.55
            + win_component * 0.35
            + game_component * 0.10
        )

        number_rows.append({
            "台番号": number,
            "日数": len(rows),
            "総差枚": total,
            "平均差枚": round(average, 1),
            "勝ち日": wins,
            "負け日": losses,
            "勝率": round(pct(wins, len(rows)), 2),
            "平均G数": round(sum(games) / len(games), 1),
            "最大差枚": max(diffs),
            "最小差枚": min(diffs),
            "スコア": round(score, 2),
        })

    number_rows.sort(
        key=lambda x: (
            -x["スコア"],
            -x["平均差枚"],
            x["台番号"]
        )
    )

    # --------------------------------------------------------
    # 機種別
    # --------------------------------------------------------

    machine_rows = []

    for machine, rows in by_machine.items():
        diffs = [r["diff"] for r in rows]
        games = [r["games"] for r in rows]

        wins = sum(d > 0 for d in diffs)

        machine_rows.append({
            "機種名": machine,
            "台数×日数": len(rows),
            "総差枚": sum(diffs),
            "平均差枚": round(
                sum(diffs) / len(diffs), 1
            ),
            "勝ち台日": wins,
            "勝率": round(
                pct(wins, len(diffs)), 2
            ),
            "平均G数": round(
                sum(games) / len(games), 1
            ),
            "最大差枚": max(diffs),
            "最小差枚": min(diffs),
        })

    machine_rows.sort(
        key=lambda x: (
            -x["平均差枚"],
            -x["勝率"]
        )
    )

    # --------------------------------------------------------
    # 日別
    # --------------------------------------------------------

    date_rows = []

    for date, rows in sorted(by_date.items()):
        diffs = [r["diff"] for r in rows]
        games = [r["games"] for r in rows]

        wins = sum(d > 0 for d in diffs)

        date_rows.append({
            "日付": date,
            "台数": len(rows),
            "総差枚": sum(diffs),
            "平均差枚": round(
                sum(diffs) / len(diffs), 1
            ),
            "勝ち台": wins,
            "勝率": round(
                pct(wins, len(rows)), 2
            ),
            "平均G数": round(
                sum(games) / len(games), 1
            ),
        })

    # --------------------------------------------------------
    # 末尾
    # --------------------------------------------------------

    tail_rows = []

    for tail, rows in sorted(by_tail.items()):
        diffs = [r["diff"] for r in rows]
        wins = sum(d > 0 for d in diffs)

        tail_rows.append({
            "末尾": tail,
            "総件数": len(rows),
            "総差枚": sum(diffs),
            "平均差枚": round(
                sum(diffs) / len(diffs), 1
            ),
            "勝率": round(
                pct(wins, len(rows)), 2
            ),
        })

    tail_rows.sort(
        key=lambda x: -x["平均差枚"]
    )

    # --------------------------------------------------------
    # 偶奇
    # --------------------------------------------------------

    parity_rows = []

    for parity in ["奇数", "偶数"]:
        rows = by_parity.get(parity, [])

        diffs = [r["diff"] for r in rows]
        wins = sum(d > 0 for d in diffs)

        parity_rows.append({
            "区分": parity,
            "件数": len(rows),
            "総差枚": sum(diffs),
            "平均差枚": round(
                sum(diffs) / len(diffs), 1
            ) if rows else 0,
            "勝率": round(
                pct(wins, len(rows)), 2
            ),
        })

    # --------------------------------------------------------
    # 曜日
    # --------------------------------------------------------

    weekday_rows = []

    for weekday in ["月", "火", "水", "木", "金", "土", "日"]:
        rows = by_weekday.get(weekday, [])

        if not rows:
            continue

        diffs = [r["diff"] for r in rows]
        wins = sum(d > 0 for d in diffs)

        weekday_rows.append({
            "曜日": weekday,
            "件数": len(rows),
            "総差枚": sum(diffs),
            "平均差枚": round(
                sum(diffs) / len(diffs), 1
            ),
            "勝率": round(
                pct(wins, len(diffs)), 2
            ),
        })

    return (
        number_rows,
        machine_rows,
        date_rows,
        tail_rows,
        parity_rows,
        weekday_rows,
    )


def write_csv(path, rows):
    if not rows:
        return

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 70)
    print("アナスロ 31日分 基礎解析【7/11追加修正版】")
    print("=" * 70)
    print()
    print(
        f"対象期間: {START_DATE} ～ {END_DATE}"
    )
    print(
        f"想定: {EXPECTED_DAYS}日 × "
        f"{EXPECTED_MACHINES_PER_DAY}台 = "
        f"{EXPECTED_RECORDS:,}件"
    )
    print()

    records = load_31_days()

    print()
    print(
        f"7/11 + 7/12～8/10 合計: "
        f"{len(records):,}件"
    )
    print()

    passed = check_quality(records)

    if not passed:
        print()
        print("品質チェックNGのため解析を中止します。")
        print("まずデータの欠落・重複を確認してください。")
        return

    (
        number_rows,
        machine_rows,
        date_rows,
        tail_rows,
        parity_rows,
        weekday_rows,
    ) = analyze(records)

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    write_csv(
        OUTPUT_DIR / "machine_number_31days.csv",
        number_rows
    )

    write_csv(
        OUTPUT_DIR / "machine_type_31days.csv",
        machine_rows
    )

    write_csv(
        OUTPUT_DIR / "date_31days.csv",
        date_rows
    )

    write_csv(
        OUTPUT_DIR / "tail_31days.csv",
        tail_rows
    )

    write_csv(
        OUTPUT_DIR / "parity_31days.csv",
        parity_rows
    )

    write_csv(
        OUTPUT_DIR / "weekday_31days.csv",
        weekday_rows
    )

    # --------------------------------------------------------
    # TOP30
    # --------------------------------------------------------

    top_file = OUTPUT_DIR / "top30_machine_numbers.txt"

    with open(
        top_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "アナスロ 31日 台番号スコア TOP30\n"
        )

        f.write("=" * 70 + "\n\n")

        for i, row in enumerate(
            number_rows[:30],
            1
        ):
            f.write(
                f"{i:2d}. "
                f"台番号={row['台番号']} "
                f"スコア={row['スコア']} "
                f"平均差枚={row['平均差枚']:+.1f} "
                f"勝率={row['勝率']:.2f}% "
                f"平均G={row['平均G数']:.0f} "
                f"日数={row['日数']}\n"
            )

    # --------------------------------------------------------
    # コンソール
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("【台番号スコア TOP30】")
    print("=" * 70)

    for i, row in enumerate(
        number_rows[:30],
        1
    ):
        print(
            f"{i:2d}. "
            f"台番号={row['台番号']} "
            f"スコア={row['スコア']} "
            f"平均差枚={row['平均差枚']:+.1f} "
            f"勝率={row['勝率']:.2f}% "
            f"平均G={row['平均G数']:.0f} "
            f"日数={row['日数']}"
        )

    print()
    print("=" * 70)
    print("【機種別 TOP20】")
    print("=" * 70)

    for i, row in enumerate(
        machine_rows[:20],
        1
    ):
        print(
            f"{i:2d}. "
            f"{row['機種名']} "
            f"平均差枚={row['平均差枚']:+.1f} "
            f"勝率={row['勝率']:.2f}% "
            f"平均G={row['平均G数']:.0f}"
        )

    print()
    print("=" * 70)
    print("【末尾別】")
    print("=" * 70)

    for row in tail_rows:
        print(
            f"末尾{row['末尾']}: "
            f"平均差枚={row['平均差枚']:+.1f} "
            f"勝率={row['勝率']:.2f}%"
        )

    print()
    print("=" * 70)
    print("【曜日別】")
    print("=" * 70)

    for row in weekday_rows:
        print(
            f"{row['曜日']}曜: "
            f"平均差枚={row['平均差枚']:+.1f} "
            f"勝率={row['勝率']:.2f}%"
        )

    print()
    print("=" * 70)
    print("★★★★★ 31日分基礎解析 完了 ★★★★★")
    print("=" * 70)
    print()
    print("解析結果保存先:")
    print(OUTPUT_DIR)
    print()
    print("次の解析:")
    print("・台番号の連番/並び")
    print("・前日→翌日の差枚関係")
    print("・曜日×台番号")
    print("・特定日候補")
    print("・狙い台スコア精密化")


if __name__ == "__main__":
    main()
