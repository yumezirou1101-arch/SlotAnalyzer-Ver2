import csv
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "maruhan_maebashi" / "machine_number"

START_DATE = date(2026, 7, 11)
END_DATE = date(2026, 8, 10)
EXPECTED_MACHINES_PER_DAY = 514

OUTPUT_CSV = DATA_DIR / "ana_slo_20260711_20260810.csv"
REPORT_FILE = DATA_DIR / "ana_slo_20260711_20260810_quality_report.txt"

REQUIRED_COLUMNS = [
    "機種名", "台番号", "G数", "差枚", "BB", "RB",
    "合成確率", "BB確率", "RB確率"
]

def dates_between(start, end):
    result = []
    d = start
    while d <= end:
        result.append(d)
        d += timedelta(days=1)
    return result

def read_csv(path):
    last_error = None
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                return reader.fieldnames or [], list(reader)
        except UnicodeDecodeError as e:
            last_error = e
    raise last_error

def text(v):
    return "" if v is None else str(v).strip()

def to_int(v):
    s = text(v).replace(",", "").replace("+", "")
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None

def main():
    dates = dates_between(START_DATE, END_DATE)
    expected_total = len(dates) * EXPECTED_MACHINES_PER_DAY

    print("=" * 70)
    print("アナスロ 31日分統合＋データ品質チェック")
    print("=" * 70)
    print(f"対象期間: {START_DATE} ～ {END_DATE}")
    print(f"対象日数: {len(dates)}日")
    print(f"想定総レコード数: {expected_total:,}")
    print()

    if not DATA_DIR.exists():
        print("[ERROR] データフォルダがありません:")
        print(DATA_DIR)
        return

    all_rows = []
    daily_counts = {}
    missing_dates = []
    invalid_dates = []
    seen_keys = set()
    duplicate_keys = []
    machine_numbers = defaultdict(list)
    missing_by_column = Counter()

    print("【日別CSV確認】")
    print("-" * 70)

    for d in dates:
        ds = d.strftime("%Y%m%d")
        path = DATA_DIR / f"ana_slo_{ds}.csv"

        if not path.exists():
            print(f"× {d} CSVなし")
            missing_dates.append(str(d))
            continue

        try:
            fieldnames, rows = read_csv(path)
        except Exception as e:
            print(f"× {d} 読み込み失敗: {e}")
            invalid_dates.append(str(d))
            continue

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing_cols:
            print(f"× {d} 必須列不足: {', '.join(missing_cols)}")
            invalid_dates.append(str(d))
            continue

        daily_counts[str(d)] = len(rows)
        status = "OK" if len(rows) == EXPECTED_MACHINES_PER_DAY else "WARNING"
        print(f"{status} {d} {len(rows)}台")

        for row in rows:
            new_row = {
                "日付": str(d),
                "機種名": text(row.get("機種名")),
                "台番号": text(row.get("台番号")),
                "G数": text(row.get("G数")),
                "差枚": text(row.get("差枚")),
                "BB": text(row.get("BB")),
                "RB": text(row.get("RB")),
                "合成確率": text(row.get("合成確率")),
                "BB確率": text(row.get("BB確率")),
                "RB確率": text(row.get("RB確率")),
            }

            for col in REQUIRED_COLUMNS:
                if not new_row[col]:
                    missing_by_column[col] += 1

            key = (new_row["日付"], new_row["台番号"])
            if key in seen_keys:
                duplicate_keys.append(key)
            seen_keys.add(key)

            if new_row["台番号"]:
                machine_numbers[str(d)].append(new_row["台番号"])

            all_rows.append(new_row)

    print()
    print("=" * 70)
    print("【統合結果】")
    print("=" * 70)
    print(f"取得日数: {len(daily_counts)} / {len(dates)}")
    print(f"総レコード数: {len(all_rows):,}")
    print(f"想定レコード数: {expected_total:,}")
    print(f"欠落日数: {len(missing_dates)}")
    print(f"異常CSV: {len(invalid_dates)}")

    print()
    print("【台番号チェック】")
    bad_machine_days = 0
    for d in dates:
        ds = str(d)
        if ds not in machine_numbers:
            continue
        unique_count = len(set(machine_numbers[ds]))
        if unique_count != EXPECTED_MACHINES_PER_DAY:
            bad_machine_days += 1
            print(f"WARNING {ds}: ユニーク台番号 {unique_count}台")

    if bad_machine_days == 0:
        print("★ 全日514台のユニーク台番号")

    print()
    print("【欠損チェック】")
    total_missing = sum(missing_by_column.values())
    if total_missing == 0:
        print("★ 必須項目の空欄なし")
    else:
        print(f"WARNING: 空欄 {total_missing:,}件")
        for col in REQUIRED_COLUMNS:
            if missing_by_column[col]:
                print(f"  {col}: {missing_by_column[col]:,}")

    invalid_machine = sum(
        1 for r in all_rows if to_int(r["台番号"]) is None
    )
    invalid_g = sum(
        1 for r in all_rows if to_int(r["G数"]) is None
    )
    invalid_diff = 0
    for r in all_rows:
        s = text(r["差枚"]).replace(",", "")
        if s:
            try:
                int(s)
            except ValueError:
                invalid_diff += 1

    print()
    print("【数値チェック】")
    print(f"台番号変換エラー: {invalid_machine}")
    print(f"G数変換エラー: {invalid_g}")
    print(f"差枚変換エラー: {invalid_diff}")

    print()
    print("【重複チェック】")
    print(f"日付＋台番号 重複: {len(duplicate_keys)}件")

    header = [
        "日付", "機種名", "台番号", "G数", "差枚",
        "BB", "RB", "合成確率", "BB確率", "RB確率"
    ]

    print()
    print("【統合CSV保存】")
    if OUTPUT_CSV.exists():
        print("[WARNING] 統合CSVは既に存在します。上書きしません。")
        print(OUTPUT_CSV)
    else:
        with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(all_rows)
        print("★ 統合CSV保存成功")
        print(OUTPUT_CSV)

    quality_ok = (
        len(daily_counts) == len(dates)
        and len(all_rows) == expected_total
        and bad_machine_days == 0
        and total_missing == 0
        and invalid_machine == 0
        and invalid_g == 0
        and invalid_diff == 0
        and len(duplicate_keys) == 0
    )

    report = []
    report.append("アナスロ 31日分データ品質チェック")
    report.append("=" * 60)
    report.append(f"対象期間: {START_DATE} ～ {END_DATE}")
    report.append(f"取得日数: {len(daily_counts)} / {len(dates)}")
    report.append(f"総レコード数: {len(all_rows):,}")
    report.append(f"想定レコード数: {expected_total:,}")
    report.append("")
    report.append("【日別件数】")
    for d in dates:
        ds = str(d)
        report.append(f"{ds}: {daily_counts.get(ds, 0)}台")
    report.append("")
    report.append(f"欠落日: {', '.join(missing_dates) if missing_dates else 'なし'}")
    report.append(f"異常CSV: {', '.join(invalid_dates) if invalid_dates else 'なし'}")
    report.append(f"日付＋台番号重複: {len(duplicate_keys)}件")
    report.append(f"必須項目空欄: {total_missing}件")
    report.append(f"台番号変換エラー: {invalid_machine}件")
    report.append(f"G数変換エラー: {invalid_g}件")
    report.append(f"差枚変換エラー: {invalid_diff}件")
    report.append("")
    report.append("総合判定: PASS" if quality_ok else "総合判定: WARNING")

    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

    print()
    print("=" * 70)
    if quality_ok:
        print("★★★★★ データ品質チェック PASS ★★★★★")
        print("31日分・15,934レコードが想定どおり揃っています。")
    else:
        print("★★★★★ データ品質チェック WARNING ★★★★★")
        print("上記の項目を確認してください。")

    print()
    print("品質レポート:")
    print(REPORT_FILE)

if __name__ == "__main__":
    main()
