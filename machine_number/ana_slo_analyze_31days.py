# -*- coding: utf-8 -*-
"""
アナスロ 31日分 基礎解析
対象期間: 2026-07-11 ～ 2026-08-10

目的:
1. 台番号別の31日集計
2. 機種別の31日集計
3. 曜日別集計
4. 台番号の末尾・偶奇集計
5. 日別集計
6. 連番・並びの基礎集計
7. 実戦向け「台番号スコア」の作成

入力:
data/maruhan_maebashi/machine_number/ana_slo_20260712_20260810.csv
および 20260711 の日別CSV

※既存CSVは変更しません。
"""

from pathlib import Path
import csv
import re
from collections import defaultdict
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "maruhan_maebashi" / "machine_number"

MERGED_CANDIDATES = [
    DATA_DIR / "ana_slo_20260712_20260810.csv",
    DATA_DIR / "ana_slo_20260712_20260810" / "ana_slo_20260712_20260810.csv",
]

OUTPUT_DIR = DATA_DIR / "analysis_31days"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2026-07-11"
END_DATE = "2026-08-10"


def find_input():
    for p in MERGED_CANDIDATES:
        if p.exists():
            return p

    # 既存の日別CSVから再構築
    files = sorted(DATA_DIR.glob("ana_slo_*.csv"))
    files = [
        p for p in files
        if re.fullmatch(r"ana_slo_\d{8}\.csv", p.name)
    ]
    if files:
        return files

    raise FileNotFoundError(
        "統合CSVまたは日別CSVが見つかりません。\n"
        f"確認場所: {DATA_DIR}"
    )


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_key(s):
    return re.sub(r"\s+", "", str(s or "")).lower()


def find_col(headers, candidates):
    normalized = {normalize_key(h): h for h in headers}
    for c in candidates:
        nc = normalize_key(c)
        if nc in normalized:
            return normalized[nc]
    for h in headers:
        nh = normalize_key(h)
        for c in candidates:
            if normalize_key(c) in nh or nh in normalize_key(c):
                return h
    return None


def to_int(v):
    if v is None:
        return None
    s = str(v).strip().replace(",", "").replace("+", "")
    if s in ("", "-", "--", "None"):
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def parse_date(row, headers):
    date_col = find_col(headers, ["日付", "date", "対象日"])
    if date_col and row.get(date_col):
        s = row[date_col].strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
            try:
                return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
            except ValueError:
                pass

    # CSV名由来の日付がない場合は空
    return ""


def load_records():
    src = find_input()

    if isinstance(src, list):
        paths = src
    else:
        paths = [src]

    records = []

    for path in paths:
        rows = read_csv(path)
        headers = rows[0].keys() if rows else []

        machine_col = find_col(headers, ["機種名", "機種", "machine", "machine_name"])
        number_col = find_col(headers, ["台番号", "台番", "台No", "台NO", "machine_no", "machine_number"])
        games_col = find_col(headers, ["G数", "G数", "ゲーム数", "games", "game"])
        diff_col = find_col(headers, ["差枚", "差枚数", "差枚数枚", "差枚数(枚)", "diff"])

        if not number_col or not diff_col:
            raise ValueError(
                f"必要列を認識できません: {path}\n"
                f"列: {list(headers)}"
            )

        # 統合CSVなら日付列を使う。日別CSVならファイル名から取得。
        filename_date = ""
        m = re.search(r"ana_slo_(\d{8})\.csv$", path.name)
        if m:
            filename_date = datetime.strptime(m.group(1), "%Y%m%d").strftime("%Y-%m-%d")

        for row in rows:
            machine_no = to_int(row.get(number_col))
            diff = to_int(row.get(diff_col))
            games = to_int(row.get(games_col)) if games_col else None

            if machine_no is None or diff is None:
                continue

            date = parse_date(row, headers) or filename_date
            machine = (row.get(machine_col, "") if machine_col else "").strip()

            records.append({
                "date": date,
                "machine_no": machine_no,
                "machine": machine,
                "games": games if games is not None else 0,
                "diff": diff,
            })

    # 日付範囲のみ
    records = [
        r for r in records
        if START_DATE <= r["date"] <= END_DATE
    ]

    # 日付＋台番号で重複除去（同一なら最初を採用）
    unique = {}
    for r in records:
        unique.setdefault((r["date"], r["machine_no"]), r)

    return list(unique.values()), src


def write_csv(path, rows, headers):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def pct(x, y):
    return round(x / y * 100, 2) if y else 0.0


def analyze(records):
    by_number = defaultdict(list)
    by_machine = defaultdict(list)
    by_date = defaultdict(list)
    by_tail = defaultdict(list)
    by_parity = defaultdict(list)
    by_weekday = defaultdict(list)

    for r in records:
        by_number[r["machine_no"]].append(r)
        by_machine[r["machine"]].append(r)
        by_date[r["date"]].append(r)

        tail = r["machine_no"] % 10
        by_tail[tail].append(r)

        parity = "偶数" if r["machine_no"] % 2 == 0 else "奇数"
        by_parity[parity].append(r)

        if r["date"]:
            wd = ["月", "火", "水", "木", "金", "土", "日"][
                datetime.strptime(r["date"], "%Y-%m-%d").weekday()
            ]
            by_weekday[wd].append(r)

    # 台番号別
    number_rows = []
    for no, rows in by_number.items():
        diffs = [r["diff"] for r in rows]
        games = [r["games"] for r in rows]
        wins = sum(d > 0 for d in diffs)
        losses = sum(d < 0 for d in diffs)
        total = sum(diffs)
        avg = total / len(diffs)

        # 実戦用スコア:
        # 平均差枚だけに偏らないよう、勝率・平均差枚・稼働を穏やかに加味。
        avg_score = avg / 100.0
        win_score = (wins / len(rows)) * 100.0
        game_score = min(sum(games) / len(games) / 7000.0, 1.2) * 20.0
        score = avg_score * 0.55 + win_score * 0.35 + game_score * 0.10

        number_rows.append({
            "台番号": no,
            "日数": len(rows),
            "総差枚": total,
            "平均差枚": round(avg, 1),
            "勝ち日": wins,
            "負け日": losses,
            "勝率": round(pct(wins, len(rows)), 2),
            "平均G数": round(sum(games) / len(games), 1),
            "最大差枚": max(diffs),
            "最小差枚": min(diffs),
            "プラス率": round(pct(sum(d > 0 for d in diffs), len(diffs)), 2),
            "スコア": round(score, 2),
        })

    number_rows.sort(key=lambda x: (-x["スコア"], -x["平均差枚"], x["台番号"]))

    # 機種別
    machine_rows = []
    for machine, rows in by_machine.items():
        diffs = [r["diff"] for r in rows]
        games = [r["games"] for r in rows]
        wins = sum(d > 0 for d in diffs)
        machine_rows.append({
            "機種名": machine,
            "台数×日数": len(rows),
            "総差枚": sum(diffs),
            "平均差枚": round(sum(diffs) / len(diffs), 1),
            "勝ち台日": wins,
            "勝率": round(pct(wins, len(diffs)), 2),
            "平均G数": round(sum(games) / len(games), 1),
            "最大差枚": max(diffs),
            "最小差枚": min(diffs),
        })
    machine_rows.sort(key=lambda x: (-x["平均差枚"], -x["勝率"]))

    # 日別
    date_rows = []
    for date, rows in sorted(by_date.items()):
        diffs = [r["diff"] for r in rows]
        games = [r["games"] for r in rows]
        wins = sum(d > 0 for d in diffs)
        date_rows.append({
            "日付": date,
            "台数": len(rows),
            "総差枚": sum(diffs),
            "平均差枚": round(sum(diffs) / len(diffs), 1),
            "勝ち台": wins,
            "勝率": round(pct(wins, len(rows)), 2),
            "平均G数": round(sum(games) / len(games), 1),
        })

    # 末尾
    tail_rows = []
    for tail, rows in sorted(by_tail.items()):
        diffs = [r["diff"] for r in rows]
        wins = sum(d > 0 for d in diffs)
        tail_rows.append({
            "末尾": tail,
            "総件数": len(rows),
            "総差枚": sum(diffs),
            "平均差枚": round(sum(diffs) / len(diffs), 1),
            "勝率": round(pct(wins, len(rows)), 2),
        })
    tail_rows.sort(key=lambda x: -x["平均差枚"])

    # 偶奇
    parity_rows = []
    for parity in ["奇数", "偶数"]:
        rows = by_parity.get(parity, [])
        diffs = [r["diff"] for r in rows]
        wins = sum(d > 0 for d in diffs)
        parity_rows.append({
            "区分": parity,
            "件数": len(rows),
            "総差枚": sum(diffs),
            "平均差枚": round(sum(diffs) / len(diffs), 1) if rows else 0,
            "勝率": round(pct(wins, len(rows)), 2),
        })

    # 曜日
    weekday_rows = []
    for wd in ["月", "火", "水", "木", "金", "土", "日"]:
        rows = by_weekday.get(wd, [])
        if not rows:
            continue
        diffs = [r["diff"] for r in rows]
        wins = sum(d > 0 for d in diffs)
        weekday_rows.append({
            "曜日": wd,
            "件数": len(rows),
            "総差枚": sum(diffs),
            "平均差枚": round(sum(diffs) / len(diffs), 1),
            "勝率": round(pct(wins, len(rows)), 2),
        })

    return number_rows, machine_rows, date_rows, tail_rows, parity_rows, weekday_rows


def main():
    print("=" * 70)
    print("アナスロ 31日分 基礎解析")
    print("=" * 70)
    print(f"対象期間: {START_DATE} ～ {END_DATE}")
    print()

    records, source = load_records()
    print(f"入力: {source}")
    print(f"解析レコード: {len(records):,}件")

    expected = 31 * 514
    if len(records) != expected:
        print(f"[WARNING] 想定 {expected:,}件に対して {len(records):,}件です。")

    (
        number_rows,
        machine_rows,
        date_rows,
        tail_rows,
        parity_rows,
        weekday_rows,
    ) = analyze(records)

    write_csv(
        OUTPUT_DIR / "machine_number_31days.csv",
        number_rows,
        list(number_rows[0].keys()),
    )
    write_csv(
        OUTPUT_DIR / "machine_type_31days.csv",
        machine_rows,
        list(machine_rows[0].keys()),
    )
    write_csv(
        OUTPUT_DIR / "date_31days.csv",
        date_rows,
        list(date_rows[0].keys()),
    )
    write_csv(
        OUTPUT_DIR / "tail_31days.csv",
        tail_rows,
        list(tail_rows[0].keys()),
    )
    write_csv(
        OUTPUT_DIR / "parity_31days.csv",
        parity_rows,
        list(parity_rows[0].keys()),
    )
    write_csv(
        OUTPUT_DIR / "weekday_31days.csv",
        weekday_rows,
        list(weekday_rows[0].keys()),
    )

    # 上位30台
    with open(OUTPUT_DIR / "top30_machine_numbers.txt", "w", encoding="utf-8") as f:
        f.write("アナスロ 31日 台番号スコア TOP30\n")
        f.write("=" * 70 + "\n\n")
        for i, r in enumerate(number_rows[:30], 1):
            f.write(
                f"{i:2d}. 台番号 {r['台番号']} "
                f"スコア={r['スコア']} "
                f"平均差枚={r['平均差枚']:+.1f} "
                f"勝率={r['勝率']:.2f}% "
                f"平均G={r['平均G数']:.0f}\n"
            )

    # コンソール表示
    print()
    print("=" * 70)
    print("【台番号スコア TOP30】")
    print("=" * 70)
    for i, r in enumerate(number_rows[:30], 1):
        print(
            f"{i:2d}. 台番号={r['台番号']} "
            f"スコア={r['スコア']} "
            f"平均差枚={r['平均差枚']:+.1f} "
            f"勝率={r['勝率']:.2f}% "
            f"平均G={r['平均G数']:.0f}"
        )

    print()
    print("=" * 70)
    print("【機種別 TOP20】")
    print("=" * 70)
    for i, r in enumerate(machine_rows[:20], 1):
        print(
            f"{i:2d}. {r['機種名']} "
            f"平均差枚={r['平均差枚']:+.1f} "
            f"勝率={r['勝率']:.2f}% "
            f"平均G={r['平均G数']:.0f}"
        )

    print()
    print("=" * 70)
    print("【末尾別】")
    print("=" * 70)
    for r in tail_rows:
        print(
            f"末尾{r['末尾']}: "
            f"平均差枚={r['平均差枚']:+.1f} "
            f"勝率={r['勝率']:.2f}%"
        )

    print()
    print("=" * 70)
    print("【曜日別】")
    print("=" * 70)
    for r in weekday_rows:
        print(
            f"{r['曜日']}曜: "
            f"平均差枚={r['平均差枚']:+.1f} "
            f"勝率={r['勝率']:.2f}%"
        )

    print()
    print("=" * 70)
    print("★★★★★ 基礎解析完了 ★★★★★")
    print("=" * 70)
    print()
    print(f"解析結果保存先:")
    print(OUTPUT_DIR)
    print()
    print("次段階では、この結果を使って")
    print("・台番号の連番/並び")
    print("・前日→翌日の差枚関係")
    print("・曜日×台番号")
    print("・特定日候補")
    print("・狙い台スコアの精密化")
    print("を分析できます。")


if __name__ == "__main__":
    main()
