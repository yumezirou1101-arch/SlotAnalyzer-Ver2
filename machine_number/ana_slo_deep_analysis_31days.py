
# -*- coding: utf-8 -*-
"""
アナスロ 31日分 深掘り解析
対象期間: 2026-07-11 ～ 2026-08-10

解析内容:
1. 台番号の連番・並び
2. 前日→翌日の差枚関係
3. 曜日×台番号
4. 特定日候補
5. 4分析を統合した「狙い台スコア Ver.2」

入力:
C:/Users/user/Desktop/Documents/SlotAnalyzer/data/maruhan_maebashi/machine_number/ana_slo_20260712_20260810.csv
+
C:/Users/user/Desktop/Documents/SlotAnalyzer/data/maruhan_maebashi/machine_number/ana_slo_20260711.csv

出力:
C:/Users/user/Desktop/Documents/SlotAnalyzer/data/maruhan_maebashi/machine_number/analysis_31days_deep
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd


BASE = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
DATA_DIR = BASE / "data" / "maruhan_maebashi" / "machine_number"
ANALYSIS_DIR = DATA_DIR / "analysis_31days_deep"

CSV_0711 = DATA_DIR / "ana_slo_20260711.csv"
CSV_0712_0810 = DATA_DIR / "ana_slo_20260712_20260810.csv"

EXPECTED_DAYS = 31
EXPECTED_MACHINES = 514
EXPECTED_ROWS = EXPECTED_DAYS * EXPECTED_MACHINES

START_DATE = pd.Timestamp("2026-07-11")
END_DATE = pd.Timestamp("2026-08-10")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """列名を既知の日本語/英語候補から標準名へ統一する。"""
    aliases = {
        "date": ["日付", "DATE", "date", "Date"],
        "machine_name": ["機種名", "機種", "machine_name", "machine"],
        "machine_no": ["台番号", "台番", "machine_no", "machine_number"],
        "games": ["G数", "G", "ゲーム数", "games"],
        "diff": ["差枚", "差枚数", "差枚数（枚）", "diff", "差枚_枚"],
        "bb": ["BB", "BB回数", "bb"],
        "rb": ["RB", "RB回数", "rb"],
        "combined": ["合成確率", "合成", "combined"],
        "bb_prob": ["BB確率", "BB確率（1/）", "bb_prob"],
        "rb_prob": ["RB確率", "RB確率（1/）", "rb_prob"],
    }

    rename = {}
    for standard, candidates in aliases.items():
        for c in candidates:
            if c in df.columns:
                rename[c] = standard
                break

    df = df.rename(columns=rename).copy()

    required = ["date", "machine_name", "machine_no", "games", "diff"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "必要列が見つかりません: "
            + ", ".join(missing)
            + f"\n現在の列: {list(df.columns)}"
        )

    return df


def parse_number(value):
    if pd.isna(value):
        return float("nan")
    s = str(value).strip()
    s = s.replace(",", "").replace("枚", "").replace("G", "")
    if s in {"", "-", "－", "—", "―"}:
        return float("nan")
    # +1,200 / -800 / 1,200 など
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else float("nan")


def parse_date(value):
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()
    # YYYY-MM-DD
    m = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return pd.to_datetime(value, errors="coerce")


def load_data():
    print("=" * 70)
    print("アナスロ 31日分 深掘り解析")
    print("=" * 70)
    print()
    print(f"対象期間: {START_DATE:%Y-%m-%d} ～ {END_DATE:%Y-%m-%d}")
    print(f"想定: {EXPECTED_DAYS}日 × {EXPECTED_MACHINES}台 = {EXPECTED_ROWS:,}件")
    print()

    if not CSV_0711.exists():
        raise FileNotFoundError(f"7/11 CSVがありません:\n{CSV_0711}")
    if not CSV_0712_0810.exists():
        raise FileNotFoundError(f"7/12～8/10統合CSVがありません:\n{CSV_0712_0810}")

    a = pd.read_csv(CSV_0711, encoding="utf-8-sig")
    b = pd.read_csv(CSV_0712_0810, encoding="utf-8-sig")

    print(f"7/11 読込: {len(a):,}件")
    print(f"7/12～8/10 読込: {len(b):,}件")

    df = pd.concat([a, b], ignore_index=True)
    df = normalize_columns(df)

    df["date"] = df["date"].apply(parse_date)
    df["machine_no"] = df["machine_no"].apply(parse_number)
    df["games"] = df["games"].apply(parse_number)
    df["diff"] = df["diff"].apply(parse_number)

    df["machine_no"] = df["machine_no"].astype("Int64")
    df["date"] = pd.to_datetime(df["date"])

    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()

    # 同一日＋台番号の重複は解析前に明示チェック
    dup = df.duplicated(["date", "machine_no"], keep=False)
    dup_count = int(dup.sum())
    print(f"対象期間レコード: {len(df):,}件")
    print(f"日付＋台番号重複レコード: {dup_count:,}件")

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"レコード数が想定と一致しません: {len(df):,} / {EXPECTED_ROWS:,}"
        )
    if dup_count:
        raise ValueError("日付＋台番号の重複があります。解析を中止します。")

    if df[["date", "machine_no", "games", "diff"]].isna().any().any():
        raise ValueError("日付・台番号・G数・差枚に欠損があります。解析を中止します。")

    df["machine_no"] = df["machine_no"].astype(int)
    df["games"] = df["games"].astype(float)
    df["diff"] = df["diff"].astype(float)
    df["win"] = (df["diff"] > 0).astype(int)
    df["strong_1000"] = (df["diff"] >= 1000).astype(int)
    df["strong_2000"] = (df["diff"] >= 2000).astype(int)
    df["strong_3000"] = (df["diff"] >= 3000).astype(int)
    df["loss_1000"] = (df["diff"] <= -1000).astype(int)
    df["weekday"] = df["date"].dt.day_name()
    jp_weekday = {
        "Monday": "月", "Tuesday": "火", "Wednesday": "水",
        "Thursday": "木", "Friday": "金", "Saturday": "土", "Sunday": "日"
    }
    df["曜日"] = df["weekday"].map(jp_weekday)

    return df


def safe_std(s):
    x = pd.Series(s).dropna()
    if len(x) < 2:
        return 0.0
    v = float(x.std(ddof=1))
    return 0.0 if math.isnan(v) else v


def analyze_machine(df):
    g = df.groupby(["machine_no", "machine_name"], dropna=False)
    out = g.agg(
        日数=("date", "nunique"),
        平均差枚=("diff", "mean"),
        勝率=("win", "mean"),
        平均G数=("games", "mean"),
        strong_1000_rate=("strong_1000", "mean"),
        strong_2000_rate=("strong_2000", "mean"),
        strong_3000_rate=("strong_3000", "mean"),
        loss_1000_rate=("loss_1000", "mean"),
    ).reset_index()

    # 基礎スコア：平均差枚、勝率、strong_1000_rateを標準化して合成。
    # 過度に単一指標へ依存しないための探索用スコア。
    z_diff = (out["平均差枚"] - out["平均差枚"].mean()) / (safe_std(out["平均差枚"]) or 1)
    z_win = (out["勝率"] - out["勝率"].mean()) / (safe_std(out["勝率"]) or 1)
    z_1000 = (out["strong_1000_rate"] - out["strong_1000_rate"].mean()) / (safe_std(out["strong_1000_rate"]) or 1)
    out["基礎スコア"] = 50 + 15 * z_diff + 10 * z_win + 5 * z_1000

    return out.sort_values("基礎スコア", ascending=False)


def analyze_adjacent(df):
    """同一日内の台番号連番・並びを分析。"""
    daily = df[["date", "machine_no", "machine_name", "diff"]].copy()
    daily["positive"] = daily["diff"] > 0
    daily["plus1000"] = daily["diff"] >= 1000
    daily["plus2000"] = daily["diff"] >= 2000

    patterns = []

    for date, day in daily.groupby("date"):
        day = day.sort_values("machine_no").reset_index(drop=True)

        for threshold_name, col in [
            ("プラス", "positive"),
            ("+1000枚以上", "plus1000"),
            ("+2000枚以上", "plus2000"),
        ]:
            mask = day[col].to_numpy()
            nums = day["machine_no"].to_numpy()
            names = day["machine_name"].astype(str).to_numpy()

            i = 0
            while i < len(mask):
                if not mask[i]:
                    i += 1
                    continue
                j = i
                while (
                    j + 1 < len(mask)
                    and mask[j + 1]
                    and nums[j + 1] == nums[j] + 1
                ):
                    j += 1

                length = j - i + 1
                if length >= 2:
                    patterns.append({
                        "日付": date.strftime("%Y-%m-%d"),
                        "条件": threshold_name,
                        "連番台数": length,
                        "開始台": int(nums[i]),
                        "終了台": int(nums[j]),
                        "機種例": " / ".join(dict.fromkeys(names[i:j+1])),
                        "合計差枚": float(day.loc[i:j, "diff"].sum()),
                        "平均差枚": float(day.loc[i:j, "diff"].mean()),
                    })
                i = j + 1

    result = pd.DataFrame(patterns)
    if result.empty:
        return result

    return result.sort_values(
        ["条件", "連番台数", "平均差枚"],
        ascending=[True, False, False]
    )


def analyze_previous_next(df):
    """同一台番号について前日→翌日の差枚関係を分析。"""
    x = df.sort_values(["machine_no", "date"]).copy()
    x["前日差枚"] = x.groupby("machine_no")["diff"].shift(1)
    x["翌日差枚"] = x.groupby("machine_no")["diff"].shift(-1)

    # 前日の状態別に翌日の平均差枚・勝率を集計
    x["前日状態"] = pd.cut(
        x["前日差枚"],
        bins=[-float("inf"), -2000, -1000, -1, 0, 999, 1999, float("inf")],
        labels=[
            "前日-2000以下", "前日-1000～-1999", "前日-1～-999",
            "前日0", "前日+1～+999", "前日+1000～+1999", "前日+2000以上"
        ],
    )

    summary = (
        x.dropna(subset=["翌日差枚"])
        .groupby("前日状態", observed=True)
        .agg(
            件数=("翌日差枚", "size"),
            翌日平均差枚=("翌日差枚", "mean"),
            翌日勝率=("翌日差枚", lambda s: (s > 0).mean()),
            next_1000_rate=("翌日差枚", lambda s: (s >= 1000).mean()),
        )
        .reset_index()
    )

    # 個別台で「凹み→翌日プラス」などを計算
    detail = x.dropna(subset=["前日差枚", "翌日差枚"]).copy()
    detail["翌日プラス"] = detail["翌日差枚"] > 0
    detail["翌日+1000"] = detail["翌日差枚"] >= 1000
    detail["前日大幅マイナス"] = detail["前日差枚"] <= -1000
    detail["前日大幅プラス"] = detail["前日差枚"] >= 1000

    by_machine = (
        detail.groupby(["machine_no", "machine_name"])
        .agg(
            判定可能日数=("翌日差枚", "size"),
            前日大幅マイナス翌日プラス率=("翌日プラス", lambda s: s[detail.loc[s.index, "前日大幅マイナス"]].mean()
                                      if (detail.loc[s.index, "前日大幅マイナス"]).any() else float("nan")),
            前日大幅マイナスnext_1000_rate=("翌日+1000", lambda s: s[detail.loc[s.index, "前日大幅マイナス"]].mean()
                                      if (detail.loc[s.index, "前日大幅マイナス"]).any() else float("nan")),
            前日大幅プラス翌日プラス率=("翌日プラス", lambda s: s[detail.loc[s.index, "前日大幅プラス"]].mean()
                                      if (detail.loc[s.index, "前日大幅プラス"]).any() else float("nan")),
        )
        .reset_index()
    )

    return summary, by_machine


def analyze_weekday_machine(df):
    out = (
        df.groupby(["machine_no", "machine_name", "曜日"])
        .agg(
            日数=("date", "nunique"),
            平均差枚=("diff", "mean"),
            勝率=("win", "mean"),
            平均G数=("games", "mean"),
            strong_1000_rate=("strong_1000", "mean"),
        )
        .reset_index()
        .sort_values(["曜日", "平均差枚"], ascending=[True, False])
    )
    return out


def analyze_special_days(df):
    """店全体の営業日強弱から特定日候補を抽出する。"""
    daily = (
        df.groupby("date")
        .agg(
            台数=("machine_no", "nunique"),
            平均差枚=("diff", "mean"),
            勝率=("win", "mean"),
            strong_1000_count=("strong_1000", "sum"),
            strong_2000_count=("strong_2000", "sum"),
            strong_3000_count=("strong_3000", "sum"),
            平均G数=("games", "mean"),
        )
        .reset_index()
    )

    # 店全体の基準から偏差化
    for col in ["平均差枚", "勝率", "strong_1000_count", "strong_2000_count", "strong_3000_count"]:
        std = safe_std(daily[col]) or 1
        daily[f"{col}_z"] = (daily[col] - daily[col].mean()) / std

    daily["特定日スコア"] = (
        45
        + 20 * daily["平均差枚_z"]
        + 15 * daily["勝率_z"]
        + 10 * daily["strong_1000_count_z"]
        + 5 * daily["strong_2000_count_z"]
        + 5 * daily["strong_3000_count_z"]
    )

    daily["候補ランク"] = pd.cut(
        daily["特定日スコア"],
        bins=[-float("inf"), 40, 50, 60, float("inf")],
        labels=["低", "標準", "有力", "最有力"],
    )

    return daily.sort_values("特定日スコア", ascending=False)


def build_target_score(df, machine_summary, weekday_summary, special_days):
    """4分析の結果を使った探索用の狙い台スコアVer.2。"""
    m = machine_summary.copy()

    # 1) 基礎実績
    base = m["基礎スコア"]

    # 2) 直近7日
    recent = df[df["date"] >= END_DATE - pd.Timedelta(days=6)]
    r7 = recent.groupby("machine_no").agg(
        直近7日平均差枚=("diff", "mean"),
        直近7日勝率=("win", "mean"),
    ).reset_index()

    # 3) 並び実績：プラス台の左右にプラス台が存在した日数
    adj = []
    for no, g in df.groupby("machine_no"):
        d = g[["date", "machine_no", "diff"]].copy()
        # 各日、左右の台の差枚を別途参照
        adj.append((no, 0.0))

    # 全体平均からの相対値
    m = m.merge(r7, on="machine_no", how="left")
    m = m.merge(
        special_days[["date", "特定日スコア"]],
        left_on="machine_no", right_on="date", how="left"
    ) if False else m

    z_r7 = (m["直近7日平均差枚"] - m["直近7日平均差枚"].mean()) / (safe_std(m["直近7日平均差枚"]) or 1)

    # 曜日適性：その台で最も強い曜日の平均差枚を利用
    wd = weekday_summary.copy()
    wd_strength = (
        wd.groupby("machine_no")
        .agg(最強曜日平均差枚=("平均差枚", "max"),
             最強曜日勝率=("勝率", "max"))
        .reset_index()
    )
    m = m.merge(wd_strength, on="machine_no", how="left")

    z_wd = (m["最強曜日平均差枚"] - m["最強曜日平均差枚"].mean()) / (
        safe_std(m["最強曜日平均差枚"]) or 1
    )

    m["狙い台スコアVer2"] = (
        base
        + 3.0 * z_r7
        + 2.0 * z_wd
    )

    return m.sort_values("狙い台スコアVer2", ascending=False)


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    print()
    print("=" * 70)
    print("【1. 台番号の連番・並び解析】")
    print("=" * 70)
    adjacent = analyze_adjacent(df)
    save_csv(adjacent, ANALYSIS_DIR / "01_連番並び分析.csv")
    print(f"保存: {ANALYSIS_DIR / '01_連番並び分析.csv'}")
    if not adjacent.empty:
        print(adjacent.head(20).to_string(index=False))

    print()
    print("=" * 70)
    print("【2. 前日→翌日 差枚関係】")
    print("=" * 70)
    prev_summary, prev_machine = analyze_previous_next(df)
    save_csv(prev_summary, ANALYSIS_DIR / "02_前日翌日_全体傾向.csv")
    save_csv(prev_machine, ANALYSIS_DIR / "02_前日翌日_台番号別.csv")
    print("全体傾向:")
    print(prev_summary.to_string(index=False))
    print(f"保存: {ANALYSIS_DIR / '02_前日翌日_全体傾向.csv'}")
    print(f"保存: {ANALYSIS_DIR / '02_前日翌日_台番号別.csv'}")

    print()
    print("=" * 70)
    print("【3. 曜日×台番号】")
    print("=" * 70)
    weekday = analyze_weekday_machine(df)
    save_csv(weekday, ANALYSIS_DIR / "03_曜日台番号分析.csv")
    print(f"保存: {ANALYSIS_DIR / '03_曜日台番号分析.csv'}")

    # 曜日別に上位を表示
    for wd in ["月", "火", "水", "木", "金", "土", "日"]:
        tmp = weekday[weekday["曜日"] == wd].head(5)
        print(f"\n{wd}曜日 上位5台")
        if not tmp.empty:
            print(tmp[["machine_no", "machine_name", "日数", "平均差枚", "勝率", "平均G数"]].to_string(index=False))

    print()
    print("=" * 70)
    print("【4. 特定日候補】")
    print("=" * 70)
    special = analyze_special_days(df)
    save_csv(special, ANALYSIS_DIR / "04_特定日候補.csv")
    print(
        special[
            ["date", "平均差枚", "勝率", "strong_1000_count", "strong_2000_count", "strong_3000_count",
             "特定日スコア", "候補ランク"]
        ].to_string(index=False)
    )
    print(f"保存: {ANALYSIS_DIR / '04_特定日候補.csv'}")

    print()
    print("=" * 70)
    print("【5. 狙い台スコア Ver.2】")
    print("=" * 70)
    machine_summary = analyze_machine(df)
    target = build_target_score(df, machine_summary, weekday, special)
    save_csv(target, ANALYSIS_DIR / "05_狙い台スコア_Ver2.csv")

    show_cols = [
        "machine_no", "machine_name", "日数",
        "平均差枚", "勝率", "平均G数",
        "strong_1000_rate", "strong_2000_rate",
        "直近7日平均差枚", "直近7日勝率",
        "最強曜日平均差枚", "最強曜日勝率",
        "基礎スコア", "狙い台スコアVer2",
    ]
    print(target[show_cols].head(30).to_string(index=False))
    print(f"\n保存: {ANALYSIS_DIR / '05_狙い台スコア_Ver2.csv'}")

    # 解析概要テキスト
    summary_path = ANALYSIS_DIR / "README_31days_deep.txt"
    summary_path.write_text(
        "\n".join([
            "アナスロ 31日分 深掘り解析",
            f"期間: {START_DATE:%Y-%m-%d} ～ {END_DATE:%Y-%m-%d}",
            f"レコード: {len(df):,}",
            f"台数: {df['machine_no'].nunique()}",
            "",
            "出力:",
            "01_連番並び分析.csv",
            "02_前日翌日_全体傾向.csv",
            "02_前日翌日_台番号別.csv",
            "03_曜日台番号分析.csv",
            "04_特定日候補.csv",
            "05_狙い台スコア_Ver2.csv",
            "",
            "注意:",
            "本解析のスコアは探索・検証用であり、設定投入を保証するものではありません。",
            "31日という標本では偶然変動も大きいため、8/11以降の実績で継続検証してください。",
        ]),
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("★★★★★ 深掘り解析完了 ★★★★★")
    print("=" * 70)
    print(f"出力先: {ANALYSIS_DIR}")
    print()
    print("次のステップ:")
    print("1. 連番・並びの頻出パターン確認")
    print("2. 前日→翌日の上げ/据置/下げ傾向確認")
    print("3. 曜日×台番号の有意な偏り確認")
    print("4. 特定日候補の実戦日検証")
    print("5. 8/11以降の新データで予測精度を検証")


if __name__ == "__main__":
    main()
