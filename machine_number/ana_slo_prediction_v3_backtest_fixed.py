# -*- coding: utf-8 -*-
"""
アナスロ 狙い台スコア Ver.3 バックテスト

目的:
- ana_slo_prediction_v3.py と同じ考え方で、過去日を「未来の日」として予測
- 予測対象日の実績を後から答え合わせ
- TOP1 / TOP5 / TOP10 / TOP20 / TOP30 の勝率・平均差枚等を集計
- 予測対象日のデータは特徴量計算に一切使用しない（Look-ahead bias防止）

対象:
2026-07-11 ～ 2026-08-10
推奨バックテスト開始日:
2026-07-26
（最低15日程度の履歴を確保）
"""

from __future__ import annotations

from pathlib import Path
import math
import pandas as pd
import numpy as np


BASE = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
DATA_DIR = BASE / "data" / "maruhan_maebashi" / "machine_number"
OUT_DIR = DATA_DIR / "analysis_31days_deep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_0711 = DATA_DIR / "ana_slo_20260711.csv"
CSV_0712_0810 = DATA_DIR / "ana_slo_20260712_20260810.csv"

SUMMARY = OUT_DIR / "07_Ver3_バックテスト_総合結果.csv"
DAILY = OUT_DIR / "07_Ver3_バックテスト_日別結果.csv"
DETAIL = OUT_DIR / "07_Ver3_バックテスト_台別結果.csv"
TXT = OUT_DIR / "07_Ver3_バックテスト_README.txt"

START = pd.Timestamp("2026-07-11")
END = pd.Timestamp("2026-08-10")
BACKTEST_START = pd.Timestamp("2026-07-26")
EXPECTED_MACHINES = 514
PRIOR_N = 15.0


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"CSVを読み込めません: {path}")


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_data():
    frames = []
    if CSV_0711.exists():
        frames.append(read_csv(CSV_0711))
    if CSV_0712_0810.exists():
        frames.append(read_csv(CSV_0712_0810))

    if not frames:
        raise FileNotFoundError("対象CSVが見つかりません。")

    df = pd.concat(frames, ignore_index=True)

    date_col = find_col(df, ["date", "日付", "譌･莉・"])
    no_col = find_col(df, ["machine_no", "台番号", "台番号"])
    name_col = find_col(df, ["machine_name", "機種名"])
    diff_col = find_col(df, ["diff", "差枚"])

    if not all([date_col, no_col, name_col, diff_col]):
        raise ValueError(
            f"必要列が見つかりません: date={date_col}, no={no_col}, "
            f"name={name_col}, diff={diff_col}"
        )

    df = df.rename(columns={
        date_col: "date",
        no_col: "machine_no",
        name_col: "machine_name",
        diff_col: "diff",
    })

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["machine_no"] = pd.to_numeric(df["machine_no"], errors="coerce")

    # +1,600 / -1,000 / 1,600 などを数値化
    df["diff"] = (
        df["diff"].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.replace("枚", "", regex=False)
        .str.strip()
    )
    df["diff"] = pd.to_numeric(df["diff"], errors="coerce")

    df = df.dropna(subset=["date", "machine_no", "diff"]).copy()
    df["machine_no"] = df["machine_no"].astype(int)
    df["machine_name"] = df["machine_name"].astype(str).str.strip()

    df = df[(df["date"] >= START) & (df["date"] <= END)].copy()
    df = df.sort_values(["date", "machine_no"])
    df = df.drop_duplicates(["date", "machine_no"], keep="last")

    df["win"] = (df["diff"] > 0).astype(int)
    df["plus1000"] = (df["diff"] >= 1000).astype(int)
    df["plus2000"] = (df["diff"] >= 2000).astype(int)
    df["plus3000"] = (df["diff"] >= 3000).astype(int)

    return df


def eb_rate(successes, n, prior_rate, prior_n=PRIOR_N):
    if n <= 0:
        return prior_rate
    return (successes + prior_rate * prior_n) / (n + prior_n)


def z_to_50(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    std = float(s.std(ddof=0))
    if std == 0:
        return pd.Series(50.0, index=s.index)
    z = (s - float(s.mean())) / std
    return (50.0 + z * 12.5).clip(0, 100)


def reliability(days):
    if days <= 0:
        return 0.0
    return math.sqrt(min(days, 31) / 31.0)


def build_features(hist: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    overall_win = float(hist["win"].mean())
    overall_p1000 = float(hist["plus1000"].mean())
    overall_p2000 = float(hist["plus2000"].mean())

    target_weekday = target_date.dayofweek

    type_stats = {}
    for name, g in hist.groupby("machine_name"):
        n = len(g)
        type_stats[name] = {
            "avg_diff": float(g["diff"].mean()),
            "win": eb_rate(int(g["win"].sum()), n, overall_win),
        }

    latest = hist.sort_values("date").groupby("machine_no").tail(1)
    candidates = sorted(latest["machine_no"].unique())

    latest_date = hist["date"].max()
    latest_day = hist[hist["date"] == latest_date].set_index("machine_no")

    rows = []

    for no in candidates:
        m = hist[hist["machine_no"] == no].sort_values("date").copy()
        if m.empty:
            continue

        name = str(m.iloc[-1]["machine_name"])
        days = len(m)

        diffs = m["diff"].astype(float).tolist()
        last = float(diffs[-1])
        prev = float(diffs[-2]) if len(diffs) >= 2 else last
        prev_change = last - prev

        recent7 = m.tail(7)
        recent3 = m.tail(3)

        avg31 = float(m["diff"].mean())
        win31_raw = float(m["win"].mean())

        win31 = eb_rate(int(m["win"].sum()), days, overall_win)
        p1000 = eb_rate(int(m["plus1000"].sum()), days, overall_p1000)
        p2000 = eb_rate(int(m["plus2000"].sum()), days, overall_p2000)

        recent7_avg = float(recent7["diff"].mean())
        recent7_win = float(recent7["win"].mean())

        # 対象曜日実績
        wd = m[m["date"].dt.dayofweek == target_weekday]
        wd_n = len(wd)
        if wd_n:
            wd_avg_raw = float(wd["diff"].mean())
            wd_win = eb_rate(int(wd["win"].sum()), wd_n, overall_win)
            wd_weight = wd_n / (wd_n + PRIOR_N)
            wd_avg = wd_avg_raw * wd_weight + avg31 * (1 - wd_weight)
        else:
            wd_avg = avg31
            wd_win = win31

        # 前日→翌日関係
        transitions = []
        prev_map = m.set_index("date")["diff"].to_dict()
        for d, val in prev_map.items():
            nxt = d + pd.Timedelta(days=1)
            if nxt in prev_map:
                transitions.append((float(val), float(prev_map[nxt])))

        if transitions:
            tdf = pd.DataFrame(transitions, columns=["prev_diff", "next_diff"])
            neg500 = tdf[tdf["prev_diff"] <= -500]["next_diff"]
            neg1000 = tdf[tdf["prev_diff"] <= -1000]["next_diff"]
            next_after_neg500 = float(neg500.mean()) if len(neg500) else 0.0
            next_after_neg1000 = float(neg1000.mean()) if len(neg1000) else 0.0
            transition_n = len(transitions)
        else:
            next_after_neg500 = 0.0
            next_after_neg1000 = 0.0
            transition_n = 0

        # 前日凹みシグナル
        bounce_signal = 0.0
        if last <= -1000:
            bounce_signal = 1.0
        elif last <= -500:
            bounce_signal = 0.5
        elif last >= 1000:
            bounce_signal = -0.25

        # 隣接台
        neighbor_values = []
        for n2 in (no - 1, no + 1):
            if n2 in latest_day.index:
                neighbor_values.append(float(latest_day.loc[n2, "diff"]))
        neighbor_avg = float(np.mean(neighbor_values)) if neighbor_values else 0.0

        ts = type_stats.get(name, {})
        type_avg = float(ts.get("avg_diff", 0.0))
        type_win = float(ts.get("win", overall_win))

        rows.append({
            "machine_no": no,
            "machine_name": name,
            "history_days": days,
            "avg_diff_31": avg31,
            "win_rate_31": win31,
            "win_rate_31_raw": win31_raw,
            "plus1000_rate": p1000,
            "plus2000_rate": p2000,
            "recent7_avg": recent7_avg,
            "recent7_win": recent7_win,
            "recent3_avg": float(recent3["diff"].mean()),
            "last_diff": last,
            "prev_change": prev_change,
            "weekday_avg": wd_avg,
            "weekday_win": wd_win,
            "weekday_n": wd_n,
            "type_avg": type_avg,
            "type_win": type_win,
            "next_after_neg500": next_after_neg500,
            "next_after_neg1000": next_after_neg1000,
            "transition_n": transition_n,
            "bounce_signal": bounce_signal,
            "neighbor_avg": neighbor_avg,
            "reliability": reliability(days),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["s_avg31"] = z_to_50(out["avg_diff_31"])
    out["s_recent7"] = z_to_50(out["recent7_avg"])
    out["s_recent7_win"] = z_to_50(out["recent7_win"])
    out["s_last"] = z_to_50(out["last_diff"])
    out["s_prev_change"] = z_to_50(out["prev_change"])
    out["s_weekday"] = z_to_50(out["weekday_avg"])
    out["s_type"] = z_to_50(out["type_avg"])
    out["s_p1000"] = z_to_50(out["plus1000_rate"])
    out["s_p2000"] = z_to_50(out["plus2000_rate"])
    out["s_neighbor"] = z_to_50(out["neighbor_avg"])

    raw = (
        out["s_avg31"] * 0.18
        + out["s_recent7"] * 0.18
        + out["s_recent7_win"] * 0.08
        + out["s_last"] * 0.08
        + out["s_prev_change"] * 0.07
        + out["s_weekday"] * 0.08
        + out["s_type"] * 0.08
        + ((out["s_p1000"] * 0.5) + (out["s_p2000"] * 0.5)) * 0.08
        + out["s_neighbor"] * 0.04
        + out["bounce_signal"] * 3.0
    )

    confidence = out["reliability"]
    out["score"] = (50.0 + (raw - 50.0) * confidence).clip(0, 100)

    out = out.sort_values(
        ["score", "history_days"],
        ascending=[False, False]
    ).reset_index(drop=True)

    out["rank"] = np.arange(1, len(out) + 1)
    return out


def evaluate_top(pred, actual, target_date, top_n):
    top = pred.head(top_n).copy()
    merged = top.merge(
        actual[["machine_no", "diff"]],
        on="machine_no",
        how="inner",
        suffixes=("", "_actual")
    )

    if merged.empty:
        return {
            "date": target_date.date(),
            "top_n": top_n,
            "predicted": len(top),
            "matched": 0,
            "coverage": 0.0,
            "avg_diff": np.nan,
            "win_rate": np.nan,
            "plus1000_rate": np.nan,
            "plus2000_rate": np.nan,
            "total_diff": np.nan,
        }

    d = merged["diff"].astype(float)
    return {
        "date": target_date.date(),
        "top_n": top_n,
        "predicted": len(top),
        "matched": len(merged),
        "coverage": len(merged) / len(top) * 100.0,
        "avg_diff": float(d.mean()),
        "win_rate": float((d > 0).mean() * 100.0),
        "plus1000_rate": float((d >= 1000).mean() * 100.0),
        "plus2000_rate": float((d >= 2000).mean() * 100.0),
        "total_diff": float(d.sum()),
    }


def main():
    print("=" * 70)
    print("アナスロ 狙い台スコア Ver.3 バックテスト")
    print("=" * 70)

    df = load_data()
    print(f"総レコード: {len(df):,}")
    print(f"バックテスト期間: {BACKTEST_START.date()} ～ {END.date()}")

    dates = sorted(
        d for d in df["date"].dropna().unique()
        if BACKTEST_START <= d <= END
    )

    daily_rows = []
    detail_rows = []

    for target_date in dates:
        target_date = pd.Timestamp(target_date)

        hist = df[df["date"] < target_date].copy()
        actual = df[df["date"] == target_date].copy()

        if hist.empty or actual.empty:
            continue

        pred = build_features(hist, target_date)

        if pred.empty:
            continue

        # TOP30までの台別答え合わせ
        top30 = pred.head(30).merge(
            actual[["machine_no", "diff", "machine_name"]],
            on="machine_no",
            how="left",
            suffixes=("", "_actual")
        )

        for _, r in top30.iterrows():
            actual_diff = r.get("diff", np.nan)
            detail_rows.append({
                "date": target_date.date(),
                "rank": int(r["rank"]),
                "machine_no": int(r["machine_no"]),
                "machine_name": r["machine_name"],
                "score": float(r["score"]),
                "actual_diff": actual_diff,
                "win": int(actual_diff > 0) if pd.notna(actual_diff) else np.nan,
                "plus1000": int(actual_diff >= 1000) if pd.notna(actual_diff) else np.nan,
                "plus2000": int(actual_diff >= 2000) if pd.notna(actual_diff) else np.nan,
            })

        # TOP別
        for n in (1, 5, 10, 20, 30):
            daily_rows.append(evaluate_top(pred, actual, target_date, n))

        # 全台ベースライン
        ad = actual["diff"].astype(float)
        daily_rows.append({
            "date": target_date.date(),
            "top_n": 0,
            "predicted": len(actual),
            "matched": len(actual),
            "coverage": 100.0,
            "avg_diff": float(ad.mean()),
            "win_rate": float((ad > 0).mean() * 100.0),
            "plus1000_rate": float((ad >= 1000).mean() * 100.0),
            "plus2000_rate": float((ad >= 2000).mean() * 100.0),
            "total_diff": float(ad.sum()),
        })

        print(
            f"{target_date.date()} "
            f"履歴={len(hist):5d}件 "
            f"対象={len(actual):3d}台 "
            f"TOP1={pred.iloc[0]['machine_no']} "
            f"TOP30平均={daily_rows[-2]['avg_diff']:+.0f}"
        )

    daily = pd.DataFrame(daily_rows)
    detail = pd.DataFrame(detail_rows)

    # 総合集計
    summaries = []
    for n in (0, 1, 5, 10, 20, 30):
        x = daily[daily["top_n"] == n].copy()
        if x.empty:
            continue

        label = "全台" if n == 0 else f"TOP{n}"

        summaries.append({
            "対象": label,
            "日数": len(x),
            "平均差枚": x["avg_diff"].mean(),
            "日別平均差枚の中央値": x["avg_diff"].median(),
            "勝率平均": x["win_rate"].mean(),
            "+1000枚率平均": x["plus1000_rate"].mean(),
            "+2000枚率平均": x["plus2000_rate"].mean(),
            "日別プラス率": (x["avg_diff"] > 0).mean() * 100.0,
            "総差枚": x["total_diff"].sum(),
        })

    summary = pd.DataFrame(summaries)

    # 全台との差
    baseline = summary.loc[summary["対象"] == "全台", "平均差枚"]
    baseline = float(baseline.iloc[0]) if len(baseline) else np.nan
    if pd.notna(baseline):
        summary["全台平均との差"] = summary["平均差枚"] - baseline
    else:
        summary["全台平均との差"] = np.nan

    summary.to_csv(SUMMARY, index=False, encoding="utf-8-sig")
    daily.to_csv(DAILY, index=False, encoding="utf-8-sig")
    detail.to_csv(DETAIL, index=False, encoding="utf-8-sig")

    lines = [
        "アナスロ 狙い台スコア Ver.3 バックテスト",
        f"対象期間: {BACKTEST_START.date()} ～ {END.date()}",
        f"データ期間: {START.date()} ～ {END.date()}",
        f"総レコード: {len(df):,}",
        "",
        "重要: 各予測日は、その予測日より前のデータだけでスコアを計算。",
        "予測対象日の差枚は答え合わせにのみ使用。",
        "",
        "===== 総合結果 =====",
    ]

    if not summary.empty:
        for _, r in summary.iterrows():
            lines.append(
                f"{r['対象']:5s} "
                f"平均差枚={r['平均差枚']:+.1f} "
                f"勝率={r['勝率平均']:.2f}% "
                f"+1000={r['+1000枚率平均']:.2f}% "
                f"+2000={r['+2000枚率平均']:.2f}% "
                f"全台平均との差={r['全台平均との差']:+.1f}"
            )

    lines += [
        "",
        "出力:",
        str(SUMMARY),
        str(DAILY),
        str(DETAIL),
        "",
        "判定:",
        "TOP30の平均差枚が全台平均との差でプラスなら、少なくともこの期間では選別効果あり。",
        "ただしバックテスト期間が短いため、モデル完成を意味しない。",
        "次段階では要因別バックテストと重み最適化を行う。",
    ]

    TXT.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("===== 総合結果 =====")
    print(summary.to_string(index=False))
    print()
    print("保存:")
    print(SUMMARY)
    print(DAILY)
    print(DETAIL)
    print(TXT)
    print()
    print("バックテスト完了")


if __name__ == "__main__":
    main()
