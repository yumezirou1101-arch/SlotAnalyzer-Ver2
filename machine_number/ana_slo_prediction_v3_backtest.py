# -*- coding: utf-8 -*-
"""
繧｢繝翫せ繝ｭ 迢吶＞蜿ｰ繧ｹ繧ｳ繧｢ Ver.3 繝舌ャ繧ｯ繝・せ繝・
逶ｮ逧・
- ana_slo_prediction_v3.py 縺ｨ蜷後§閠・∴譁ｹ縺ｧ縲・℃蜴ｻ譌･繧偵梧悴譚･縺ｮ譌･縲阪→縺励※莠域ｸｬ
- 莠域ｸｬ蟇ｾ雎｡譌･縺ｮ螳溽ｸｾ繧貞ｾ後°繧臥ｭ斐∴蜷医ｏ縺・- TOP1 / TOP5 / TOP10 / TOP20 / TOP30 縺ｮ蜍晉紫繝ｻ蟷ｳ蝮・ｷｮ譫夂ｭ峨ｒ髮・ｨ・- 莠域ｸｬ蟇ｾ雎｡譌･縺ｮ繝・・繧ｿ縺ｯ迚ｹ蠕ｴ驥剰ｨ育ｮ励↓荳蛻・ｽｿ逕ｨ縺励↑縺・ｼ・ook-ahead bias髦ｲ豁｢・・
蟇ｾ雎｡:
2026-07-11 ・・2026-08-10
謗ｨ螂ｨ繝舌ャ繧ｯ繝・せ繝磯幕蟋区律:
2026-07-26
・域怙菴・5譌･遞句ｺｦ縺ｮ螻･豁ｴ繧堤｢ｺ菫晢ｼ・"""

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

SUMMARY = OUT_DIR / "07_Ver3_繝舌ャ繧ｯ繝・せ繝・邱丞粋邨先棡.csv"
DAILY = OUT_DIR / "07_Ver3_繝舌ャ繧ｯ繝・せ繝・譌･蛻･邨先棡.csv"
DETAIL = OUT_DIR / "07_Ver3_繝舌ャ繧ｯ繝・せ繝・蜿ｰ蛻･邨先棡.csv"
TXT = OUT_DIR / "07_Ver3_繝舌ャ繧ｯ繝・せ繝・README.txt"

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
    raise RuntimeError(f"CSV繧定ｪｭ縺ｿ霎ｼ繧√∪縺帙ｓ: {path}")


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
        raise FileNotFoundError("蟇ｾ雎｡CSV縺瑚ｦ九▽縺九ｊ縺ｾ縺帙ｓ縲・)

    df = pd.concat(frames, ignore_index=True)

    date_col = find_col(df, ["date", "譌･莉・, "隴鯉ｽ･闔峨・"])
    no_col = find_col(df, ["machine_no", "蜿ｰ逡ｪ蜿ｷ", "蜿ｰ逡ｪ蜿ｷ"])
    name_col = find_col(df, ["machine_name", "讖溽ｨｮ蜷・])
    diff_col = find_col(df, ["diff", "蟾ｮ譫・])

    if not all([date_col, no_col, name_col, diff_col]):
        raise ValueError(
            f"蠢・ｦ∝・縺瑚ｦ九▽縺九ｊ縺ｾ縺帙ｓ: date={date_col}, no={no_col}, "
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

    # +1,600 / -1,000 / 1,600 縺ｪ縺ｩ繧呈焚蛟､蛹・    df["diff"] = (
        df["diff"].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.replace("譫・, "", regex=False)
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

        # 蟇ｾ雎｡譖懈律螳溽ｸｾ
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

        # 蜑肴律竊堤ｿ梧律髢｢菫・        transitions = []
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

        # 蜑肴律蜃ｹ縺ｿ繧ｷ繧ｰ繝翫Ν
        bounce_signal = 0.0
        if last <= -1000:
            bounce_signal = 1.0
        elif last <= -500:
            bounce_signal = 0.5
        elif last >= 1000:
            bounce_signal = -0.25

        # 髫｣謗･蜿ｰ
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
    print("繧｢繝翫せ繝ｭ 迢吶＞蜿ｰ繧ｹ繧ｳ繧｢ Ver.3 繝舌ャ繧ｯ繝・せ繝・)
    print("=" * 70)

    df = load_data()
    print(f"邱上Ξ繧ｳ繝ｼ繝・ {len(df):,}")
    print(f"繝舌ャ繧ｯ繝・せ繝域悄髢・ {BACKTEST_START.date()} ・・{END.date()}")

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

        # TOP30縺ｾ縺ｧ縺ｮ蜿ｰ蛻･遲斐∴蜷医ｏ縺・        top30 = pred.head(30).merge(
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

        # TOP蛻･
        for n in (1, 5, 10, 20, 30):
            daily_rows.append(evaluate_top(pred, actual, target_date, n))

        # 蜈ｨ蜿ｰ繝吶・繧ｹ繝ｩ繧､繝ｳ
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
            f"螻･豁ｴ={len(hist):5d}莉ｶ "
            f"蟇ｾ雎｡={len(actual):3d}蜿ｰ "
            f"TOP1={pred.iloc[0]['machine_no']} "
            f"TOP30蟷ｳ蝮・{daily_rows[-2]['avg_diff']:+.0f}"
        )

    daily = pd.DataFrame(daily_rows)
    detail = pd.DataFrame(detail_rows)

    # 邱丞粋髮・ｨ・    summaries = []
    for n in (0, 1, 5, 10, 20, 30):
        x = daily[daily["top_n"] == n].copy()
        if x.empty:
            continue

        label = "蜈ｨ蜿ｰ" if n == 0 else f"TOP{n}"

        summaries.append({
            "蟇ｾ雎｡": label,
            "譌･謨ｰ": len(x),
            "蟷ｳ蝮・ｷｮ譫・: x["avg_diff"].mean(),
            "譌･蛻･蟷ｳ蝮・ｷｮ譫壹・荳ｭ螟ｮ蛟､": x["avg_diff"].median(),
            "蜍晉紫蟷ｳ蝮・: x["win_rate"].mean(),
            "+1000譫夂紫蟷ｳ蝮・: x["plus1000_rate"].mean(),
            "+2000譫夂紫蟷ｳ蝮・: x["plus2000_rate"].mean(),
            "譌･蛻･繝励Λ繧ｹ邇・: (x["avg_diff"] > 0).mean() * 100.0,
            "邱丞ｷｮ譫・: x["total_diff"].sum(),
        })

    summary = pd.DataFrame(summaries)

    # 蜈ｨ蜿ｰ縺ｨ縺ｮ蟾ｮ
    baseline = summary.loc[summary["蟇ｾ雎｡"] == "蜈ｨ蜿ｰ", "蟷ｳ蝮・ｷｮ譫・]
    baseline = float(baseline.iloc[0]) if len(baseline) else np.nan
    if pd.notna(baseline):
        summary["蜈ｨ蜿ｰ蟷ｳ蝮・→縺ｮ蟾ｮ"] = summary["蟷ｳ蝮・ｷｮ譫・] - baseline
    else:
        summary["蜈ｨ蜿ｰ蟷ｳ蝮・→縺ｮ蟾ｮ"] = np.nan

    summary.to_csv(SUMMARY, index=False, encoding="utf-8-sig")
    daily.to_csv(DAILY, index=False, encoding="utf-8-sig")
    detail.to_csv(DETAIL, index=False, encoding="utf-8-sig")

    lines = [
        "繧｢繝翫せ繝ｭ 迢吶＞蜿ｰ繧ｹ繧ｳ繧｢ Ver.3 繝舌ャ繧ｯ繝・せ繝・,
        f"蟇ｾ雎｡譛滄俣: {BACKTEST_START.date()} ・・{END.date()}",
        f"繝・・繧ｿ譛滄俣: {START.date()} ・・{END.date()}",
        f"邱上Ξ繧ｳ繝ｼ繝・ {len(df):,}",
        "",
        "驥崎ｦ・ 蜷・ｺ域ｸｬ譌･縺ｯ縲√◎縺ｮ莠域ｸｬ譌･繧医ｊ蜑阪・繝・・繧ｿ縺縺代〒繧ｹ繧ｳ繧｢繧定ｨ育ｮ励・,
        "莠域ｸｬ蟇ｾ雎｡譌･縺ｮ蟾ｮ譫壹・遲斐∴蜷医ｏ縺帙↓縺ｮ縺ｿ菴ｿ逕ｨ縲・,
        "",
        "===== 邱丞粋邨先棡 =====",
    ]

    if not summary.empty:
        for _, r in summary.iterrows():
            lines.append(
                f"{r['蟇ｾ雎｡']:5s} "
                f"蟷ｳ蝮・ｷｮ譫・{r['蟷ｳ蝮・ｷｮ譫・]:+.1f} "
                f"蜍晉紫={r['蜍晉紫蟷ｳ蝮・]:.2f}% "
                f"+1000={r['+1000譫夂紫蟷ｳ蝮・]:.2f}% "
                f"+2000={r['+2000譫夂紫蟷ｳ蝮・]:.2f}% "
                f"蜈ｨ蜿ｰ蟷ｳ蝮・→縺ｮ蟾ｮ={r['蜈ｨ蜿ｰ蟷ｳ蝮・→縺ｮ蟾ｮ']:+.1f}"
            )

    lines += [
        "",
        "蜃ｺ蜉・",
        str(SUMMARY),
        str(DAILY),
        str(DETAIL),
        "",
        "蛻､螳・",
        "TOP30縺ｮ蟷ｳ蝮・ｷｮ譫壹′蜈ｨ蜿ｰ蟷ｳ蝮・→縺ｮ蟾ｮ縺ｧ繝励Λ繧ｹ縺ｪ繧峨∝ｰ代↑縺上→繧ゅ％縺ｮ譛滄俣縺ｧ縺ｯ驕ｸ蛻･蜉ｹ譫懊≠繧翫・,
        "縺溘□縺励ヰ繝・け繝・せ繝域悄髢薙′遏ｭ縺・◆繧√√Δ繝・Ν螳梧・繧呈э蜻ｳ縺励↑縺・・,
        "谺｡谿ｵ髫弱〒縺ｯ隕∝屏蛻･繝舌ャ繧ｯ繝・せ繝医→驥阪∩譛驕ｩ蛹悶ｒ陦後≧縲・,
    ]

    TXT.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("===== 邱丞粋邨先棡 =====")
    print(summary.to_string(index=False))
    print()
    print("菫晏ｭ・")
    print(SUMMARY)
    print(DAILY)
    print(DETAIL)
    print(TXT)
    print()
    print("繝舌ャ繧ｯ繝・せ繝亥ｮ御ｺ・)


if __name__ == "__main__":
    main()
