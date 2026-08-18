# -*- coding: utf-8 -*-
"""
繧｢繝翫せ繝ｭ 迢吶＞蜿ｰ繧ｹ繧ｳ繧｢ Ver.3 隕∝屏蛻･繝舌ャ繧ｯ繝・せ繝・蟇ｾ雎｡: 繝槭Ν繝上Φ蜑肴ｩ九う繝ｳ繧ｿ繝ｼ
譛滄俣: 2026-07-26 ・・2026-08-10

逶ｮ逧・
Ver.3縺ｧ菴ｿ逕ｨ縺励※縺・ｋ蜷・ｦ∝屏縺後∫ｿ梧律縺ｮ螳溽ｸｾ蟾ｮ譫壹→譛ｬ蠖薙↓髢｢菫ゅ＠縺ｦ縺・ｋ縺九ｒ
譎らｳｻ蛻励ヰ繝・け繝・せ繝医〒讀懆ｨｼ縺吶ｋ縲・
驥崎ｦ・
蜷・ｺ域ｸｬ譌･縺ｯ縲後◎縺ｮ譌･繧医ｊ蜑阪阪・繝・・繧ｿ縺縺代ｒ菴ｿ逕ｨ縺吶ｋ縲・譛ｪ譚･繝・・繧ｿ縺ｯ迚ｹ蠕ｴ驥剰ｨ育ｮ励↓菴ｿ逕ｨ縺励↑縺・・"""

from pathlib import Path
import math
import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
DATA_DIR = BASE / "data" / "maruhan_maebashi" / "machine_number"
OUT_DIR = DATA_DIR / "analysis_31days_deep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_0711 = DATA_DIR / "ana_slo_20260711.csv"
CSV_0712_0810 = DATA_DIR / "ana_slo_20260712_20260810.csv"

BT_START = pd.Timestamp("2026-07-26")
BT_END = pd.Timestamp("2026-08-10")


def read_csv(path):
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"CSV繧定ｪｭ繧√∪縺帙ｓ: {path}")


def load_data():
    frames = []
    for p in (CSV_0711, CSV_0712_0810):
        if p.exists():
            frames.append(read_csv(p))

    if not frames:
        raise FileNotFoundError("対象CSVがありません。")

    df = pd.concat(frames, ignore_index=True)

    def find(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    date_col = find(["date", "日付"])
    no_col = find(["machine_no", "台番号"])
    name_col = find(["machine_name", "機種名"])
    diff_col = find(["diff", "差枚"])

    if not all([date_col, no_col, name_col, diff_col]):
        raise ValueError(
            f"蠢・ｦ∝・縺後≠繧翫∪縺帙ｓ: date={date_col}, no={no_col}, "
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
    df["diff"] = (
        df["diff"].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.strip()
    )
    df["diff"] = pd.to_numeric(df["diff"], errors="coerce")

    df = df.dropna(subset=["date", "machine_no", "diff"]).copy()
    df["machine_no"] = df["machine_no"].astype(int)
    df["machine_name"] = df["machine_name"].astype(str).str.strip()

    df = df.sort_values(["date", "machine_no"])
    df = df.drop_duplicates(["date", "machine_no"], keep="last")

    df["win"] = (df["diff"] > 0).astype(int)
    df["plus1000"] = (df["diff"] >= 1000).astype(int)
    df["plus2000"] = (df["diff"] >= 2000).astype(int)

    return df


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def build_features(df, target_date):
    hist = df[df["date"] < target_date].copy()
    actual = df[df["date"] == target_date][
        ["machine_no", "machine_name", "diff"]
    ].copy()

    if hist.empty or actual.empty:
        return pd.DataFrame()

    target_weekday = target_date.dayofweek
    latest_date = hist["date"].max()
    latest_day = hist[hist["date"] == latest_date].set_index("machine_no")

    # 讖溽ｨｮ蛻･縺ｮ驕主悉螳溽ｸｾ
    type_stats = (
        hist.groupby("machine_name")["diff"]
        .agg(type_avg="mean")
        .reset_index()
    )

    rows = []

    for no, m in hist.groupby("machine_no"):
        m = m.sort_values("date")
        if m.empty:
            continue

        name = str(m.iloc[-1]["machine_name"])
        diffs = m["diff"].astype(float).to_numpy()

        recent7 = m.tail(7)
        recent3 = m.tail(3)

        avg31 = float(m["diff"].mean())
        recent7_avg = float(recent7["diff"].mean())
        recent7_win = float(recent7["win"].mean())
        recent3_avg = float(recent3["diff"].mean())
        last_diff = float(diffs[-1])
        prev_diff = float(diffs[-2]) if len(diffs) >= 2 else last_diff
        prev_change = last_diff - prev_diff

        # 蟇ｾ雎｡譖懈律縺ｮ驕主悉螳溽ｸｾ
        wd = m[m["date"].dt.dayofweek == target_weekday]
        weekday_n = len(wd)
        weekday_avg_raw = float(wd["diff"].mean()) if weekday_n else avg31

        # V3縺ｨ蜷後§閠・∴譁ｹ縺ｧ譖懈律蟷ｳ蝮・ｒ螻･豁ｴ驥上↓蠢懊§縺ｦ邵ｮ蟆・        prior_n = 15.0
        wd_weight = weekday_n / (weekday_n + 15.0)
        weekday_avg = (
            weekday_avg_raw * wd_weight
            + avg31 * (1.0 - wd_weight)
        )

        plus1000_rate = float(m["plus1000"].mean())
        plus2000_rate = float(m["plus2000"].mean())

        type_row = type_stats[type_stats["machine_name"] == name]
        type_avg = (
            float(type_row.iloc[0]["type_avg"])
            if len(type_row) else 0.0
        )

        neighbor_values = []
        for n2 in (no - 1, no + 1):
            if n2 in latest_day.index:
                neighbor_values.append(float(latest_day.loc[n2, "diff"]))
        neighbor_avg = (
            float(np.mean(neighbor_values))
            if neighbor_values else 0.0
        )

        if last_diff <= -1000:
            bounce_signal = 1.0
        elif last_diff <= -500:
            bounce_signal = 0.5
        elif last_diff >= 1000:
            bounce_signal = -0.25
        else:
            bounce_signal = 0.0

        rows.append({
            "target_date": target_date.date(),
            "machine_no": int(no),
            "machine_name": name,
            "history_days": len(m),
            "avg31": avg31,
            "recent7_avg": recent7_avg,
            "recent7_win": recent7_win,
            "recent3_avg": recent3_avg,
            "last_diff": last_diff,
            "prev_change": prev_change,
            "weekday_avg": weekday_avg,
            "weekday_n": weekday_n,
            "type_avg": type_avg,
            "plus1000_rate": plus1000_rate,
            "plus2000_rate": plus2000_rate,
            "neighbor_avg": neighbor_avg,
            "bounce_signal": bounce_signal,
        })

    feat = pd.DataFrame(rows)
    return feat.merge(
        actual.rename(columns={"diff": "actual_diff"}),
        on="machine_no",
        how="inner",
        suffixes=("", "_actual")
    )


FACTORS = [
    "avg31",
    "recent7_avg",
    "recent7_win",
    "last_diff",
    "prev_change",
    "weekday_avg",
    "type_avg",
    "plus1000_rate",
    "plus2000_rate",
    "neighbor_avg",
    "bounce_signal",
]


def evaluate_factor(df, factor):
    x = pd.to_numeric(df[factor], errors="coerce")
    y = pd.to_numeric(df["actual_diff"], errors="coerce")
    ok = x.notna() & y.notna()
    x = x[ok]
    y = y[ok]

    if len(x) < 20 or x.nunique() < 2:
        return None

    pearson = float(x.corr(y, method="pearson"))
    spearman = float(x.rank().corr(y.rank()))

    q = max(1, int(len(x) * 0.10))
    order = x.sort_values(ascending=False).index
    top = y.loc[order[:q]]
    bottom = y.loc[order[-q:]]

    return {
        "factor": factor,
        "n": len(x),
        "pearson_r": pearson,
        "spearman_r": spearman,
        "top10_avg_diff": float(top.mean()),
        "bottom10_avg_diff": float(bottom.mean()),
        "top10_lift_vs_bottom": float(top.mean() - bottom.mean()),
        "top10_win_rate": float((top > 0).mean() * 100),
        "top10_plus2000_rate": float((top >= 2000).mean() * 100),
    }


def main():
    print("=" * 70)
    print("アナスロ 狙い台スコア Ver.3 要因別バックテスト")
    print("=" * 70)

    df = load_data()
    print(f"邱上Ξ繧ｳ繝ｼ繝・ {len(df):,}")
    print(f"蛻・梵譛滄俣: {BT_START.date()} ・・{BT_END.date()}")

    all_rows = []

    for target_date in pd.date_range(BT_START, BT_END):
        result = build_features(df, target_date)
        if result.empty:
            continue
        all_rows.append(result)
        print(
            f"{target_date.date()} "
            f"螻･豁ｴ={len(df[df['date'] < target_date]):5d}莉ｶ "
            f"蟇ｾ雎｡={len(result):3d}蜿ｰ"
        )

    panel = pd.concat(all_rows, ignore_index=True)

    # 譌･蛻･縺ｫ縺ｾ縺ｨ繧√◆逶ｸ髢｢繧貞性繧√∝・譛滄俣繧偵・繝ｼ繝ｫ縺励※讀懆ｨｼ
    summary = []
    for factor in FACTORS:
        r = evaluate_factor(panel, factor)
        if r:
            summary.append(r)

    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values(
        "top10_lift_vs_bottom", ascending=False
    ).reset_index(drop=True)

    # 譌･蛻･縺ｮ繝医ャ繝・0謌千ｸｾ
    daily_rows = []
    for date, g in panel.groupby("target_date"):
        for factor in FACTORS:
            x = pd.to_numeric(g[factor], errors="coerce")
            q = max(1, min(10, len(g)))
            idx = x.nlargest(q).index
            top = g.loc[idx, "actual_diff"]
            daily_rows.append({
                "date": date,
                "factor": factor,
                "top_n": q,
                "avg_diff": float(top.mean()),
                "win_rate": float((top > 0).mean() * 100),
                "plus2000_rate": float((top >= 2000).mean() * 100),
            })

    daily_df = pd.DataFrame(daily_rows)

    factor_daily = (
        daily_df.groupby("factor")
        .agg(
            days=("date", "nunique"),
            top10_daily_avg=("avg_diff", "mean"),
            top10_daily_median=("avg_diff", "median"),
            daily_positive_rate=("avg_diff", lambda s: float((s > 0).mean() * 100)),
            avg_win_rate=("win_rate", "mean"),
            avg_plus2000_rate=("plus2000_rate", "mean"),
        )
        .reset_index()
        .sort_values("top10_daily_avg", ascending=False)
    )

    out1 = OUT_DIR / "08_Ver3_隕∝屏蛻･逶ｸ髢｢蛻・梵.csv"
    out2 = OUT_DIR / "08_Ver3_隕∝屏蛻･_TOP10譌･蛻･謌千ｸｾ.csv"
    out3 = OUT_DIR / "08_Ver3_隕∝屏蛻･繝代ロ繝ｫ繝・・繧ｿ.csv"

    summary_df.to_csv(out1, index=False, encoding="utf-8-sig")
    factor_daily.to_csv(out2, index=False, encoding="utf-8-sig")
    panel.to_csv(out3, index=False, encoding="utf-8-sig")

    print()
    print("===== 隕∝屏蛻･隧穂ｾ｡ =====")
    print(summary_df.to_string(index=False))
    print()
    print("===== 隕∝屏蛻･ TOP10 譌･蛻･謌千ｸｾ =====")
    print(factor_daily.to_string(index=False))
    print()
    print("菫晏ｭ・")
    print(out1)
    print(out2)
    print(out3)
    print()
    print("要因別バックテスト完了")


if __name__ == "__main__":
    main()
