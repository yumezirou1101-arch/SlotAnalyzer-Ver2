# -*- coding: utf-8 -*-
"""
Ana-Slo Ver.4.2 Switch Trigger Candidate OOS Validation

目的:
- 既存の long-term Adaptive rank-band daily CSV を入力
- TOP10 を基本戦略とし、過去OOSだけから計算した候補条件で
  TOP4_5へ切り替えるかをブロック単位Walk-Forwardで判定
- 当該テストブロックの結果は、そのブロックの選択条件には使用しない

候補条件:
A  prior_last_edge > 0
B  prior_last_edge >= +300
C  prior_last_edge >= +500
D  prior_last_edge >= +750
E  prior_last2_edge_mean > 0
F  prior_last2_edge_mean >= +300
G  prior_last_edge > 0 AND prior_last2_edge_mean > 0
H  prior_last_edge > 0 AND prior_last3_edge_mean > 0

重要:
- 閾値は候補比較用であり、現時点で本番採用しない
- サンプルが短いため統計的有意性は判定しない
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATH
# ============================================================

INPUT_DAILY = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer\data"
    r"\maruhan_maebashi\machine_number\analysis_31days_deep"
    r"\39_Ver4_2_long_term_adaptive_rank_band_fixed_daily.csv"
)

OUT_DIR = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer\data"
    r"\maruhan_maebashi\machine_number\analysis_31days_deep"
    r"\46_Ver4_2_switch_trigger_candidate_oos"
)

OUT_DAILY = OUT_DIR / "46_Ver4_2_switch_trigger_candidate_oos_daily.csv"
OUT_BLOCK = OUT_DIR / "46_Ver4_2_switch_trigger_candidate_oos_blocks.csv"
OUT_SUMMARY = OUT_DIR / "46_Ver4_2_switch_trigger_candidate_oos_summary.csv"
OUT_DIAGNOSTIC = OUT_DIR / "46_Ver4_2_switch_trigger_candidate_oos_diagnostic.csv"
OUT_MONTHLY = OUT_DIR / "46_Ver4_2_switch_trigger_candidate_oos_monthly.csv"


# ============================================================
# CONFIG
# ============================================================

TOP10 = "TOP10"
TOP45 = "TOP4_5"

# 少なくとも2つの過去ブロックを使用してから条件判定
MIN_PRIOR_BLOCKS = 2


# ============================================================
# LOAD
# ============================================================

def load_daily():
    if not INPUT_DAILY.exists():
        raise FileNotFoundError(
            f"INPUT_DAILY not found:\n{INPUT_DAILY}"
        )

    print(f"Loading: {INPUT_DAILY}")
    df = pd.read_csv(INPUT_DAILY)

    required = {
        "date",
        "block",
        "block_start",
        "block_end",
        "rule",
        "selected_rule",
        "machines",
        "avg_diff",
        "total_diff",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"必要列がありません: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["block_start"] = pd.to_datetime(df["block_start"], errors="coerce")
    df["block_end"] = pd.to_datetime(df["block_end"], errors="coerce")
    df["avg_diff"] = pd.to_numeric(df["avg_diff"], errors="coerce")
    df["total_diff"] = pd.to_numeric(df["total_diff"], errors="coerce")
    df["block"] = pd.to_numeric(df["block"], errors="coerce").astype("Int64")

    df = df.dropna(
        subset=["date", "block", "avg_diff", "total_diff"]
    ).copy()

    print(f"records = {len(df):,}")
    print(f"columns = {list(df.columns)}")
    return df


# ============================================================
# BUILD BLOCK TABLE
# ============================================================

def build_block_table(df):
    print()
    print("Building rule/block table...")

    rows = []

    for block, g in df.groupby("block", sort=True):
        start = g["block_start"].dropna().min()
        end = g["block_end"].dropna().max()

        row = {
            "block": int(block),
            "block_start": start,
            "block_end": end,
        }

        for rule in [TOP10, TOP45]:
            sub = g[g["rule"] == rule]

            if sub.empty:
                row[f"{rule}_avg"] = np.nan
                row[f"{rule}_total"] = np.nan
                row[f"{rule}_machines"] = np.nan
            else:
                row[f"{rule}_avg"] = sub["avg_diff"].mean()
                row[f"{rule}_total"] = sub["total_diff"].sum()
                row[f"{rule}_machines"] = sub["machines"].sum()

        row["top45_minus_top10"] = (
            row["TOP4_5_avg"] - row["TOP10_avg"]
            if pd.notna(row["TOP4_5_avg"])
            and pd.notna(row["TOP10_avg"])
            else np.nan
        )

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("block").reset_index(drop=True)

    if out.empty:
        raise ValueError("block table が空です。")

    return out


# ============================================================
# PRIOR FEATURES
# ============================================================

def add_prior_features(blocks):
    out = blocks.copy()

    out["prior_blocks_used"] = np.arange(len(out))

    prior_cols = [
        "prior_edge_mean",
        "prior_edge_last",
        "prior_edge_last2_mean",
        "prior_edge_last3_mean",
        "prior_top45_positive_rate",
        "prior_top10_positive_rate",
    ]

    for c in prior_cols:
        out[c] = np.nan

    for i in range(len(out)):
        if i == 0:
            continue

        prior = out.iloc[:i].copy()

        out.loc[i, "prior_edge_mean"] = (
            prior["top45_minus_top10"].mean()
        )

        out.loc[i, "prior_edge_last"] = (
            prior["top45_minus_top10"].iloc[-1]
        )

        out.loc[i, "prior_edge_last2_mean"] = (
            prior["top45_minus_top10"].tail(2).mean()
        )

        out.loc[i, "prior_edge_last3_mean"] = (
            prior["top45_minus_top10"].tail(3).mean()
        )

        out.loc[i, "prior_top45_positive_rate"] = (
            (prior["top45_minus_top10"] > 0).mean() * 100.0
        )

        out.loc[i, "prior_top10_positive_rate"] = (
            (prior["TOP10_avg"] > 0).mean() * 100.0
        )

    return out


# ============================================================
# CANDIDATES
# ============================================================

def candidate_rules(row):
    n = int(row["prior_blocks_used"])

    if n < MIN_PRIOR_BLOCKS:
        return {
            "A_LAST_POSITIVE": False,
            "B_LAST_GE_300": False,
            "C_LAST_GE_500": False,
            "D_LAST_GE_750": False,
            "E_LAST2_POSITIVE": False,
            "F_LAST2_GE_300": False,
            "G_LAST_AND_LAST2_POSITIVE": False,
            "H_LAST_AND_LAST3_POSITIVE": False,
        }

    last = row["prior_edge_last"]
    last2 = row["prior_edge_last2_mean"]
    last3 = row["prior_edge_last3_mean"]

    return {
        "A_LAST_POSITIVE": bool(pd.notna(last) and last > 0),
        "B_LAST_GE_300": bool(pd.notna(last) and last >= 300),
        "C_LAST_GE_500": bool(pd.notna(last) and last >= 500),
        "D_LAST_GE_750": bool(pd.notna(last) and last >= 750),
        "E_LAST2_POSITIVE": bool(pd.notna(last2) and last2 > 0),
        "F_LAST2_GE_300": bool(pd.notna(last2) and last2 >= 300),
        "G_LAST_AND_LAST2_POSITIVE": bool(
            pd.notna(last)
            and pd.notna(last2)
            and last > 0
            and last2 > 0
        ),
        "H_LAST_AND_LAST3_POSITIVE": bool(
            pd.notna(last)
            and pd.notna(last3)
            and last > 0
            and last3 > 0
        ),
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_candidate(blocks, candidate_name):
    rows = []

    for i, row in blocks.iterrows():
        n = int(row["prior_blocks_used"])
        flags = candidate_rules(row)
        switch = flags[candidate_name] if n >= MIN_PRIOR_BLOCKS else False

        selected = TOP45 if switch else TOP10

        selected_avg = (
            row["TOP4_5_avg"] if switch else row["TOP10_avg"]
        )
        selected_total = (
            row["TOP4_5_total"] if switch else row["TOP10_total"]
        )

        vs_top10 = selected_total - row["TOP10_total"]

        rows.append({
            "candidate": candidate_name,
            "block": int(row["block"]),
            "block_start": row["block_start"],
            "block_end": row["block_end"],
            "prior_blocks_used": n,
            "selected_rule": selected,
            "switch": bool(switch),
            "prior_edge_mean": row["prior_edge_mean"],
            "prior_edge_last": row["prior_edge_last"],
            "prior_edge_last2_mean": row["prior_edge_last2_mean"],
            "prior_edge_last3_mean": row["prior_edge_last3_mean"],
            "prior_top45_positive_rate": row["prior_top45_positive_rate"],
            "prior_top10_positive_rate": row["prior_top10_positive_rate"],
            "selected_avg_diff": selected_avg,
            "selected_total_diff": selected_total,
            "top10_avg_diff": row["TOP10_avg"],
            "top10_total_diff": row["TOP10_total"],
            "top45_avg_diff": row["TOP4_5_avg"],
            "top45_total_diff": row["TOP4_5_total"],
            "vs_top10": vs_top10,
        })

    return pd.DataFrame(rows)


def summarize(daily, candidate_name):
    total_days = len(daily)
    switch_days = int(daily["switch"].sum())

    selected_total = daily["selected_total_diff"].sum()
    top10_total = daily["top10_total_diff"].sum()

    selected_avg = daily["selected_avg_diff"].mean()
    top10_avg = daily["top10_avg_diff"].mean()

    switch_df = daily[daily["switch"]].copy()

    if switch_days:
        switch_oos_vs_top10 = switch_df["vs_top10"]
        switch_vs_mean = switch_oos_vs_top10.mean()
        switch_positive = int((switch_oos_vs_top10 > 0).sum())
        switch_negative = int((switch_oos_vs_top10 < 0).sum())
        switch_total = switch_oos_vs_top10.sum()
    else:
        switch_vs_mean = np.nan
        switch_positive = 0
        switch_negative = 0
        switch_total = 0.0

    daily_vs = daily["vs_top10"]

    return {
        "candidate": candidate_name,
        "days": total_days,
        "eligible_switch_days": switch_days,
        "switch_rate": switch_days / total_days * 100 if total_days else np.nan,
        "selected_total_diff": selected_total,
        "top10_total_diff": top10_total,
        "selected_vs_top10_total": selected_total - top10_total,
        "selected_avg_diff": selected_avg,
        "top10_avg_diff": top10_avg,
        "switch_day_mean_vs_top10": switch_vs_mean,
        "switch_day_total_vs_top10": switch_total,
        "switch_day_positive": switch_positive,
        "switch_day_negative": switch_negative,
        "switch_day_positive_rate": (
            switch_positive / switch_days * 100
            if switch_days else np.nan
        ),
        "all_day_positive_vs_top10": int((daily_vs > 0).sum()),
        "all_day_negative_vs_top10": int((daily_vs < 0).sum()),
        "all_day_tie": int((daily_vs == 0).sum()),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("Ana-Slo Ver.4.2 Switch Trigger Candidate OOS Validation")
    print("=" * 72)

    df = load_daily()
    blocks = build_block_table(df)
    blocks = add_prior_features(blocks)

    print()
    print(f"blocks = {len(blocks)}")
    print(f"minimum prior blocks = {MIN_PRIOR_BLOCKS}")

    candidates = [
        "A_LAST_POSITIVE",
        "B_LAST_GE_300",
        "C_LAST_GE_500",
        "D_LAST_GE_750",
        "E_LAST2_POSITIVE",
        "F_LAST2_GE_300",
        "G_LAST_AND_LAST2_POSITIVE",
        "H_LAST_AND_LAST3_POSITIVE",
    ]

    all_daily = []
    summaries = []

    for name in candidates:
        daily = evaluate_candidate(blocks, name)
        all_daily.append(daily)
        summaries.append(summarize(daily, name))

    daily_out = pd.concat(all_daily, ignore_index=True)
    summary_out = pd.DataFrame(summaries)

    # 条件候補の比較ランキング。
    # 最優先は「switch後のOOSでTOP10を上回った総差」。
    summary_out = summary_out.sort_values(
        [
            "switch_day_total_vs_top10",
            "switch_day_positive_rate",
            "switch_rate",
        ],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)

    summary_out["ranking"] = np.arange(1, len(summary_out) + 1)

    # ブロック単位の診断。
    diagnostic_rows = []

    for name in candidates:
        sub = daily_out[daily_out["candidate"] == name].copy()

        for _, r in sub.iterrows():
            diagnostic_rows.append({
                "candidate": name,
                "block": r["block"],
                "selected_rule": r["selected_rule"],
                "switch": r["switch"],
                "prior_edge_last": r["prior_edge_last"],
                "prior_edge_last2_mean": r["prior_edge_last2_mean"],
                "prior_edge_last3_mean": r["prior_edge_last3_mean"],
                "oos_vs_top10": r["vs_top10"],
            })

    diagnostic = pd.DataFrame(diagnostic_rows)

    # 月別。
    daily_out["month"] = pd.to_datetime(
        daily_out["block_start"]
    ).dt.to_period("M").astype(str)

    monthly = (
        daily_out.groupby(["candidate", "month"], as_index=False)
        .agg(
            days=("block", "count"),
            switch_days=("switch", "sum"),
            selected_total_diff=("selected_total_diff", "sum"),
            top10_total_diff=("top10_total_diff", "sum"),
            selected_avg_diff=("selected_avg_diff", "mean"),
            vs_top10_total=("vs_top10", "sum"),
        )
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    daily_out.to_csv(
        OUT_DAILY, index=False, encoding="utf-8-sig"
    )
    blocks.to_csv(
        OUT_BLOCK, index=False, encoding="utf-8-sig"
    )
    summary_out.to_csv(
        OUT_SUMMARY, index=False, encoding="utf-8-sig"
    )
    diagnostic.to_csv(
        OUT_DIAGNOSTIC, index=False, encoding="utf-8-sig"
    )
    monthly.to_csv(
        OUT_MONTHLY, index=False, encoding="utf-8-sig"
    )

    print()
    print("=" * 72)
    print("CANDIDATE SUMMARY")
    print("=" * 72)

    print(
        summary_out[
            [
                "ranking",
                "candidate",
                "eligible_switch_days",
                "switch_rate",
                "selected_vs_top10_total",
                "switch_day_total_vs_top10",
                "switch_day_mean_vs_top10",
                "switch_day_positive",
                "switch_day_negative",
                "switch_day_positive_rate",
            ]
        ].to_string(index=False)
    )

    best = summary_out.iloc[0]

    print()
    print("=" * 72)
    print("BEST CANDIDATE BY CURRENT OOS DIAGNOSTIC")
    print("=" * 72)
    print(f"candidate                  : {best['candidate']}")
    print(f"switch days                : {int(best['eligible_switch_days'])}")
    print(f"switch rate                : {best['switch_rate']:.2f}%")
    print(
        f"switch-day total vs TOP10 : "
        f"{best['switch_day_total_vs_top10']:+.2f}"
    )
    print(
        f"switch-day mean vs TOP10  : "
        f"{best['switch_day_mean_vs_top10']:+.2f}"
    )
    print(
        f"switch-day positive       : "
        f"{int(best['switch_day_positive'])}"
    )
    print(
        f"switch-day negative       : "
        f"{int(best['switch_day_negative'])}"
    )

    print()
    print("IMPORTANT:")
    print(
        "This is a candidate trigger OOS diagnostic. "
        "The current sample is short."
    )
    print(
        "Candidate ranking is exploratory and must not be adopted "
        "as production tuning without a fresh OOS test."
    )
    print(
        "No current/test-block result is used to construct prior features."
    )

    print()
    print("FILES SAVED")
    print("=" * 72)
    print(OUT_DAILY)
    print(OUT_BLOCK)
    print(OUT_SUMMARY)
    print(OUT_DIAGNOSTIC)
    print(OUT_MONTHLY)
    print()
    print("Switch trigger candidate OOS validation complete.")


if __name__ == "__main__":
    main()
