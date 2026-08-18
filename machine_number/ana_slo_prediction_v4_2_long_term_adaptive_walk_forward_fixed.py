
from pathlib import Path
import math
import numpy as np
import pandas as pd


print("=" * 72)
print("Ana-Slo Ver.4.2 Rank Band Walk-Forward Validation")
print("=" * 72)


# ============================================================
# PATH
# ============================================================

BASE = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")

DATA_DIR = BASE / "data" / "maruhan_maebashi" / "machine_number"
OUT_DIR = DATA_DIR / "analysis_31days_deep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAY11_FILE = DATA_DIR / "ana_slo_20260711.csv"
MERGED_FILE = DATA_DIR / "ana_slo_20260712_20260810.csv"

# ============================================================
# PERIOD / WALK-FORWARD
# ============================================================

DATA_START = pd.Timestamp("2026-04-01")
TEST_START = pd.Timestamp("2026-04-15")
TEST_END = pd.Timestamp("2026-08-10")
BLOCK_SIZE = 4

# ============================================================
# V4.2-C
# recent7_win / bounce_signal を除外
# 元V4固定重みを除外後に再正規化
# ============================================================

RAW_WEIGHTS = {
    "avg31": 0.067095,
    "recent7_avg": 0.051649,
    "recent7_win": 0.066030,
    "last_diff": 0.123823,
    "prev_change": 0.104847,
    "weekday_avg": 0.056727,
    "type_avg": 0.058437,
    "plus1000_rate": 0.177254,
    "plus2000_rate": 0.132989,
    "neighbor_avg": 0.061613,
    "bounce_signal": 0.099536,
}

EXCLUDED = {"recent7_win", "bounce_signal"}

FEATURES = [
    k for k in RAW_WEIGHTS
    if k not in EXCLUDED
]

weight_sum = sum(RAW_WEIGHTS[k] for k in FEATURES)

WEIGHTS = {
    k: RAW_WEIGHTS[k] / weight_sum
    for k in FEATURES
}

# 今回比較する順位帯
RULES = {
    "TOP10": list(range(1, 11)),
    "TOP4_10": list(range(4, 11)),
    "TOP4_5": list(range(4, 6)),
    "TOP4_8": list(range(4, 9)),
    "TOP5_10": list(range(5, 11)),
    "TOP6_10": list(range(6, 11)),
}

OUT_DAILY = OUT_DIR / "36_Ver4_2_rank_band_walk_forward_daily.csv"
OUT_SUMMARY = OUT_DIR / "36_Ver4_2_rank_band_walk_forward_summary.csv"
OUT_BLOCKS = OUT_DIR / "36_Ver4_2_rank_band_walk_forward_blocks.csv"
OUT_DIAGNOSTIC = OUT_DIR / "36_Ver4_2_rank_band_walk_forward_diagnostic.csv"


# ============================================================
# CSV
# ============================================================

def read_csv(path):
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"CSV read failed: {path}")


def find_col(df, candidates):
    normalized = {
        str(c).strip().lower().replace(" ", ""): c
        for c in df.columns
    }

    for candidate in candidates:
        key = str(candidate).strip().lower().replace(" ", "")
        if key in normalized:
            return normalized[key]

    for c in df.columns:
        nc = str(c).strip().lower().replace(" ", "")
        for candidate in candidates:
            key = str(candidate).strip().lower().replace(" ", "")
            if key in nc or nc in key:
                return c

    return None


def load_data():
    # Long-term mode: load all Ana-Slo CSVs already stored in machine_number.
    candidates = sorted(DATA_DIR.glob("ana_slo*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Ana-Slo CSVが見つかりません: {DATA_DIR}")

    frames = []
    for p in candidates:
        print(f"Loading: {p}")
        frames.append(read_csv(p))

    df = pd.concat(frames, ignore_index=True)

    date_col = find_col(df, ["date", "日付", "対象日"])
    no_col = find_col(df, ["machine_no", "machine_number", "台番号", "台番", "台No", "台NO"])
    name_col = find_col(df, ["machine_name", "machine", "機種名", "機種"])
    diff_col = find_col(df, ["diff", "差枚", "差枚数", "差枚数(枚)"])

    if not all([date_col, no_col, name_col, diff_col]):
        raise ValueError(
            f"必要列が見つかりません: date={date_col}, "
            f"no={no_col}, name={name_col}, diff={diff_col}"
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
    df = df[(df["date"] >= DATA_START) & (df["date"] <= TEST_END)].copy()
    df = df.sort_values(["date", "machine_no"])
    df = df.drop_duplicates(["date", "machine_no"], keep="last")
    df["plus1000"] = (df["diff"] >= 1000).astype(int)
    df["plus2000"] = (df["diff"] >= 2000).astype(int)

    print(f"records = {len(df):,}")
    return df

def zscore(s):
    x = pd.to_numeric(s, errors="coerce").fillna(0.0)
    std = float(x.std(ddof=0))

    if std == 0 or math.isnan(std):
        return pd.Series(0.0, index=x.index)

    return (x - float(x.mean())) / std


def build_features(df, target_date):
    hist = df[df["date"] < target_date].copy()
    actual = df[df["date"] == target_date][
        ["machine_no", "machine_name", "diff"]
    ].copy()

    if hist.empty or actual.empty:
        return pd.DataFrame()

    latest_date = hist["date"].max()

    # 直近履歴に同じ台番号が存在する台だけを予測対象にする。
    # 機種変更台は当日の機種と前日機種が一致しない場合に除外。
    latest_machine = (
        hist[hist["date"] == latest_date]
        [["machine_no", "machine_name"]]
        .drop_duplicates("machine_no", keep="last")
        .rename(columns={"machine_name": "hist_machine_name"})
    )

    actual = actual.merge(
        latest_machine,
        on="machine_no",
        how="left",
    )

    actual = actual[
        actual["hist_machine_name"].notna()
        & (
            actual["machine_name"]
            == actual["hist_machine_name"]
        )
    ].copy()

    if actual.empty:
        return pd.DataFrame()

    type_stats = (
        hist.groupby("machine_name")["diff"]
        .mean()
        .to_dict()
    )

    target_weekday = target_date.dayofweek

    latest_day = (
        hist[hist["date"] == latest_date]
        .set_index("machine_no")
    )

    rows = []

    for machine_no, m in hist.groupby("machine_no"):
        m = m.sort_values("date").copy()

        if m.empty:
            continue

        name = str(m.iloc[-1]["machine_name"])

        avg31 = float(m["diff"].mean())

        recent7 = m.tail(7)
        recent7_avg = float(recent7["diff"].mean())

        last_diff = float(m.iloc[-1]["diff"])

        if len(m) >= 2:
            prev_diff = float(m.iloc[-2]["diff"])
        else:
            prev_diff = last_diff

        prev_change = last_diff - prev_diff

        wd = m[
            m["date"].dt.dayofweek == target_weekday
        ]

        weekday_n = len(wd)

        if weekday_n:
            weekday_raw = float(wd["diff"].mean())
        else:
            weekday_raw = avg31

        # 少数曜日サンプルを平均へ縮約
        prior_n = 15.0
        wd_weight = weekday_n / (weekday_n + prior_n)

        weekday_avg = (
            weekday_raw * wd_weight
            + avg31 * (1.0 - wd_weight)
        )

        plus1000_rate = float(
            m["plus1000"].mean()
        )

        plus2000_rate = float(
            m["plus2000"].mean()
        )

        type_avg = float(
            type_stats.get(name, avg31)
        )

        neighbors = []

        for n2 in (machine_no - 1, machine_no + 1):
            if n2 in latest_day.index:
                neighbors.append(
                    float(latest_day.loc[n2, "diff"])
                )

        if neighbors:
            neighbor_avg = float(np.mean(neighbors))
        else:
            neighbor_avg = 0.0

        rows.append(
            {
                "machine_no": int(machine_no),
                "machine_name": name,
                "avg31": avg31,
                "recent7_avg": recent7_avg,
                "last_diff": last_diff,
                "prev_change": prev_change,
                "weekday_avg": weekday_avg,
                "type_avg": type_avg,
                "plus1000_rate": plus1000_rate,
                "plus2000_rate": plus2000_rate,
                "neighbor_avg": neighbor_avg,
            }
        )

    features = pd.DataFrame(rows)

    if features.empty:
        return pd.DataFrame()

    panel = features.merge(
        actual[["machine_no", "machine_name", "diff"]],
        on=["machine_no", "machine_name"],
        how="inner",
    )

    return panel


def score_panel(panel):
    panel = panel.copy()

    for feature in FEATURES:
        panel[f"{feature}_z"] = zscore(
            panel[feature]
        )

    panel["score"] = 0.0

    for feature in FEATURES:
        panel["score"] += (
            panel[f"{feature}_z"]
            * WEIGHTS[feature]
        )

    panel = panel.sort_values(
        ["score", "machine_no"],
        ascending=[False, True],
    ).reset_index(drop=True)

    panel["rank"] = np.arange(len(panel)) + 1

    return panel


# ============================================================
# WALK-FORWARD BLOCKS
# ============================================================

def make_blocks(df):
    # Use only dates that actually exist in the loaded dataset.
    dates = sorted(
        pd.to_datetime(
            df.loc[
                (df["date"] >= TEST_START) & (df["date"] <= TEST_END),
                "date"
            ]
        ).dropna().dt.normalize().unique()
    )

    if not dates:
        raise RuntimeError(
            f"検証期間 {TEST_START.date()}～{TEST_END.date()} に"
            "実データの日付がありません。"
        )

    blocks = []
    for i in range(0, len(dates), BLOCK_SIZE):
        d = dates[i:i + BLOCK_SIZE]
        blocks.append({
            "block": len(blocks) + 1,
            "start": pd.Timestamp(d[0]),
            "end": pd.Timestamp(d[-1]),
            "observed_days": len(d),
        })
    return blocks

def main():
    print()
    print("NORMALIZED WEIGHTS")
    print("-" * 72)

    for feature in FEATURES:
        print(
            f"{feature:18s}: "
            f"{WEIGHTS[feature] * 100:8.3f}%"
        )

    print(
        f"{'weight sum':18s}: "
        f"{sum(WEIGHTS.values()) * 100:8.3f}%"
    )

    df = load_data()
    blocks = make_blocks(df)

    print()
    print(f"walk-forward blocks = {len(blocks)}")

    for b in blocks:
        print(
            f"BLOCK {b['block']}: "
            f"{b['start'].date()} to "
            f"{b['end'].date()}"
        )

    daily_rows = []
    block_rows = []

    for b in blocks:
        block_no = b["block"]
        block_start = b["start"]
        block_end = b["end"]

        print()
        print(
            f"BLOCK {block_no}: "
            f"{block_start.date()} to "
            f"{block_end.date()}"
        )

        block_daily = []

        dates = pd.date_range(
            block_start,
            block_end,
            freq="D",
        )

        for target_date in dates:
            panel = build_features(
                df,
                target_date,
            )

            if panel.empty:
                print(
                    f"{target_date.date()} eligible=0"
                )
                continue

            panel = score_panel(panel)

            print(
                f"{target_date.date()} "
                f"eligible={len(panel)}"
            )

            for rule_name, ranks in RULES.items():
                selected = panel[
                    panel["rank"].isin(ranks)
                ].copy()

                if selected.empty:
                    continue

                row = {
                    "date": target_date,
                    "block": block_no,
                    "block_start": block_start,
                    "block_end": block_end,
                    "rule": rule_name,
                    "machines": len(selected),
                    "avg_diff": float(
                        selected["diff"].mean()
                    ),
                    "median_diff": float(
                        selected["diff"].median()
                    ),
                    "win_rate": float(
                        selected["diff"].gt(0).mean()
                        * 100
                    ),
                    "plus1000_rate": float(
                        selected["diff"].ge(1000).mean()
                        * 100
                    ),
                    "plus2000_rate": float(
                        selected["diff"].ge(2000).mean()
                        * 100
                    ),
                    "positive": int(
                        selected["diff"].gt(0).sum()
                    ),
                    "total_diff": float(
                        selected["diff"].sum()
                    ),
                }

                daily_rows.append(row)
                block_daily.append(row)

        block_df = pd.DataFrame(block_daily)

        for rule_name in RULES:
            sub = block_df[
                block_df["rule"] == rule_name
            ]

            if sub.empty:
                continue

            block_rows.append(
                {
                    "block": block_no,
                    "block_start": block_start,
                    "block_end": block_end,
                    "rule": rule_name,
                    "days": len(sub),
                    "avg_diff": float(
                        sub["avg_diff"].mean()
                    ),
                    "total_diff": float(
                        sub["total_diff"].sum()
                    ),
                    "win_rate": float(
                        sub["win_rate"].mean()
                    ),
                    "plus1000_rate": float(
                        sub["plus1000_rate"].mean()
                    ),
                    "plus2000_rate": float(
                        sub["plus2000_rate"].mean()
                    ),
                    "positive_days": int(
                        sub["avg_diff"].gt(0).sum()
                    ),
                    "negative_days": int(
                        sub["avg_diff"].lt(0).sum()
                    ),
                    "positive_day_rate": float(
                        sub["avg_diff"].gt(0).mean()
                        * 100
                    ),
                }
            )

    daily = pd.DataFrame(daily_rows)
    blocks_df = pd.DataFrame(block_rows)

    if daily.empty:
        raise RuntimeError(
            "Walk-forward結果が空です。"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows = []

    for rule_name in RULES:
        sub = daily[
            daily["rule"] == rule_name
        ].copy()

        if sub.empty:
            continue

        avg = sub["avg_diff"]

        cumulative = sub["total_diff"].cumsum()
        peak = cumulative.cummax()
        drawdown = cumulative - peak

        max_losing = 0
        current_losing = 0

        for value in avg:
            if value < 0:
                current_losing += 1
                max_losing = max(
                    max_losing,
                    current_losing,
                )
            else:
                current_losing = 0

        bsub = blocks_df[
            blocks_df["rule"] == rule_name
        ]

        summary_rows.append(
            {
                "rule": rule_name,
                "days": len(sub),
                "blocks": len(bsub),
                "avg_diff": float(avg.mean()),
                "median_daily_avg": float(avg.median()),
                "std_daily_avg": float(
                    avg.std(ddof=0)
                ),
                "best_day": float(avg.max()),
                "worst_day": float(avg.min()),
                "win_rate": float(
                    sub["win_rate"].mean()
                ),
                "plus1000_rate": float(
                    sub["plus1000_rate"].mean()
                ),
                "plus2000_rate": float(
                    sub["plus2000_rate"].mean()
                ),
                "positive_days": int(
                    avg.gt(0).sum()
                ),
                "negative_days": int(
                    avg.lt(0).sum()
                ),
                "positive_day_rate": float(
                    avg.gt(0).mean() * 100
                ),
                "positive_blocks": int(
                    bsub["avg_diff"].gt(0).sum()
                ),
                "negative_blocks": int(
                    bsub["avg_diff"].lt(0).sum()
                ),
                "positive_block_rate": float(
                    bsub["avg_diff"].gt(0).mean()
                    * 100
                ),
                "max_losing_streak": max_losing,
                "total_diff": float(
                    sub["total_diff"].sum()
                ),
                "per_machine_avg_diff": float(
                    sub["total_diff"].sum()
                    / sub["machines"].sum()
                ),
                "max_drawdown": float(
                    drawdown.min()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    # TOP10との比較
    top10 = summary[
        summary["rule"] == "TOP10"
    ].iloc[0]

    summary["avg_diff_vs_top10"] = (
        summary["avg_diff"]
        - float(top10["avg_diff"])
    )

    summary["total_diff_vs_top10"] = (
        summary["total_diff"]
        - float(top10["total_diff"])
    )

    summary["avg_rank"] = (
        summary["avg_diff"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    summary["block_stability_rank"] = (
        summary["positive_block_rate"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    # ========================================================
    # BLOCK COMPARISON
    # ========================================================

    top10_blocks = blocks_df[
        blocks_df["rule"] == "TOP10"
    ][
        ["block", "avg_diff", "total_diff"]
    ].rename(
        columns={
            "avg_diff": "top10_avg_diff",
            "total_diff": "top10_total_diff",
        }
    )

    block_compare = blocks_df.merge(
        top10_blocks,
        on="block",
        how="left",
    )

    block_compare["avg_diff_vs_top10"] = (
        block_compare["avg_diff"]
        - block_compare["top10_avg_diff"]
    )

    block_compare["total_diff_vs_top10"] = (
        block_compare["total_diff"]
        - block_compare["top10_total_diff"]
    )

    # ========================================================
    # DIAGNOSTIC
    # ========================================================

    diagnostic_rows = []

    for rule_name in RULES:
        sub = block_compare[
            block_compare["rule"] == rule_name
        ]

        if sub.empty:
            continue

        better = int(
            sub["avg_diff_vs_top10"].gt(0).sum()
        )

        worse = int(
            sub["avg_diff_vs_top10"].lt(0).sum()
        )

        ties = int(
            sub["avg_diff_vs_top10"].eq(0).sum()
        )

        diagnostic_rows.append(
            {
                "rule": rule_name,
                "blocks": len(sub),
                "better_than_top10_blocks": better,
                "worse_than_top10_blocks": worse,
                "tie_blocks": ties,
                "better_block_rate": (
                    better / len(sub) * 100
                ),
                "mean_block_improvement": float(
                    sub["avg_diff_vs_top10"].mean()
                ),
                "minimum_block_improvement": float(
                    sub["avg_diff_vs_top10"].min()
                ),
                "maximum_block_improvement": float(
                    sub["avg_diff_vs_top10"].max()
                ),
            }
        )

    diagnostic = pd.DataFrame(
        diagnostic_rows
    )

    # ========================================================
    # SAVE
    # ========================================================

    daily.to_csv(
        OUT_DAILY,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    block_compare.to_csv(
        OUT_BLOCKS,
        index=False,
        encoding="utf-8-sig",
    )

    diagnostic.to_csv(
        OUT_DIAGNOSTIC,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("=" * 72)
    print("VER.4.2 RANK BAND WALK-FORWARD RESULT")
    print("=" * 72)

    display_cols = [
        "rule",
        "days",
        "blocks",
        "avg_diff",
        "median_daily_avg",
        "best_day",
        "worst_day",
        "win_rate",
        "plus1000_rate",
        "plus2000_rate",
        "positive_day_rate",
        "positive_block_rate",
        "max_losing_streak",
        "total_diff",
        "per_machine_avg_diff",
        "max_drawdown",
        "avg_diff_vs_top10",
    ]

    print(
        summary[
            display_cols
        ]
        .sort_values(
            "avg_diff",
            ascending=False,
        )
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print()
    print("=" * 72)
    print("BLOCK RESULT")
    print("=" * 72)

    print(
        block_compare[
            [
                "block",
                "block_start",
                "block_end",
                "rule",
                "avg_diff",
                "total_diff",
                "positive_day_rate",
                "avg_diff_vs_top10",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print()
    print("=" * 72)
    print("ROBUSTNESS DIAGNOSTIC")
    print("=" * 72)

    print(
        diagnostic.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    best = summary.loc[
        summary["avg_diff"].idxmax()
    ]

    best_diag = diagnostic[
        diagnostic["rule"] == best["rule"]
    ]

    better_blocks = int(
        best_diag[
            "better_than_top10_blocks"
        ].iloc[0]
    )

    print()
    print(
        f"Best rule by OOS average diff : "
        f"{best['rule']}"
    )
    print(
        f"Average diff                  : "
        f"{best['avg_diff']:+.2f}"
    )
    print(
        f"Total diff                    : "
        f"{best['total_diff']:+.0f}"
    )
    print(
        f"Positive block rate           : "
        f"{best['positive_block_rate']:.2f}%"
    )
    print(
        f"Better blocks vs TOP10        : "
        f"{better_blocks}/{len(blocks)}"
    )

    print()
    print("=" * 72)
    print("FILES SAVED")
    print("=" * 72)

    print(OUT_DAILY)
    print(OUT_SUMMARY)
    print(OUT_BLOCKS)
    print(OUT_DIAGNOSTIC)

    print()
    print(
        "Ver.4.2 rank band walk-forward validation complete."
    )



def adaptive_main():
    print()
    print("=" * 72)
    print("Ana-Slo Ver.4.2 Adaptive Rank Band Nested Walk-Forward")
    print("=" * 72)

    df = load_data()
    blocks = make_blocks()

    print(f"walk-forward blocks = {len(blocks)}")

    daily_rows = []
    block_rows = []
    selection_rows = []

    # Rule selection uses ONLY completed prior test blocks.
    # Block 1 has no prior OOS result, so TOP10 is the neutral baseline.
    for b in blocks:
        prior_blocks = pd.DataFrame(block_rows)

        if prior_blocks.empty:
            selected_rule = "TOP10"
            reason = "NO_PRIOR_OOS"
        else:
            agg = (
                prior_blocks.groupby("rule")
                .agg(
                    mean_avg_diff=("avg_diff", "mean"),
                    total_diff=("total_diff", "sum"),
                    positive_blocks=("avg_diff", lambda x: int((x > 0).sum())),
                )
                .reset_index()
            )
            rule_order = {r: i for i, r in enumerate(RULES)}
            agg["order"] = agg["rule"].map(rule_order)
            agg = agg.sort_values(
                ["mean_avg_diff", "total_diff", "positive_blocks", "order"],
                ascending=[False, False, False, True],
            )
            selected_rule = str(agg.iloc[0]["rule"])
            reason = "PRIOR_OOS_BEST"

        print()
        print(
            f"BLOCK {b['block']}: "
            f"{b['start'].date()} to {b['end'].date()}"
        )
        print(f"Selected rule = {selected_rule} ({reason})")

        selection_rows.append({
            "block": b["block"],
            "block_start": b["start"],
            "block_end": b["end"],
            "selected_rule": selected_rule,
            "selection_reason": reason,
            "prior_blocks_used": b["block"] - 1,
        })

        block_daily = []

        for target_date in pd.date_range(b["start"], b["end"], freq="D"):
            panel = build_features(df, target_date)
            if panel.empty:
                continue

            panel = score_panel(panel)

            for rule_name, ranks in RULES.items():
                selected = panel[panel["rank"].isin(ranks)]
                if selected.empty:
                    continue

                row = {
                    "date": target_date,
                    "block": b["block"],
                    "block_start": b["start"],
                    "block_end": b["end"],
                    "rule": rule_name,
                    "selected_rule": selected_rule,
                    "is_adaptive_selection": rule_name == selected_rule,
                    "machines": len(selected),
                    "avg_diff": float(selected["diff"].mean()),
                    "median_diff": float(selected["diff"].median()),
                    "win_rate": float(selected["diff"].gt(0).mean() * 100),
                    "plus1000_rate": float(selected["diff"].ge(1000).mean() * 100),
                    "plus2000_rate": float(selected["diff"].ge(2000).mean() * 100),
                    "positive": int(selected["diff"].gt(0).sum()),
                    "total_diff": float(selected["diff"].sum()),
                }
                daily_rows.append(row)
                block_daily.append(row)

        block_df = pd.DataFrame(block_daily)

        if block_df.empty:
            print(
                f"  NO OOS DATA in block {b['block']}; block skipped."
            )
            continue

        for rule_name in RULES:
            sub = block_df[block_df["rule"] == rule_name]
            if sub.empty:
                continue

            block_rows.append({
                "block": b["block"],
                "block_start": b["start"],
                "block_end": b["end"],
                "rule": rule_name,
                "selected_rule": selected_rule,
                "is_adaptive_selection": rule_name == selected_rule,
                "days": len(sub),
                "avg_diff": float(sub["avg_diff"].mean()),
                "total_diff": float(sub["total_diff"].sum()),
                "win_rate": float(sub["win_rate"].mean()),
                "plus1000_rate": float(sub["plus1000_rate"].mean()),
                "plus2000_rate": float(sub["plus2000_rate"].mean()),
                "positive_days": int(sub["avg_diff"].gt(0).sum()),
                "negative_days": int(sub["avg_diff"].lt(0).sum()),
                "positive_day_rate": float(sub["avg_diff"].gt(0).mean() * 100),
            })

    daily = pd.DataFrame(daily_rows)
    blocks_df = pd.DataFrame(block_rows)
    selections = pd.DataFrame(selection_rows)

    if daily.empty or blocks_df.empty:
        raise RuntimeError(
            "OOS結果が1件も作成されませんでした。"
            "対象期間のAna-Sloデータを確認してください。"
        )

    adaptive = daily[daily["is_adaptive_selection"]].copy()

    adaptive_monthly = adaptive.copy()
    adaptive_monthly["month"] = pd.to_datetime(
        adaptive_monthly["date"]
    ).dt.to_period("M").astype(str)
    monthly = (
        adaptive_monthly.groupby("month")
        .agg(
            days=("date", "count"),
            avg_diff=("avg_diff", "mean"),
            total_diff=("total_diff", "sum"),
            positive_day_rate=("avg_diff", lambda x: float((x > 0).mean() * 100)),
        )
        .reset_index()
    )
    adaptive_blocks = blocks_df[blocks_df["is_adaptive_selection"]].copy()

    cumulative = adaptive["total_diff"].cumsum()
    drawdown = cumulative - cumulative.cummax()

    max_losing = 0
    streak = 0
    for x in adaptive["avg_diff"]:
        if x < 0:
            streak += 1
            max_losing = max(max_losing, streak)
        else:
            streak = 0

    adaptive_summary = pd.DataFrame([{
        "model": "V4.2_C_ADAPTIVE",
        "days": len(adaptive),
        "blocks": len(adaptive_blocks),
        "avg_diff": float(adaptive["avg_diff"].mean()),
        "median_daily_avg": float(adaptive["avg_diff"].median()),
        "best_day": float(adaptive["avg_diff"].max()),
        "worst_day": float(adaptive["avg_diff"].min()),
        "win_rate": float(adaptive["win_rate"].mean()),
        "plus1000_rate": float(adaptive["plus1000_rate"].mean()),
        "plus2000_rate": float(adaptive["plus2000_rate"].mean()),
        "positive_day_rate": float(adaptive["avg_diff"].gt(0).mean() * 100),
        "positive_block_rate": float(adaptive_blocks["avg_diff"].gt(0).mean() * 100),
        "max_losing_streak": max_losing,
        "total_diff": float(adaptive["total_diff"].sum()),
        "per_machine_avg_diff": float(adaptive["total_diff"].sum() / adaptive["machines"].sum()),
        "max_drawdown": float(drawdown.min()),
    }])

    comparison = []
    for rule_name in RULES:
        sub = daily[daily["rule"] == rule_name]
        comparison.append({
            "rule": rule_name,
            "days": len(sub),
            "avg_diff": float(sub["avg_diff"].mean()),
            "total_diff": float(sub["total_diff"].sum()),
            "positive_day_rate": float(sub["avg_diff"].gt(0).mean() * 100),
        })

    comparison.append({
        "rule": "ADAPTIVE",
        "days": len(adaptive),
        "avg_diff": float(adaptive["avg_diff"].mean()),
        "total_diff": float(adaptive["total_diff"].sum()),
        "positive_day_rate": float(adaptive["avg_diff"].gt(0).mean() * 100),
    })
    comparison = pd.DataFrame(comparison)

    diagnostic = selections.copy()
    if not diagnostic.empty and not adaptive_blocks.empty:
        diagnostic["adaptive_avg_diff"] = diagnostic["block"].map(
            adaptive_blocks.set_index("block")["avg_diff"]
        )
    else:
        diagnostic["adaptive_avg_diff"] = np.nan
    top10_blocks = blocks_df[blocks_df["rule"] == "TOP10"][
        ["block", "avg_diff"]
    ].rename(columns={"avg_diff": "top10_avg_diff"})
    diagnostic = diagnostic.merge(top10_blocks, on="block", how="left")
    diagnostic["adaptive_vs_top10"] = (
        diagnostic["adaptive_avg_diff"] - diagnostic["top10_avg_diff"]
    )

    out_dir = OUT_DIR
    out_daily = out_dir / "39_Ver4_2_long_term_adaptive_rank_band_fixed_daily.csv"
    out_block = out_dir / "39_Ver4_2_long_term_adaptive_rank_band_fixed_blocks.csv"
    out_selection = out_dir / "39_Ver4_2_long_term_adaptive_rank_band_fixed_selection.csv"
    out_summary = out_dir / "39_Ver4_2_long_term_adaptive_rank_band_fixed_summary.csv"
    out_diag = out_dir / "39_Ver4_2_long_term_adaptive_rank_band_fixed_diagnostic.csv"
    out_monthly = out_dir / "39_Ver4_2_long_term_adaptive_rank_band_fixed_monthly.csv"

    daily.to_csv(out_daily, index=False, encoding="utf-8-sig")
    blocks_df.to_csv(out_block, index=False, encoding="utf-8-sig")
    selections.to_csv(out_selection, index=False, encoding="utf-8-sig")
    adaptive_summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
    diagnostic.to_csv(out_diag, index=False, encoding="utf-8-sig")
    monthly.to_csv(out_monthly, index=False, encoding="utf-8-sig")

    print()
    print("=" * 72)
    print("ADAPTIVE SELECTION")
    print("=" * 72)
    print(selections.to_string(index=False))

    print()
    print("=" * 72)
    print("ADAPTIVE RESULT")
    print("=" * 72)
    print(adaptive_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print()
    print("=" * 72)
    print("FIXED RULE COMPARISON")
    print("=" * 72)
    print(comparison.sort_values("avg_diff", ascending=False).to_string(
        index=False, float_format=lambda x: f"{x:.2f}"
    ))

    print()
    print("=" * 72)
    print("ADAPTIVE vs TOP10 BY BLOCK")
    print("=" * 72)
    print(diagnostic.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print()
    print("FILES SAVED")
    print(out_daily)
    print(out_block)
    print(out_selection)
    print(out_summary)
    print(out_diag)
    print(out_monthly)
    print()
    print("Ver.4.2 long-term adaptive rank band walk-forward FIXED complete.")


if __name__ == "__main__":
    adaptive_main()
