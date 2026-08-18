# Ana-Slo Ver.4.2 Adaptive Switch Trigger Analysis
# Purpose:
#   Reverse-engineer WHY the existing Adaptive rule switched to TOP4_5.
#   This is a diagnostic analysis only. It does not tune a production rule.
#
# Input:
#   39_Ver4_2_long_term_adaptive_rank_band_fixed_daily.csv
#
# Output:
#   daily trigger panel
#   block trigger summary
#   switch-context analysis
#   trigger-feature analysis
#   diagnostic summary
#
# IMPORTANT:
#   - Only PRIOR OOS information is used for trigger features.
#   - The current/test block result is never used to construct its trigger.
#   - No threshold is selected automatically for production use.

from pathlib import Path
import numpy as np
import pandas as pd

INPUT_DAILY = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer\data\maruhan\_maebashi"
    r"\machine_number\analysis\_31days\_deep\39\_Ver4_2_long_term_adaptive_rank_band_fixed_daily.csv"
)

OUT_DIR = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer\data\maruhan\_maebashi"
    r"\machine_number\analysis\_31days\_deep\45\_Ver4_2_adaptive_switch_trigger"
)

OUT_DAILY = OUT_DIR / "45_Ver4_2_adaptive_switch_trigger_daily.csv"
OUT_BLOCK = OUT_DIR / "45_Ver4_2_adaptive_switch_trigger_block.csv"
OUT_CONTEXT = OUT_DIR / "45_Ver4_2_adaptive_switch_trigger_context.csv"
OUT_FEATURE = OUT_DIR / "45_Ver4_2_adaptive_switch_trigger_feature.csv"
OUT_DIAGNOSTIC = OUT_DIR / "45_Ver4_2_adaptive_switch_trigger_diagnostic.csv"

TARGET_RULE = "TOP4_5"
BASELINE_RULE = "TOP10"


def norm_rule(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def load_daily():
    if not INPUT_DAILY.exists():
        raise FileNotFoundError(
            f"Input daily CSV not found:\n{INPUT_DAILY}"
        )

    print(f"Loading: {INPUT_DAILY}")
    df = pd.read_csv(INPUT_DAILY)
    print(f"records = {len(df)}")
    print(f"columns = {list(df.columns)}")

    required = {
        "date",
        "block",
        "rule",
        "selected_rule",
        "avg_diff",
        "total_diff",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["block"] = pd.to_numeric(df["block"], errors="coerce")
    df["rule_norm"] = df["rule"].map(norm_rule)
    df["selected_rule_norm"] = df["selected_rule"].map(norm_rule)
    df["avg_diff"] = pd.to_numeric(df["avg_diff"], errors="coerce")
    df["total_diff"] = pd.to_numeric(df["total_diff"], errors="coerce")

    df = df.dropna(
        subset=["date", "block", "avg_diff", "total_diff"]
    ).copy()

    return df


def make_rule_block_table(df):
    # One row per block/rule.
    g = (
        df.groupby(["block", "rule_norm"], as_index=False)
        .agg(
            block_start=("date", "min"),
            block_end=("date", "max"),
            days=("date", "nunique"),
            avg_diff=("avg_diff", "mean"),
            total_diff=("total_diff", "sum"),
            median_daily_avg=("avg_diff", "median"),
            positive_days=("avg_diff", lambda x: int((x > 0).sum())),
            negative_days=("avg_diff", lambda x: int((x < 0).sum())),
        )
    )

    return g


def add_prior_features(block_table):
    rows = []

    for block_no in sorted(block_table["block"].unique()):
        prior = block_table[
            block_table["block"] < block_no
        ].copy()

        current = block_table[
            block_table["block"] == block_no
        ].copy()

        # Current selection is known from the source file.
        selected_values = current["rule_norm"].unique().tolist()

        selected_rule = ""
        if "selected_rule_norm" in current.columns:
            vals = current["selected_rule_norm"].dropna().unique().tolist()
            if vals:
                selected_rule = vals[0]

        # selected_rule is more reliably recovered directly from source.
        source_current = df_global[df_global["block"] == block_no]
        src_sel = source_current["selected_rule_norm"].dropna().unique()
        if len(src_sel):
            selected_rule = src_sel[0]

        def prior_rule_stats(rule):
            p = prior[prior["rule_norm"] == rule]
            if p.empty:
                return {
                    "blocks": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "last": np.nan,
                    "last2_mean": np.nan,
                    "last3_mean": np.nan,
                    "positive_blocks": 0,
                }

            vals = (
                p.sort_values("block")["avg_diff"]
                .astype(float)
                .to_numpy()
            )
            return {
                "blocks": len(vals),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "last": float(vals[-1]),
                "last2_mean": float(np.mean(vals[-2:])),
                "last3_mean": float(np.mean(vals[-3:])),
                "positive_blocks": int(np.sum(vals > 0)),
            }

        t = prior_rule_stats(TARGET_RULE)
        b = prior_rule_stats(BASELINE_RULE)

        def edge(a, c):
            if pd.isna(a) or pd.isna(c):
                return np.nan
            return a - c

        row = {
            "block": block_no,
            "block_start": current["date"].min(),
            "block_end": current["date"].max(),
            "selected_rule": selected_rule,
            "prior_blocks_used": len(
                prior["block"].unique()
            ),
            "prior_top45_mean": t["mean"],
            "prior_top10_mean": b["mean"],
            "prior_top45_minus_top10": edge(
                t["mean"], b["mean"]
            ),
            "prior_top45_median": t["median"],
            "prior_top10_median": b["median"],
            "prior_top45_last": t["last"],
            "prior_top10_last": b["last"],
            "prior_top45_last2_mean": t["last2_mean"],
            "prior_top10_last2_mean": b["last2_mean"],
            "prior_top45_last3_mean": t["last3_mean"],
            "prior_top10_last3_mean": b["last3_mean"],
            "prior_last_edge": edge(
                t["last"], b["last"]
            ),
            "prior_last2_edge": edge(
                t["last2_mean"], b["last2_mean"]
            ),
            "prior_last3_edge": edge(
                t["last3_mean"], b["last3_mean"]
            ),
            "prior_top45_positive_blocks": t["positive_blocks"],
            "prior_top10_positive_blocks": b["positive_blocks"],
            "prior_top45_positive_rate": (
                t["positive_blocks"] / t["blocks"] * 100
                if t["blocks"] else np.nan
            ),
            "prior_top10_positive_rate": (
                b["positive_blocks"] / b["blocks"] * 100
                if b["blocks"] else np.nan
            ),
        }

        row["selected_is_top45"] = (
            selected_rule == TARGET_RULE
        )
        row["selected_is_top10"] = (
            selected_rule == BASELINE_RULE
        )

        rows.append(row)

    return pd.DataFrame(rows)


def make_context_table(block_table, trigger):
    records = []

    for _, r in trigger.iterrows():
        block_no = int(r["block"])

        current = block_table[
            block_table["block"] == block_no
        ].copy()

        t = current[
            current["rule_norm"] == TARGET_RULE
        ]
        b = current[
            current["rule_norm"] == BASELINE_RULE
        ]

        if t.empty or b.empty:
            continue

        records.append(
            {
                "block": block_no,
                "block_start": r["block_start"],
                "block_end": r["block_end"],
                "selected_rule": r["selected_rule"],
                "prior_blocks_used": r["prior_blocks_used"],
                "prior_edge_mean": r["prior_top45_minus_top10"],
                "prior_edge_last": r["prior_last_edge"],
                "prior_edge_last2": r["prior_last2_edge"],
                "prior_edge_last3": r["prior_last3_edge"],
                "current_top45_avg": float(t["avg_diff"].iloc[0]),
                "current_top10_avg": float(b["avg_diff"].iloc[0]),
                "current_top45_total": float(t["total_diff"].iloc[0]),
                "current_top10_total": float(b["total_diff"].iloc[0]),
                "current_top45_minus_top10": (
                    float(t["avg_diff"].iloc[0])
                    - float(b["avg_diff"].iloc[0])
                ),
                "current_top45_selected": bool(
                    r["selected_is_top45"]
                ),
            }
        )

    return pd.DataFrame(records)


def feature_analysis(trigger):
    features = [
        "prior_top45_mean",
        "prior_top10_mean",
        "prior_top45_minus_top10",
        "prior_top45_last",
        "prior_top10_last",
        "prior_last_edge",
        "prior_top45_last2_mean",
        "prior_top10_last2_mean",
        "prior_last2_edge",
        "prior_top45_last3_mean",
        "prior_top10_last3_mean",
        "prior_last3_edge",
        "prior_top45_positive_rate",
        "prior_top10_positive_rate",
    ]

    rows = []

    sw = trigger[trigger["selected_is_top45"]].copy()
    ns = trigger[~trigger["selected_is_top45"]].copy()

    for col in features:
        if col not in trigger.columns:
            continue

        a = pd.to_numeric(sw[col], errors="coerce").dropna()
        b = pd.to_numeric(ns[col], errors="coerce").dropna()

        rows.append(
            {
                "feature": col,
                "switch_blocks": len(a),
                "non_switch_blocks": len(b),
                "switch_mean": (
                    float(a.mean()) if len(a) else np.nan
                ),
                "non_switch_mean": (
                    float(b.mean()) if len(b) else np.nan
                ),
                "switch_median": (
                    float(a.median()) if len(a) else np.nan
                ),
                "non_switch_median": (
                    float(b.median()) if len(b) else np.nan
                ),
                "difference_switch_minus_non_switch": (
                    float(a.mean() - b.mean())
                    if len(a) and len(b)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def diagnostic(trigger, context, features):
    rows = []

    switch_blocks = trigger[
        trigger["selected_is_top45"]
    ].copy()

    non_switch = trigger[
        ~trigger["selected_is_top45"]
    ].copy()

    rows.append(
        {
            "metric": "total_blocks",
            "value": len(trigger),
        }
    )
    rows.append(
        {
            "metric": "switch_to_top45_blocks",
            "value": len(switch_blocks),
        }
    )
    rows.append(
        {
            "metric": "switch_rate",
            "value": (
                len(switch_blocks) / len(trigger) * 100
                if len(trigger)
                else np.nan
            ),
        }
    )
    rows.append(
        {
            "metric": "first_switch_block",
            "value": (
                int(switch_blocks["block"].min())
                if len(switch_blocks)
                else np.nan
            ),
        }
    )
    rows.append(
        {
            "metric": "last_switch_block",
            "value": (
                int(switch_blocks["block"].max())
                if len(switch_blocks)
                else np.nan
            ),
        }
    )

    if len(switch_blocks):
        rows.append(
            {
                "metric": "switch_prior_mean_edge_avg",
                "value": float(
                    switch_blocks[
                        "prior_top45_minus_top10"
                    ].mean()
                ),
            }
        )
        rows.append(
            {
                "metric": "switch_prior_last_edge_avg",
                "value": float(
                    switch_blocks[
                        "prior_last_edge"
                    ].mean()
                ),
            }
        )

    if len(context):
        rows.append(
            {
                "metric": "switch_current_mean_edge_avg",
                "value": float(
                    context[
                        context["current_top45_selected"]
                    ]["current_top45_minus_top10"].mean()
                ),
            }
        )

    # Simple monotonic diagnostic:
    # Does the prior edge increase before switches?
    if len(switch_blocks):
        rows.append(
            {
                "metric": "switch_prior_edge_positive_rate",
                "value": float(
                    (
                        switch_blocks[
                            "prior_top45_minus_top10"
                        ] > 0
                    ).mean()
                    * 100
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    print("=" * 72)
    print("Ana-Slo Ver.4.2 Adaptive Switch Trigger Analysis")
    print("=" * 72)

    df = load_daily()

    global df_global
    df_global = df.copy()

    print()
    print("Building rule/block table...")
    block_table = make_rule_block_table(df)

    # Recover selected_rule at block level.
    selected_map = (
        df.groupby("block")["selected_rule_norm"]
        .agg(lambda x: next(
            (v for v in x.dropna().tolist() if v),
            ""
        ))
        .to_dict()
    )

    trigger = add_prior_features(block_table)

    trigger["selected_rule"] = trigger["block"].map(
        selected_map
    ).fillna(trigger["selected_rule"])

    trigger["selected_is_top45"] = (
        trigger["selected_rule"].map(norm_rule)
        == TARGET_RULE
    )
    trigger["selected_is_top10"] = (
        trigger["selected_rule"].map(norm_rule)
        == BASELINE_RULE
    )

    context = make_context_table(
        block_table,
        trigger,
    )

    features = feature_analysis(trigger)
    diag = diagnostic(
        trigger,
        context,
        features,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    trigger.to_csv(
        OUT_DAILY,
        index=False,
        encoding="utf-8-sig",
    )
    block_table.to_csv(
        OUT_BLOCK,
        index=False,
        encoding="utf-8-sig",
    )
    context.to_csv(
        OUT_CONTEXT,
        index=False,
        encoding="utf-8-sig",
    )
    features.to_csv(
        OUT_FEATURE,
        index=False,
        encoding="utf-8-sig",
    )
    diag.to_csv(
        OUT_DIAGNOSTIC,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 72)
    print("SWITCH SUMMARY")
    print("=" * 72)

    print(
        f"blocks                 : {len(trigger)}"
    )
    print(
        f"TOP4_5 switch blocks   : "
        f"{int(trigger['selected_is_top45'].sum())}"
    )
    print(
        f"switch rate            : "
        f"{trigger['selected_is_top45'].mean() * 100:.2f}%"
    )

    print()
    print(
        trigger[
            [
                "block",
                "block_start",
                "block_end",
                "selected_rule",
                "prior_blocks_used",
                "prior_top45_minus_top10",
                "prior_last_edge",
                "prior_last2_edge",
                "prior_last3_edge",
                "prior_top45_positive_rate",
                "prior_top10_positive_rate",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 72)
    print("FEATURE COMPARISON: SWITCH vs NON-SWITCH")
    print("=" * 72)
    print(features.to_string(index=False))

    print()
    print("=" * 72)
    print("SWITCH CONTEXT")
    print("=" * 72)

    if len(context):
        print(
            context.to_string(index=False)
        )
    else:
        print("No context rows available.")

    print()
    print("=" * 72)
    print("FILES SAVED")
    print("=" * 72)

    print(OUT_DAILY)
    print(OUT_BLOCK)
    print(OUT_CONTEXT)
    print(OUT_FEATURE)
    print(OUT_DIAGNOSTIC)

    print()
    print("IMPORTANT:")
    print(
        "This is a trigger reverse-engineering diagnostic."
    )
    print(
        "It does not prove that any trigger is predictive."
    )
    print(
        "No current/test-block result is used to construct prior features."
    )
    print(
        "Do not adopt a threshold from this analysis without a new OOS test."
    )


if __name__ == "__main__":
    main()
