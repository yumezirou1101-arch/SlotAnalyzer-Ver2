from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Ana-Slo Ver.4.2 Normal Day Comparison
#
# Purpose:
#   Check whether V4.2_C remains superior to V4_BASE
#   after excluding the machine-change day (2026-08-03).
#
# Input:
#   24_Ver4_2_rolling_daily.csv
#
# Models:
#   V4_BASE
#   V4.2_C
#
# Top N:
#   5 / 10 / 20 / 30
#
# ============================================================


print("=" * 70)
print("Ana-Slo Ver.4.2 Normal Day Comparison")
print("=" * 70)


# ============================================================
# PATH
# ============================================================

BASE = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

DATA_DIR = (
    BASE
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
)

OUT_DIR = (
    DATA_DIR
    / "analysis_31days_deep"
)

INPUT = (
    OUT_DIR
    / "24_Ver4_2_rolling_daily.csv"
)

OUTPUT_DAILY = (
    OUT_DIR
    / "27_Ver4_2_normal_day_daily.csv"
)

OUTPUT_SUMMARY = (
    OUT_DIR
    / "27_Ver4_2_normal_day_summary.csv"
)

OUTPUT_COMPARE = (
    OUT_DIR
    / "27_Ver4_2_normal_day_compare.csv"
)

OUTPUT_EXCLUDED = (
    OUT_DIR
    / "27_Ver4_2_excluded_day_effect.csv"
)


# ============================================================
# SETTINGS
# ============================================================

BASE_MODEL = "V4_BASE"
CANDIDATE_MODEL = "V4.2_C"

TOP_NS = [5, 10, 20, 30]

# Machine-change day confirmed by previous diagnostics.
EXCLUDE_DATES = {
    pd.Timestamp("2026-08-03")
}


# ============================================================
# LOAD
# ============================================================

if not INPUT.exists():

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT}"
    )


df = pd.read_csv(
    INPUT,
    encoding="utf-8-sig"
)


print()
print("INPUT")
print("-" * 70)
print(INPUT)
print(f"records = {len(df):,}")

print()
print("columns =")
print(list(df.columns))


# ============================================================
# NORMALIZE
# ============================================================

required = [
    "date",
    "model",
    "top_n",
    "avg_diff",
    "total_diff",
    "win_rate",
    "positive",
]


missing = [
    c for c in required
    if c not in df.columns
]

if missing:

    raise ValueError(
        "Required columns missing: "
        + ", ".join(missing)
    )


df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["model"] = (
    df["model"]
    .astype(str)
    .str.strip()
)

df["top_n"] = pd.to_numeric(
    df["top_n"],
    errors="coerce"
)

for col in [
    "avg_diff",
    "total_diff",
    "win_rate",
    "positive",
]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df = df.dropna(
    subset=[
        "date",
        "model",
        "top_n",
        "avg_diff",
        "total_diff",
    ]
).copy()


df["top_n"] = (
    df["top_n"]
    .astype(int)
)


# ============================================================
# SELECT MODELS
# ============================================================

df = df[
    df["model"].isin(
        [
            BASE_MODEL,
            CANDIDATE_MODEL
        ]
    )
    &
    df["top_n"].isin(TOP_NS)
].copy()


df = df.sort_values(
    [
        "date",
        "top_n",
        "model"
    ]
)


# ============================================================
# DATE INFORMATION
# ============================================================

all_dates = sorted(
    df["date"].unique()
)

normal_dates = [
    d
    for d in all_dates
    if d not in EXCLUDE_DATES
]

excluded_dates = [
    d
    for d in all_dates
    if d in EXCLUDE_DATES
]


print()
print("=" * 70)
print("DATE SPLIT")
print("=" * 70)

print(
    f"ALL DAYS      : {len(all_dates)}"
)

print(
    f"NORMAL DAYS   : {len(normal_dates)}"
)

print(
    f"EXCLUDED DAYS : {len(excluded_dates)}"
)

print()
print(
    "EXCLUDED:"
)

for d in excluded_dates:

    print(
        f"  {d.strftime('%Y-%m-%d')}"
    )


# ============================================================
# PAIRED COMPARISON FUNCTION
# ============================================================

def build_comparison(
    source_df,
    label
):

    rows = []

    for top_n in TOP_NS:

        sub = source_df[
            source_df["top_n"] == top_n
        ].copy()

        base = sub[
            sub["model"] == BASE_MODEL
        ][
            [
                "date",
                "avg_diff",
                "total_diff",
                "win_rate",
                "positive",
            ]
        ].rename(
            columns={
                "avg_diff":
                    "v4_avg_diff",

                "total_diff":
                    "v4_total_diff",

                "win_rate":
                    "v4_win_rate",

                "positive":
                    "v4_positive",
            }
        )

        candidate = sub[
            sub["model"] == CANDIDATE_MODEL
        ][
            [
                "date",
                "avg_diff",
                "total_diff",
                "win_rate",
                "positive",
            ]
        ].rename(
            columns={
                "avg_diff":
                    "v42c_avg_diff",

                "total_diff":
                    "v42c_total_diff",

                "win_rate":
                    "v42c_win_rate",

                "positive":
                    "v42c_positive",
            }
        )

        paired = base.merge(
            candidate,
            on="date",
            how="inner"
        )

        if paired.empty:
            continue

        paired["top_n"] = top_n
        paired["period"] = label

        paired["difference"] = (
            paired["v42c_avg_diff"]
            - paired["v4_avg_diff"]
        )

        paired["total_difference"] = (
            paired["v42c_total_diff"]
            - paired["v4_total_diff"]
        )

        paired["v42c_better"] = (
            paired["difference"] > 0
        ).astype(int)

        paired["v4_better"] = (
            paired["difference"] < 0
        ).astype(int)

        paired["tie"] = (
            paired["difference"] == 0
        ).astype(int)

        rows.append(paired)

    if not rows:

        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True
    )


# ============================================================
# BUILD FULL / NORMAL
# ============================================================

full_compare = build_comparison(
    df,
    "ALL_DAYS"
)

normal_df = df[
    df["date"].isin(normal_dates)
].copy()

normal_compare = build_comparison(
    normal_df,
    "NORMAL_DAYS"
)


daily_compare = pd.concat(
    [
        full_compare,
        normal_compare
    ],
    ignore_index=True
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

for period in [
    "ALL_DAYS",
    "NORMAL_DAYS"
]:

    sub = daily_compare[
        daily_compare["period"] == period
    ]

    for top_n in TOP_NS:

        x = sub[
            sub["top_n"] == top_n
        ].copy()

        if x.empty:
            continue

        v4_mean = (
            x["v4_avg_diff"]
            .mean()
        )

        v42c_mean = (
            x["v42c_avg_diff"]
            .mean()
        )

        improvement = (
            v42c_mean
            - v4_mean
        )

        total_v4 = (
            x["v4_total_diff"]
            .sum()
        )

        total_v42c = (
            x["v42c_total_diff"]
            .sum()
        )

        total_improvement = (
            total_v42c
            - total_v4
        )

        better_days = (
            x["v42c_better"]
            .sum()
        )

        v4_better_days = (
            x["v4_better"]
            .sum()
        )

        tie_days = (
            x["tie"]
            .sum()
        )

        days = len(x)

        better_rate = (
            better_days
            / days
            * 100
            if days
            else 0.0
        )

        summary_rows.append({

            "period":
                period,

            "top_n":
                top_n,

            "days":
                days,

            "v4_mean":
                v4_mean,

            "v42c_mean":
                v42c_mean,

            "improvement":
                improvement,

            "v4_total_diff":
                total_v4,

            "v42c_total_diff":
                total_v42c,

            "total_improvement":
                total_improvement,

            "v42c_better_days":
                better_days,

            "v4_better_days":
                v4_better_days,

            "tie_days":
                tie_days,

            "v42c_better_rate":
                better_rate,

        })


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# EXCLUDED DAY EFFECT
# ============================================================

effect_rows = []

for top_n in TOP_NS:

    full = full_compare[
        full_compare["top_n"] == top_n
    ]

    normal = normal_compare[
        normal_compare["top_n"] == top_n
    ]

    if full.empty or normal.empty:
        continue

    excluded = full_compare[
        (full_compare["top_n"] == top_n)
        &
        (
            full_compare["date"]
            .isin(EXCLUDE_DATES)
        )
    ]

    full_improvement = (
        full["difference"].mean()
    )

    normal_improvement = (
        normal["difference"].mean()
    )

    if excluded.empty:

        excluded_improvement = 0.0

    else:

        excluded_improvement = (
            excluded["difference"].mean()
        )

    effect_rows.append({

        "top_n":
            top_n,

        "all_days":
            len(full),

        "normal_days":
            len(normal),

        "excluded_days":
            len(excluded),

        "all_day_improvement":
            full_improvement,

        "normal_day_improvement":
            normal_improvement,

        "excluded_day_improvement":
            excluded_improvement,

        "normal_vs_all_change":
            normal_improvement
            - full_improvement,

    })


excluded_effect = pd.DataFrame(
    effect_rows
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("VER.4.2 NORMAL DAY COMPARISON")
print("=" * 70)

print()
print("SUMMARY")
print("-" * 70)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


print()
print("=" * 70)
print("NORMAL DAY RESULT")
print("=" * 70)

normal_summary = summary[
    summary["period"] == "NORMAL_DAYS"
].copy()

print(
    normal_summary[
        [
            "top_n",
            "days",
            "v4_mean",
            "v42c_mean",
            "improvement",
            "v42c_better_days",
            "v4_better_days",
            "tie_days",
            "v42c_better_rate",
            "total_improvement",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


print()
print("=" * 70)
print("EXCLUDED DAY EFFECT")
print("=" * 70)

print(
    excluded_effect.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# TOP10 DIAGNOSTIC
# ============================================================

print()
print("=" * 70)
print("TOP10 DIAGNOSTIC")
print("=" * 70)

top10_normal = normal_compare[
    normal_compare["top_n"] == 10
].copy()

if not top10_normal.empty:

    print(
        f"V4 BASE mean      : "
        f"{top10_normal['v4_avg_diff'].mean():+.2f}"
    )

    print(
        f"V4.2_C mean       : "
        f"{top10_normal['v42c_avg_diff'].mean():+.2f}"
    )

    print(
        f"Improvement       : "
        f"{top10_normal['difference'].mean():+.2f}"
    )

    print(
        f"V4.2_C better     : "
        f"{int(top10_normal['v42c_better'].sum())}"
        f"/{len(top10_normal)}"
    )

    print(
        f"V4 better         : "
        f"{int(top10_normal['v4_better'].sum())}"
        f"/{len(top10_normal)}"
    )

    print(
        f"Tie               : "
        f"{int(top10_normal['tie'].sum())}"
    )

    print(
        f"Total improvement : "
        f"{top10_normal['total_difference'].sum():+.0f}"
    )


# ============================================================
# JUDGMENT
# ============================================================

print()
print("=" * 70)
print("JUDGMENT")
print("=" * 70)

top10_row = normal_summary[
    normal_summary["top_n"] == 10
]

if not top10_row.empty:

    row = top10_row.iloc[0]

    improvement = (
        float(row["improvement"])
    )

    better_rate = (
        float(row["v42c_better_rate"])
    )

    total_improvement = (
        float(row["total_improvement"])
    )

    if (
        improvement > 0
        and better_rate >= 50
        and total_improvement > 0
    ):

        print(
            "V4.2_C remains positive "
            "after excluding the machine-change day."
        )

        print(
            "This supports robustness, "
            "but does NOT prove statistical significance."
        )

    elif improvement > 0:

        print(
            "V4.2_C remains numerically better, "
            "but the evidence is not strong."
        )

    else:

        print(
            "V4.2_C advantage disappears "
            "after excluding the machine-change day."
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This is a robustness check, "
        "not a significance test."
    )


# ============================================================
# SAVE
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


daily_compare.to_csv(
    OUTPUT_DAILY,
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)

normal_compare.to_csv(
    OUTPUT_COMPARE,
    index=False,
    encoding="utf-8-sig"
)

excluded_effect.to_csv(
    OUTPUT_EXCLUDED,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(OUTPUT_DAILY)
print(OUTPUT_SUMMARY)
print(OUTPUT_COMPARE)
print(OUTPUT_EXCLUDED)

print()
print(
    "Ver.4.2 normal day comparison complete."
)