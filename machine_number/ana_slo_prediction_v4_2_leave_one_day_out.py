from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Ana-Slo Ver.4.2 Leave-One-Day-Out Sensitivity Analysis
#
# Purpose:
#   Test whether V4.2_C advantage depends heavily on any
#   single test day.
#
# For each test day:
#   1. Remove that day
#   2. Compare V4_BASE vs V4.2_C
#   3. Calculate TOP5 / TOP10 / TOP20 / TOP30 results
#
# No weights are optimized in this script.
# ============================================================


print("=" * 70)
print("Ana-Slo Ver.4.2 Leave-One-Day-Out Analysis")
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
    / "28_Ver4_2_leave_one_day_out_daily.csv"
)

OUTPUT_SUMMARY = (
    OUT_DIR
    / "28_Ver4_2_leave_one_day_out_summary.csv"
)

OUTPUT_TOP10 = (
    OUT_DIR
    / "28_Ver4_2_leave_one_day_out_top10.csv"
)

OUTPUT_RANKING = (
    OUT_DIR
    / "28_Ver4_2_leave_one_day_out_ranking.csv"
)


BASE_MODEL = "V4_BASE"
CANDIDATE_MODEL = "V4.2_C"

TOP_NS = [5, 10, 20, 30]


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


# ============================================================
# VALIDATE
# ============================================================

required = [
    "date",
    "model",
    "top_n",
    "avg_diff",
    "total_diff",
]


missing = [
    col
    for col in required
    if col not in df.columns
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

df["avg_diff"] = pd.to_numeric(
    df["avg_diff"],
    errors="coerce"
)

df["total_diff"] = pd.to_numeric(
    df["total_diff"],
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


df = df[
    df["model"].isin(
        [
            BASE_MODEL,
            CANDIDATE_MODEL,
        ]
    )
    &
    df["top_n"].isin(TOP_NS)
].copy()


dates = sorted(
    df["date"].unique()
)


print(
    f"test days = {len(dates)}"
)

print()

for d in dates:

    print(
        d.strftime("%Y-%m-%d")
    )


# ============================================================
# BUILD PAIRED DAILY DATA
# ============================================================

rows = []


for top_n in TOP_NS:

    sub = df[
        df["top_n"] == top_n
    ].copy()


    base = sub[
        sub["model"] == BASE_MODEL
    ][
        [
            "date",
            "avg_diff",
            "total_diff",
        ]
    ].rename(
        columns={
            "avg_diff":
                "v4_avg_diff",

            "total_diff":
                "v4_total_diff",
        }
    )


    candidate = sub[
        sub["model"] == CANDIDATE_MODEL
    ][
        [
            "date",
            "avg_diff",
            "total_diff",
        ]
    ].rename(
        columns={
            "avg_diff":
                "v42c_avg_diff",

            "total_diff":
                "v42c_total_diff",
        }
    )


    paired = base.merge(
        candidate,
        on="date",
        how="inner"
    )


    paired["top_n"] = top_n


    paired["difference"] = (
        paired["v42c_avg_diff"]
        - paired["v4_avg_diff"]
    )


    paired["total_difference"] = (
        paired["v42c_total_diff"]
        - paired["v4_total_diff"]
    )


    rows.append(
        paired
    )


daily = pd.concat(
    rows,
    ignore_index=True
)


print()
print(
    f"paired records = {len(daily)}"
)


# ============================================================
# LEAVE-ONE-DAY-OUT
# ============================================================

result_rows = []


for excluded_date in dates:

    print(
        f"Analyzing exclusion: "
        f"{excluded_date.strftime('%Y-%m-%d')}"
    )


    remaining = daily[
        daily["date"] != excluded_date
    ].copy()


    for top_n in TOP_NS:

        x = remaining[
            remaining["top_n"] == top_n
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


        v4_total = (
            x["v4_total_diff"]
            .sum()
        )

        v42c_total = (
            x["v42c_total_diff"]
            .sum()
        )


        total_improvement = (
            v42c_total
            - v4_total
        )


        difference = (
            x["difference"]
        )


        better_days = int(
            (
                difference > 0
            ).sum()
        )


        base_better_days = int(
            (
                difference < 0
            ).sum()
        )


        tie_days = int(
            (
                difference == 0
            ).sum()
        )


        days = len(x)


        better_rate = (
            better_days
            / days
            * 100
        )


        result_rows.append({

            "excluded_date":
                excluded_date,

            "top_n":
                top_n,

            "remaining_days":
                days,

            "v4_mean":
                v4_mean,

            "v42c_mean":
                v42c_mean,

            "improvement":
                improvement,

            "v4_total_diff":
                v4_total,

            "v42c_total_diff":
                v42c_total,

            "total_improvement":
                total_improvement,

            "v42c_better_days":
                better_days,

            "v4_better_days":
                base_better_days,

            "tie_days":
                tie_days,

            "v42c_better_rate":
                better_rate,

        })


results = pd.DataFrame(
    result_rows
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


for top_n in TOP_NS:

    x = results[
        results["top_n"] == top_n
    ].copy()


    if x.empty:

        continue


    positive_cases = int(
        (
            x["improvement"] > 0
        ).sum()
    )


    negative_cases = int(
        (
            x["improvement"] < 0
        ).sum()
    )


    zero_cases = int(
        (
            x["improvement"] == 0
        ).sum()
    )


    min_improvement = (
        x["improvement"].min()
    )


    max_improvement = (
        x["improvement"].max()
    )


    mean_improvement = (
        x["improvement"].mean()
    )


    median_improvement = (
        x["improvement"].median()
    )


    summary_rows.append({

        "top_n":
            top_n,

        "leave_one_out_cases":
            len(x),

        "positive_cases":
            positive_cases,

        "negative_cases":
            negative_cases,

        "zero_cases":
            zero_cases,

        "positive_case_rate":
            positive_cases
            / len(x)
            * 100,

        "mean_improvement":
            mean_improvement,

        "median_improvement":
            median_improvement,

        "minimum_improvement":
            min_improvement,

        "maximum_improvement":
            max_improvement,

        "all_positive":
            int(
                negative_cases == 0
            ),

    })


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# TOP10 FOCUS
# ============================================================

top10 = results[
    results["top_n"] == 10
].copy()


top10 = top10.sort_values(
    "improvement",
    ascending=True
)


# ============================================================
# RANKING
# ============================================================

ranking_rows = []


for top_n in TOP_NS:

    x = results[
        results["top_n"] == top_n
    ].copy()


    x = x.sort_values(
        "improvement"
    )


    if x.empty:

        continue


    worst = x.iloc[0]

    best = x.iloc[-1]


    ranking_rows.append({

        "top_n":
            top_n,

        "worst_excluded_date":
            worst["excluded_date"],

        "worst_improvement":
            worst["improvement"],

        "best_excluded_date":
            best["excluded_date"],

        "best_improvement":
            best["improvement"],

        "mean_improvement":
            x["improvement"].mean(),

        "median_improvement":
            x["improvement"].median(),

        "positive_case_rate":
            (
                x["improvement"] > 0
            ).mean()
            * 100,

    })


ranking = pd.DataFrame(
    ranking_rows
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 70)
print("LEAVE-ONE-DAY-OUT SUMMARY")
print("=" * 70)


print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.2f}"
    )
)


print()
print("=" * 70)
print("TOP10 SENSITIVITY")
print("=" * 70)


print(
    top10[
        [
            "excluded_date",
            "remaining_days",
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
        float_format=lambda x:
            f"{x:.2f}"
    )
)


# ============================================================
# TOP10 DIAGNOSTIC
# ============================================================

print()
print("=" * 70)
print("TOP10 DIAGNOSTIC")
print("=" * 70)


if not top10.empty:

    worst = top10.iloc[0]

    best = top10.iloc[-1]


    print(
        "Worst exclusion:"
    )

    print(
        f"  date = "
        f"{worst['excluded_date']}"
    )

    print(
        f"  improvement = "
        f"{worst['improvement']:+.2f}"
    )


    print()

    print(
        "Best exclusion:"
    )

    print(
        f"  date = "
        f"{best['excluded_date']}"
    )

    print(
        f"  improvement = "
        f"{best['improvement']:+.2f}"
    )


    print()

    print(
        f"Mean improvement = "
        f"{top10['improvement'].mean():+.2f}"
    )

    print(
        f"Median improvement = "
        f"{top10['improvement'].median():+.2f}"
    )

    print(
        f"Positive exclusions = "
        f"{int((top10['improvement'] > 0).sum())}"
        f"/{len(top10)}"
    )

    print(
        f"Negative exclusions = "
        f"{int((top10['improvement'] < 0).sum())}"
        f"/{len(top10)}"
    )


# ============================================================
# JUDGMENT
# ============================================================

print()
print("=" * 70)
print("JUDGMENT")
print("=" * 70)


if not top10.empty:

    positive_rate = (
        (
            top10["improvement"] > 0
        ).mean()
        * 100
    )

    min_value = (
        top10["improvement"].min()
    )

    median_value = (
        top10["improvement"].median()
    )


    if (
        positive_rate >= 80
        and min_value > 0
    ):

        print(
            "STRONG ROBUSTNESS:"
        )

        print(
            "V4.2_C remains better "
            "under every single-day exclusion."
        )


    elif (
        positive_rate >= 70
        and median_value > 0
    ):

        print(
            "GOOD ROBUSTNESS:"
        )

        print(
            "V4.2_C remains positive "
            "under most single-day exclusions."
        )


    elif median_value > 0:

        print(
            "MODERATE ROBUSTNESS:"
        )

        print(
            "V4.2_C advantage exists on average, "
            "but some individual days have substantial influence."
        )


    else:

        print(
            "WEAK ROBUSTNESS:"
        )

        print(
            "V4.2_C advantage is sensitive "
            "to individual days."
        )


    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Leave-One-Day-Out is a sensitivity test."
    )

    print(
        "It does NOT establish statistical significance."
    )


# ============================================================
# SAVE
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


daily.to_csv(
    OUTPUT_DAILY,
    index=False,
    encoding="utf-8-sig"
)


results.to_csv(
    OUTPUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)


top10.to_csv(
    OUTPUT_TOP10,
    index=False,
    encoding="utf-8-sig"
)


ranking.to_csv(
    OUTPUT_RANKING,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(OUTPUT_DAILY)
print(OUTPUT_SUMMARY)
print(OUTPUT_TOP10)
print(OUTPUT_RANKING)

print()
print(
    "Ver.4.2 Leave-One-Day-Out analysis complete."
)