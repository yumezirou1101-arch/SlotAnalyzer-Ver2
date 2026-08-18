from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Ana-Slo Ver.4.2 Condition Comparison
#
# V4_BASE vs V4.2_C
#
# Conditions:
#   1. TOP5 / TOP10 / TOP20
#   2. Weekday
#   3. Machine-change day
#   4. Normal day
#   5. Candidate better / base better
#
# IMPORTANT:
# This script does NOT optimize any weights.
# It only compares already-tested results.
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


ROLLING_FILE = (
    OUT_DIR
    / "24_Ver4_2_rolling_daily.csv"
)

CHANGE_FILE = (
    OUT_DIR
    / "17_Ver4_machine_change_diagnostics.csv"
)


OUT_DAILY = (
    OUT_DIR
    / "26_Ver4_2_condition_daily.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "26_Ver4_2_condition_summary.csv"
)

OUT_WEEKDAY = (
    OUT_DIR
    / "26_Ver4_2_condition_weekday.csv"
)

OUT_CHANGE = (
    OUT_DIR
    / "26_Ver4_2_condition_machine_change.csv"
)

OUT_MODEL = (
    OUT_DIR
    / "26_Ver4_2_condition_model_compare.csv"
)


BASE_MODEL = "V4_BASE"
CANDIDATE_MODEL = "V4.2_C"


# ============================================================
# Read CSV
# ============================================================

def read_csv(path):

    for enc in (
        "utf-8-sig",
        "utf-8",
        "cp932",
    ):

        try:

            return pd.read_csv(
                path,
                encoding=enc
            )

        except Exception:

            pass

    raise RuntimeError(
        "CSV read failed: "
        + str(path)
    )


# ============================================================
# Column helper
# ============================================================

def find_column(
    df,
    candidates
):

    for col in candidates:

        if col in df.columns:

            return col

    return None


# ============================================================
# Load rolling results
# ============================================================

print("=" * 70)
print("Ana-Slo Ver.4.2 Condition Comparison")
print("=" * 70)

print()
print("ROLLING RESULT")
print("-" * 70)
print(ROLLING_FILE)

if not ROLLING_FILE.exists():

    raise FileNotFoundError(
        "Rolling result not found:\n"
        + str(ROLLING_FILE)
    )


rolling = read_csv(
    ROLLING_FILE
)

print(
    "records =",
    len(rolling)
)


print()
print("columns =")
print(list(rolling.columns))


# ============================================================
# Normalize rolling data
# ============================================================

model_col = find_column(
    rolling,
    [
        "model",
        "Model",
    ]
)

date_col = find_column(
    rolling,
    [
        "date",
        "Date",
    ]
)

topn_col = find_column(
    rolling,
    [
        "top_n",
        "topN",
    ]
)

avg_col = find_column(
    rolling,
    [
        "avg_diff",
        "average_diff",
        "avg",
    ]
)

total_col = find_column(
    rolling,
    [
        "total_diff",
    ]
)

win_col = find_column(
    rolling,
    [
        "win_rate",
    ]
)

positive_col = find_column(
    rolling,
    [
        "positive",
        "positive_days",
    ]
)


required = [
    model_col,
    date_col,
    topn_col,
    avg_col,
]

if not all(required):

    raise ValueError(
        "\nRequired columns missing.\n"
        f"model = {model_col}\n"
        f"date = {date_col}\n"
        f"top_n = {topn_col}\n"
        f"avg_diff = {avg_col}\n"
        "\nDetected columns:\n"
        + str(list(rolling.columns))
    )


rolling[model_col] = (
    rolling[model_col]
    .astype(str)
    .str.strip()
)

rolling[date_col] = pd.to_datetime(
    rolling[date_col],
    errors="coerce"
)

rolling[topn_col] = pd.to_numeric(
    rolling[topn_col],
    errors="coerce"
)

rolling[avg_col] = pd.to_numeric(
    rolling[avg_col],
    errors="coerce"
)

if total_col:

    rolling[total_col] = pd.to_numeric(
        rolling[total_col],
        errors="coerce"
    )

if win_col:

    rolling[win_col] = pd.to_numeric(
        rolling[win_col],
        errors="coerce"
    )

if positive_col:

    rolling[positive_col] = pd.to_numeric(
        rolling[positive_col],
        errors="coerce"
    )


rolling = rolling.dropna(
    subset=[
        model_col,
        date_col,
        topn_col,
        avg_col,
    ]
).copy()


# ============================================================
# Keep only V4 and V4.2_C
# ============================================================

rolling = rolling[
    rolling[model_col].isin(
        [
            BASE_MODEL,
            CANDIDATE_MODEL,
        ]
    )
].copy()


# ============================================================
# Build paired daily data
# ============================================================

rows = []


for top_n in sorted(
    rolling[topn_col]
    .unique()
):

    base = rolling[
        (rolling[model_col] == BASE_MODEL)
        &
        (rolling[topn_col] == top_n)
    ].copy()

    candidate = rolling[
        (rolling[model_col] == CANDIDATE_MODEL)
        &
        (rolling[topn_col] == top_n)
    ].copy()


    base = base[
        [
            date_col,
            avg_col,
        ]
    ].rename(
        columns={
            avg_col:
                "v4_avg_diff"
        }
    )


    candidate = candidate[
        [
            date_col,
            avg_col,
        ]
    ].rename(
        columns={
            avg_col:
                "v42c_avg_diff"
        }
    )


    paired = pd.merge(
        base,
        candidate,
        on=date_col,
        how="inner"
    )


    for _, row in paired.iterrows():

        date = row[date_col]

        v4 = float(
            row["v4_avg_diff"]
        )

        v42 = float(
            row["v42c_avg_diff"]
        )

        diff = v42 - v4


        rows.append(
            {
                "date":
                    date,

                "top_n":
                    int(top_n),

                "v4_avg_diff":
                    v4,

                "v42c_avg_diff":
                    v42,

                "difference":
                    diff,

                "v42c_better":
                    int(diff > 0),

                "v4_better":
                    int(diff < 0),

                "tie":
                    int(diff == 0),

                "weekday":
                    date.day_name(),

                "weekday_jp":
                    [
                        "月",
                        "火",
                        "水",
                        "木",
                        "金",
                        "土",
                        "日",
                    ][date.dayofweek],
            }
        )


daily = pd.DataFrame(
    rows
)


if daily.empty:

    raise ValueError(
        "No paired V4 / V4.2_C records found."
    )


# ============================================================
# Machine change diagnostics
# ============================================================

print()
print("=" * 70)
print("MACHINE CHANGE INFORMATION")
print("=" * 70)


if CHANGE_FILE.exists():

    change = read_csv(
        CHANGE_FILE
    )

    print(
        "diagnostic records =",
        len(change)
    )

    change_date_col = find_column(
        change,
        [
            "date",
            "Date",
        ]
    )

    change_count_col = find_column(
        change,
        [
            "no_same_machine_history",
            "machine_change",
            "machine_change_count",
        ]
    )


    if (
        change_date_col
        and change_count_col
    ):

        change[change_date_col] = (
            pd.to_datetime(
                change[change_date_col],
                errors="coerce"
            )
        )

        change[change_count_col] = (
            pd.to_numeric(
                change[change_count_col],
                errors="coerce"
            )
            .fillna(0)
        )


        change_small = change[
            [
                change_date_col,
                change_count_col,
            ]
        ].rename(
            columns={
                change_date_col:
                    "date",

                change_count_col:
                    "machine_change_count",
            }
        )


        daily = daily.merge(
            change_small,
            on="date",
            how="left"
        )


    else:

        print(
            "WARNING: machine change columns "
            "not detected."
        )

        daily[
            "machine_change_count"
        ] = np.nan


else:

    print(
        "WARNING: machine change diagnostics "
        "not found."
    )

    daily[
        "machine_change_count"
    ] = np.nan


daily[
    "machine_change_count"
] = pd.to_numeric(
    daily[
        "machine_change_count"
    ],
    errors="coerce"
)


daily[
    "machine_change_day"
] = (
    daily[
        "machine_change_count"
    ]
    .fillna(0)
    > 0
)


# ============================================================
# Overall comparison
# ============================================================

def summarize(
    data,
    label
):

    if data.empty:

        return {
            "condition":
                label,

            "top_n":
                np.nan,

            "days":
                0,

            "v4_mean":
                np.nan,

            "v42c_mean":
                np.nan,

            "mean_difference":
                np.nan,

            "median_difference":
                np.nan,

            "v42c_better_days":
                0,

            "v4_better_days":
                0,

            "tie_days":
                0,

            "v42c_better_rate":
                np.nan,

            "total_difference":
                np.nan,
        }


    diff = (
        data["difference"]
        .to_numpy(
            dtype=float
        )
    )


    return {
        "condition":
            label,

        "top_n":
            int(
                data["top_n"].iloc[0]
            )
            if data["top_n"].nunique() == 1
            else "ALL",

        "days":
            len(data),

        "v4_mean":
            float(
                data["v4_avg_diff"].mean()
            ),

        "v42c_mean":
            float(
                data["v42c_avg_diff"].mean()
            ),

        "mean_difference":
            float(
                np.mean(diff)
            ),

        "median_difference":
            float(
                np.median(diff)
            ),

        "v42c_better_days":
            int(
                np.sum(diff > 0)
            ),

        "v4_better_days":
            int(
                np.sum(diff < 0)
            ),

        "tie_days":
            int(
                np.sum(diff == 0)
            ),

        "v42c_better_rate":
            float(
                np.mean(diff > 0)
                * 100
            ),

        "total_difference":
            float(
                np.sum(diff)
            ),
    }


# ============================================================
# Overall summary
# ============================================================

summary_rows = []


for top_n in sorted(
    daily["top_n"].unique()
):

    subset = daily[
        daily["top_n"] == top_n
    ]

    summary_rows.append(
        summarize(
            subset,
            f"ALL_TOP{top_n}"
        )
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# Weekday comparison
# ============================================================

weekday_rows = []


for top_n in sorted(
    daily["top_n"].unique()
):

    for wd in [
        "月",
        "火",
        "水",
        "木",
        "金",
        "土",
        "日",
    ]:

        subset = daily[
            (daily["top_n"] == top_n)
            &
            (daily["weekday_jp"] == wd)
        ]

        if subset.empty:

            continue


        result = summarize(
            subset,
            f"{wd}_TOP{top_n}"
        )

        result[
            "weekday"
        ] = wd

        weekday_rows.append(
            result
        )


weekday_summary = pd.DataFrame(
    weekday_rows
)


# ============================================================
# Machine change comparison
# ============================================================

change_rows = []


for top_n in sorted(
    daily["top_n"].unique()
):

    for change_flag, label in [
        (False, "NORMAL_DAY"),
        (True, "MACHINE_CHANGE_DAY"),
    ]:

        subset = daily[
            (daily["top_n"] == top_n)
            &
            (
                daily[
                    "machine_change_day"
                ] == change_flag
            )
        ]


        if subset.empty:

            continue


        result = summarize(
            subset,
            f"{label}_TOP{top_n}"
        )

        result[
            "machine_change_day"
        ] = change_flag

        change_rows.append(
            result
        )


change_summary = pd.DataFrame(
    change_rows
)


# ============================================================
# Model comparison
# ============================================================

model_rows = []


for top_n in sorted(
    daily["top_n"].unique()
):

    subset = daily[
        daily["top_n"] == top_n
    ]

    v4_mean = float(
        subset[
            "v4_avg_diff"
        ].mean()
    )

    v42_mean = float(
        subset[
            "v42c_avg_diff"
        ].mean()
    )

    diff = (
        subset[
            "difference"
        ]
    )

    model_rows.append(
        {
            "top_n":
                int(top_n),

            "days":
                len(subset),

            "v4_mean":
                v4_mean,

            "v42c_mean":
                v42_mean,

            "improvement":
                v42_mean - v4_mean,

            "v42c_better_days":
                int(
                    (diff > 0).sum()
                ),

            "v4_better_days":
                int(
                    (diff < 0).sum()
                ),

            "tie_days":
                int(
                    (diff == 0).sum()
                ),

            "v42c_better_rate":
                float(
                    (diff > 0).mean()
                    * 100
                ),

            "total_improvement":
                float(
                    diff.sum()
                ),
        }
    )


model_summary = pd.DataFrame(
    model_rows
)


# ============================================================
# Save
# ============================================================

daily.to_csv(
    OUT_DAILY,
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)

weekday_summary.to_csv(
    OUT_WEEKDAY,
    index=False,
    encoding="utf-8-sig"
)

change_summary.to_csv(
    OUT_CHANGE,
    index=False,
    encoding="utf-8-sig"
)

model_summary.to_csv(
    OUT_MODEL,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Console output
# ============================================================

print()
print("=" * 70)
print("VER.4.2 CONDITION COMPARISON")
print("=" * 70)

print()
print("OVERALL")
print("-" * 70)

print(
    model_summary.to_string(
        index=False
    )
)


print()
print("WEEKDAY")
print("-" * 70)

if not weekday_summary.empty:

    print(
        weekday_summary[
            [
                "weekday",
                "top_n",
                "days",
                "mean_difference",
                "v42c_better_rate",
                "total_difference",
            ]
        ]
        .to_string(
            index=False
        )
    )


print()
print("MACHINE CHANGE")
print("-" * 70)

if not change_summary.empty:

    print(
        change_summary[
            [
                "condition",
                "top_n",
                "days",
                "mean_difference",
                "v42c_better_rate",
                "total_difference",
            ]
        ]
        .to_string(
            index=False
        )
    )


print()
print("DAILY WIN / LOSS")
print("-" * 70)

for top_n in sorted(
    daily["top_n"].unique()
):

    subset = daily[
        daily["top_n"] == top_n
    ]

    print()
    print(
        f"TOP{top_n}"
    )

    print(
        subset[
            [
                "date",
                "v4_avg_diff",
                "v42c_avg_diff",
                "difference",
                "v42c_better",
                "machine_change_day",
            ]
        ]
        .to_string(
            index=False
        )
    )


print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    OUT_DAILY
)

print(
    OUT_SUMMARY
)

print(
    OUT_WEEKDAY
)

print(
    OUT_CHANGE
)

print(
    OUT_MODEL
)

print()
print(
    "Ver.4.2 condition comparison complete."
)