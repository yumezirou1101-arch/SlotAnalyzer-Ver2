from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Ana-Slo Ver.4.2 Significance Test
#
# V4_BASE vs V4.2_C
# - Paired daily difference
# - Bootstrap 95% CI
# - Sign permutation test
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
    / "25_Ver4_2_significance_daily.csv"
)

OUTPUT_SUMMARY = (
    OUT_DIR
    / "25_Ver4_2_significance_summary.csv"
)

OUTPUT_BOOTSTRAP = (
    OUT_DIR
    / "25_Ver4_2_bootstrap.csv"
)

OUTPUT_PERMUTATION = (
    OUT_DIR
    / "25_Ver4_2_permutation.csv"
)

RANDOM_SEED = 20260817

BOOTSTRAP_TRIALS = 10000

PERMUTATION_TRIALS = 10000


# ============================================================
# Utility
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


def find_column(df, candidates):

    for col in candidates:

        if col in df.columns:

            return col

    return None


# ============================================================
# Load
# ============================================================

print("=" * 70)
print("Ana-Slo Ver.4.2 Significance Test")
print("=" * 70)

print()
print("INPUT")
print("-" * 70)
print(INPUT)

if not INPUT.exists():

    raise FileNotFoundError(
        "Input file not found:\n"
        + str(INPUT)
    )


df = read_csv(INPUT)

print()
print("columns =")
print(list(df.columns))

print()
print("records =", len(df))


# ============================================================
# Detect columns
# ============================================================

model_col = find_column(
    df,
    [
        "model",
        "Model",
    ]
)

split_col = find_column(
    df,
    [
        "split",
        "Split",
    ]
)

date_col = find_column(
    df,
    [
        "date",
        "Date",
    ]
)

topn_col = find_column(
    df,
    [
        "top_n",
        "topN",
        "top_n_value",
    ]
)

avg_col = find_column(
    df,
    [
        "avg_diff",
        "average_diff",
        "avg",
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
        "\nRequired columns not found.\n"
        f"model_col={model_col}\n"
        f"split_col={split_col}\n"
        f"date_col={date_col}\n"
        f"topn_col={topn_col}\n"
        f"avg_col={avg_col}\n"
        "\nColumns detected:\n"
        + str(list(df.columns))
    )


# ============================================================
# Normalize
# ============================================================

df[model_col] = (
    df[model_col]
    .astype(str)
    .str.strip()
)

df[date_col] = pd.to_datetime(
    df[date_col],
    errors="coerce"
)

df[topn_col] = pd.to_numeric(
    df[topn_col],
    errors="coerce"
)

df[avg_col] = pd.to_numeric(
    df[avg_col],
    errors="coerce"
)

df = df.dropna(
    subset=[
        model_col,
        date_col,
        topn_col,
        avg_col,
    ]
).copy()


# ============================================================
# Select TOP10
# ============================================================

top10 = df[
    df[topn_col] == 10
].copy()

if top10.empty:

    raise ValueError(
        "TOP10 records not found."
    )


# ============================================================
# Select models
# ============================================================

BASE_MODEL = "V4_BASE"

CANDIDATE_MODEL = "V4.2_C"

base = top10[
    top10[model_col] == BASE_MODEL
][
    [
        date_col,
        avg_col,
    ]
].copy()

candidate = top10[
    top10[model_col] == CANDIDATE_MODEL
][
    [
        date_col,
        avg_col,
    ]
].copy()


if base.empty:

    raise ValueError(
        "V4_BASE TOP10 data not found."
    )


if candidate.empty:

    raise ValueError(
        "V4.2_C TOP10 data not found."
    )


base = base.rename(
    columns={
        avg_col: "v4_avg_diff"
    }
)

candidate = candidate.rename(
    columns={
        avg_col: "v42c_avg_diff"
    }
)


# ============================================================
# Paired comparison
# ============================================================

paired = pd.merge(
    base,
    candidate,
    on=date_col,
    how="inner"
)

paired = paired.sort_values(
    date_col
).reset_index(
    drop=True
)

paired["daily_difference"] = (
    paired["v42c_avg_diff"]
    - paired["v4_avg_diff"]
)

paired["candidate_better"] = (
    paired["daily_difference"] > 0
).astype(int)

paired["base_better"] = (
    paired["daily_difference"] < 0
).astype(int)

paired["tie"] = (
    paired["daily_difference"] == 0
).astype(int)


if len(paired) < 5:

    raise ValueError(
        "Too few paired observations: "
        + str(len(paired))
    )


# ============================================================
# Basic statistics
# ============================================================

differences = (
    paired["daily_difference"]
    .to_numpy(
        dtype=float
    )
)

n = len(differences)

mean_diff = float(
    np.mean(differences)
)

median_diff = float(
    np.median(differences)
)

std_diff = float(
    np.std(
        differences,
        ddof=1
    )
)

sem_diff = (
    std_diff
    / np.sqrt(n)
)

candidate_better_days = int(
    np.sum(
        differences > 0
    )
)

base_better_days = int(
    np.sum(
        differences < 0
    )
)

tie_days = int(
    np.sum(
        differences == 0
    )
)

candidate_win_rate = (
    candidate_better_days
    / n
    * 100
)

base_win_rate = (
    base_better_days
    / n
    * 100
)


# ============================================================
# Bootstrap
# ============================================================

print()
print("=" * 70)
print("BOOTSTRAP")
print("=" * 70)

rng = np.random.default_rng(
    RANDOM_SEED
)

bootstrap_means = np.empty(
    BOOTSTRAP_TRIALS,
    dtype=float
)

for i in range(
    BOOTSTRAP_TRIALS
):

    sample = rng.choice(
        differences,
        size=n,
        replace=True
    )

    bootstrap_means[i] = (
        np.mean(sample)
    )


bootstrap_lower = float(
    np.percentile(
        bootstrap_means,
        2.5
    )
)

bootstrap_upper = float(
    np.percentile(
        bootstrap_means,
        97.5
    )
)


# ============================================================
# Sign permutation test
#
# Null hypothesis:
# V4.2_C and V4 have no systematic
# paired advantage.
#
# Under H0, each daily difference
# can randomly change sign.
# ============================================================

print()
print("=" * 70)
print("SIGN PERMUTATION TEST")
print("=" * 70)

observed = mean_diff

permutation_means = np.empty(
    PERMUTATION_TRIALS,
    dtype=float
)

for i in range(
    PERMUTATION_TRIALS
):

    signs = rng.choice(
        np.array(
            [
                -1.0,
                1.0
            ]
        ),
        size=n
    )

    permutation_means[i] = np.mean(
        differences * signs
    )


# Two-sided p-value

p_value = float(
    (
        np.sum(
            np.abs(
                permutation_means
            )
            >= abs(observed)
        )
        + 1
    )
    / (
        PERMUTATION_TRIALS
        + 1
    )
)


perm_lower = float(
    np.percentile(
        permutation_means,
        2.5
    )
)

perm_upper = float(
    np.percentile(
        permutation_means,
        97.5
    )
)


# ============================================================
# Effect size
#
# Paired standardized effect:
# mean difference / SD difference
# ============================================================

if std_diff > 0:

    cohens_d = (
        mean_diff
        / std_diff
    )

else:

    cohens_d = 0.0


# ============================================================
# Total improvement
# ============================================================

total_difference = float(
    np.sum(differences)
)

base_total = float(
    paired["v4_avg_diff"].sum()
)

candidate_total = float(
    paired["v42c_avg_diff"].sum()
)


# ============================================================
# Judgment
# ============================================================

if (
    p_value < 0.01
    and bootstrap_lower > 0
):

    judgment = (
        "STRONG EVIDENCE"
    )

elif (
    p_value < 0.05
    and bootstrap_lower > 0
):

    judgment = (
        "MODERATE EVIDENCE"
    )

elif (
    p_value < 0.10
    and bootstrap_lower > 0
):

    judgment = (
        "WEAK POSITIVE EVIDENCE"
    )

else:

    judgment = (
        "NO CLEAR SIGNIFICANT ADVANTAGE"
    )


# ============================================================
# Daily output
# ============================================================

daily_output = paired.copy()

daily_output["model_difference"] = (
    daily_output["daily_difference"]
)

daily_output["candidate_better"] = (
    daily_output["candidate_better"]
)

daily_output["base_better"] = (
    daily_output["base_better"]
)

daily_output["tie"] = (
    daily_output["tie"]
)


daily_output.to_csv(
    OUTPUT_DAILY,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Summary
# ============================================================

summary = pd.DataFrame(
    [
        {
            "comparison":
                "V4.2_C_vs_V4_BASE",

            "top_n":
                10,

            "paired_days":
                n,

            "v4_mean_avg_diff":
                float(
                    paired[
                        "v4_avg_diff"
                    ].mean()
                ),

            "v42c_mean_avg_diff":
                float(
                    paired[
                        "v42c_avg_diff"
                    ].mean()
                ),

            "mean_difference":
                mean_diff,

            "median_difference":
                median_diff,

            "std_difference":
                std_diff,

            "candidate_better_days":
                candidate_better_days,

            "base_better_days":
                base_better_days,

            "tie_days":
                tie_days,

            "candidate_better_rate":
                candidate_win_rate,

            "base_better_rate":
                base_win_rate,

            "v4_total_avg_diff":
                base_total,

            "v42c_total_avg_diff":
                candidate_total,

            "total_improvement":
                total_difference,

            "cohens_d":
                cohens_d,

            "bootstrap_95_lower":
                bootstrap_lower,

            "bootstrap_95_upper":
                bootstrap_upper,

            "permutation_p_value":
                p_value,

            "permutation_95_lower":
                perm_lower,

            "permutation_95_upper":
                perm_upper,

            "bootstrap_trials":
                BOOTSTRAP_TRIALS,

            "permutation_trials":
                PERMUTATION_TRIALS,

            "judgment":
                judgment,
        }
    ]
)


summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Bootstrap distribution output
# ============================================================

bootstrap_output = pd.DataFrame(
    {
        "trial":
            np.arange(
                1,
                BOOTSTRAP_TRIALS + 1
            ),

        "bootstrap_mean_difference":
            bootstrap_means,
    }
)

bootstrap_output.to_csv(
    OUTPUT_BOOTSTRAP,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Permutation distribution output
# ============================================================

permutation_output = pd.DataFrame(
    {
        "trial":
            np.arange(
                1,
                PERMUTATION_TRIALS + 1
            ),

        "permutation_mean_difference":
            permutation_means,
    }
)

permutation_output.to_csv(
    OUTPUT_PERMUTATION,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Console report
# ============================================================

print()
print("=" * 70)
print("VER.4.2 SIGNIFICANCE RESULT")
print("=" * 70)

print()
print(
    f"paired TOP10 days        : {n}"
)

print(
    f"V4_BASE mean             : "
    f"{paired['v4_avg_diff'].mean():+.2f}"
)

print(
    f"V4.2_C mean              : "
    f"{paired['v42c_avg_diff'].mean():+.2f}"
)

print(
    f"mean improvement         : "
    f"{mean_diff:+.2f}"
)

print(
    f"median improvement       : "
    f"{median_diff:+.2f}"
)

print(
    f"candidate better days    : "
    f"{candidate_better_days}"
)

print(
    f"base better days         : "
    f"{base_better_days}"
)

print(
    f"tie days                 : "
    f"{tie_days}"
)

print(
    f"candidate better rate    : "
    f"{candidate_win_rate:.2f}%"
)

print(
    f"total improvement        : "
    f"{total_difference:+.0f}"
)

print(
    f"Cohen's d                : "
    f"{cohens_d:+.4f}"
)

print()
print(
    "BOOTSTRAP 95% CI"
)

print(
    f"lower                    : "
    f"{bootstrap_lower:+.2f}"
)

print(
    f"upper                    : "
    f"{bootstrap_upper:+.2f}"
)

print()
print(
    "PERMUTATION TEST"
)

print(
    f"p-value                  : "
    f"{p_value:.5f}"
)

print(
    f"95% null range           : "
    f"{perm_lower:+.2f} to "
    f"{perm_upper:+.2f}"
)

print()
print(
    "JUDGMENT:"
)

print(
    judgment
)

print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    OUTPUT_DAILY
)

print(
    OUTPUT_SUMMARY
)

print(
    OUTPUT_BOOTSTRAP
)

print(
    OUTPUT_PERMUTATION
)

print()
print(
    "Ver.4.2 significance test complete."
)