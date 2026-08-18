# -*- coding: utf-8 -*-
"""
Ana-Slo Ver.4.2 Adaptive vs TOP10 Statistical Significance Test

Purpose
-------
Compare the OOS daily performance of:
    1. V4.2_C Adaptive Rank Band
    2. Fixed TOP10

Tests
-----
- Paired daily difference
- Bootstrap 95% CI of mean improvement
- Sign permutation test
- Cohen's d for paired differences
- Win / loss / tie counts
- Exclusion sensitivity (leave-one-day-out)
- Monthly / half-period diagnostics when available

IMPORTANT
---------
This is a statistical test of the existing OOS results.
It does not create new predictions and does not establish future profitability.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

# The long-term validation output folder may differ slightly depending on
# the exact script version/path used on the PC.  Locate the 39-series output
# recursively instead of assuming one hard-coded directory name.
ANALYSIS_ROOT = (
    BASE_DIR / "data" / "maruhan_maebashi" / "machine_number"
    / "analysis" / "31days_deep"
)

INPUT_DAILY = None
INPUT_SELECTION = None

# Search the entire SlotAnalyzer project, not only one assumed data folder.
# This handles differences in folder naming/location between validation runs.
PROJECT_ROOT = BASE_DIR

def find_project_file(filename: str):
    candidates = []
    if PROJECT_ROOT.exists():
        candidates = sorted(
            p for p in PROJECT_ROOT.rglob(filename)
            if p.is_file()
        )
    return candidates[-1] if candidates else None

INPUT_DAILY = find_project_file(
    "39_Ver4_2_long_term_adaptive_rank_band_fixed_daily.csv"
)
INPUT_SELECTION = find_project_file(
    "39_Ver4_2_long_term_adaptive_rank_band_fixed_selection.csv"
)

if INPUT_DAILY is None and PROJECT_ROOT.exists():
    candidates = sorted(
        p for p in PROJECT_ROOT.rglob("*adaptive*rank*band*fixed*daily*.csv")
        if p.is_file() and "39" in p.name
    )
    if candidates:
        INPUT_DAILY = candidates[-1]

if INPUT_SELECTION is None and PROJECT_ROOT.exists():
    candidates = sorted(
        p for p in PROJECT_ROOT.rglob("*adaptive*rank*band*fixed*selection*.csv")
        if p.is_file() and "39" in p.name
    )
    if candidates:
        INPUT_SELECTION = candidates[-1]

OUTPUT_DIR = (
    BASE_DIR / "data" / "maruhan_maebashi" / "machine_number"
    / "analysis" / "31days_deep"
    / "40_Ver4_2_adaptive_vs_top10_significance"
)

OUT_DAILY = OUTPUT_DIR / "40_Ver4_2_adaptive_vs_top10_daily.csv"
OUT_SUMMARY = OUTPUT_DIR / "40_Ver4_2_adaptive_vs_top10_summary.csv"
OUT_BOOTSTRAP = OUTPUT_DIR / "40_Ver4_2_adaptive_vs_top10_bootstrap.csv"
OUT_PERMUTATION = OUTPUT_DIR / "40_Ver4_2_adaptive_vs_top10_permutation.csv"
OUT_LOO = OUTPUT_DIR / "40_Ver4_2_adaptive_vs_top10_leave_one_out.csv"
OUT_MONTHLY = OUTPUT_DIR / "40_Ver4_2_adaptive_vs_top10_monthly.csv"

OUTPUT_DIR = BASE_DIR / "data" / "maruhan_maebashi" / "machine_number" / "analysis" / "31days_deep" / "40_Ver4_2_adaptive_vs_top10_significance"

OUT_DAILY = OUTPUT_DIR / "40_Ver4_2_adaptive_vs_top10_daily.csv"
OUT_SUMMARY = OUTPUT_DIR / "40_Ver4_2_adaptive_vs_top10_summary.csv"
OUT_BOOTSTRAP = OUTPUT_DIR / "40_Ver4_2_adaptive_vs_top10_bootstrap.csv"
OUT_PERMUTATION = OUTPUT_DIR / "40_Ver4_2_adaptive_vs_top10_permutation.csv"
OUT_LOO = OUTPUT_DIR / "40_Ver4_2_adaptive_vs_top10_leave_one_out.csv"
OUT_MONTHLY = OUTPUT_DIR / "40_Ver4_2_adaptive_vs_top10_monthly.csv"


# ============================================================
# SETTINGS
# ============================================================

N_BOOTSTRAP = 20000
N_PERMUTATION = 20000
RANDOM_SEED = 20260818


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}

    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]

    for c in df.columns:
        s = str(c).lower()
        for name in candidates:
            if name.lower() in s:
                return c

    return None


def load_daily():
    if INPUT_DAILY is None or not INPUT_DAILY.exists():
        raise FileNotFoundError(
            "39_Ver4_2 long-term adaptive のdaily CSVが見つかりません。\n"
            f"検索先: {ANALYSIS_ROOT}\n"
            "39_Ver4_2_long_term_adaptive_rank_band_fixed_daily.csv を"
            "確認してください。"
        )

    print(f"Loading: {INPUT_DAILY}")
    df = pd.read_csv(INPUT_DAILY, encoding="utf-8-sig")

    print(f"records = {len(df):,}")
    print(f"columns = {list(df.columns)}")

    date_col = find_column(df, ["date", "日付"])
    model_col = find_column(df, ["model"])
    rule_col = find_column(df, ["rule", "selected_rule"])
    avg_col = find_column(df, ["avg_diff"])
    total_col = find_column(df, ["total_diff"])
    block_col = find_column(df, ["block"])

    required = {
        "date": date_col,
        "model": model_col,
        "rule": rule_col,
        "avg_diff": avg_col,
        "total_diff": total_col,
        "block": block_col,
    }

    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            "必要列が見つかりません: "
            + ", ".join(missing)
        )

    df = df.rename(
        columns={
            date_col: "date",
            model_col: "model",
            rule_col: "rule",
            avg_col: "avg_diff",
            total_col: "total_diff",
            block_col: "block",
        }
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["avg_diff"] = pd.to_numeric(df["avg_diff"], errors="coerce")
    df["total_diff"] = pd.to_numeric(df["total_diff"], errors="coerce")
    df["block"] = pd.to_numeric(df["block"], errors="coerce")

    df = df.dropna(
        subset=["date", "avg_diff", "total_diff", "block"]
    ).copy()

    return df


def select_comparable_days(df):
    """
    The adaptive result may be represented either as a dedicated model row
    or by a selected-rule row. We first try explicit adaptive model rows.
    Otherwise reconstruct the daily adaptive result from selected_rule.
    """

    # Explicit adaptive model rows
    model_text = df["model"].astype(str).str.upper()
    explicit = df[
        model_text.str.contains("ADAPTIVE", na=False)
    ].copy()

    # Fixed TOP10 rows
    top10 = df[
        df["rule"].astype(str).str.upper().eq("TOP10")
    ].copy()

    if not explicit.empty:
        adaptive = explicit.copy()
    else:
        # Reconstruct from the adaptive-selected rule.
        adaptive = df[
            df["rule"].astype(str).str.upper().isin(
                ["TOP10", "TOP4_5", "TOP4_8", "TOP4_10", "TOP5_10", "TOP6_10"]
            )
        ].copy()

        # If the file contains multiple rule rows per date/block, use the
        # selection file to identify the actual adaptive rule.
        if INPUT_SELECTION is not None and INPUT_SELECTION.exists():
            print(f"Loading selection file: {INPUT_SELECTION}")
            sel = pd.read_csv(INPUT_SELECTION, encoding="utf-8-sig")

            sel_date = find_column(sel, ["date", "block_start"])
            sel_rule = find_column(sel, ["selected_rule", "rule"])

            if sel_rule is not None:
                if sel_date is not None:
                    if str(sel_date) == "block_start":
                        sel["block_start"] = pd.to_datetime(
                            sel[sel_date], errors="coerce"
                        )
                        sel["_selection_date"] = sel["block_start"]
                    else:
                        sel["_selection_date"] = pd.to_datetime(
                            sel[sel_date], errors="coerce"
                        )

                if "block" in sel.columns:
                    adaptive = adaptive.merge(
                        sel[["block", sel_rule]].rename(
                            columns={sel_rule: "_selected_rule"}
                        ),
                        on="block",
                        how="inner",
                    )
                    adaptive = adaptive[
                        adaptive["rule"].astype(str).str.upper()
                        == adaptive["_selected_rule"].astype(str).str.upper()
                    ].copy()

    # If explicit adaptive rows have several entries per day, aggregate them.
    adaptive_daily = (
        adaptive.groupby("date", as_index=False)
        .agg(
            adaptive_avg_diff=("avg_diff", "mean"),
            adaptive_total_diff=("total_diff", "sum"),
            adaptive_blocks=("block", "nunique"),
        )
    )

    top10_daily = (
        top10.groupby("date", as_index=False)
        .agg(
            top10_avg_diff=("avg_diff", "mean"),
            top10_total_diff=("total_diff", "sum"),
            top10_blocks=("block", "nunique"),
        )
    )

    paired = adaptive_daily.merge(
        top10_daily,
        on="date",
        how="inner",
    )

    if paired.empty:
        raise RuntimeError(
            "AdaptiveとTOP10を同じ日付で比較できませんでした。\n"
            "39のdaily CSVとselection CSVの内容を確認してください。"
        )

    paired["improvement"] = (
        paired["adaptive_avg_diff"]
        - paired["top10_avg_diff"]
    )

    paired["total_improvement"] = (
        paired["adaptive_total_diff"]
        - paired["top10_total_diff"]
    )

    paired["adaptive_better"] = paired["improvement"] > 0
    paired["top10_better"] = paired["improvement"] < 0
    paired["tie"] = paired["improvement"] == 0

    return paired.sort_values("date").reset_index(drop=True)


def bootstrap_mean(x, n=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        return np.array([])

    indices = rng.integers(
        0, len(x), size=(n, len(x))
    )
    samples = x[indices]
    return samples.mean(axis=1)


def permutation_mean(x, n=N_PERMUTATION, seed=RANDOM_SEED):
    """
    Paired sign permutation:
    Under H0, each paired improvement has an equally likely + or - sign.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        return np.array([])

    signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=(n, len(x))
    )
    return (signs * x).mean(axis=1)


def paired_cohens_d(x):
    x = np.asarray(x, dtype=float)

    if len(x) < 2:
        return np.nan

    sd = np.std(x, ddof=1)

    if sd == 0:
        return np.nan

    return float(np.mean(x) / sd)


def percentile_ci(samples):
    if len(samples) == 0:
        return np.nan, np.nan

    return (
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    )


def exact_sign_pvalue(x):
    """
    Exact two-sided sign-flip enumeration for small samples.
    Used only when n <= 20.
    """
    x = np.asarray(x, dtype=float)
    x = x[x != 0]

    n = len(x)

    if n == 0:
        return 1.0

    if n > 20:
        return np.nan

    observed = abs(np.mean(x))

    vals = []
    for mask in range(1 << n):
        signs = np.ones(n)
        for i in range(n):
            if (mask >> i) & 1:
                signs[i] = -1
        vals.append(abs(np.mean(x * signs)))

    vals = np.asarray(vals)

    return float(
        (np.sum(vals >= observed) + 1)
        / (len(vals) + 1)
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("Ana-Slo Ver.4.2 Adaptive vs TOP10 Statistical Significance Test")
    print("=" * 72)

    df = load_daily()

    paired = select_comparable_days(df)

    x = paired["improvement"].to_numpy(dtype=float)

    n = len(x)

    print()
    print("======================================================================")
    print("PAIRED OOS DATA")
    print("======================================================================")
    print(f"paired days            : {n}")
    print(f"date range             : {paired['date'].min().date()} to {paired['date'].max().date()}")

    adaptive_mean = float(paired["adaptive_avg_diff"].mean())
    top10_mean = float(paired["top10_avg_diff"].mean())
    mean_improvement = float(x.mean())
    median_improvement = float(np.median(x))

    adaptive_total = float(paired["adaptive_total_diff"].sum())
    top10_total = float(paired["top10_total_diff"].sum())
    total_improvement = float(
        paired["total_improvement"].sum()
    )

    adaptive_better_days = int((x > 0).sum())
    top10_better_days = int((x < 0).sum())
    tie_days = int((x == 0).sum())

    print(f"Adaptive mean          : {adaptive_mean:+.2f}")
    print(f"TOP10 mean             : {top10_mean:+.2f}")
    print(f"mean improvement       : {mean_improvement:+.2f}")
    print(f"median improvement     : {median_improvement:+.2f}")
    print(f"Adaptive better days   : {adaptive_better_days}")
    print(f"TOP10 better days      : {top10_better_days}")
    print(f"tie days               : {tie_days}")
    print(f"Adaptive better rate   : {adaptive_better_days / n * 100:.2f}%")
    print(f"total Adaptive diff    : {adaptive_total:+.0f}")
    print(f"total TOP10 diff       : {top10_total:+.0f}")
    print(f"total improvement      : {total_improvement:+.0f}")

    # Bootstrap
    boot = bootstrap_mean(x)
    boot_low, boot_high = percentile_ci(boot)

    # Permutation
    perm = permutation_mean(x)
    observed = mean_improvement

    if len(perm):
        p_perm = float(
            (np.sum(np.abs(perm) >= abs(observed)) + 1)
            / (len(perm) + 1)
        )
        null_low, null_high = percentile_ci(perm)
    else:
        p_perm = np.nan
        null_low = np.nan
        null_high = np.nan

    exact_p = exact_sign_pvalue(x)

    d = paired_cohens_d(x)

    print()
    print("======================================================================")
    print("BOOTSTRAP")
    print("======================================================================")
    print(f"bootstrap samples      : {N_BOOTSTRAP:,}")
    print(f"95% CI lower           : {boot_low:+.2f}")
    print(f"95% CI upper           : {boot_high:+.2f}")

    print()
    print("======================================================================")
    print("PAIRED SIGN PERMUTATION")
    print("======================================================================")
    print(f"permutation samples    : {N_PERMUTATION:,}")
    print(f"p-value                : {p_perm:.5f}")
    print(f"95% null range         : {null_low:+.2f} to {null_high:+.2f}")

    print()
    print("======================================================================")
    print("EFFECT SIZE")
    print("======================================================================")
    print(f"Cohen's d (paired)     : {d:+.4f}")

    if not np.isnan(exact_p):
        print(f"Exact sign p-value     : {exact_p:.5f}")

    # Leave-one-day-out sensitivity
    loo_rows = []

    for i, row in paired.iterrows():
        sub = paired.drop(index=i)
        imp = sub["improvement"].to_numpy(dtype=float)

        loo_rows.append(
            {
                "excluded_date": row["date"],
                "remaining_days": len(sub),
                "mean_improvement": float(imp.mean()),
                "median_improvement": float(np.median(imp)),
                "adaptive_better_days": int((imp > 0).sum()),
                "top10_better_days": int((imp < 0).sum()),
                "tie_days": int((imp == 0).sum()),
                "minimum_improvement": float(imp.min()),
                "maximum_improvement": float(imp.max()),
            }
        )

    loo = pd.DataFrame(loo_rows)

    # Monthly diagnostics
    monthly = paired.copy()
    monthly["month"] = (
        pd.to_datetime(monthly["date"])
        .dt.to_period("M")
        .astype(str)
    )

    monthly_result = (
        monthly.groupby("month")
        .agg(
            days=("date", "count"),
            adaptive_mean=("adaptive_avg_diff", "mean"),
            top10_mean=("top10_avg_diff", "mean"),
            improvement=("improvement", "mean"),
            adaptive_total=("adaptive_total_diff", "sum"),
            top10_total=("top10_total_diff", "sum"),
            total_improvement=("total_improvement", "sum"),
            adaptive_better_days=("adaptive_better", "sum"),
        )
        .reset_index()
    )

    # Summary
    judgment = "NO CLEAR SIGNIFICANT ADVANTAGE"

    if (
        boot_low > 0
        and p_perm < 0.05
    ):
        judgment = "CLEAR SIGNIFICANT ADVANTAGE FOR ADAPTIVE"
    elif (
        boot_low > 0
        and p_perm < 0.10
    ):
        judgment = "PROMISING ADVANTAGE, BUT NOT 5% SIGNIFICANT"
    elif (
        boot_high < 0
    ):
        judgment = "SIGNIFICANT DISADVANTAGE FOR ADAPTIVE"
    elif mean_improvement > 0:
        judgment = "POSITIVE POINT ESTIMATE, BUT NOT STATISTICALLY CLEAR"

    print()
    print("======================================================================")
    print("JUDGMENT")
    print("======================================================================")
    print(judgment)

    print()
    print("IMPORTANT:")
    print("This test evaluates the existing OOS sample only.")
    print("It does NOT prove future profitability.")
    print("The sample is short if only 2026-07-11 to 2026-08-10 is available.")
    print("A non-significant result does not mean the Adaptive rule is useless;")
    print("it means the available evidence is insufficient to establish superiority.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paired.to_csv(
        OUT_DAILY,
        index=False,
        encoding="utf-8-sig"
    )

    summary = pd.DataFrame(
        [
            {
                "paired_days": n,
                "date_start": paired["date"].min(),
                "date_end": paired["date"].max(),
                "adaptive_mean": adaptive_mean,
                "top10_mean": top10_mean,
                "mean_improvement": mean_improvement,
                "median_improvement": median_improvement,
                "adaptive_better_days": adaptive_better_days,
                "top10_better_days": top10_better_days,
                "tie_days": tie_days,
                "adaptive_better_rate": adaptive_better_days / n * 100,
                "adaptive_total_diff": adaptive_total,
                "top10_total_diff": top10_total,
                "total_improvement": total_improvement,
                "bootstrap_ci_lower": boot_low,
                "bootstrap_ci_upper": boot_high,
                "permutation_p_value": p_perm,
                "permutation_null_low": null_low,
                "permutation_null_high": null_high,
                "exact_sign_p_value": exact_p,
                "cohens_d_paired": d,
                "judgment": judgment,
            }
        ]
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        {
            "bootstrap_mean_improvement": boot
        }
    ).to_csv(
        OUT_BOOTSTRAP,
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        {
            "permutation_mean_improvement": perm
        }
    ).to_csv(
        OUT_PERMUTATION,
        index=False,
        encoding="utf-8-sig"
    )

    loo.to_csv(
        OUT_LOO,
        index=False,
        encoding="utf-8-sig"
    )

    monthly_result.to_csv(
        OUT_MONTHLY,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("======================================================================")
    print("FILES SAVED")
    print("======================================================================")
    print(OUT_DAILY)
    print(OUT_SUMMARY)
    print(OUT_BOOTSTRAP)
    print(OUT_PERMUTATION)
    print(OUT_LOO)
    print(OUT_MONTHLY)

    print()
    print("Adaptive vs TOP10 significance test complete.")


if __name__ == "__main__":
    main()
