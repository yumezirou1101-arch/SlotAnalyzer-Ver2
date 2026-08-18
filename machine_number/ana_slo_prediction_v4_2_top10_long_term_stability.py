from __future__ import annotations

"""
Ana-Slo Ver.4.2 TOP10 Long-Term Stability Analysis

Purpose
-------
Analyze the stability of the existing V4.2 TOP10 OOS results across all
available long-term rank-band daily CSV files.

Important
---------
- This script does NOT rebuild the V4.2 model.
- It does NOT tune TOP10 or any threshold.
- It uses only already-generated OOS daily result CSVs.
- Duplicate dates are removed deterministically.
- If the available history is short, the script explicitly reports that
  long-term validation is insufficient.

Expected source CSV
-------------------
A file containing columns such as:
date, rule, selected_rule, machines, avg_diff, median_diff,
win_rate, plus1000_rate, plus2000_rate, positive, total_diff

The script recursively searches under DATA_ROOT for files whose name
contains:
    rank_band
and ends with:
    daily.csv
"""

from pathlib import Path
import math
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

DATA_ROOT = PROJECT_ROOT / "data"

OUTPUT_ROOT = (
    DATA_ROOT
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
    / "47_Ver4_2_TOP10_long_term_stability"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

TOP10_RULE = "TOP10"

# Minimum history levels used for interpretation.
MIN_DAYS_WARNING = 60
MIN_DAYS_REASONABLE = 90
MIN_DAYS_STRONG = 180

# Rolling windows.
ROLLING_WINDOWS = [7, 14, 30]

# Block size in calendar observations, not necessarily calendar days.
BLOCK_SIZE = 7


# ============================================================
# HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def max_losing_streak(values: pd.Series) -> int:
    best = 0
    current = 0

    for value in values:
        if pd.isna(value):
            continue

        if float(value) < 0:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return int(best)


def max_drawdown(values: pd.Series) -> float:
    s = pd.to_numeric(values, errors="coerce").fillna(0.0)
    cumulative = s.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max

    if len(drawdown) == 0:
        return 0.0

    return float(drawdown.min())


def profit_factor_simple(values: pd.Series) -> float:
    s = pd.to_numeric(values, errors="coerce").dropna()

    gross_profit = s[s > 0].sum()
    gross_loss = -s[s < 0].sum()

    if gross_loss == 0:
        if gross_profit > 0:
            return float("inf")
        return np.nan

    return float(gross_profit / gross_loss)


def wilson_interval(successes: int, n: int, z: float = 1.96):
    if n <= 0:
        return np.nan, np.nan

    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (
        z
        * math.sqrt(
            (p * (1 - p) / n)
            + (z * z / (4 * n * n))
        )
        / denom
    )

    return center - half, center + half


# ============================================================
# INPUT DISCOVERY
# ============================================================

def discover_daily_files() -> list[Path]:
    candidates = []

    for p in DATA_ROOT.rglob("*rank_band*daily.csv"):
        if p.is_file():
            candidates.append(p)

    # Also allow the exact currently-known file pattern.
    for p in DATA_ROOT.rglob("*_adaptive_rank_band_fixed_daily.csv"):
        if p.is_file():
            candidates.append(p)

    unique = sorted(set(candidates))

    # Do not accidentally consume this script's own output.
    unique = [
        p for p in unique
        if OUTPUT_ROOT not in p.parents
    ]

    return unique


def load_all_sources() -> pd.DataFrame:
    print_header("SOURCE DISCOVERY")

    files = discover_daily_files()

    if not files:
        raise FileNotFoundError(
            "V4.2 rank-band daily CSVが見つかりません。\n"
            f"検索先: {DATA_ROOT}"
        )

    print(f"candidate files = {len(files)}")

    frames = []

    required_any = {
        "date",
        "rule",
        "avg_diff",
        "total_diff",
    }

    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"SKIP read error: {path}")
            print(f"  {exc}")
            continue

        if not required_any.issubset(df.columns):
            print(f"SKIP missing columns: {path}")
            continue

        sub = df[df["rule"].astype(str).str.upper() == TOP10_RULE].copy()

        if sub.empty:
            continue

        sub["source_file"] = str(path)
        frames.append(sub)

        print(
            f"Loading: {path}\n"
            f"  TOP10 rows = {len(sub)}"
        )

    if not frames:
        raise ValueError(
            "TOP10行を含むV4.2 rank-band daily CSVが見つかりません。"
        )

    all_df = pd.concat(frames, ignore_index=True)

    all_df["date"] = pd.to_datetime(
        all_df["date"],
        errors="coerce"
    )

    all_df = all_df.dropna(subset=["date"]).copy()

    for col in [
        "avg_diff",
        "median_diff",
        "win_rate",
        "plus1000_rate",
        "plus2000_rate",
        "machines",
        "total_diff",
    ]:
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(
                all_df[col],
                errors="coerce"
            )

    # Prefer the most recent source file for duplicate dates.
    all_df = all_df.sort_values(
        ["date", "source_file"]
    ).drop_duplicates(
        subset=["date"],
        keep="last"
    )

    all_df = all_df.sort_values("date").reset_index(drop=True)

    print()
    print(f"unique TOP10 days = {len(all_df)}")
    print(
        "date range         = "
        f"{all_df['date'].min().date()} to "
        f"{all_df['date'].max().date()}"
    )

    return all_df


# ============================================================
# CORE ANALYSIS
# ============================================================

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    diffs = pd.to_numeric(
        df["avg_diff"],
        errors="coerce"
    ).dropna()

    n = len(diffs)

    positive = int((diffs > 0).sum())
    negative = int((diffs < 0).sum())
    tie = int((diffs == 0).sum())

    ci_low, ci_high = wilson_interval(
        positive,
        n
    )

    result = {
        "model": "V4.2_C_TOP10",
        "days": n,
        "date_start": df["date"].min().date()
        if n else None,
        "date_end": df["date"].max().date()
        if n else None,
        "avg_diff": diffs.mean() if n else np.nan,
        "median_diff": diffs.median() if n else np.nan,
        "std_diff": diffs.std(ddof=1)
        if n > 1 else np.nan,
        "best_day": diffs.max() if n else np.nan,
        "worst_day": diffs.min() if n else np.nan,
        "total_diff": diffs.sum() if n else 0.0,
        "positive_days": positive,
        "negative_days": negative,
        "tie_days": tie,
        "positive_day_rate": (
            positive / n * 100
            if n else np.nan
        ),
        "positive_day_rate_wilson_low": (
            ci_low * 100
            if n else np.nan
        ),
        "positive_day_rate_wilson_high": (
            ci_high * 100
            if n else np.nan
        ),
        "max_losing_streak": max_losing_streak(diffs),
        "max_drawdown": max_drawdown(diffs),
        "profit_factor_simple": profit_factor_simple(diffs),
    }

    if "machines" in df.columns:
        machines = pd.to_numeric(
            df["machines"],
            errors="coerce"
        )

        result["avg_machines_per_day"] = machines.mean()
        result["total_machine_selections"] = machines.sum()
        result["avg_diff_per_machine"] = (
            diffs.sum() / machines.sum()
            if machines.sum() > 0
            else np.nan
        )

    return pd.DataFrame([result])


def build_rolling(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    work = df.sort_values("date").reset_index(drop=True)

    for window in ROLLING_WINDOWS:
        if len(work) < window:
            continue

        rolling = (
            work["avg_diff"]
            .rolling(window)
            .mean()
        )

        rolling_sum = (
            work["avg_diff"]
            .rolling(window)
            .sum()
        )

        for i in range(window - 1, len(work)):
            values = work.loc[
                i - window + 1:i,
                "avg_diff"
            ]

            rows.append(
                {
                    "date": work.loc[i, "date"],
                    "window_days": window,
                    "rolling_avg_diff": rolling.iloc[i],
                    "rolling_total_diff": rolling_sum.iloc[i],
                    "rolling_positive_rate": (
                        (values > 0).mean() * 100
                    ),
                    "rolling_max_losing_streak": (
                        max_losing_streak(values)
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_blocks(df: pd.DataFrame) -> pd.DataFrame:
    work = df.sort_values("date").reset_index(drop=True)

    rows = []

    for block_id, start in enumerate(
        range(0, len(work), BLOCK_SIZE),
        start=1
    ):
        block = work.iloc[
            start:start + BLOCK_SIZE
        ].copy()

        if block.empty:
            continue

        diffs = block["avg_diff"].dropna()

        rows.append(
            {
                "block": block_id,
                "block_start": block["date"].min().date(),
                "block_end": block["date"].max().date(),
                "days": len(diffs),
                "avg_diff": diffs.mean(),
                "median_diff": diffs.median(),
                "total_diff": diffs.sum(),
                "positive_days": int(
                    (diffs > 0).sum()
                ),
                "negative_days": int(
                    (diffs < 0).sum()
                ),
                "positive_day_rate": (
                    (diffs > 0).mean() * 100
                ),
                "max_losing_streak": (
                    max_losing_streak(diffs)
                ),
                "max_drawdown": max_drawdown(diffs),
            }
        )

    return pd.DataFrame(rows)


def build_monthly(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    work["month"] = work["date"].dt.to_period(
        "M"
    ).astype(str)

    rows = []

    for month, g in work.groupby("month"):
        diffs = g["avg_diff"].dropna()

        rows.append(
            {
                "month": month,
                "days": len(diffs),
                "avg_diff": diffs.mean(),
                "median_diff": diffs.median(),
                "total_diff": diffs.sum(),
                "positive_days": int(
                    (diffs > 0).sum()
                ),
                "negative_days": int(
                    (diffs < 0).sum()
                ),
                "positive_day_rate": (
                    (diffs > 0).mean() * 100
                ),
                "max_losing_streak": (
                    max_losing_streak(diffs)
                ),
                "max_drawdown": max_drawdown(diffs),
                "profit_factor_simple": (
                    profit_factor_simple(diffs)
                ),
            }
        )

    return pd.DataFrame(rows)


def build_half_split(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    if n < 4:
        return pd.DataFrame()

    mid = n // 2

    first = df.iloc[:mid]["avg_diff"].dropna()
    second = df.iloc[mid:]["avg_diff"].dropna()

    first_avg = first.mean()
    second_avg = second.mean()

    return pd.DataFrame(
        [
            {
                "first_days": len(first),
                "second_days": len(second),
                "first_avg_diff": first_avg,
                "second_avg_diff": second_avg,
                "second_minus_first": (
                    second_avg - first_avg
                ),
                "first_total_diff": first.sum(),
                "second_total_diff": second.sum(),
            }
        ]
    )


def build_recent_windows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for window in [7, 14, 21, 30, 45, 60, 90]:
        if len(df) < window:
            continue

        g = df.tail(window)
        diffs = g["avg_diff"].dropna()

        rows.append(
            {
                "window_days": window,
                "start": g["date"].min().date(),
                "end": g["date"].max().date(),
                "avg_diff": diffs.mean(),
                "median_diff": diffs.median(),
                "total_diff": diffs.sum(),
                "positive_day_rate": (
                    (diffs > 0).mean() * 100
                ),
                "max_losing_streak": (
                    max_losing_streak(diffs)
                ),
                "max_drawdown": max_drawdown(diffs),
            }
        )

    return pd.DataFrame(rows)


def interpretation(summary: pd.DataFrame) -> pd.DataFrame:
    row = summary.iloc[0]
    days = int(row["days"])

    if days < MIN_DAYS_WARNING:
        level = "INSUFFICIENT_LONG_TERM_HISTORY"
        message = (
            "長期安定性を判断するにはサンプル不足。"
            "現状は暫定評価のみ。"
        )
    elif days < MIN_DAYS_REASONABLE:
        level = "SHORT_HISTORY"
        message = (
            "30日超の確認はできるが、長期運用判断にはまだ不足。"
        )
    elif days < MIN_DAYS_STRONG:
        level = "REASONABLE_BUT_NOT_STRONG"
        message = (
            "中期的な安定性評価が可能。"
            "季節性・店舗イベント等の影響には注意。"
        )
    else:
        level = "STRONGER_HISTORY"
        message = (
            "比較的長いOOS履歴。"
            "それでも将来利益を保証するものではない。"
        )

    return pd.DataFrame(
        [
            {
                "assessment": level,
                "message": message,
                "days": days,
                "min_warning_days": MIN_DAYS_WARNING,
                "min_reasonable_days": MIN_DAYS_REASONABLE,
                "min_strong_days": MIN_DAYS_STRONG,
            }
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print_header(
        "Ana-Slo Ver.4.2 TOP10 Long-Term Stability Analysis"
    )

    df = load_all_sources()

    summary = build_summary(df)
    rolling = build_rolling(df)
    blocks = build_blocks(df)
    monthly = build_monthly(df)
    half_split = build_half_split(df)
    recent = build_recent_windows(df)
    assessment = interpretation(summary)

    print_header("TOP10 SUMMARY")

    s = summary.iloc[0]

    print(f"days                    : {int(s['days'])}")
    print(
        f"date range              : "
        f"{s['date_start']} to {s['date_end']}"
    )
    print(
        f"average diff            : "
        f"{s['avg_diff']:+.2f}"
    )
    print(
        f"median diff             : "
        f"{s['median_diff']:+.2f}"
    )
    print(
        f"total diff              : "
        f"{s['total_diff']:+.0f}"
    )
    print(
        f"positive day rate       : "
        f"{s['positive_day_rate']:.2f}%"
    )
    print(
        f"Wilson 95% CI           : "
        f"{s['positive_day_rate_wilson_low']:.2f}% "
        f"to "
        f"{s['positive_day_rate_wilson_high']:.2f}%"
    )
    print(
        f"max losing streak       : "
        f"{int(s['max_losing_streak'])}"
    )
    print(
        f"max drawdown             : "
        f"{s['max_drawdown']:+.0f}"
    )
    print(
        f"profit factor           : "
        f"{s['profit_factor_simple']:.3f}"
    )

    if "avg_diff_per_machine" in s:
        print(
            f"avg diff / machine      : "
            f"{s['avg_diff_per_machine']:+.2f}"
        )

    print_header("MONTHLY")

    if monthly.empty:
        print("monthly data unavailable")
    else:
        print(monthly.to_string(index=False))

    print_header("7-DAY BLOCK STABILITY")

    if blocks.empty:
        print("block data unavailable")
    else:
        print(blocks.to_string(index=False))

    print_header("RECENT WINDOW")

    if recent.empty:
        print("not enough data")
    else:
        print(recent.to_string(index=False))

    print_header("FIRST HALF vs SECOND HALF")

    if half_split.empty:
        print("not enough data")
    else:
        print(half_split.to_string(index=False))

    print_header("ASSESSMENT")

    a = assessment.iloc[0]

    print(f"status                  : {a['assessment']}")
    print(f"message                 : {a['message']}")

    print()
    print(
        "NOTE: TOP10 is evaluated here as the already-generated "
        "V4.2 OOS result."
    )
    print(
        "This analysis does not tune the model and does not prove "
        "future profitability."
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    out_daily = OUTPUT_ROOT / (
        "47_Ver4_2_TOP10_long_term_stability_daily.csv"
    )
    out_summary = OUTPUT_ROOT / (
        "47_Ver4_2_TOP10_long_term_stability_summary.csv"
    )
    out_monthly = OUTPUT_ROOT / (
        "47_Ver4_2_TOP10_long_term_stability_monthly.csv"
    )
    out_blocks = OUTPUT_ROOT / (
        "47_Ver4_2_TOP10_long_term_stability_blocks.csv"
    )
    out_rolling = OUTPUT_ROOT / (
        "47_Ver4_2_TOP10_long_term_stability_rolling.csv"
    )
    out_recent = OUTPUT_ROOT / (
        "47_Ver4_2_TOP10_long_term_stability_recent.csv"
    )
    out_half = OUTPUT_ROOT / (
        "47_Ver4_2_TOP10_long_term_stability_half_split.csv"
    )
    out_assessment = OUTPUT_ROOT / (
        "47_Ver4_2_TOP10_long_term_stability_assessment.csv"
    )

    df.to_csv(
        out_daily,
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        out_summary,
        index=False,
        encoding="utf-8-sig"
    )

    monthly.to_csv(
        out_monthly,
        index=False,
        encoding="utf-8-sig"
    )

    blocks.to_csv(
        out_blocks,
        index=False,
        encoding="utf-8-sig"
    )

    rolling.to_csv(
        out_rolling,
        index=False,
        encoding="utf-8-sig"
    )

    recent.to_csv(
        out_recent,
        index=False,
        encoding="utf-8-sig"
    )

    half_split.to_csv(
        out_half,
        index=False,
        encoding="utf-8-sig"
    )

    assessment.to_csv(
        out_assessment,
        index=False,
        encoding="utf-8-sig"
    )

    print_header("FILES SAVED")

    print(out_daily)
    print(out_summary)
    print(out_monthly)
    print(out_blocks)
    print(out_rolling)
    print(out_recent)
    print(out_half)
    print(out_assessment)

    print()
    print("Ver.4.2 TOP10 long-term stability analysis complete.")


if __name__ == "__main__":
    main()
