from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "ana_slo_20260711_20260818.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
    / "55_Ver4_2_machine_number_position_diagnostic"
)

MIN_GROUP_N = 20


# ============================================================
# HELPERS
# ============================================================

def header(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def read_csv_flexible(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"CSV read failed: {path}")


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def group_summary(
    df: pd.DataFrame,
    group_col: str,
    label: str,
) -> pd.DataFrame:

    x = df.copy()

    out = (
        x.groupby(group_col, dropna=False)
        .agg(
            records=("diff", "size"),
            days=("date", "nunique"),
            machines=("machine_no", "nunique"),
            avg_diff=("diff", "mean"),
            median_diff=("diff", "median"),
            total_diff=("diff", "sum"),
            win_rate=("win", "mean"),
            plus1000_rate=("plus1000", "mean"),
            plus2000_rate=("plus2000", "mean"),
            avg_g=("G", "mean"),
        )
        .reset_index()
    )

    for col in ("win_rate", "plus1000_rate", "plus2000_rate"):
        out[col] = out[col] * 100.0

    out.insert(0, "feature", label)

    return out


def daily_group_summary(
    df: pd.DataFrame,
    group_col: str,
    label: str,
) -> pd.DataFrame:

    daily = (
        df.groupby(["date", group_col], dropna=False)
        .agg(
            n=("diff", "size"),
            avg_diff=("diff", "mean"),
            total_diff=("diff", "sum"),
            win_rate=("win", "mean"),
        )
        .reset_index()
    )

    daily["win_rate"] *= 100.0

    out = (
        daily.groupby(group_col, dropna=False)
        .agg(
            observed_days=("date", "nunique"),
            mean_daily_avg_diff=("avg_diff", "mean"),
            median_daily_avg_diff=("avg_diff", "median"),
            positive_day_rate=("total_diff", lambda s: float((s > 0).mean() * 100.0)),
            mean_daily_win_rate=("win_rate", "mean"),
        )
        .reset_index()
    )

    out.insert(0, "feature", label)
    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    header("55 - Machine Number / Position Characteristic Diagnostic")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = read_csv_flexible(INPUT_CSV)

    date_col = find_col(df, ["date", "\u65e5\u4ed8"])
    no_col = find_col(df, ["machine_no", "\u53f0\u756a\u53f7"])
    name_col = find_col(df, ["machine_name", "\u6a5f\u7a2e\u540d"])
    diff_col = find_col(df, ["diff", "\u5dee\u679a"])
    g_col = find_col(df, ["G", "G\u6570", "games"])

    if not all([date_col, no_col, name_col, diff_col]):
        raise RuntimeError(
            "Required columns not found: "
            f"date={date_col}, no={no_col}, name={name_col}, diff={diff_col}"
        )

    rename = {
        date_col: "date",
        no_col: "machine_no",
        name_col: "machine_name",
        diff_col: "diff",
    }

    if g_col:
        rename[g_col] = "G"

    df = df.rename(columns=rename)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["machine_no"] = pd.to_numeric(df["machine_no"], errors="coerce")
    df["diff"] = clean_numeric(df["diff"])

    if "G" in df.columns:
        df["G"] = clean_numeric(df["G"])
    else:
        df["G"] = np.nan

    df["machine_name"] = df["machine_name"].astype(str).str.strip()

    df = df.dropna(
        subset=["date", "machine_no", "machine_name", "diff"]
    ).copy()

    df["machine_no"] = df["machine_no"].astype(int)

    df = (
        df.sort_values(["date", "machine_no"])
        .drop_duplicates(["date", "machine_no"], keep="last")
        .reset_index(drop=True)
    )

    df["win"] = (df["diff"] > 0).astype(int)
    df["plus1000"] = (df["diff"] >= 1000).astype(int)
    df["plus2000"] = (df["diff"] >= 2000).astype(int)

    print(f"records              : {len(df):,}")
    print(f"days                 : {df['date'].nunique()}")
    print(f"unique machine nos   : {df['machine_no'].nunique()}")
    print(f"date range           : {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"machine no range     : {df['machine_no'].min()} to {df['machine_no'].max()}")

    # --------------------------------------------------------
    # Features based only on machine number.
    # These are diagnostics, not physical-island claims.
    # --------------------------------------------------------

    df["last_digit"] = df["machine_no"] % 10
    df["last_two_digits"] = df["machine_no"] % 100

    df["number_band_10"] = (
        (df["machine_no"] // 10) * 10
    ).astype(int)

    df["number_band_50"] = (
        (df["machine_no"] // 50) * 50
    ).astype(int)

    # Position within each contiguous machine-number run.
    # This does NOT claim a physical island.
    unique_nos = sorted(df["machine_no"].unique())

    run_map: dict[int, tuple[int, int, int, int]] = {}
    runs: list[list[int]] = []

    current_run: list[int] = []

    for no in unique_nos:
        if not current_run or no == current_run[-1] + 1:
            current_run.append(no)
        else:
            runs.append(current_run)
            current_run = [no]

    if current_run:
        runs.append(current_run)

    for run_id, run in enumerate(runs, start=1):
        run_len = len(run)
        for pos, no in enumerate(run, start=1):
            run_map[no] = (run_id, pos, run_len, min(pos - 1, run_len - pos))

    df["number_run_id"] = df["machine_no"].map(lambda x: run_map[x][0])
    df["position_in_number_run"] = df["machine_no"].map(lambda x: run_map[x][1])
    df["number_run_length"] = df["machine_no"].map(lambda x: run_map[x][2])
    df["distance_from_number_run_edge"] = df["machine_no"].map(lambda x: run_map[x][3])

    def edge_bucket(distance: int) -> str:
        if distance == 0:
            return "EDGE_1"
        if distance == 1:
            return "EDGE_2"
        if distance == 2:
            return "EDGE_3"
        if distance <= 4:
            return "EDGE_4_5"
        return "INTERIOR_6_PLUS"

    df["number_run_edge_bucket"] = (
        df["distance_from_number_run_edge"]
        .map(edge_bucket)
    )

    header("CONTIGUOUS NUMBER RUNS - REFERENCE ONLY")

    run_rows = []
    for run_id, run in enumerate(runs, start=1):
        run_rows.append(
            {
                "number_run_id": run_id,
                "start_no": min(run),
                "end_no": max(run),
                "machines": len(run),
            }
        )

    runs_df = pd.DataFrame(run_rows)
    print(runs_df.to_string(index=False))
    print()
    print(
        "NOTE: contiguous number runs are NOT assumed to be physical islands."
    )

    # --------------------------------------------------------
    # Aggregate diagnostics
    # --------------------------------------------------------

    summary_frames = []
    daily_frames = []

    diagnostics = [
        ("last_digit", "LAST_DIGIT"),
        ("number_band_10", "NUMBER_BAND_10"),
        ("number_band_50", "NUMBER_BAND_50"),
        ("number_run_edge_bucket", "NUMBER_RUN_EDGE_BUCKET"),
    ]

    for col, label in diagnostics:
        summary_frames.append(group_summary(df, col, label))
        daily_frames.append(daily_group_summary(df, col, label))

    summary_df = pd.concat(summary_frames, ignore_index=True)
    daily_summary_df = pd.concat(daily_frames, ignore_index=True)

    header("LAST DIGIT SUMMARY")
    print(
        summary_df[
            summary_df["feature"] == "LAST_DIGIT"
        ].to_string(index=False)
    )

    header("NUMBER-RUN EDGE BUCKET SUMMARY")
    print(
        summary_df[
            summary_df["feature"] == "NUMBER_RUN_EDGE_BUCKET"
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Machine-level long-run summary.
    # Helps identify whether group results are driven by a few machines.
    # --------------------------------------------------------

    machine_df = (
        df.groupby(["machine_no"])
        .agg(
            records=("diff", "size"),
            avg_diff=("diff", "mean"),
            median_diff=("diff", "median"),
            total_diff=("diff", "sum"),
            win_rate=("win", "mean"),
            plus1000_rate=("plus1000", "mean"),
            plus2000_rate=("plus2000", "mean"),
        )
        .reset_index()
    )

    machine_df["win_rate"] *= 100.0
    machine_df["plus1000_rate"] *= 100.0
    machine_df["plus2000_rate"] *= 100.0
    machine_df["last_digit"] = machine_df["machine_no"] % 10
    machine_df["number_band_10"] = (machine_df["machine_no"] // 10) * 10
    machine_df["number_band_50"] = (machine_df["machine_no"] // 50) * 50
    machine_df["number_run_edge_bucket"] = (
        machine_df["machine_no"]
        .map(lambda x: edge_bucket(run_map[x][3]))
    )

    # --------------------------------------------------------
    # Shrunk machine-number estimate:
    # regularizes each machine's mean toward the store mean.
    # Diagnostic only; prevents overreacting to 39 observations.
    # --------------------------------------------------------

    store_mean = float(df["diff"].mean())
    prior_n = 30.0

    machine_df["shrunk_avg_diff"] = (
        (
            machine_df["avg_diff"] * machine_df["records"]
            + store_mean * prior_n
        )
        / (machine_df["records"] + prior_n)
    )

    machine_rank_df = machine_df.sort_values(
        ["shrunk_avg_diff", "records"],
        ascending=[False, False],
    ).reset_index(drop=True)

    header("TOP 20 MACHINE NUMBERS - SHRUNK AVG DIFF (DIAGNOSTIC ONLY)")
    print(
        machine_rank_df[
            [
                "machine_no",
                "records",
                "avg_diff",
                "shrunk_avg_diff",
                "win_rate",
                "plus1000_rate",
                "plus2000_rate",
            ]
        ].head(20).to_string(index=False)
    )

    # --------------------------------------------------------
    # Simple stability checks: first half vs second half.
    # This is temporal consistency, not a production backtest.
    # --------------------------------------------------------

    dates = sorted(df["date"].unique())
    split_idx = len(dates) // 2
    first_dates = set(dates[:split_idx])
    second_dates = set(dates[split_idx:])

    stability_rows = []

    for col, label in diagnostics:
        first = (
            df[df["date"].isin(first_dates)]
            .groupby(col)["diff"]
            .agg(["size", "mean"])
            .reset_index()
            .rename(columns={"size": "n_first", "mean": "avg_first"})
        )

        second = (
            df[df["date"].isin(second_dates)]
            .groupby(col)["diff"]
            .agg(["size", "mean"])
            .reset_index()
            .rename(columns={"size": "n_second", "mean": "avg_second"})
        )

        merged = first.merge(second, on=col, how="inner")
        merged = merged[
            (merged["n_first"] >= MIN_GROUP_N)
            & (merged["n_second"] >= MIN_GROUP_N)
        ].copy()

        for _, row in merged.iterrows():
            first_mean = float(row["avg_first"])
            second_mean = float(row["avg_second"])

            stability_rows.append(
                {
                    "feature": label,
                    "group": str(row[col]),
                    "n_first": int(row["n_first"]),
                    "n_second": int(row["n_second"]),
                    "avg_first": first_mean,
                    "avg_second": second_mean,
                    "same_sign": bool(
                        np.sign(first_mean) == np.sign(second_mean)
                        and first_mean != 0
                        and second_mean != 0
                    ),
                    "absolute_shift": float(abs(second_mean - first_mean)),
                }
            )

    stability_df = pd.DataFrame(stability_rows)

    header("FIRST-HALF vs SECOND-HALF STABILITY - LAST DIGIT")
    print(
        stability_df[
            stability_df["feature"] == "LAST_DIGIT"
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Assessment
    # --------------------------------------------------------

    last_digit_stability = stability_df[
        stability_df["feature"] == "LAST_DIGIT"
    ].copy()

    if not last_digit_stability.empty:
        stable_sign_rate = float(
            last_digit_stability["same_sign"].mean() * 100.0
        )
    else:
        stable_sign_rate = np.nan

    assessment_df = pd.DataFrame(
        [{
            "status": "EXPLORATORY_POSITION_DIAGNOSTIC_ONLY",
            "records": len(df),
            "days": int(df["date"].nunique()),
            "machines": int(df["machine_no"].nunique()),
            "last_digit_same_sign_rate_first_vs_second_pct": stable_sign_rate,
            "physical_island_map_available": False,
            "production_rule_adopted": False,
            "note": (
                "Machine-number patterns may reflect layout, machine mix, or chance. "
                "Do not interpret number-run edges as physical island corners. "
                "Any promising pattern must be tested with leakage-safe rolling OOS "
                "before adding it to the prediction model."
            ),
        }]
    )

    header("ASSESSMENT")
    print(assessment_df.to_string(index=False))

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "55_machine_number_position_summary.csv": summary_df,
        "55_machine_number_position_daily_summary.csv": daily_summary_df,
        "55_machine_number_position_machine_rank.csv": machine_rank_df,
        "55_machine_number_position_stability.csv": stability_df,
        "55_machine_number_position_number_runs.csv": runs_df,
        "55_machine_number_position_assessment.csv": assessment_df,
    }

    header("FILES SAVED")

    for filename, frame in outputs.items():
        path = OUTPUT_DIR / filename
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(path)

    print()
    print("55 machine-number / position diagnostic complete.")
    print(
        "No production feature has been added. "
        "Promising patterns, if any, require rolling OOS validation next."
    )


if __name__ == "__main__":
    main()
