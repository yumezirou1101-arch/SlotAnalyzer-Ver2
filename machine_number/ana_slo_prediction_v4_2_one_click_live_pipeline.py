from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys
import time
import re

import pandas as pd


# ============================================================
# 79 - One-Click Live Prediction Pipeline
# ============================================================
#
# Purpose
# -------
# Run the full prediction preparation flow for one target date:
#
#   64 auto  -> Normal prediction
#   74       -> A-type separated prediction
#   75       -> Juggler separated prediction
#   77       -> Integrated practical report
#
# Safety
# -------
# - Existing prediction scripts are called as-is.
# - This launcher does NOT recalculate scores itself.
# - Each stage must finish successfully before the next starts.
# - Expected output files are checked after each stage.
# - 76 evaluation and 78 lottery filtering are NOT run here.
#
# Recommended live flow
# ---------------------
# Before lottery:
#   79 -> 64 / 74 / 75 / 77
#
# After lottery:
#   78 HISTORY_V2
#
# After actual results become available:
#   76
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

MACHINE_SCRIPT_DIR = (
    PROJECT_ROOT
    / "machine_number"
)

ANALYSIS_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
)

SCRIPT_64 = (
    MACHINE_SCRIPT_DIR
    / "ana_slo_prediction_v4_2_future_top10_auto.py"
)

SCRIPT_74 = (
    MACHINE_SCRIPT_DIR
    / "ana_slo_prediction_v4_2_A_type_separated.py"
)

SCRIPT_75 = (
    MACHINE_SCRIPT_DIR
    / "ana_slo_prediction_v4_2_Juggler_separated.py"
)

SCRIPT_77 = (
    MACHINE_SCRIPT_DIR
    / "ana_slo_prediction_v4_2_live_integrated_report.py"
)

DIR_64 = (
    ANALYSIS_DIR
    / "64_Ver4_2_future_top10"
)

DIR_74 = (
    ANALYSIS_DIR
    / "74_Ver4_2_A_type_prediction"
)

DIR_75 = (
    ANALYSIS_DIR
    / "75_Ver4_2_Juggler_prediction"
)

DIR_77 = (
    ANALYSIS_DIR
    / "77_live_integrated_prediction_report"
)

LOG_DIR = (
    ANALYSIS_DIR
    / "79_one_click_prediction_pipeline"
)


def header(title: str) -> None:
    print()
    print("=" * 118)
    print(title)
    print("=" * 118)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run 64 -> 74 -> 75 -> 77 for one prediction target date."
        )
    )

    parser.add_argument(
        "--target-date",
        default=None,
        help=(
            "Prediction target date YYYY-MM-DD. "
            "If omitted, 64 auto uses latest validated daily CSV + 1 day."
        ),
    )

    parser.add_argument(
        "--allow-gap",
        action="store_true",
        help=(
            "Pass --allow-gap to 64 auto when target-date is intentionally "
            "more than one day after the latest daily CSV."
        ),
    )

    return parser.parse_args()


def run_stage(
    stage_name: str,
    script_path: Path,
    args: list[str],
) -> tuple[int, float]:

    if not script_path.exists():
        raise FileNotFoundError(
            f"{stage_name}: script not found:\n{script_path}"
        )

    cmd = [
        sys.executable,
        str(script_path),
        *args,
    ]

    header(
        f"RUN {stage_name}"
    )

    print(
        "command               : "
        + " ".join(
            f'"{x}"' if " " in x else x
            for x in cmd
        )
    )

    started = time.perf_counter()

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    print()
    print(
        f"{stage_name} return code    : {result.returncode}"
    )
    print(
        f"{stage_name} elapsed sec    : {elapsed:.2f}"
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{stage_name} failed with return code "
            f"{result.returncode}."
        )

    return (
        result.returncode,
        elapsed,
    )


def latest_64_target_date() -> pd.Timestamp:

    rx = re.compile(
        r"^64_prediction_(\d{8})_top10\.csv$",
        re.IGNORECASE,
    )

    dates = []

    if not DIR_64.exists():
        raise FileNotFoundError(
            f"64 output directory not found:\n{DIR_64}"
        )

    for path in DIR_64.glob(
        "64_prediction_????????_top10.csv"
    ):
        m = rx.fullmatch(
            path.name
        )

        if not m:
            continue

        dt = pd.to_datetime(
            m.group(1),
            format="%Y%m%d",
            errors="coerce",
        )

        if not pd.isna(
            dt
        ):
            dates.append(
                pd.Timestamp(dt).normalize()
            )

    if not dates:
        raise RuntimeError(
            "No 64 top10 output found after stage 64."
        )

    return max(
        dates
    )


def require_file(
    path: Path,
    label: str,
) -> None:

    if not path.exists():
        raise FileNotFoundError(
            f"{label} output missing:\n{path}"
        )

    if path.stat().st_size <= 0:
        raise RuntimeError(
            f"{label} output is empty:\n{path}"
        )

    print(
        f"{label:<22}: OK  {path.name}"
    )


def main() -> None:
    args = parse_args()

    header(
        "79 - One-Click Live Prediction Pipeline"
    )

    print(
        f"project root          : {PROJECT_ROOT}"
    )

    print(
        f"requested target      : "
        f"{args.target_date if args.target_date else '(auto)'}"
    )

    print(
        f"allow gap             : {args.allow_gap}"
    )

    stage_rows = []

    # --------------------------------------------------------
    # 64
    # --------------------------------------------------------

    args64 = []

    if args.target_date:
        args64 += [
            "--target-date",
            args.target_date,
        ]

    if args.allow_gap:
        args64.append(
            "--allow-gap"
        )

    _, elapsed = run_stage(
        "64 NORMAL",
        SCRIPT_64,
        args64,
    )

    stage_rows.append(
        {
            "stage": "64_NORMAL",
            "elapsed_sec": elapsed,
            "status": "OK",
        }
    )

    # Determine the actual target date from the generated 64 file.
    if args.target_date:
        target_date = pd.Timestamp(
            pd.to_datetime(
                args.target_date,
                format="%Y-%m-%d",
                errors="raise",
            )
        ).normalize()
    else:
        target_date = latest_64_target_date()

    ymd = target_date.strftime(
        "%Y%m%d"
    )

    # Validate 64 immediately.
    normal_top10 = (
        DIR_64
        / f"64_prediction_{ymd}_top10.csv"
    )

    normal_all = (
        DIR_64
        / f"64_prediction_{ymd}_all514.csv"
    )

    normal_meta = (
        DIR_64
        / f"64_prediction_{ymd}_metadata.csv"
    )

    header(
        "CHECK 64 OUTPUT"
    )

    require_file(
        normal_top10,
        "64 top10"
    )

    require_file(
        normal_all,
        "64 all514"
    )

    require_file(
        normal_meta,
        "64 metadata"
    )

    # --------------------------------------------------------
    # 74
    # --------------------------------------------------------

    _, elapsed = run_stage(
        "74 A-TYPE",
        SCRIPT_74,
        [
            "--target-date",
            target_date.strftime(
                "%Y-%m-%d"
            ),
        ],
    )

    stage_rows.append(
        {
            "stage": "74_A_TYPE",
            "elapsed_sec": elapsed,
            "status": "OK",
        }
    )

    a_top10 = (
        DIR_74
        / f"74_A_type_prediction_{ymd}_top10.csv"
    )

    header(
        "CHECK 74 OUTPUT"
    )

    require_file(
        a_top10,
        "74 A-type top10"
    )

    # --------------------------------------------------------
    # 75
    # --------------------------------------------------------

    _, elapsed = run_stage(
        "75 JUGGLER",
        SCRIPT_75,
        [
            "--target-date",
            target_date.strftime(
                "%Y-%m-%d"
            ),
        ],
    )

    stage_rows.append(
        {
            "stage": "75_JUGGLER",
            "elapsed_sec": elapsed,
            "status": "OK",
        }
    )

    j_top10 = (
        DIR_75
        / f"75_Juggler_prediction_{ymd}_top10.csv"
    )

    header(
        "CHECK 75 OUTPUT"
    )

    require_file(
        j_top10,
        "75 Juggler top10"
    )

    # --------------------------------------------------------
    # 77
    # --------------------------------------------------------

    _, elapsed = run_stage(
        "77 INTEGRATED",
        SCRIPT_77,
        [
            "--target-date",
            target_date.strftime(
                "%Y-%m-%d"
            ),
        ],
    )

    stage_rows.append(
        {
            "stage": "77_INTEGRATED",
            "elapsed_sec": elapsed,
            "status": "OK",
        }
    )

    integrated = (
        DIR_77
        / f"77_integrated_prediction_{ymd}.csv"
    )

    integrated_summary = (
        DIR_77
        / f"77_integrated_prediction_{ymd}_summary.csv"
    )

    header(
        "CHECK 77 OUTPUT"
    )

    require_file(
        integrated,
        "77 integrated"
    )

    require_file(
        integrated_summary,
        "77 summary"
    )

    # --------------------------------------------------------
    # Save pipeline run log
    # --------------------------------------------------------

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        LOG_DIR
        / f"79_pipeline_{ymd}_status.csv"
    )

    status_df = pd.DataFrame(
        stage_rows
    )

    status_df[
        "target_date"
    ] = target_date.date()

    status_df[
        "pipeline_complete"
    ] = True

    status_df.to_csv(
        log_path,
        index=False,
        encoding="utf-8-sig",
    )

    header(
        "PIPELINE COMPLETE"
    )

    print(
        f"target date           : {target_date.date()}"
    )

    print(
        "64 normal prediction : OK"
    )

    print(
        "74 A-type prediction : OK"
    )

    print(
        "75 Juggler prediction: OK"
    )

    print(
        "77 integrated report : OK"
    )

    print(
        f"status log            : {log_path}"
    )

    print()
    print(
        "Next step after lottery:"
    )

    print(
        "Run 78 HISTORY_V2 with actual lottery numbers "
        "and unavailable machine numbers."
    )

    print()
    print(
        "79 one-click prediction pipeline complete."
    )


if __name__ == "__main__":
    main()
