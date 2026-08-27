from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 73 - V4.2_C Backtest + Live Forward Integrated Evaluation
# ============================================================
#
# Purpose
# -------
# Integrate:
#   72 = historical walk-forward backtest
#   71 = live forward rank-band tracking
#
# Important:
# - Backtest and live-forward results remain SEPARATE.
# - This script does NOT average them into one "final score".
# - No model weights are changed.
# - No prediction files are modified.
#
# Main question:
#   "Is a rank band strong in historical OOS AND also holding up
#    in genuinely live forward predictions?"
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

ANALYSIS_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
)

BACKTEST_FILE = (
    ANALYSIS_DIR
    / "72_Ver4_2_C_walk_forward_rank_band_backtest"
    / "72_rank_band_overall.csv"
)

LIVE_FILE = (
    ANALYSIS_DIR
    / "71_Ver4_2_rank_band_forward_tracker"
    / "71_rank_band_overall.csv"
)

OUTPUT_DIR = (
    ANALYSIS_DIR
    / "73_Ver4_2_C_backtest_live_integrated_evaluation"
)

FIRST_LIVE_REVIEW_DAYS = 21

BAND_ORDER = [
    "TOP1",
    "TOP3",
    "TOP5",
    "TOP10",
    "PRIMARY_1_5",
    "NEXT_6_10",
    "RANK_7_9",
]


def header(title: str) -> None:
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error = None

    for enc in (
        "utf-8-sig",
        "utf-8",
        "cp932",
    ):
        try:
            return pd.read_csv(
                path,
                encoding=enc,
            )
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"CSV read failed: {path}\n"
        f"last_error={last_error}"
    )


def require_columns(
    df: pd.DataFrame,
    cols: list[str],
    label: str,
) -> None:

    missing = [
        c
        for c in cols
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{label}: required columns missing: {missing}"
        )


def prepare_backtest(
    df: pd.DataFrame,
) -> pd.DataFrame:

    require_columns(
        df,
        [
            "band",
            "evaluated_days",
            "selected_rows",
            "avg_diff_per_machine",
            "total_diff",
            "win_rate",
            "plus1000_rate",
            "plus2000_rate",
            "positive_day_rate",
            "mean_store_avg_diff",
            "mean_lift_vs_store",
            "daily_lift_ci95_low",
            "daily_lift_ci95_high",
        ],
        "72 backtest",
    )

    x = df.copy()

    rename = {
        c: f"bt_{c}"
        for c in x.columns
        if c != "band"
    }

    x = x.rename(
        columns=rename
    )

    return x


def prepare_live(
    df: pd.DataFrame,
) -> pd.DataFrame:

    require_columns(
        df,
        [
            "band",
            "evaluated_days",
            "selected_rows",
            "avg_diff_per_machine",
            "total_diff",
            "win_rate",
            "plus1000_rate",
            "plus2000_rate",
            "positive_day_rate",
            "machine_avg_diff_ci95_low",
            "machine_avg_diff_ci95_high",
        ],
        "71 live forward",
    )

    x = df.copy()

    rename = {
        c: f"live_{c}"
        for c in x.columns
        if c != "band"
    }

    x = x.rename(
        columns=rename
    )

    return x


def sign_label(value: float) -> str:

    if pd.isna(value):
        return "UNKNOWN"

    if value > 0:
        return "POSITIVE"

    if value < 0:
        return "NEGATIVE"

    return "ZERO"


def build_integrated_table(
    bt: pd.DataFrame,
    live: pd.DataFrame,
) -> pd.DataFrame:

    merged = bt.merge(
        live,
        on="band",
        how="outer",
        validate="one_to_one",
    )

    merged["backtest_direction"] = (
        merged[
            "bt_avg_diff_per_machine"
        ].apply(
            sign_label
        )
    )

    merged["live_direction"] = (
        merged[
            "live_avg_diff_per_machine"
        ].apply(
            sign_label
        )
    )

    merged[
        "same_direction"
    ] = (
        merged[
            "backtest_direction"
        ]
        == merged[
            "live_direction"
        ]
    )

    merged[
        "live_minus_backtest_avg_diff"
    ] = (
        merged[
            "live_avg_diff_per_machine"
        ]
        - merged[
            "bt_avg_diff_per_machine"
        ]
    )

    merged[
        "live_minus_backtest_win_rate"
    ] = (
        merged[
            "live_win_rate"
        ]
        - merged[
            "bt_win_rate"
        ]
    )

    merged[
        "live_minus_backtest_plus2000_rate"
    ] = (
        merged[
            "live_plus2000_rate"
        ]
        - merged[
            "bt_plus2000_rate"
        ]
    )

    merged[
        "live_review_ready"
    ] = (
        merged[
            "live_evaluated_days"
        ]
        >= FIRST_LIVE_REVIEW_DAYS
    )

    statuses = []

    for row in merged.itertuples(
        index=False
    ):

        bt_avg = getattr(
            row,
            "bt_avg_diff_per_machine",
        )

        live_avg = getattr(
            row,
            "live_avg_diff_per_machine",
        )

        live_days = int(
            getattr(
                row,
                "live_evaluated_days",
            )
        )

        if live_days < FIRST_LIVE_REVIEW_DAYS:

            if bt_avg > 0 and live_avg > 0:
                status = (
                    "PROMISING_BUT_LIVE_SAMPLE_SMALL"
                )

            elif bt_avg > 0 and live_avg <= 0:
                status = (
                    "BACKTEST_POSITIVE_LIVE_WEAK"
                )

            elif bt_avg <= 0 and live_avg > 0:
                status = (
                    "LIVE_POSITIVE_BACKTEST_WEAK"
                )

            else:
                status = (
                    "WEAK_BOTH_SAMPLE_SMALL"
                )

        else:

            if bt_avg > 0 and live_avg > 0:
                status = (
                    "CONSISTENT_POSITIVE_REVIEW"
                )

            elif bt_avg > 0 and live_avg <= 0:
                status = (
                    "BACKTEST_LIVE_DIVERGENCE_REVIEW"
                )

            elif bt_avg <= 0 and live_avg > 0:
                status = (
                    "LIVE_ONLY_POSITIVE_REVIEW"
                )

            else:
                status = (
                    "CONSISTENT_WEAK_REVIEW"
                )

        statuses.append(
            status
        )

    merged[
        "integrated_status"
    ] = statuses

    order = {
        name: i
        for i, name
        in enumerate(
            BAND_ORDER
        )
    }

    merged[
        "_order"
    ] = (
        merged["band"]
        .map(order)
        .fillna(999)
    )

    merged = (
        merged.sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return merged


def build_focus_comparison(
    integrated: pd.DataFrame,
) -> pd.DataFrame:

    focus = integrated[
        integrated["band"].isin(
            [
                "PRIMARY_1_5",
                "NEXT_6_10",
                "RANK_7_9",
                "TOP10",
            ]
        )
    ].copy()

    return focus[
        [
            "band",
            "bt_evaluated_days",
            "bt_selected_rows",
            "bt_avg_diff_per_machine",
            "bt_total_diff",
            "bt_win_rate",
            "bt_plus2000_rate",
            "bt_positive_day_rate",
            "bt_mean_lift_vs_store",
            "live_evaluated_days",
            "live_selected_rows",
            "live_avg_diff_per_machine",
            "live_total_diff",
            "live_win_rate",
            "live_plus2000_rate",
            "live_positive_day_rate",
            "live_minus_backtest_avg_diff",
            "same_direction",
            "integrated_status",
        ]
    ]


def build_status_summary(
    integrated: pd.DataFrame,
) -> pd.DataFrame:

    live_days = int(
        integrated[
            "live_evaluated_days"
        ].max()
    )

    rows = [
        {
            "current_model":
                "CHAMPION_V4.2_C",
            "backtest_source":
                str(BACKTEST_FILE),
            "live_source":
                str(LIVE_FILE),
            "backtest_days":
                int(
                    integrated[
                        "bt_evaluated_days"
                    ].max()
                ),
            "live_days":
                live_days,
            "first_live_review_days":
                FIRST_LIVE_REVIEW_DAYS,
            "days_remaining_to_first_live_review":
                max(
                    0,
                    FIRST_LIVE_REVIEW_DAYS
                    - live_days,
                ),
            "model_weights_changed":
                False,
            "automatic_model_change":
                False,
            "status":
                (
                    "ACCUMULATING_LIVE_FORWARD_DATA"
                    if live_days
                    < FIRST_LIVE_REVIEW_DAYS
                    else
                    "READY_FOR_MANUAL_INTEGRATED_REVIEW"
                ),
        }
    ]

    return pd.DataFrame(
        rows
    )


def main() -> None:

    header(
        "73 - V4.2_C Backtest + Live Forward Integrated Evaluation"
    )

    if not BACKTEST_FILE.exists():
        raise FileNotFoundError(
            f"72 backtest file not found: "
            f"{BACKTEST_FILE}"
        )

    if not LIVE_FILE.exists():
        raise FileNotFoundError(
            f"71 live-forward file not found: "
            f"{LIVE_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    bt_raw = read_csv_flexible(
        BACKTEST_FILE
    )

    live_raw = read_csv_flexible(
        LIVE_FILE
    )

    bt = prepare_backtest(
        bt_raw
    )

    live = prepare_live(
        live_raw
    )

    integrated = (
        build_integrated_table(
            bt,
            live,
        )
    )

    focus = (
        build_focus_comparison(
            integrated
        )
    )

    status = (
        build_status_summary(
            integrated
        )
    )

    print(
        f"backtest file         : {BACKTEST_FILE}"
    )
    print(
        f"live-forward file     : {LIVE_FILE}"
    )
    print(
        f"backtest bands        : {len(bt)}"
    )
    print(
        f"live-forward bands    : {len(live)}"
    )
    print(
        f"backtest days         : "
        f"{int(integrated['bt_evaluated_days'].max())}"
    )
    print(
        f"live-forward days     : "
        f"{int(integrated['live_evaluated_days'].max())}"
    )

    header(
        "FOCUS COMPARISON"
    )

    print(
        focus.to_string(
            index=False
        )
    )

    header(
        "ALL BAND INTEGRATED STATUS"
    )

    display_cols = [
        "band",
        "bt_avg_diff_per_machine",
        "live_avg_diff_per_machine",
        "live_minus_backtest_avg_diff",
        "bt_win_rate",
        "live_win_rate",
        "bt_plus2000_rate",
        "live_plus2000_rate",
        "same_direction",
        "live_review_ready",
        "integrated_status",
    ]

    print(
        integrated[
            display_cols
        ].to_string(
            index=False
        )
    )

    header(
        "PROJECT STATUS"
    )

    print(
        status.to_string(
            index=False
        )
    )

    paths = {
        "integrated":
            OUTPUT_DIR
            / "73_integrated_rank_band_evaluation.csv",
        "focus":
            OUTPUT_DIR
            / "73_focus_primary_next_rank7_9_top10.csv",
        "status":
            OUTPUT_DIR
            / "73_integrated_status.csv",
    }

    integrated.to_csv(
        paths["integrated"],
        index=False,
        encoding="utf-8-sig",
    )

    focus.to_csv(
        paths["focus"],
        index=False,
        encoding="utf-8-sig",
    )

    status.to_csv(
        paths["status"],
        index=False,
        encoding="utf-8-sig",
    )

    header(
        "FILES SAVED"
    )

    for path in paths.values():
        print(path)

    print()
    print(
        "73 integrated evaluation complete."
    )
    print(
        "Backtest and live-forward results remain separate."
    )
    print(
        "No model weights or predictions were changed."
    )


if __name__ == "__main__":
    main()
