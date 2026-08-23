from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SlotAnalyzer
# Min-Repo Candidate Robustness Diagnostic
# ============================================================
#
# Input:
#   65_minrepo_external_daily.csv
#   65_minrepo_external_top10_picks.csv
#
# Output:
#   analysis_31days_deep\66_Ver4_2_minrepo_candidate_robustness\
#
# Focus candidates:
#   - MINREPO_MACHINE_DIFF3
#   - MINREPO_MACHINE_WIN3
#   - MINREPO_MACHINE_TAIL_DIFF3
#
# This script does NOT modify V4.2_C.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
    / "65_Ver4_2_minrepo_external_feature_oos"
)

DAILY_CSV = (
    SOURCE_DIR
    / "65_minrepo_external_daily.csv"
)

TOP10_CSV = (
    SOURCE_DIR
    / "65_minrepo_external_top10_picks.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
    / "66_Ver4_2_minrepo_candidate_robustness"
)

BASE_MODE = "BASE_V4.2_C"

CANDIDATES = [
    "MINREPO_MACHINE_DIFF3",
    "MINREPO_MACHINE_WIN3",
    "MINREPO_MACHINE_TAIL_DIFF3",
]

TOP_NS = [1, 3, 5, 10]


# ============================================================
# HELPERS
# ============================================================

def header(title: str) -> None:
    print()
    print("=" * 116)
    print(title)
    print("=" * 116)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required CSV not found: {path}"
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

    return df


def max_drawdown_from_series(
    values: pd.Series,
) -> float:
    x = pd.to_numeric(
        values,
        errors="coerce",
    ).fillna(0.0)

    equity = x.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max

    return float(
        drawdown.min()
    )


def spearman_rank_corr(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> float:
    a = left[
        ["machine_no", "prediction_rank"]
    ].rename(
        columns={
            "prediction_rank": "rank_left",
        }
    )

    b = right[
        ["machine_no", "prediction_rank"]
    ].rename(
        columns={
            "prediction_rank": "rank_right",
        }
    )

    merged = a.merge(
        b,
        on="machine_no",
        how="inner",
    )

    if len(merged) < 2:
        return np.nan

    return float(
        merged[
            ["rank_left", "rank_right"]
        ].corr(
            method="spearman"
        ).iloc[
            0,
            1,
        ]
    )


# ============================================================
# DAILY ROBUSTNESS
# ============================================================

def build_daily_pairwise(
    daily: pd.DataFrame,
) -> pd.DataFrame:

    base = daily[
        daily["mode"]
        == BASE_MODE
    ].copy()

    rows = []

    for mode in CANDIDATES:
        challenger = daily[
            daily["mode"]
            == mode
        ].copy()

        merged = challenger.merge(
            base,
            on=[
                "date",
                "top_n",
            ],
            suffixes=(
                "_challenger",
                "_base",
            ),
            how="inner",
        )

        merged["mode"] = mode

        merged["avg_diff_change"] = (
            merged[
                "avg_diff_challenger"
            ]
            - merged[
                "avg_diff_base"
            ]
        )

        merged["total_diff_change"] = (
            merged[
                "total_diff_challenger"
            ]
            - merged[
                "total_diff_base"
            ]
        )

        rows.append(
            merged
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def summarize_pairwise(
    pairwise: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        mode,
        top_n,
    ), g in pairwise.groupby(
        [
            "mode",
            "top_n",
        ]
    ):

        g = g.sort_values(
            "date"
        ).copy()

        changes = g[
            "total_diff_change"
        ].astype(float)

        biggest_gain_idx = (
            changes.idxmax()
            if len(changes)
            else None
        )

        biggest_loss_idx = (
            changes.idxmin()
            if len(changes)
            else None
        )

        best1_removed = (
            changes.drop(
                index=[
                    biggest_gain_idx
                ]
            ).sum()
            if len(changes) > 1
            else np.nan
        )

        best2_idx = (
            changes.nlargest(
                min(
                    2,
                    len(changes),
                )
            ).index
        )

        best2_removed = (
            changes.drop(
                index=best2_idx
            ).sum()
            if len(changes) > 2
            else np.nan
        )

        worst1_removed = (
            changes.drop(
                index=[
                    biggest_loss_idx
                ]
            ).sum()
            if len(changes) > 1
            else np.nan
        )

        rows.append(
            {
                "mode":
                    mode,

                "top_n":
                    int(
                        top_n
                    ),

                "days":
                    len(g),

                "better_days":
                    int(
                        (
                            changes > 0
                        ).sum()
                    ),

                "same_days":
                    int(
                        (
                            changes == 0
                        ).sum()
                    ),

                "worse_days":
                    int(
                        (
                            changes < 0
                        ).sum()
                    ),

                "mean_total_diff_change":
                    float(
                        changes.mean()
                    ),

                "median_total_diff_change":
                    float(
                        changes.median()
                    ),

                "std_total_diff_change":
                    float(
                        changes.std(
                            ddof=0
                        )
                    ),

                "total_diff_change":
                    float(
                        changes.sum()
                    ),

                "max_gain_day_change":
                    float(
                        changes.max()
                    ),

                "max_loss_day_change":
                    float(
                        changes.min()
                    ),

                "total_change_without_best1":
                    float(
                        best1_removed
                    )
                    if pd.notna(
                        best1_removed
                    )
                    else np.nan,

                "total_change_without_best2":
                    float(
                        best2_removed
                    )
                    if pd.notna(
                        best2_removed
                    )
                    else np.nan,

                "total_change_without_worst1":
                    float(
                        worst1_removed
                    )
                    if pd.notna(
                        worst1_removed
                    )
                    else np.nan,

                "change_max_drawdown":
                    max_drawdown_from_series(
                        changes
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PICK OVERLAP / REPLACEMENT
# ============================================================

def build_pick_comparison(
    picks: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    detail_rows = []
    summary_rows = []

    dates = sorted(
        picks["date"]
        .dropna()
        .unique()
        .tolist()
    )

    for mode in CANDIDATES:

        for date in dates:

            base_day = picks[
                (
                    picks["date"]
                    == date
                )
                & (
                    picks["mode"]
                    == BASE_MODE
                )
            ].copy()

            challenger_day = picks[
                (
                    picks["date"]
                    == date
                )
                & (
                    picks["mode"]
                    == mode
                )
            ].copy()

            if (
                base_day.empty
                or challenger_day.empty
            ):
                continue

            rank_corr = spearman_rank_corr(
                base_day,
                challenger_day,
            )

            for top_n in TOP_NS:

                base_top = base_day[
                    base_day[
                        "prediction_rank"
                    ]
                    <= top_n
                ].copy()

                chall_top = challenger_day[
                    challenger_day[
                        "prediction_rank"
                    ]
                    <= top_n
                ].copy()

                base_set = set(
                    base_top[
                        "machine_no"
                    ].astype(int)
                )

                chall_set = set(
                    chall_top[
                        "machine_no"
                    ].astype(int)
                )

                overlap = (
                    base_set
                    & chall_set
                )

                entered = (
                    chall_set
                    - base_set
                )

                removed = (
                    base_set
                    - chall_set
                )

                entered_df = chall_top[
                    chall_top[
                        "machine_no"
                    ]
                    .astype(int)
                    .isin(
                        entered
                    )
                ].copy()

                removed_df = base_top[
                    base_top[
                        "machine_no"
                    ]
                    .astype(int)
                    .isin(
                        removed
                    )
                ].copy()

                entered_total = float(
                    pd.to_numeric(
                        entered_df[
                            "diff"
                        ],
                        errors="coerce",
                    ).sum()
                )

                removed_total = float(
                    pd.to_numeric(
                        removed_df[
                            "diff"
                        ],
                        errors="coerce",
                    ).sum()
                )

                replacement_delta = (
                    entered_total
                    - removed_total
                )

                summary_rows.append(
                    {
                        "date":
                            date,

                        "mode":
                            mode,

                        "top_n":
                            top_n,

                        "overlap_count":
                            len(
                                overlap
                            ),

                        "entered_count":
                            len(
                                entered
                            ),

                        "removed_count":
                            len(
                                removed
                            ),

                        "overlap_rate":
                            (
                                len(
                                    overlap
                                )
                                / top_n
                                * 100.0
                            ),

                        "entered_total_diff":
                            entered_total,

                        "removed_total_diff":
                            removed_total,

                        "replacement_delta":
                            replacement_delta,

                        "top10_rank_spearman":
                            rank_corr,
                    }
                )

                for kind, df_part in (
                    (
                        "ENTERED",
                        entered_df,
                    ),
                    (
                        "REMOVED",
                        removed_df,
                    ),
                ):
                    for _, row in (
                        df_part.iterrows()
                    ):
                        detail_rows.append(
                            {
                                "date":
                                    date,

                                "mode":
                                    mode,

                                "top_n":
                                    top_n,

                                "change_type":
                                    kind,

                                "machine_no":
                                    int(
                                        row[
                                            "machine_no"
                                        ]
                                    ),

                                "machine_name":
                                    row[
                                        "machine_name"
                                    ],

                                "prediction_rank":
                                    int(
                                        row[
                                            "prediction_rank"
                                        ]
                                    ),

                                "diff":
                                    float(
                                        row[
                                            "diff"
                                        ]
                                    ),
                            }
                        )

    return (
        pd.DataFrame(
            summary_rows
        ),
        pd.DataFrame(
            detail_rows
        ),
    )


def summarize_pick_overlap(
    pick_summary: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        mode,
        top_n,
    ), g in pick_summary.groupby(
        [
            "mode",
            "top_n",
        ]
    ):

        rows.append(
            {
                "mode":
                    mode,

                "top_n":
                    int(
                        top_n
                    ),

                "days":
                    len(g),

                "avg_overlap_count":
                    float(
                        g[
                            "overlap_count"
                        ].mean()
                    ),

                "avg_overlap_rate":
                    float(
                        g[
                            "overlap_rate"
                        ].mean()
                    ),

                "avg_entered_count":
                    float(
                        g[
                            "entered_count"
                        ].mean()
                    ),

                "replacement_positive_days":
                    int(
                        (
                            g[
                                "replacement_delta"
                            ]
                            > 0
                        ).sum()
                    ),

                "replacement_negative_days":
                    int(
                        (
                            g[
                                "replacement_delta"
                            ]
                            < 0
                        ).sum()
                    ),

                "replacement_total_delta":
                    float(
                        g[
                            "replacement_delta"
                        ].sum()
                    ),

                "median_replacement_delta":
                    float(
                        g[
                            "replacement_delta"
                        ].median()
                    ),

                "avg_top10_rank_spearman":
                    float(
                        g[
                            "top10_rank_spearman"
                        ].mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    header(
        "66 - Min-Repo Candidate Robustness Diagnostic"
    )

    daily = read_csv(
        DAILY_CSV
    )

    picks = read_csv(
        TOP10_CSV
    )

    print(
        f"daily rows          : {len(daily)}"
    )

    print(
        f"pick rows           : {len(picks)}"
    )

    print(
        f"dates               : {daily['date'].nunique()}"
    )

    print(
        f"modes               : {daily['mode'].nunique()}"
    )

    pairwise = build_daily_pairwise(
        daily
    )

    robustness = summarize_pairwise(
        pairwise
    )

    (
        pick_comparison,
        pick_detail,
    ) = build_pick_comparison(
        picks
    )

    pick_overall = summarize_pick_overlap(
        pick_comparison
    )

    header(
        "ROBUSTNESS - FOCUS VIEW"
    )

    focus = robustness[
        (
            (
                robustness["mode"]
                == "MINREPO_MACHINE_DIFF3"
            )
            & (
                robustness["top_n"]
                == 3
            )
        )
        | (
            (
                robustness["mode"]
                == "MINREPO_MACHINE_WIN3"
            )
            & (
                robustness["top_n"]
                .isin(
                    [
                        3,
                        5,
                    ]
                )
            )
        )
        | (
            (
                robustness["mode"]
                == "MINREPO_MACHINE_TAIL_DIFF3"
            )
            & (
                robustness["top_n"]
                == 10
            )
        )
    ].copy()

    print(
        focus.to_string(
            index=False
        )
    )

    header(
        "PICK OVERLAP - FOCUS VIEW"
    )

    overlap_focus = pick_overall[
        (
            (
                pick_overall["mode"]
                == "MINREPO_MACHINE_DIFF3"
            )
            & (
                pick_overall["top_n"]
                == 3
            )
        )
        | (
            (
                pick_overall["mode"]
                == "MINREPO_MACHINE_WIN3"
            )
            & (
                pick_overall["top_n"]
                .isin(
                    [
                        3,
                        5,
                    ]
                )
            )
        )
        | (
            (
                pick_overall["mode"]
                == "MINREPO_MACHINE_TAIL_DIFF3"
            )
            & (
                pick_overall["top_n"]
                == 10
            )
        )
    ].copy()

    print(
        overlap_focus.to_string(
            index=False
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = {
        "66_candidate_pairwise_daily.csv":
            pairwise,

        "66_candidate_robustness_summary.csv":
            robustness,

        "66_candidate_pick_overlap_daily.csv":
            pick_comparison,

        "66_candidate_pick_overlap_summary.csv":
            pick_overall,

        "66_candidate_pick_changes_detail.csv":
            pick_detail,
    }

    header(
        "FILES SAVED"
    )

    for filename, frame in (
        files.items()
    ):
        path = (
            OUTPUT_DIR
            / filename
        )

        frame.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            path
        )

    print()
    print(
        "Interpretation:"
    )

    print(
        "- total_change_without_best1 / best2 tests outlier dependence."
    )

    print(
        "- replacement_total_delta isolates the machines that actually changed."
    )

    print(
        "- overlap_rate shows how much each challenger truly changes the pick set."
    )

    print(
        "- This script does not promote any model and does not modify V4.2_C."
    )


if __name__ == "__main__":
    main()
