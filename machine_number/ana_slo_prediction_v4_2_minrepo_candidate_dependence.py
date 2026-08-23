from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SlotAnalyzer
# 68 - Min-Repo Candidate Dependence / Generalization Diagnostic
# ============================================================
#
# Purpose:
#   Test whether the apparent TOP3 improvement of
#   MINREPO_MACHINE_DIFF3 / MINREPO_MACHINE_WIN3
#   is broadly reproducible or concentrated in:
#     - a small number of dates
#     - a small number of machine models
#     - especially ヤバチバ
#
# Inputs:
#   67_top3_change_enriched.csv
#   67_top3_daily_replacement.csv
#
# Existing V4.2_C is NOT modified.
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
    / "67_Ver4_2_minrepo_top3_mechanism"
)

ENRICHED_CSV = (
    SOURCE_DIR
    / "67_top3_change_enriched.csv"
)

DAILY_CSV = (
    SOURCE_DIR
    / "67_top3_daily_replacement.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
    / "68_Ver4_2_minrepo_candidate_dependence"
)

CANDIDATES = [
    "MINREPO_MACHINE_DIFF3",
    "MINREPO_MACHINE_WIN3",
]

FOCUS_MACHINE = "ヤバチバ"


# ============================================================
# HELPERS
# ============================================================

def header(title: str) -> None:
    print()
    print("=" * 118)
    print(title)
    print("=" * 118)


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


def safe_sum(series: pd.Series) -> float:
    return float(
        pd.to_numeric(
            series,
            errors="coerce",
        ).fillna(0.0).sum()
    )


# ============================================================
# EVENT-LEVEL REPLACEMENT DATA
# ============================================================

def build_replacement_events(
    enriched: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per candidate/date.
    Keeps entered/removed machine lists so we can identify
    machine-model dependence.
    """

    rows = []

    for (
        mode,
        date,
    ), g in enriched.groupby(
        [
            "mode",
            "date",
        ]
    ):

        entered = g[
            g["change_type"]
            == "ENTERED"
        ].copy()

        removed = g[
            g["change_type"]
            == "REMOVED"
        ].copy()

        entered_total = safe_sum(
            entered["diff"]
        )

        removed_total = safe_sum(
            removed["diff"]
        )

        entered_names = (
            entered[
                "machine_name"
            ]
            .astype(str)
            .tolist()
        )

        removed_names = (
            removed[
                "machine_name"
            ]
            .astype(str)
            .tolist()
        )

        rows.append(
            {
                "mode":
                    mode,

                "date":
                    date,

                "entered_count":
                    len(entered),

                "removed_count":
                    len(removed),

                "entered_total_diff":
                    entered_total,

                "removed_total_diff":
                    removed_total,

                "replacement_delta":
                    entered_total
                    - removed_total,

                "entered_machine_names":
                    "|".join(
                        entered_names
                    ),

                "removed_machine_names":
                    "|".join(
                        removed_names
                    ),

                "focus_machine_entered":
                    FOCUS_MACHINE
                    in entered_names,

                "focus_machine_entered_count":
                    entered_names.count(
                        FOCUS_MACHINE
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# FOCUS MACHINE DEPENDENCE
# ============================================================

def summarize_focus_dependence(
    events: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for mode in CANDIDATES:
        g = events[
            events["mode"]
            == mode
        ].copy()

        with_focus = g[
            g[
                "focus_machine_entered"
            ]
        ].copy()

        without_focus = g[
            ~g[
                "focus_machine_entered"
            ]
        ].copy()

        total = safe_sum(
            g[
                "replacement_delta"
            ]
        )

        focus_total = safe_sum(
            with_focus[
                "replacement_delta"
            ]
        )

        nonfocus_total = safe_sum(
            without_focus[
                "replacement_delta"
            ]
        )

        rows.append(
            {
                "mode":
                    mode,

                "all_change_days":
                    len(g),

                "all_total_delta":
                    total,

                "focus_days":
                    len(
                        with_focus
                    ),

                "focus_total_delta":
                    focus_total,

                "focus_share_of_total_percent":
                    (
                        focus_total
                        / total
                        * 100.0
                        if total != 0
                        else np.nan
                    ),

                "nonfocus_days":
                    len(
                        without_focus
                    ),

                "nonfocus_total_delta":
                    nonfocus_total,

                "nonfocus_mean_delta":
                    (
                        float(
                            without_focus[
                                "replacement_delta"
                            ].mean()
                        )
                        if len(
                            without_focus
                        )
                        else np.nan
                    ),

                "nonfocus_median_delta":
                    (
                        float(
                            without_focus[
                                "replacement_delta"
                            ].median()
                        )
                        if len(
                            without_focus
                        )
                        else np.nan
                    ),

                "nonfocus_positive_days":
                    int(
                        (
                            without_focus[
                                "replacement_delta"
                            ]
                            > 0
                        ).sum()
                    ),

                "nonfocus_negative_days":
                    int(
                        (
                            without_focus[
                                "replacement_delta"
                            ]
                            < 0
                        ).sum()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MACHINE-MODEL CONTRIBUTION
# ============================================================

def machine_model_contribution(
    enriched: pd.DataFrame,
) -> pd.DataFrame:
    """
    Contribution of entered machines only.
    This is not identical to replacement_delta because removed
    machines also matter, but it identifies concentration.
    """

    entered = enriched[
        enriched["change_type"]
        == "ENTERED"
    ].copy()

    if entered.empty:
        return pd.DataFrame()

    result = (
        entered.groupby(
            [
                "mode",
                "machine_name",
            ],
            as_index=False,
        )
        .agg(
            entered_count=(
                "machine_no",
                "size",
            ),

            entered_days=(
                "date",
                "nunique",
            ),

            entered_total_diff=(
                "diff",
                "sum",
            ),

            entered_avg_diff=(
                "diff",
                "mean",
            ),

            positive_entries=(
                "diff",
                lambda s:
                    int(
                        (
                            pd.to_numeric(
                                s,
                                errors="coerce",
                            )
                            > 0
                        ).sum()
                    ),
            ),
        )
    )

    result[
        "abs_entered_total_diff"
    ] = result[
        "entered_total_diff"
    ].abs()

    result = result.sort_values(
        [
            "mode",
            "abs_entered_total_diff",
        ],
        ascending=[
            True,
            False,
        ],
    )

    return result


# ============================================================
# LEAVE-ONE-DAY-OUT
# ============================================================

def leave_one_day_out(
    events: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for mode in CANDIDATES:
        g = events[
            events["mode"]
            == mode
        ].copy()

        total = safe_sum(
            g[
                "replacement_delta"
            ]
        )

        for _, row in (
            g.iterrows()
        ):
            rows.append(
                {
                    "mode":
                        mode,

                    "excluded_date":
                        row[
                            "date"
                        ],

                    "excluded_day_delta":
                        float(
                            row[
                                "replacement_delta"
                            ]
                        ),

                    "remaining_total_delta":
                        total
                        - float(
                            row[
                                "replacement_delta"
                            ]
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def summarize_lodo(
    lodo: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for mode in CANDIDATES:
        g = lodo[
            lodo["mode"]
            == mode
        ].copy()

        rows.append(
            {
                "mode":
                    mode,

                "lodo_runs":
                    len(g),

                "min_remaining_total_delta":
                    float(
                        g[
                            "remaining_total_delta"
                        ].min()
                    ),

                "median_remaining_total_delta":
                    float(
                        g[
                            "remaining_total_delta"
                        ].median()
                    ),

                "max_remaining_total_delta":
                    float(
                        g[
                            "remaining_total_delta"
                        ].max()
                    ),

                "positive_lodo_runs":
                    int(
                        (
                            g[
                                "remaining_total_delta"
                            ]
                            > 0
                        ).sum()
                    ),

                "nonpositive_lodo_runs":
                    int(
                        (
                            g[
                                "remaining_total_delta"
                            ]
                            <= 0
                        ).sum()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# COMMON vs DIVERGENT CANDIDATE DAYS
# ============================================================

def compare_candidates(
    events: pd.DataFrame,
) -> pd.DataFrame:

    a = events[
        events["mode"]
        == CANDIDATES[0]
    ][
        [
            "date",
            "replacement_delta",
            "entered_machine_names",
        ]
    ].rename(
        columns={
            "replacement_delta":
                "diff3_delta",

            "entered_machine_names":
                "diff3_entered",
        }
    )

    b = events[
        events["mode"]
        == CANDIDATES[1]
    ][
        [
            "date",
            "replacement_delta",
            "entered_machine_names",
        ]
    ].rename(
        columns={
            "replacement_delta":
                "win3_delta",

            "entered_machine_names":
                "win3_entered",
        }
    )

    merged = a.merge(
        b,
        on="date",
        how="outer",
    )

    merged[
        "same_entered_models"
    ] = (
        merged[
            "diff3_entered"
        ]
        == merged[
            "win3_entered"
        ]
    )

    merged[
        "delta_difference"
    ] = (
        merged[
            "diff3_delta"
        ]
        - merged[
            "win3_delta"
        ]
    )

    return merged.sort_values(
        "date"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    header(
        "68 - Min-Repo Candidate Dependence / Generalization Diagnostic"
    )

    enriched = read_csv(
        ENRICHED_CSV
    )

    daily = read_csv(
        DAILY_CSV
    )

    print(
        f"enriched rows       : {len(enriched)}"
    )

    print(
        f"daily rows          : {len(daily)}"
    )

    events = build_replacement_events(
        enriched
    )

    focus = summarize_focus_dependence(
        events
    )

    model_contrib = machine_model_contribution(
        enriched
    )

    lodo = leave_one_day_out(
        events
    )

    lodo_summary = summarize_lodo(
        lodo
    )

    candidate_compare = compare_candidates(
        events
    )

    header(
        f"FOCUS MACHINE DEPENDENCE: {FOCUS_MACHINE}"
    )

    print(
        focus.to_string(
            index=False
        )
    )

    header(
        "LEAVE-ONE-DAY-OUT ROBUSTNESS"
    )

    print(
        lodo_summary.to_string(
            index=False
        )
    )

    header(
        "DIFF3 vs WIN3 DAILY DIVERGENCE"
    )

    print(
        candidate_compare.to_string(
            index=False
        )
    )

    header(
        "ENTERED MACHINE CONTRIBUTION"
    )

    print(
        model_contrib.head(
            30
        ).to_string(
            index=False
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = {
        "68_replacement_events.csv":
            events,

        "68_focus_machine_dependence.csv":
            focus,

        "68_machine_model_contribution.csv":
            model_contrib,

        "68_leave_one_day_out.csv":
            lodo,

        "68_leave_one_day_out_summary.csv":
            lodo_summary,

        "68_diff3_vs_win3_daily_divergence.csv":
            candidate_compare,
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
        "- If nonfocus_total_delta collapses, the challenger is machine-specific rather than general."
    )

    print(
        "- If every leave-one-day-out run remains positive, the result is not dependent on one single date."
    )

    print(
        "- MACHINE_DIFF3 and MACHINE_WIN3 should not be treated as independent evidence if their picks are nearly identical."
    )

    print(
        "- This script does not modify or promote V4.2_C."
    )


if __name__ == "__main__":
    main()
