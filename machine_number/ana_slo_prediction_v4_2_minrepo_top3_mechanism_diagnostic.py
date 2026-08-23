from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SlotAnalyzer
# 67 - Min-Repo TOP3 Correction Mechanism Diagnostic
# ============================================================
#
# Purpose:
#   Diagnose WHY MINREPO_MACHINE_DIFF3 / MINREPO_MACHINE_WIN3
#   changed V4.2_C TOP3 selections and when those changes helped.
#
# Inputs:
#   65_minrepo_external_top10_picks.csv
#   66_candidate_pick_changes_detail.csv
#
# Existing V4.2_C is NOT modified.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

SOURCE65_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
    / "65_Ver4_2_minrepo_external_feature_oos"
)

SOURCE66_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
    / "66_Ver4_2_minrepo_candidate_robustness"
)

TOP10_CSV = (
    SOURCE65_DIR
    / "65_minrepo_external_top10_picks.csv"
)

CHANGES_CSV = (
    SOURCE66_DIR
    / "66_candidate_pick_changes_detail.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
    / "67_Ver4_2_minrepo_top3_mechanism"
)

BASE_MODE = "BASE_V4.2_C"

CANDIDATES = [
    "MINREPO_MACHINE_DIFF3",
    "MINREPO_MACHINE_WIN3",
]

TOP_N = 3


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


def safe_mean(series: pd.Series) -> float:
    x = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    return float(x.mean()) if len(x) else np.nan


def safe_median(series: pd.Series) -> float:
    x = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    return float(x.median()) if len(x) else np.nan


# ============================================================
# BUILD TOP3 CHANGE DETAIL WITH SOURCE65 FEATURES
# ============================================================

def enrich_changes(
    picks: pd.DataFrame,
    changes: pd.DataFrame,
) -> pd.DataFrame:

    x = changes[
        (
            changes["top_n"] == TOP_N
        )
        & (
            changes["mode"].isin(
                CANDIDATES
            )
        )
    ].copy()

    pick_cols = [
        "date",
        "mode",
        "machine_no",
        "prediction_rank",
        "score",
        "ext_machine_diff3",
        "ext_machine_win3",
        "ext_tail_diff3",
        "ext_tail_win3",
    ]

    candidate_pick_data = picks[
        picks["mode"].isin(
            CANDIDATES
        )
    ][
        pick_cols
    ].copy()

    candidate_pick_data = (
        candidate_pick_data
        .drop_duplicates(
            subset=[
                "date",
                "mode",
                "machine_no",
            ],
            keep="first",
        )
    )

    x = x.merge(
        candidate_pick_data,
        on=[
            "date",
            "mode",
            "machine_no",
        ],
        how="left",
        suffixes=(
            "",
            "_source65",
        ),
        validate="many_to_one",
    )

    # Base pick info for same machine/date.
    base = picks[
        picks["mode"] == BASE_MODE
    ][
        [
            "date",
            "machine_no",
            "prediction_rank",
            "score",
        ]
    ].copy()

    base = base.rename(
        columns={
            "prediction_rank":
                "base_prediction_rank",

            "score":
                "base_score",
        }
    )

    base = base.drop_duplicates(
        subset=[
            "date",
            "machine_no",
        ],
        keep="first",
    )

    x = x.merge(
        base,
        on=[
            "date",
            "machine_no",
        ],
        how="left",
        validate="many_to_one",
    )

    # Candidate ranking rank from Source65.
    x["candidate_prediction_rank"] = (
        pd.to_numeric(
            x["prediction_rank_source65"],
            errors="coerce",
        )
    )

    x["candidate_score"] = (
        pd.to_numeric(
            x["score"],
            errors="coerce",
        )
    )

    x["base_score"] = pd.to_numeric(
        x["base_score"],
        errors="coerce",
    )

    x["score_delta_vs_base"] = (
        x["candidate_score"]
        - x["base_score"]
    )

    return x


# ============================================================
# DAILY ENTERED vs REMOVED PAIRS
# ============================================================

def build_daily_replacement_pairs(
    enriched: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        date,
        mode,
    ), g in enriched.groupby(
        [
            "date",
            "mode",
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

        entered_total = float(
            pd.to_numeric(
                entered["diff"],
                errors="coerce",
            ).sum()
        )

        removed_total = float(
            pd.to_numeric(
                removed["diff"],
                errors="coerce",
            ).sum()
        )

        replacement_delta = (
            entered_total
            - removed_total
        )

        rows.append(
            {
                "date":
                    date,

                "mode":
                    mode,

                "entered_count":
                    len(
                        entered
                    ),

                "removed_count":
                    len(
                        removed
                    ),

                "entered_total_diff":
                    entered_total,

                "removed_total_diff":
                    removed_total,

                "replacement_delta":
                    replacement_delta,

                "success":
                    replacement_delta > 0,

                "entered_machine_diff3_mean":
                    safe_mean(
                        entered[
                            "ext_machine_diff3"
                        ]
                    ),

                "removed_machine_diff3_mean":
                    safe_mean(
                        removed[
                            "ext_machine_diff3"
                        ]
                    ),

                "machine_diff3_gap":
                    (
                        safe_mean(
                            entered[
                                "ext_machine_diff3"
                            ]
                        )
                        - safe_mean(
                            removed[
                                "ext_machine_diff3"
                            ]
                        )
                    ),

                "entered_machine_win3_mean":
                    safe_mean(
                        entered[
                            "ext_machine_win3"
                        ]
                    ),

                "removed_machine_win3_mean":
                    safe_mean(
                        removed[
                            "ext_machine_win3"
                        ]
                    ),

                "machine_win3_gap":
                    (
                        safe_mean(
                            entered[
                                "ext_machine_win3"
                            ]
                        )
                        - safe_mean(
                            removed[
                                "ext_machine_win3"
                            ]
                        )
                    ),

                "entered_candidate_score_mean":
                    safe_mean(
                        entered[
                            "candidate_score"
                        ]
                    ),

                "removed_base_score_mean":
                    safe_mean(
                        removed[
                            "base_score"
                        ]
                    ),

                "score_gap":
                    (
                        safe_mean(
                            entered[
                                "candidate_score"
                            ]
                        )
                        - safe_mean(
                            removed[
                                "base_score"
                            ]
                        )
                    ),

                "entered_base_rank_mean":
                    safe_mean(
                        entered[
                            "base_prediction_rank"
                        ]
                    ),

                "removed_base_rank_mean":
                    safe_mean(
                        removed[
                            "base_prediction_rank"
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# DIFF3 vs WIN3 AGREEMENT
# ============================================================

def build_candidate_agreement(
    picks: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    dates = sorted(
        picks["date"]
        .dropna()
        .unique()
        .tolist()
    )

    for date in dates:
        daily_sets = {}

        for mode in CANDIDATES:
            top3 = picks[
                (
                    picks["date"] == date
                )
                & (
                    picks["mode"] == mode
                )
                & (
                    picks["prediction_rank"]
                    <= TOP_N
                )
            ].copy()

            daily_sets[
                mode
            ] = set(
                top3[
                    "machine_no"
                ].astype(int)
            )

        a = daily_sets[
            CANDIDATES[0]
        ]

        b = daily_sets[
            CANDIDATES[1]
        ]

        overlap = a & b
        only_a = a - b
        only_b = b - a

        rows.append(
            {
                "date":
                    date,

                "diff3_win3_same_top3":
                    a == b,

                "overlap_count":
                    len(
                        overlap
                    ),

                "diff3_only_count":
                    len(
                        only_a
                    ),

                "win3_only_count":
                    len(
                        only_b
                    ),

                "diff3_only_machines":
                    ",".join(
                        map(
                            str,
                            sorted(
                                only_a
                            ),
                        )
                    ),

                "win3_only_machines":
                    ",".join(
                        map(
                            str,
                            sorted(
                                only_b
                            ),
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MACHINE DEPENDENCE
# ============================================================

def build_machine_dependence(
    enriched: pd.DataFrame,
) -> pd.DataFrame:

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

            total_diff=(
                "diff",
                "sum",
            ),

            avg_diff=(
                "diff",
                "mean",
            ),

            positive_count=(
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
        .sort_values(
            [
                "mode",
                "entered_count",
                "total_diff",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
    )

    return result


# ============================================================
# SUCCESS vs FAILURE PROFILE
# ============================================================

def build_success_failure_profile(
    daily_pairs: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for mode in CANDIDATES:
        g = daily_pairs[
            daily_pairs["mode"]
            == mode
        ].copy()

        for label, mask in (
            (
                "SUCCESS",
                g["replacement_delta"] > 0,
            ),
            (
                "FAILURE",
                g["replacement_delta"] < 0,
            ),
            (
                "SAME",
                g["replacement_delta"] == 0,
            ),
        ):
            x = g[
                mask
            ].copy()

            if x.empty:
                continue

            rows.append(
                {
                    "mode":
                        mode,

                    "result_type":
                        label,

                    "days":
                        len(
                            x
                        ),

                    "avg_replacement_delta":
                        safe_mean(
                            x[
                                "replacement_delta"
                            ]
                        ),

                    "median_replacement_delta":
                        safe_median(
                            x[
                                "replacement_delta"
                            ]
                        ),

                    "avg_machine_diff3_gap":
                        safe_mean(
                            x[
                                "machine_diff3_gap"
                            ]
                        ),

                    "avg_machine_win3_gap":
                        safe_mean(
                            x[
                                "machine_win3_gap"
                            ]
                        ),

                    "avg_score_gap":
                        safe_mean(
                            x[
                                "score_gap"
                            ]
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
        "67 - Min-Repo TOP3 Correction Mechanism Diagnostic"
    )

    picks = read_csv(
        TOP10_CSV
    )

    changes = read_csv(
        CHANGES_CSV
    )

    print(
        f"source65 pick rows     : {len(picks)}"
    )

    print(
        f"source66 change rows   : {len(changes)}"
    )

    enriched = enrich_changes(
        picks,
        changes,
    )

    daily_pairs = (
        build_daily_replacement_pairs(
            enriched
        )
    )

    agreement = (
        build_candidate_agreement(
            picks
        )
    )

    machine_dependence = (
        build_machine_dependence(
            enriched
        )
    )

    profile = (
        build_success_failure_profile(
            daily_pairs
        )
    )

    header(
        "DAILY TOP3 REPLACEMENT"
    )

    display_cols = [
        "date",
        "mode",
        "entered_count",
        "replacement_delta",
        "machine_diff3_gap",
        "machine_win3_gap",
        "score_gap",
    ]

    print(
        daily_pairs[
            display_cols
        ].to_string(
            index=False
        )
    )

    header(
        "DIFF3 vs WIN3 AGREEMENT"
    )

    print(
        agreement.to_string(
            index=False
        )
    )

    header(
        "SUCCESS / FAILURE PROFILE"
    )

    print(
        profile.to_string(
            index=False
        )
    )

    header(
        "TOP ENTERED MACHINE DEPENDENCE"
    )

    if machine_dependence.empty:
        print(
            "No entered-machine data."
        )
    else:
        print(
            machine_dependence.head(
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
        "67_top3_change_enriched.csv":
            enriched,

        "67_top3_daily_replacement.csv":
            daily_pairs,

        "67_diff3_vs_win3_agreement.csv":
            agreement,

        "67_success_failure_profile.csv":
            profile,

        "67_machine_dependence.csv":
            machine_dependence,
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
        "- Compare SUCCESS vs FAILURE gaps to see whether a future gate is plausible."
    )

    print(
        "- DIFF3 vs WIN3 agreement shows whether they are effectively the same challenger."
    )

    print(
        "- Machine dependence reveals whether gains are concentrated in a small set of models."
    )

    print(
        "- This script does not modify or promote V4.2_C."
    )


if __name__ == "__main__":
    main()
