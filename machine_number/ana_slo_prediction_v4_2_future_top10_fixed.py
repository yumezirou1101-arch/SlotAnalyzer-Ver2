from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util

import numpy as np
import pandas as pd


# ============================================================
# 64 - V4.2_C Future TOP10 Prediction
# ============================================================
#
# Exact-model policy:
# - Data assembly is inherited from 63.
# - Feature construction and scoring are inherited from 56.
# - Champion weights are the exact V42_C_WEIGHTS from 56.
# - Challenger weights are NOT used.
# - Target-day actual results are NOT used.
#
# Future prediction trick:
# Source 56 build_features() requires target-day rows only so that it
# can merge machine_no / machine_name / diff as "actual".
# For a true future date, this script adds synthetic target-day rows
# using the latest known machine_no / machine_name and diff=0.
# These synthetic rows are NOT part of history because source 56 uses
# only df["date"] < target_date for every feature.
# Therefore they do not affect feature values or scores.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
)

SOURCE_56 = (
    PROJECT_ROOT
    / "machine_number"
    / "ana_slo_prediction_v4_2_machine_number_position_ablation_oos.py"
)

SOURCE_63 = (
    PROJECT_ROOT
    / "machine_number"
    / "ana_slo_prediction_v4_2_forward_champion_challenger.py"
)

TARGET_DATE = pd.Timestamp(
    "2026-08-22"
)

EXPECTED_LATEST_DATA_DATE = pd.Timestamp(
    "2026-08-20"
)

EXPECTED_MACHINES = 514

TOP_N = 10
PRIMARY_N = 5

OUTPUT_DIR = (
    DATA_DIR
    / "analysis_31days_deep"
    / "64_Ver4_2_future_top10"
)


# ============================================================
# HELPERS
# ============================================================

def header(
    title: str,
) -> None:

    print()
    print("=" * 112)
    print(title)
    print("=" * 112)


def load_module(
    path: Path,
    module_name: str,
):

    if not path.exists():
        raise FileNotFoundError(
            f"Source script not found: {path}"
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            module_name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not import: {path}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def weight_fingerprint(
    weights: dict[str, float],
) -> str:

    text = "|".join(
        f"{k}:{weights[k]:.15f}"
        for k in sorted(
            weights
        )
    )

    return (
        hashlib.sha256(
            text.encode(
                "utf-8"
            )
        )
        .hexdigest()[:16]
    )


def build_future_panel(
    m56,
    df: pd.DataFrame,
    target_date: pd.Timestamp,
) -> tuple[
    pd.DataFrame,
    pd.Timestamp,
]:

    # Hard leakage guard.
    hist = df[
        df["date"]
        < target_date
    ].copy()

    if hist.empty:
        raise RuntimeError(
            "No historical data before target date."
        )

    latest_date = pd.Timestamp(
        hist["date"].max()
    )

    latest_snapshot = (
        hist[
            hist["date"]
            == latest_date
        ][
            [
                "machine_no",
                "machine_name",
            ]
        ]
        .drop_duplicates(
            subset=[
                "machine_no",
            ],
            keep="last",
        )
        .sort_values(
            "machine_no"
        )
        .reset_index(
            drop=True
        )
    )

    latest_machine_count = int(
        latest_snapshot[
            "machine_no"
        ].nunique()
    )

    if (
        latest_machine_count
        != EXPECTED_MACHINES
    ):
        raise RuntimeError(
            "Latest snapshot machine count mismatch: "
            f"{latest_machine_count} != {EXPECTED_MACHINES}"
        )

    # Synthetic future "actual" rows.
    # diff=0 exists only to satisfy source 56's merge.
    # It never enters history because build_features() uses date < target_date.
    synthetic = latest_snapshot.copy()

    synthetic[
        "date"
    ] = target_date

    synthetic[
        "diff"
    ] = 0.0

    synthetic[
        "win"
    ] = 0

    synthetic[
        "plus1000"
    ] = 0

    synthetic[
        "plus2000"
    ] = 0

    # 63's canonicalized dataset still contains original auxiliary
    # columns such as G数 / BB / RB / probability columns.
    # They are not needed by source 56 build_features(), but hist
    # contains them. Reindex the synthetic future rows to the exact
    # historical schema so pd.concat() is safe without inventing
    # target-day values for those unused columns.
    synthetic = synthetic.reindex(
        columns=hist.columns
    )

    working = pd.concat(
        [
            hist,
            synthetic,
        ],
        ignore_index=True,
    )

    edge_distance_map = (
        m56.build_number_edge_distance(
            hist[
                "machine_no"
            ].tolist()
        )
    )

    panel = m56.build_features(
        working,
        target_date,
        edge_distance_map,
    )

    if panel.empty:
        raise RuntimeError(
            "Future feature panel is empty."
        )

    # The merged synthetic diff is not a prediction feature
    # and must never be interpreted as an expected payout.
    panel = panel.rename(
        columns={
            "diff":
                "synthetic_actual_placeholder",
        }
    )

    return (
        panel,
        latest_date,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    header(
        "64 - V4.2_C Future TOP10 Prediction"
    )

    m56 = load_module(
        SOURCE_56,
        "slotanalyzer_source56",
    )

    m63 = load_module(
        SOURCE_63,
        "slotanalyzer_source63",
    )

    champion_weights = (
        m56.V42_C_WEIGHTS.copy()
    )

    fingerprint = (
        weight_fingerprint(
            champion_weights
        )
    )

    # Reuse exactly the data-discovery path already used by 63.
    df, sources = (
        m63.assemble_dataset()
    )

    # Absolute future guard:
    # even if target-day or later CSVs are added in the future,
    # they are discarded before feature construction.
    df = df[
        df["date"]
        < TARGET_DATE
    ].copy()

    if df.empty:
        raise RuntimeError(
            "No usable data before target date."
        )

    latest_date = pd.Timestamp(
        df["date"].max()
    )

    header(
        "INPUT / SAFETY"
    )

    print(
        f"target_date          : {TARGET_DATE.date()}"
    )

    print(
        f"model                : CHAMPION_V4.2_C"
    )

    print(
        f"weight_sum           : {sum(champion_weights.values()):.12f}"
    )

    print(
        f"weight_fingerprint   : {fingerprint}"
    )

    print(
        f"records              : {len(df):,}"
    )

    print(
        f"days                 : {df['date'].nunique()}"
    )

    print(
        f"latest_data_date     : {latest_date.date()}"
    )

    print(
        f"expected_latest_date : {EXPECTED_LATEST_DATA_DATE.date()}"
    )

    print()
    print(
        "Data sources:"
    )

    for path in sources:
        print(
            f"  {path}"
        )

    if latest_date >= TARGET_DATE:

        raise RuntimeError(
            "LEAKAGE RISK: latest data date is on/after target date."
        )

    if (
        latest_date
        != EXPECTED_LATEST_DATA_DATE
    ):

        print()
        print(
            "[WARNING] Latest data date differs from the expected 2026-08-20."
        )
        print(
            "Review inputs before treating this as the planned 2026-08-22 prediction."
        )

    latest_snapshot = df[
        df["date"]
        == latest_date
    ]

    latest_machine_count = int(
        latest_snapshot[
            "machine_no"
        ].nunique()
    )

    latest_duplicates = int(
        latest_snapshot.duplicated(
            subset=[
                "machine_no",
            ]
        ).sum()
    )

    print(
        f"latest machines      : {latest_machine_count}"
    )

    print(
        f"latest duplicates    : {latest_duplicates}"
    )

    if (
        latest_machine_count
        != EXPECTED_MACHINES
    ):

        raise RuntimeError(
            "Latest snapshot is incomplete."
        )

    if latest_duplicates != 0:

        raise RuntimeError(
            "Latest snapshot has duplicate machine numbers."
        )

    print(
        "future leakage check : OK"
    )

    print(
        "latest snapshot check: OK"
    )

    # --------------------------------------------------------
    # Exact source-56 feature construction
    # --------------------------------------------------------

    panel, panel_latest_date = (
        build_future_panel(
            m56,
            df,
            TARGET_DATE,
        )
    )

    print(
        f"prediction panel     : {len(panel)}"
    )

    if (
        len(panel)
        != EXPECTED_MACHINES
    ):

        raise RuntimeError(
            "Prediction panel machine count mismatch: "
            f"{len(panel)} != {EXPECTED_MACHINES}"
        )

    # Restore the exact column name source-56 rank_score expects
    # only if it were ever a weighted factor. It is not weighted,
    # so the placeholder is simply retained as metadata.
    ranked_input = panel.copy()

    ranked = m56.rank_score(
        ranked_input,
        champion_weights,
    ).reset_index(
        drop=True
    )

    ranked[
        "prediction_rank"
    ] = np.arange(
        1,
        len(ranked) + 1,
    )

    ranked[
        "tier"
    ] = "OUTSIDE_TOP10"

    ranked.loc[
        ranked[
            "prediction_rank"
        ]
        <= TOP_N,
        "tier",
    ] = "NEXT"

    ranked.loc[
        ranked[
            "prediction_rank"
        ]
        <= PRIMARY_N,
        "tier",
    ] = "PRIMARY"

    ranked[
        "target_date"
    ] = TARGET_DATE

    ranked[
        "latest_data_date"
    ] = panel_latest_date

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    top10 = ranked.head(
        TOP_N
    ).copy()

    header(
        f"{TARGET_DATE.date()} PREDICTION TOP10"
    )

    display_columns = [
        "prediction_rank",
        "tier",
        "machine_no",
        "machine_name",
        "score",
        "avg31",
        "recent7_avg",
        "last_diff",
        "prev_change",
        "weekday_avg",
        "type_avg",
        "plus1000_rate",
        "plus2000_rate",
        "neighbor_avg",
    ]

    print(
        top10[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "PRIMARY = ranks 1-5"
    )

    print(
        "NEXT    = ranks 6-10"
    )

    print()
    print(
        "Important assumptions:"
    )

    print(
        f"- Machine names / placement are assumed unchanged from {latest_date.date()}."
    )

    print(
        "- Target-day actual diff is not used."
    )

    print(
        "- Challenger models are not used."
    )

    print(
        "- This is a ranking prediction, not a guarantee of a positive result."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ymd = TARGET_DATE.strftime(
        "%Y%m%d"
    )

    all_path = (
        OUTPUT_DIR
        / f"64_prediction_{ymd}_all514.csv"
    )

    top10_path = (
        OUTPUT_DIR
        / f"64_prediction_{ymd}_top10.csv"
    )

    metadata_path = (
        OUTPUT_DIR
        / f"64_prediction_{ymd}_metadata.csv"
    )

    ranked.to_csv(
        all_path,
        index=False,
        encoding="utf-8-sig",
    )

    top10.to_csv(
        top10_path,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = pd.DataFrame(
        [
            {
                "target_date":
                    TARGET_DATE.date(),

                "latest_data_date":
                    latest_date.date(),

                "model":
                    "CHAMPION_V4.2_C",

                "weight_fingerprint":
                    fingerprint,

                "weight_sum":
                    float(
                        sum(
                            champion_weights.values()
                        )
                    ),

                "machines_ranked":
                    int(
                        len(ranked)
                    ),

                "top_n":
                    TOP_N,

                "primary_n":
                    PRIMARY_N,

                "target_actual_used":
                    False,

                "challenger_used":
                    False,

                "machine_name_assumption":
                    (
                        "Latest snapshot retained from "
                        f"{latest_date.date()}"
                    ),
            }
        ]
    )

    metadata.to_csv(
        metadata_path,
        index=False,
        encoding="utf-8-sig",
    )

    header(
        "FILES SAVED"
    )

    for path in (
        all_path,
        top10_path,
        metadata_path,
    ):

        print(path)

    print()
    print(
        "64 future prediction complete."
    )

    print(
        f"Keep this prediction unchanged until the {TARGET_DATE.date()} actual data is collected."
    )


if __name__ == "__main__":
    main()

