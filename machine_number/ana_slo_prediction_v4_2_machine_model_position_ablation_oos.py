from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

SOURCE_56 = (
    PROJECT_ROOT
    / "machine_number"
    / "ana_slo_prediction_v4_2_machine_number_position_ablation_oos.py"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
    / "58_Ver4_2_machine_model_position_ablation_oos"
)

EXPECTED_BASE_TOTAL_DIFF = 130400.0

# Conservative fixed test weight.
# We are testing whether each feature adds signal, not optimizing weights yet.
ADDED_FEATURE_WEIGHT = 0.10

TOP_N = 10


# ============================================================
# HELPERS
# ============================================================

def header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def load_source_56():
    if not SOURCE_56.exists():
        raise FileNotFoundError(
            f"56 source script not found: {SOURCE_56}"
        )

    spec = importlib.util.spec_from_file_location(
        "slotanalyzer_56",
        SOURCE_56,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not import 56 source."
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)

    std = float(s.std(ddof=0))

    if std == 0 or np.isnan(std):
        return pd.Series(
            0.0,
            index=s.index,
        )

    return (
        s - s.mean()
    ) / std


def add_weight(
    base_weights: dict[str, float],
    feature: str | None,
) -> dict[str, float]:
    if feature is None:
        return base_weights.copy()

    scale = 1.0 - ADDED_FEATURE_WEIGHT

    weights = {
        k: v * scale
        for k, v in base_weights.items()
    }

    weights[feature] = ADDED_FEATURE_WEIGHT

    total = sum(weights.values())

    if not np.isclose(
        total,
        1.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"Weight sum error: {total}"
        )

    return weights


def build_model_position_features(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Derive only same-day structural features from machine_no and machine_name.

    No target-day outcome is used.

    For each machine_name:
      model_count       : number of installed machines
      model_pos_pct     : normalized position from low to high machine_no (0..1)
      model_edge_score  : high near either end of the model's installed range
      model_center_score: high near the middle
      model_end_flag    : 1 for the lowest/highest numbered unit in the model
    """
    x = panel.copy()

    if x.empty:
        return x

    x["machine_no"] = pd.to_numeric(
        x["machine_no"],
        errors="coerce",
    )

    x["machine_name"] = (
        x["machine_name"]
        .astype(str)
        .str.strip()
    )

    grp = x.groupby(
        "machine_name",
        dropna=False,
    )

    x["model_count"] = (
        grp["machine_no"]
        .transform("count")
        .astype(float)
    )

    x["model_min_no"] = (
        grp["machine_no"]
        .transform("min")
        .astype(float)
    )

    x["model_max_no"] = (
        grp["machine_no"]
        .transform("max")
        .astype(float)
    )

    span = (
        x["model_max_no"]
        - x["model_min_no"]
    )

    x["model_pos_pct"] = np.where(
        span > 0,
        (
            x["machine_no"]
            - x["model_min_no"]
        ) / span,
        0.5,
    )

    x["model_edge_score"] = (
        np.abs(
            x["model_pos_pct"] - 0.5
        ) * 2.0
    )

    x["model_center_score"] = (
        1.0
        - x["model_edge_score"]
    )

    x["model_end_flag"] = (
        (
            x["machine_no"]
            == x["model_min_no"]
        )
        |
        (
            x["machine_no"]
            == x["model_max_no"]
        )
    ).astype(float)

    # A singleton has no meaningful "end" or relative position.
    singleton = (
        x["model_count"] <= 1
    )

    x.loc[
        singleton,
        "model_pos_pct",
    ] = 0.5

    x.loc[
        singleton,
        "model_edge_score",
    ] = 0.0

    x.loc[
        singleton,
        "model_center_score",
    ] = 0.0

    x.loc[
        singleton,
        "model_end_flag",
    ] = 0.0

    return x


def evaluate_day(
    panel: pd.DataFrame,
    weights: dict[str, float],
) -> dict | None:
    if panel.empty:
        return None

    x = panel.copy()

    score = pd.Series(
        0.0,
        index=x.index,
    )

    for factor, weight in weights.items():
        if factor not in x.columns:
            raise RuntimeError(
                f"Feature missing from panel: {factor}"
            )

        z = zscore(
            x[factor]
        )

        component = (
            50.0
            + z * 12.5
        ).clip(
            0,
            100,
        )

        score += (
            component
            * weight
        )

    x["score"] = score

    ranked = x.sort_values(
        "score",
        ascending=False,
    )

    selected = ranked.head(
        TOP_N
    )

    if selected.empty:
        return None

    diffs = pd.to_numeric(
        selected["diff"],
        errors="coerce",
    ).dropna()

    if diffs.empty:
        return None

    return {
        "avg_diff":
            float(diffs.mean()),

        "median_diff":
            float(diffs.median()),

        "win_rate":
            float((diffs > 0).mean() * 100.0),

        "plus1000_rate":
            float((diffs >= 1000).mean() * 100.0),

        "plus2000_rate":
            float((diffs >= 2000).mean() * 100.0),

        "positive":
            int(diffs.sum() > 0),

        "total_diff":
            float(diffs.sum()),

        "machines":
            int(len(panel)),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    header(
        "58 - V4.2_C TOP10 Machine-Model Position Feature Ablation OOS"
    )

    m56 = load_source_56()

    df = m56.load_data()

    base_weights = (
        m56.V42_C_WEIGHTS.copy()
    )

    rolling_splits = (
        m56.ROLLING_SPLITS
    )

    print(
        f"records              : {len(df):,}"
    )
    print(
        f"days                 : {df['date'].nunique()}"
    )
    print(
        f"machines             : {df['machine_no'].nunique()}"
    )
    print(
        f"added feature weight : {ADDED_FEATURE_WEIGHT:.3f}"
    )
    print()
    print(
        "Important:"
    )
    print(
        "- Baseline is the unchanged V4.2_C TOP10 model."
    )
    print(
        "- Historical features come directly from the validated 56 logic."
    )
    print(
        "- Position features use only machine_no / machine_name structure."
    )
    print(
        "- No target-day diff is used to construct position features."
    )
    print(
        "- Each candidate is added separately at a fixed 10% weight."
    )

    edge_distance_map = (
        m56.build_number_edge_distance(
            df["machine_no"].tolist()
        )
    )

    all_test_dates = sorted(
        {
            d
            for (
                _,
                _,
                _,
                test_start,
                test_end,
            ) in rolling_splits
            for d in pd.date_range(
                test_start,
                test_end,
            )
        }
    )

    panels = {}

    header(
        "BUILDING LEAKAGE-SAFE PANELS"
    )

    for target_date in all_test_dates:
        panel = m56.build_features(
            df,
            target_date,
            edge_distance_map,
        )

        panel = (
            build_model_position_features(
                panel
            )
        )

        panels[target_date] = panel

        print(
            f"{target_date.date()} "
            f"machines={len(panel)}"
        )

    modes = {
        "BASE":
            None,

        "ADD_MODEL_COUNT":
            "model_count",

        "ADD_MODEL_POSITION":
            "model_pos_pct",

        "ADD_MODEL_EDGE":
            "model_edge_score",

        "ADD_MODEL_CENTER":
            "model_center_score",

        "ADD_MODEL_END_FLAG":
            "model_end_flag",
    }

    daily_rows = []

    header(
        "ROLLING OOS"
    )

    for (
        split_name,
        train_start,
        train_end,
        test_start,
        test_end,
    ) in rolling_splits:

        print(
            f"{split_name}: "
            f"{test_start.date()} "
            f"to {test_end.date()}"
        )

        for mode, feature in modes.items():
            weights = add_weight(
                base_weights,
                feature,
            )

            for target_date in pd.date_range(
                test_start,
                test_end,
            ):
                panel = panels.get(
                    target_date
                )

                if (
                    panel is None
                    or panel.empty
                ):
                    continue

                result = evaluate_day(
                    panel,
                    weights,
                )

                if result is None:
                    continue

                result.update(
                    {
                        "mode":
                            mode,

                        "feature":
                            (
                                "NONE"
                                if feature is None
                                else feature
                            ),

                        "split":
                            split_name,

                        "model":
                            "V4.2_C",

                        "top_n":
                            TOP_N,

                        "date":
                            target_date,

                        "train_start":
                            train_start,

                        "train_end":
                            train_end,

                        "test_start":
                            test_start,

                        "test_end":
                            test_end,
                    }
                )

                daily_rows.append(
                    result
                )

    daily_df = pd.DataFrame(
        daily_rows
    )

    if daily_df.empty:
        raise RuntimeError(
            "No OOS results."
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    split_df = (
        daily_df.groupby(
            [
                "mode",
                "feature",
                "split",
            ],
            as_index=False,
        )
        .agg(
            days=("date", "nunique"),
            avg_diff=("avg_diff", "mean"),
            win_rate=("win_rate", "mean"),
            positive_days=("positive", "mean"),
            total_diff=("total_diff", "sum"),
        )
    )

    split_df["positive_days"] *= 100.0

    overall_df = (
        daily_df.groupby(
            [
                "mode",
                "feature",
            ],
            as_index=False,
        )
        .agg(
            days=("date", "nunique"),
            avg_diff=("avg_diff", "mean"),
            win_rate=("win_rate", "mean"),
            positive_days=("positive", "mean"),
            total_diff=("total_diff", "sum"),
        )
    )

    overall_df["positive_days"] *= 100.0

    stability = (
        split_df.groupby(
            [
                "mode",
                "feature",
            ],
            as_index=False,
        )
        .agg(
            min_split_avg=(
                "avg_diff",
                "min",
            ),

            max_split_avg=(
                "avg_diff",
                "max",
            ),

            positive_split_rate=(
                "total_diff",
                lambda s:
                    float(
                        (s > 0).mean()
                        * 100.0
                    ),
            ),
        )
    )

    overall_df = overall_df.merge(
        stability,
        on=[
            "mode",
            "feature",
        ],
        how="left",
    )

    base = overall_df[
        overall_df["mode"]
        == "BASE"
    ]

    if base.empty:
        raise RuntimeError(
            "BASE result missing."
        )

    base_total = float(
        base.iloc[0]["total_diff"]
    )

    base_avg = float(
        base.iloc[0]["avg_diff"]
    )

    base_ok = bool(
        np.isclose(
            base_total,
            EXPECTED_BASE_TOTAL_DIFF,
            atol=0.01,
        )
    )

    header(
        "BASELINE SAFETY CHECK"
    )

    print(
        f"actual   : {base_total:+.1f}"
    )
    print(
        f"expected : {EXPECTED_BASE_TOTAL_DIFF:+.1f}"
    )
    print(
        f"match    : {base_ok}"
    )

    if not base_ok:
        raise RuntimeError(
            "BASELINE SAFETY CHECK FAILED. "
            "Do not interpret results."
        )

    overall_df[
        "total_change_vs_base"
    ] = (
        overall_df["total_diff"]
        - base_total
    )

    overall_df[
        "avg_change_vs_base"
    ] = (
        overall_df["avg_diff"]
        - base_avg
    )

    # --------------------------------------------------------
    # Paired daily comparisons
    # --------------------------------------------------------

    base_daily = daily_df[
        daily_df["mode"] == "BASE"
    ][
        [
            "date",
            "split",
            "total_diff",
        ]
    ].rename(
        columns={
            "total_diff":
                "base_total_diff",
        }
    )

    paired_frames = []

    for mode in modes:
        if mode == "BASE":
            continue

        candidate = daily_df[
            daily_df["mode"] == mode
        ][
            [
                "date",
                "split",
                "feature",
                "total_diff",
            ]
        ].rename(
            columns={
                "total_diff":
                    "candidate_total_diff",
            }
        )

        pair = base_daily.merge(
            candidate,
            on=[
                "date",
                "split",
            ],
            how="inner",
            validate="one_to_one",
        )

        pair["mode"] = mode

        pair["change_vs_base"] = (
            pair["candidate_total_diff"]
            - pair["base_total_diff"]
        )

        paired_frames.append(
            pair
        )

    paired_df = pd.concat(
        paired_frames,
        ignore_index=True,
    )

    paired_summary_df = (
        paired_df.groupby(
            [
                "mode",
                "feature",
            ],
            as_index=False,
        )
        .agg(
            paired_days=(
                "date",
                "size",
            ),

            better_days=(
                "change_vs_base",
                lambda s:
                    int((s > 0).sum()),
            ),

            same_days=(
                "change_vs_base",
                lambda s:
                    int((s == 0).sum()),
            ),

            worse_days=(
                "change_vs_base",
                lambda s:
                    int((s < 0).sum()),
            ),

            mean_daily_change=(
                "change_vs_base",
                "mean",
            ),

            median_daily_change=(
                "change_vs_base",
                "median",
            ),

            total_change=(
                "change_vs_base",
                "sum",
            ),

            min_daily_change=(
                "change_vs_base",
                "min",
            ),

            max_daily_change=(
                "change_vs_base",
                "max",
            ),
        )
    )

    paired_summary_df[
        "better_rate_ex_ties"
    ] = (
        paired_summary_df[
            "better_days"
        ]
        / (
            paired_summary_df[
                "better_days"
            ]
            + paired_summary_df[
                "worse_days"
            ]
        ).replace(
            0,
            np.nan,
        )
        * 100.0
    )

    # --------------------------------------------------------
    # Leave-one-day / leave-one-split robustness
    # --------------------------------------------------------

    robustness_rows = []
    loo_rows = []
    loso_rows = []

    for mode in modes:
        if mode == "BASE":
            continue

        p = paired_df[
            paired_df["mode"] == mode
        ].copy()

        feature = str(
            p.iloc[0]["feature"]
        )

        for idx, row in p.iterrows():
            remaining = p.drop(
                index=idx
            )

            loo_rows.append(
                {
                    "mode":
                        mode,

                    "feature":
                        feature,

                    "excluded_date":
                        row["date"],

                    "excluded_split":
                        row["split"],

                    "excluded_change":
                        float(
                            row["change_vs_base"]
                        ),

                    "remaining_total_change":
                        float(
                            remaining[
                                "change_vs_base"
                            ].sum()
                        ),
                }
            )

        for split in (
            p["split"]
            .drop_duplicates()
        ):
            remaining = p[
                p["split"] != split
            ]

            loso_rows.append(
                {
                    "mode":
                        mode,

                    "feature":
                        feature,

                    "excluded_split":
                        split,

                    "remaining_total_change":
                        float(
                            remaining[
                                "change_vs_base"
                            ].sum()
                        ),

                    "remaining_mean_change":
                        float(
                            remaining[
                                "change_vs_base"
                            ].mean()
                        ),

                    "remaining_median_change":
                        float(
                            remaining[
                                "change_vs_base"
                            ].median()
                        ),
                }
            )

    loo_df = pd.DataFrame(
        loo_rows
    )

    loso_df = pd.DataFrame(
        loso_rows
    )

    base_split = split_df[
        split_df["mode"] == "BASE"
    ][
        [
            "split",
            "avg_diff",
        ]
    ].rename(
        columns={
            "avg_diff":
                "base_avg_diff",
        }
    )

    for mode in modes:
        if mode == "BASE":
            continue

        o = overall_df[
            overall_df["mode"] == mode
        ].iloc[0]

        p = paired_summary_df[
            paired_summary_df["mode"]
            == mode
        ].iloc[0]

        candidate_split = split_df[
            split_df["mode"] == mode
        ][
            [
                "split",
                "avg_diff",
            ]
        ]

        sc = candidate_split.merge(
            base_split,
            on="split",
            how="inner",
        )

        improved_splits = int(
            (
                sc["avg_diff"]
                > sc["base_avg_diff"]
            ).sum()
        )

        lp = loo_df[
            loo_df["mode"] == mode
        ]

        sp = loso_df[
            loso_df["mode"] == mode
        ]

        loo_all_positive = bool(
            (
                lp[
                    "remaining_total_change"
                ]
                > 0
            ).all()
        )

        loso_all_positive = bool(
            (
                sp[
                    "remaining_total_change"
                ]
                > 0
            ).all()
        )

        if (
            o["total_change_vs_base"] > 0
            and p["median_daily_change"] >= 0
            and p["better_days"] >= p["worse_days"]
            and improved_splits >= 4
            and loo_all_positive
            and loso_all_positive
        ):
            status = "ROBUST_PROMISING"

        elif (
            o["total_change_vs_base"] > 0
        ):
            status = "POSITIVE_BUT_NOT_ROBUST"

        else:
            status = "NO_OOS_IMPROVEMENT"

        robustness_rows.append(
            {
                "mode":
                    mode,

                "feature":
                    o["feature"],

                "status":
                    status,

                "total_diff":
                    float(
                        o["total_diff"]
                    ),

                "total_change_vs_base":
                    float(
                        o[
                            "total_change_vs_base"
                        ]
                    ),

                "avg_diff":
                    float(
                        o["avg_diff"]
                    ),

                "win_rate":
                    float(
                        o["win_rate"]
                    ),

                "positive_days":
                    float(
                        o["positive_days"]
                    ),

                "better_days":
                    int(
                        p["better_days"]
                    ),

                "same_days":
                    int(
                        p["same_days"]
                    ),

                "worse_days":
                    int(
                        p["worse_days"]
                    ),

                "median_daily_change":
                    float(
                        p[
                            "median_daily_change"
                        ]
                    ),

                "improved_splits":
                    improved_splits,

                "loo_all_positive":
                    loo_all_positive,

                "loso_all_positive":
                    loso_all_positive,

                "production_adopted":
                    False,
            }
        )

    robustness_df = pd.DataFrame(
        robustness_rows
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    header(
        "OVERALL RESULTS"
    )

    print(
        overall_df.sort_values(
            "total_diff",
            ascending=False,
        ).to_string(
            index=False
        )
    )

    header(
        "PAIRED DAILY COMPARISON VS BASE"
    )

    print(
        paired_summary_df.sort_values(
            "total_change",
            ascending=False,
        ).to_string(
            index=False
        )
    )

    header(
        "ROBUSTNESS ASSESSMENT"
    )

    print(
        robustness_df.sort_values(
            "total_change_vs_base",
            ascending=False,
        ).to_string(
            index=False
        )
    )

    header(
        "BY SPLIT"
    )

    print(
        split_df.sort_values(
            [
                "mode",
                "split",
            ]
        ).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "58_machine_model_position_daily.csv":
            daily_df,

        "58_machine_model_position_by_split.csv":
            split_df,

        "58_machine_model_position_overall.csv":
            overall_df,

        "58_machine_model_position_paired_daily.csv":
            paired_df,

        "58_machine_model_position_paired_summary.csv":
            paired_summary_df,

        "58_machine_model_position_leave_one_day_out.csv":
            loo_df,

        "58_machine_model_position_leave_one_split_out.csv":
            loso_df,

        "58_machine_model_position_robustness.csv":
            robustness_df,
    }

    header(
        "FILES SAVED"
    )

    for filename, frame in outputs.items():
        path = OUTPUT_DIR / filename

        frame.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

        print(path)

    print()
    print(
        "58 machine-model position ablation OOS complete."
    )
    print(
        "No production model change has been made."
    )


if __name__ == "__main__":
    main()
