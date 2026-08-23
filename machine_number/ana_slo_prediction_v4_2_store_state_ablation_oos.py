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
    / "59_Ver4_2_store_state_ablation_oos"
)

EXPECTED_BASE_TOTAL_DIFF = 130400.0
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

    std = float(
        s.std(ddof=0)
    )

    if (
        std == 0
        or np.isnan(std)
    ):
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

    scale = (
        1.0
        - ADDED_FEATURE_WEIGHT
    )

    weights = {
        k: v * scale
        for k, v
        in base_weights.items()
    }

    weights[feature] = (
        ADDED_FEATURE_WEIGHT
    )

    total = sum(
        weights.values()
    )

    if not np.isclose(
        total,
        1.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"Weight sum error: {total}"
        )

    return weights


def mean_or_zero(
    series: pd.Series,
) -> float:

    s = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if s.empty:
        return 0.0

    return float(
        s.mean()
    )


def build_store_state_features(
    raw_df: pd.DataFrame,
    panel: pd.DataFrame,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Leakage-safe store/model state features.

    Important:
    A pure store-wide value is identical for every machine on a target day.
    Because ranking is cross-sectional, such a constant cannot change rank.

    Therefore this test uses store state as a benchmark and measures each
    machine model's recent performance RELATIVE to the store.

    Features:
      model_vs_store_prev1
      model_vs_store_recent3
      model_vs_store_recent7
      model_recent3_trend_vs_store

    Only dates strictly before target_date are used.
    """

    x = panel.copy()

    if x.empty:
        return x

    hist = raw_df[
        raw_df["date"] < target_date
    ].copy()

    if hist.empty:
        for col in (
            "model_vs_store_prev1",
            "model_vs_store_recent3",
            "model_vs_store_recent7",
            "model_recent3_trend_vs_store",
        ):
            x[col] = 0.0

        return x

    hist = hist.sort_values(
        "date"
    )

    hist_dates = sorted(
        hist["date"]
        .dropna()
        .unique()
    )

    def date_window(n: int):
        return hist_dates[-n:]

    prev1_dates = date_window(1)
    recent3_dates = date_window(3)
    recent7_dates = date_window(7)

    prev1 = hist[
        hist["date"].isin(
            prev1_dates
        )
    ]

    recent3 = hist[
        hist["date"].isin(
            recent3_dates
        )
    ]

    recent7 = hist[
        hist["date"].isin(
            recent7_dates
        )
    ]

    store_prev1 = mean_or_zero(
        prev1["diff"]
    )

    store_recent3 = mean_or_zero(
        recent3["diff"]
    )

    store_recent7 = mean_or_zero(
        recent7["diff"]
    )

    model_prev1 = (
        prev1.groupby(
            "machine_name"
        )["diff"]
        .mean()
        .to_dict()
    )

    model_recent3 = (
        recent3.groupby(
            "machine_name"
        )["diff"]
        .mean()
        .to_dict()
    )

    model_recent7 = (
        recent7.groupby(
            "machine_name"
        )["diff"]
        .mean()
        .to_dict()
    )

    x["model_vs_store_prev1"] = (
        x["machine_name"]
        .map(model_prev1)
        .fillna(store_prev1)
        .astype(float)
        - store_prev1
    )

    x["model_vs_store_recent3"] = (
        x["machine_name"]
        .map(model_recent3)
        .fillna(store_recent3)
        .astype(float)
        - store_recent3
    )

    x["model_vs_store_recent7"] = (
        x["machine_name"]
        .map(model_recent7)
        .fillna(store_recent7)
        .astype(float)
        - store_recent7
    )

    # Model short-term acceleration relative to store-wide acceleration.
    model_r3 = (
        x["machine_name"]
        .map(model_recent3)
        .fillna(store_recent3)
        .astype(float)
    )

    model_r7 = (
        x["machine_name"]
        .map(model_recent7)
        .fillna(store_recent7)
        .astype(float)
    )

    store_trend = (
        store_recent3
        - store_recent7
    )

    x[
        "model_recent3_trend_vs_store"
    ] = (
        (model_r3 - model_r7)
        - store_trend
    )

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

    selected = (
        x.sort_values(
            "score",
            ascending=False,
        )
        .head(TOP_N)
    )

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
            float(
                (diffs > 0).mean()
                * 100.0
            ),

        "plus1000_rate":
            float(
                (diffs >= 1000).mean()
                * 100.0
            ),

        "plus2000_rate":
            float(
                (diffs >= 2000).mean()
                * 100.0
            ),

        "positive":
            int(
                diffs.sum() > 0
            ),

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
        "59 - V4.2_C TOP10 Store-State / Model-Relative-State Ablation OOS"
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
    print("Important:")
    print(
        "- Baseline is unchanged V4.2_C TOP10."
    )
    print(
        "- Only dates before each target date are used."
    )
    print(
        "- Pure store-wide state is constant across machines and cannot rank them."
    )
    print(
        "- Therefore store state is used as a benchmark for model-relative features."
    )
    print(
        "- Each candidate is tested separately at fixed 10% weight."
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
            build_store_state_features(
                df,
                panel,
                target_date,
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

        "ADD_MODEL_VS_STORE_PREV1":
            "model_vs_store_prev1",

        "ADD_MODEL_VS_STORE_RECENT3":
            "model_vs_store_recent3",

        "ADD_MODEL_VS_STORE_RECENT7":
            "model_vs_store_recent7",

        "ADD_MODEL_TREND_VS_STORE":
            "model_recent3_trend_vs_store",
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

    split_df[
        "positive_days"
    ] *= 100.0

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

    overall_df[
        "positive_days"
    ] *= 100.0

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

        pair[
            "change_vs_base"
        ] = (
            pair[
                "candidate_total_diff"
            ]
            - pair[
                "base_total_diff"
            ]
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

    loo_rows = []
    loso_rows = []
    robustness_rows = []

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
                            row[
                                "change_vs_base"
                            ]
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
            overall_df["mode"]
            == mode
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
            status = (
                "POSITIVE_BUT_NOT_ROBUST"
            )

        else:
            status = (
                "NO_OOS_IMPROVEMENT"
            )

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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "59_store_state_daily.csv":
            daily_df,

        "59_store_state_by_split.csv":
            split_df,

        "59_store_state_overall.csv":
            overall_df,

        "59_store_state_paired_daily.csv":
            paired_df,

        "59_store_state_paired_summary.csv":
            paired_summary_df,

        "59_store_state_leave_one_day_out.csv":
            loo_df,

        "59_store_state_leave_one_split_out.csv":
            loso_df,

        "59_store_state_robustness.csv":
            robustness_df,
    }

    header(
        "FILES SAVED"
    )

    for filename, frame in outputs.items():

        path = (
            OUTPUT_DIR
            / filename
        )

        frame.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

        print(path)

    print()
    print(
        "59 store-state ablation OOS complete."
    )
    print(
        "No production model change has been made."
    )


if __name__ == "__main__":
    main()
