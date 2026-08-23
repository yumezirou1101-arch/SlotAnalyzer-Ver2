from __future__ import annotations

from pathlib import Path
import importlib.util
import re
import unicodedata

import numpy as np
import pandas as pd


# ============================================================
# SlotAnalyzer
# V4.2_C + Min-Repo External Feature OOS Diagnostic
# ============================================================
#
# IMPORTANT
# - Existing Source 56 / V4.2_C is imported and NOT modified.
# - Existing V4.2_C weights remain the baseline.
# - Min-Repo features are constructed beforehand with date < target_date only.
# - External feature weights are FIXED in advance; no tuning on these OOS dates.
# - Comparison uses only dates for which Min-Repo historical features exist.
#
# Why there is no direct "STORE" additive mode:
# A store-level value is identical for all 514 machines on the same target date.
# Source-56 rank_score() z-scores factors cross-sectionally within that day,
# so a constant store feature has zero variance and cannot change machine ranking.
# Store context should later be tested as a regime/switch/interaction feature,
# not as a simple additive machine-ranking factor.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

SOURCE_56 = (
    PROJECT_ROOT
    / "machine_number"
    / "ana_slo_prediction_v4_2_machine_number_position_ablation_oos.py"
)

MINREPO_HISTORY_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "external_features"
    / "minrepo"
    / "history_features"
)

STORE_HISTORY_CSV = (
    MINREPO_HISTORY_DIR
    / "store_history_features.csv"
)

MACHINE_HISTORY_CSV = (
    MINREPO_HISTORY_DIR
    / "machine_history_features.csv"
)

TAIL_HISTORY_CSV = (
    MINREPO_HISTORY_DIR
    / "tail_history_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
    / "65_Ver4_2_minrepo_external_feature_oos"
)

TOP_NS = (1, 3, 5, 10)

# Fixed, pre-declared challenger weights.
# No optimization is performed on OOS results.
SINGLE_EXTERNAL_WEIGHT = 0.10
COMBINED_EACH_WEIGHT = 0.05

# Require at least this many prior Min-Repo collected dates for the store.
# With the current 13-date dataset this starts evaluation from 2026-08-04.
MIN_STORE_HISTORY_DAYS = 3

MODES = {
    "BASE_V4.2_C": [],
    "MINREPO_MACHINE_DIFF3": [
        ("ext_machine_diff3", SINGLE_EXTERNAL_WEIGHT),
    ],
    "MINREPO_MACHINE_WIN3": [
        ("ext_machine_win3", SINGLE_EXTERNAL_WEIGHT),
    ],
    "MINREPO_TAIL_DIFF3": [
        ("ext_tail_diff3", SINGLE_EXTERNAL_WEIGHT),
    ],
    "MINREPO_TAIL_WIN3": [
        ("ext_tail_win3", SINGLE_EXTERNAL_WEIGHT),
    ],
    "MINREPO_MACHINE_TAIL_DIFF3": [
        ("ext_machine_diff3", COMBINED_EACH_WEIGHT),
        ("ext_tail_diff3", COMBINED_EACH_WEIGHT),
    ],
}


# ============================================================
# HELPERS
# ============================================================

def header(title: str) -> None:
    print()
    print("=" * 118)
    print(title)
    print("=" * 118)


def load_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Required source not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import module: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv_flexible(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required CSV not found: {path}"
        )

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
        except Exception:
            pass

    raise RuntimeError(
        f"Could not read CSV: {path}"
    )


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["date"] = pd.to_datetime(
        x["date"],
        errors="coerce",
    )

    if x["date"].isna().any():
        raise RuntimeError(
            "Invalid date found in external feature CSV."
        )

    return x


def normalize_machine_name(value) -> str:
    if pd.isna(value):
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(value),
    )

    text = re.sub(
        r"\s+",
        "",
        text,
    )

    # Must mirror Min-Repo history construction logic.
    text = re.sub(
        r"^(スマスロ|Lパチスロ|Lスロット|L|スロット)",
        "",
        text,
    )

    return text


def make_weights(
    base_weights: dict[str, float],
    extra_specs: list[tuple[str, float]],
) -> dict[str, float]:

    if not extra_specs:
        return base_weights.copy()

    extra_total = sum(
        weight
        for _, weight in extra_specs
    )

    if (
        extra_total <= 0
        or extra_total >= 1
    ):
        raise ValueError(
            f"Invalid total external weight: {extra_total}"
        )

    scale = 1.0 - extra_total

    weights = {
        key: value * scale
        for key, value
        in base_weights.items()
    }

    for factor, weight in extra_specs:
        weights[factor] = weight

    total = sum(
        weights.values()
    )

    if not np.isclose(
        total,
        1.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"Weight sum is not 1.0: {total}"
        )

    return weights


def neutral_fill(
    series: pd.Series,
) -> pd.Series:
    """
    Prediction-time neutral imputation.
    Uses only the cross-sectional median of already historical features
    for the same target date. It never uses target-day actual diff.
    """
    x = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid = x.dropna()

    fill_value = (
        float(valid.median())
        if len(valid)
        else 0.0
    )

    return x.fillna(
        fill_value
    )


# ============================================================
# EXTERNAL FEATURE LOADING
# ============================================================

def load_external_features():
    store = normalize_dates(
        read_csv_flexible(
            STORE_HISTORY_CSV
        )
    )

    machine = normalize_dates(
        read_csv_flexible(
            MACHINE_HISTORY_CSV
        )
    )

    tail = normalize_dates(
        read_csv_flexible(
            TAIL_HISTORY_CSV
        )
    )

    required_store = {
        "date",
        "store_hist_days",
    }

    required_machine = {
        "date",
        "machine_key",
        "machine_hist_days",
        "machine_avg_diff_mean3",
        "machine_win_rate_mean3",
    }

    required_tail = {
        "date",
        "tail",
        "tail_hist_days",
        "tail_avg_diff_mean3",
        "tail_win_rate_mean3",
    }

    for label, df, required in (
        ("store", store, required_store),
        ("machine", machine, required_machine),
        ("tail", tail, required_tail),
    ):
        missing = required - set(
            df.columns
        )

        if missing:
            raise RuntimeError(
                f"{label} history missing columns: "
                f"{sorted(missing)}"
            )

    machine["machine_key"] = (
        machine["machine_key"]
        .astype(str)
    )

    tail["tail"] = pd.to_numeric(
        tail["tail"],
        errors="coerce",
    )

    return store, machine, tail


def eligible_dates(
    store: pd.DataFrame,
) -> list[pd.Timestamp]:

    x = store[
        pd.to_numeric(
            store["store_hist_days"],
            errors="coerce",
        )
        >= MIN_STORE_HISTORY_DAYS
    ].copy()

    return sorted(
        x["date"]
        .drop_duplicates()
        .tolist()
    )


# ============================================================
# PANEL AUGMENTATION
# ============================================================

def add_external_features(
    panel: pd.DataFrame,
    target_date: pd.Timestamp,
    machine_hist: pd.DataFrame,
    tail_hist: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    x = panel.copy()

    if "machine_no" not in x.columns:
        raise RuntimeError(
            "Source-56 panel has no machine_no column."
        )

    if "machine_name" not in x.columns:
        raise RuntimeError(
            "Source-56 panel has no machine_name column."
        )

    x["_machine_key_ext"] = (
        x["machine_name"]
        .map(
            normalize_machine_name
        )
    )

    x["_tail_ext"] = (
        pd.to_numeric(
            x["machine_no"],
            errors="coerce",
        )
        .astype("Int64")
        % 10
    )

    m = machine_hist[
        machine_hist["date"]
        == target_date
    ][
        [
            "machine_key",
            "machine_hist_days",
            "machine_avg_diff_mean3",
            "machine_win_rate_mean3",
        ]
    ].copy()

    m = m.drop_duplicates(
        subset=["machine_key"],
        keep="first",
    )

    m = m.rename(
        columns={
            "machine_key":
                "_machine_key_ext",

            "machine_hist_days":
                "_machine_hist_days_ext",

            "machine_avg_diff_mean3":
                "ext_machine_diff3",

            "machine_win_rate_mean3":
                "ext_machine_win3",
        }
    )

    t = tail_hist[
        tail_hist["date"]
        == target_date
    ][
        [
            "tail",
            "tail_hist_days",
            "tail_avg_diff_mean3",
            "tail_win_rate_mean3",
        ]
    ].copy()

    t["tail"] = pd.to_numeric(
        t["tail"],
        errors="coerce",
    ).astype("Int64")

    t = t.drop_duplicates(
        subset=["tail"],
        keep="first",
    )

    t = t.rename(
        columns={
            "tail":
                "_tail_ext",

            "tail_hist_days":
                "_tail_hist_days_ext",

            "tail_avg_diff_mean3":
                "ext_tail_diff3",

            "tail_win_rate_mean3":
                "ext_tail_win3",
        }
    )

    x = x.merge(
        m,
        on="_machine_key_ext",
        how="left",
        validate="many_to_one",
    )

    x = x.merge(
        t,
        on="_tail_ext",
        how="left",
        validate="many_to_one",
    )

    machine_matched = int(
        x[
            "_machine_hist_days_ext"
        ].notna().sum()
    )

    tail_matched = int(
        x[
            "_tail_hist_days_ext"
        ].notna().sum()
    )

    for col in (
        "ext_machine_diff3",
        "ext_machine_win3",
        "ext_tail_diff3",
        "ext_tail_win3",
    ):
        x[col] = neutral_fill(
            x[col]
        )

    diagnostic = {
        "date":
            target_date,

        "panel_rows":
            len(x),

        "machine_feature_matches":
            machine_matched,

        "tail_feature_matches":
            tail_matched,

        "machine_match_rate":
            (
                machine_matched
                / len(x)
                * 100.0
                if len(x)
                else np.nan
            ),

        "tail_match_rate":
            (
                tail_matched
                / len(x)
                * 100.0
                if len(x)
                else np.nan
            ),
    }

    return x, diagnostic


# ============================================================
# EVALUATION
# ============================================================

def evaluate_ranked(
    ranked: pd.DataFrame,
    mode: str,
    target_date: pd.Timestamp,
) -> list[dict]:

    rows = []

    for top_n in TOP_NS:
        top = ranked.head(
            min(
                top_n,
                len(ranked),
            )
        ).copy()

        d = pd.to_numeric(
            top["diff"],
            errors="coerce",
        ).dropna()

        if d.empty:
            continue

        rows.append(
            {
                "date":
                    target_date,

                "mode":
                    mode,

                "top_n":
                    top_n,

                "machines":
                    len(d),

                "avg_diff":
                    float(
                        d.mean()
                    ),

                "median_diff":
                    float(
                        d.median()
                    ),

                "win_rate":
                    float(
                        (
                            d > 0
                        ).mean()
                        * 100.0
                    ),

                "plus1000_rate":
                    float(
                        (
                            d >= 1000
                        ).mean()
                        * 100.0
                    ),

                "plus2000_rate":
                    float(
                        (
                            d >= 2000
                        ).mean()
                        * 100.0
                    ),

                "total_diff":
                    float(
                        d.sum()
                    ),
            }
        )

    return rows


def build_overall(
    daily: pd.DataFrame,
) -> pd.DataFrame:

    return (
        daily.groupby(
            [
                "mode",
                "top_n",
            ],
            as_index=False,
        )
        .agg(
            days=(
                "date",
                "nunique",
            ),

            avg_diff=(
                "avg_diff",
                "mean",
            ),

            median_daily_avg_diff=(
                "avg_diff",
                "median",
            ),

            win_rate=(
                "win_rate",
                "mean",
            ),

            plus1000_rate=(
                "plus1000_rate",
                "mean",
            ),

            plus2000_rate=(
                "plus2000_rate",
                "mean",
            ),

            positive_days=(
                "total_diff",
                lambda s:
                    float(
                        (
                            pd.to_numeric(
                                s,
                                errors="coerce",
                            )
                            > 0
                        ).mean()
                        * 100.0
                    ),
            ),

            total_diff=(
                "total_diff",
                "sum",
            ),
        )
    )


def paired_vs_base(
    daily: pd.DataFrame,
) -> pd.DataFrame:

    base = daily[
        daily["mode"]
        == "BASE_V4.2_C"
    ][
        [
            "date",
            "top_n",
            "avg_diff",
            "total_diff",
        ]
    ].rename(
        columns={
            "avg_diff":
                "base_avg_diff",

            "total_diff":
                "base_total_diff",
        }
    )

    rows = []

    for mode in daily["mode"].unique():
        if mode == "BASE_V4.2_C":
            continue

        challenger = daily[
            daily["mode"]
            == mode
        ][
            [
                "date",
                "top_n",
                "avg_diff",
                "total_diff",
            ]
        ].copy()

        merged = challenger.merge(
            base,
            on=[
                "date",
                "top_n",
            ],
            how="inner",
            validate="one_to_one",
        )

        merged[
            "avg_diff_change"
        ] = (
            merged[
                "avg_diff"
            ]
            - merged[
                "base_avg_diff"
            ]
        )

        merged[
            "total_diff_change"
        ] = (
            merged[
                "total_diff"
            ]
            - merged[
                "base_total_diff"
            ]
        )

        for top_n, g in merged.groupby(
            "top_n"
        ):
            rows.append(
                {
                    "mode":
                        mode,

                    "top_n":
                        int(
                            top_n
                        ),

                    "paired_days":
                        len(g),

                    "better_days":
                        int(
                            (
                                g[
                                    "total_diff_change"
                                ]
                                > 0
                            ).sum()
                        ),

                    "same_days":
                        int(
                            (
                                g[
                                    "total_diff_change"
                                ]
                                == 0
                            ).sum()
                        ),

                    "worse_days":
                        int(
                            (
                                g[
                                    "total_diff_change"
                                ]
                                < 0
                            ).sum()
                        ),

                    "mean_avg_diff_change":
                        float(
                            g[
                                "avg_diff_change"
                            ].mean()
                        ),

                    "median_avg_diff_change":
                        float(
                            g[
                                "avg_diff_change"
                            ].median()
                        ),

                    "total_diff_change":
                        float(
                            g[
                                "total_diff_change"
                            ].sum()
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
        "65 - V4.2_C + Min-Repo External Feature OOS Diagnostic"
    )

    m56 = load_module(
        SOURCE_56,
        "slotanalyzer_source56",
    )

    base_weights = (
        m56.V42_C_WEIGHTS.copy()
    )

    print(
        f"source               : {SOURCE_56}"
    )

    print(
        f"base weight sum      : "
        f"{sum(base_weights.values()):.12f}"
    )

    print(
        f"single ext weight    : "
        f"{SINGLE_EXTERNAL_WEIGHT:.2f}"
    )

    print(
        f"combined each weight : "
        f"{COMBINED_EACH_WEIGHT:.2f}"
    )

    print(
        f"min store hist days  : "
        f"{MIN_STORE_HISTORY_DAYS}"
    )

    df = m56.load_data()

    edge_distance_map = (
        m56.build_number_edge_distance(
            df[
                "machine_no"
            ].tolist()
        )
    )

    (
        store_hist,
        machine_hist,
        tail_hist,
    ) = load_external_features()

    dates = eligible_dates(
        store_hist
    )

    # Only evaluate dates that also exist in Source-56 actual data.
    available_actual_dates = set(
        pd.to_datetime(
            df["date"]
        ).drop_duplicates()
    )

    dates = [
        pd.Timestamp(
            date
        )
        for date in dates
        if pd.Timestamp(
            date
        ) in available_actual_dates
    ]

    header(
        "ELIGIBLE OOS DATES"
    )

    print(
        f"dates                : {len(dates)}"
    )

    print(
        ", ".join(
            str(
                d.date()
            )
            for d in dates
        )
    )

    if len(dates) < 5:
        raise RuntimeError(
            "Too few eligible dates for even a diagnostic comparison."
        )

    daily_rows = []
    coverage_rows = []
    top10_rows = []

    header(
        "DAILY EVALUATION"
    )

    for target_date in dates:

        base_panel = m56.build_features(
            df,
            target_date,
            edge_distance_map,
        )

        if base_panel.empty:
            print(
                f"{target_date.date()} SKIP empty panel"
            )
            continue

        panel, coverage = (
            add_external_features(
                base_panel,
                target_date,
                machine_hist,
                tail_hist,
            )
        )

        coverage_rows.append(
            coverage
        )

        print(
            f"{target_date.date()} "
            f"panel={len(panel)} "
            f"machine_match="
            f"{coverage['machine_match_rate']:.1f}% "
            f"tail_match="
            f"{coverage['tail_match_rate']:.1f}%"
        )

        for mode, extra_specs in MODES.items():

            weights = make_weights(
                base_weights,
                extra_specs,
            )

            ranked = m56.rank_score(
                panel,
                weights,
            )

            daily_rows.extend(
                evaluate_ranked(
                    ranked,
                    mode,
                    target_date,
                )
            )

            top10 = ranked.head(
                10
            ).copy()

            top10[
                "prediction_rank"
            ] = np.arange(
                1,
                len(top10) + 1,
            )

            top10[
                "date"
            ] = target_date

            top10[
                "mode"
            ] = mode

            keep_cols = [
                "date",
                "mode",
                "prediction_rank",
                "machine_no",
                "machine_name",
                "diff",
                "score",
                "ext_machine_diff3",
                "ext_machine_win3",
                "ext_tail_diff3",
                "ext_tail_win3",
            ]

            top10_rows.append(
                top10[
                    keep_cols
                ]
            )

    daily = pd.DataFrame(
        daily_rows
    )

    if daily.empty:
        raise RuntimeError(
            "No OOS diagnostic results."
        )

    coverage_df = pd.DataFrame(
        coverage_rows
    )

    top10_df = pd.concat(
        top10_rows,
        ignore_index=True,
    )

    overall = build_overall(
        daily
    )

    paired = paired_vs_base(
        daily
    )

    # --------------------------------------------------------
    # Display TOP10 comparison first
    # --------------------------------------------------------

    header(
        "OVERALL - TOP10"
    )

    top10_overall = overall[
        overall[
            "top_n"
        ]
        == 10
    ].sort_values(
        [
            "total_diff",
        ],
        ascending=False,
    )

    print(
        top10_overall.to_string(
            index=False
        )
    )

    header(
        "PAIRED VS BASE - TOP10"
    )

    top10_paired = paired[
        paired[
            "top_n"
        ]
        == 10
    ].sort_values(
        [
            "total_diff_change",
        ],
        ascending=False,
    )

    print(
        top10_paired.to_string(
            index=False
        )
    )

    header(
        "ALL TOP-N OVERALL"
    )

    print(
        overall.sort_values(
            [
                "top_n",
                "total_diff",
            ],
            ascending=[
                True,
                False,
            ],
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

    files = {
        "65_minrepo_external_daily.csv":
            daily,

        "65_minrepo_external_overall.csv":
            overall,

        "65_minrepo_external_paired_vs_base.csv":
            paired,

        "65_minrepo_external_feature_coverage.csv":
            coverage_df,

        "65_minrepo_external_top10_picks.csv":
            top10_df,
    }

    header(
        "FILES SAVED"
    )

    for filename, frame in files.items():

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
        "ASSESSMENT RULE:"
    )

    print(
        "- This is a diagnostic OOS comparison only."
    )

    print(
        "- No challenger is promoted from this script."
    )

    print(
        "- BASE V4.2_C remains unchanged."
    )

    print(
        "- Direct STORE additive scoring is intentionally excluded "
        "because a same-day store feature is constant across all machines."
    )

    print(
        "- If an external factor looks promising, collect more dates "
        "and confirm with a larger locked forward/OOS sample before use."
    )


if __name__ == "__main__":
    main()
