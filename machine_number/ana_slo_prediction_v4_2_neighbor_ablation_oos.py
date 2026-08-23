from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
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

INPUT_CSV = (
    DATA_DIR
    / "ana_slo_20260711_20260818.csv"
)

OUTPUT_DIR = (
    DATA_DIR
    / "analysis_31days_deep"
    / "53_Ver4_2_neighbor_ablation_oos"
)

START = pd.Timestamp("2026-07-11")
END = pd.Timestamp("2026-08-18")

TOP_NS = [
    5,
    10,
    20,
    30,
]

NEIGHBOR_MODES = [
    "CURRENT_PM1",
    "NO_NEIGHBOR",
    "SAME_MACHINE_PM1",
]


# ============================================================
# Ver.4 / Ver.4.2 weights
# ============================================================

V4_WEIGHTS = {
    "avg31": 0.0670952025611345,
    "recent7_avg": 0.05164896703284082,
    "recent7_win": 0.06602967770818714,
    "last_diff": 0.12382294629381808,
    "prev_change": 0.10484738021281044,
    "weekday_avg": 0.05672674990073483,
    "type_avg": 0.05843723530102936,
    "plus1000_rate": 0.17725354845070532,
    "plus2000_rate": 0.13298938481323394,
    "neighbor_avg": 0.06161296683628432,
    "bounce_signal": 0.09953594088922124,
}

V42_A = V4_WEIGHTS.copy()
V42_A.pop("recent7_win")

V42_B = V4_WEIGHTS.copy()
V42_B.pop("bounce_signal")

V42_C = V4_WEIGHTS.copy()
V42_C.pop("recent7_win")
V42_C.pop("bounce_signal")


def normalize_weights(
    weights: dict[str, float],
) -> dict[str, float]:

    total = sum(
        weights.values()
    )

    if total <= 0:
        raise ValueError(
            "Weight sum must be positive."
        )

    return {
        key: value / total
        for key, value
        in weights.items()
    }


MODELS = {
    "V4_BASE":
        normalize_weights(
            V4_WEIGHTS
        ),

    "V4.2_A":
        normalize_weights(
            V42_A
        ),

    "V4.2_B":
        normalize_weights(
            V42_B
        ),

    "V4.2_C":
        normalize_weights(
            V42_C
        ),
}


# ============================================================
# Rolling OOS periods
# ============================================================

ROLLING_SPLITS = [
    (
        "ROLL1",
        pd.Timestamp("2026-07-11"),
        pd.Timestamp("2026-07-20"),
        pd.Timestamp("2026-07-21"),
        pd.Timestamp("2026-07-24"),
    ),
    (
        "ROLL2",
        pd.Timestamp("2026-07-11"),
        pd.Timestamp("2026-07-24"),
        pd.Timestamp("2026-07-25"),
        pd.Timestamp("2026-07-28"),
    ),
    (
        "ROLL3",
        pd.Timestamp("2026-07-11"),
        pd.Timestamp("2026-07-28"),
        pd.Timestamp("2026-07-29"),
        pd.Timestamp("2026-08-01"),
    ),
    (
        "ROLL4",
        pd.Timestamp("2026-07-11"),
        pd.Timestamp("2026-08-01"),
        pd.Timestamp("2026-08-02"),
        pd.Timestamp("2026-08-05"),
    ),
    (
        "ROLL5",
        pd.Timestamp("2026-07-11"),
        pd.Timestamp("2026-08-05"),
        pd.Timestamp("2026-08-06"),
        pd.Timestamp("2026-08-10"),
    ),
    (
        "ROLL6",
        pd.Timestamp("2026-07-11"),
        pd.Timestamp("2026-08-10"),
        pd.Timestamp("2026-08-11"),
        pd.Timestamp("2026-08-14"),
    ),
    (
        "ROLL7",
        pd.Timestamp("2026-07-11"),
        pd.Timestamp("2026-08-14"),
        pd.Timestamp("2026-08-15"),
        pd.Timestamp("2026-08-18"),
    ),
]


# ============================================================
# Helpers
# ============================================================

def print_header(
    title: str,
) -> None:

    print()
    print("=" * 86)
    print(title)
    print("=" * 86)


def read_csv_flexible(
    path: Path,
) -> pd.DataFrame:

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
        f"CSV read failed: {path}"
    )


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    for col in candidates:

        if col in df.columns:
            return col

    return None


# ============================================================
# Data loading
# ============================================================

def load_data() -> pd.DataFrame:

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}"
        )

    df = read_csv_flexible(
        INPUT_CSV
    )

    date_col = find_column(
        df,
        [
            "date",
            "\u65e5\u4ed8",
        ],
    )

    no_col = find_column(
        df,
        [
            "machine_no",
            "\u53f0\u756a\u53f7",
        ],
    )

    name_col = find_column(
        df,
        [
            "machine_name",
            "\u6a5f\u7a2e\u540d",
        ],
    )

    diff_col = find_column(
        df,
        [
            "diff",
            "\u5dee\u679a",
        ],
    )

    if not all(
        [
            date_col,
            no_col,
            name_col,
            diff_col,
        ]
    ):

        raise ValueError(
            "Required columns not found: "
            f"date={date_col}, "
            f"machine_no={no_col}, "
            f"machine_name={name_col}, "
            f"diff={diff_col}"
        )

    df = df.rename(
        columns={
            date_col:
                "date",

            no_col:
                "machine_no",

            name_col:
                "machine_name",

            diff_col:
                "diff",
        }
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["machine_no"] = pd.to_numeric(
        df["machine_no"],
        errors="coerce",
    )

    df["diff"] = (
        df["diff"]
        .astype(str)
        .str.replace(
            ",",
            "",
            regex=False,
        )
        .str.replace(
            "+",
            "",
            regex=False,
        )
        .str.strip()
    )

    df["diff"] = pd.to_numeric(
        df["diff"],
        errors="coerce",
    )

    df["machine_name"] = (
        df["machine_name"]
        .astype(str)
        .str.strip()
    )

    df = df.dropna(
        subset=[
            "date",
            "machine_no",
            "machine_name",
            "diff",
        ]
    ).copy()

    df["machine_no"] = (
        df["machine_no"]
        .astype(int)
    )

    df = df[
        (df["date"] >= START)
        & (df["date"] <= END)
    ].copy()

    df = (
        df.sort_values(
            [
                "date",
                "machine_no",
            ]
        )
        .drop_duplicates(
            [
                "date",
                "machine_no",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    df["win"] = (
        df["diff"] > 0
    ).astype(int)

    df["plus1000"] = (
        df["diff"] >= 1000
    ).astype(int)

    df["plus2000"] = (
        df["diff"] >= 2000
    ).astype(int)

    return df


# ============================================================
# Neighbor feature
# ============================================================

def calculate_neighbor_avg(
    no: int,
    current_name: str,
    latest_day: pd.DataFrame,
    mode: str,
) -> tuple[float, int]:

    if mode == "NO_NEIGHBOR":

        return 0.0, 0

    values = []

    for n2 in (
        no - 1,
        no + 1,
    ):

        if n2 not in latest_day.index:
            continue

        neighbor_row = latest_day.loc[
            n2
        ]

        if isinstance(
            neighbor_row,
            pd.DataFrame,
        ):
            neighbor_row = (
                neighbor_row.iloc[-1]
            )

        if mode == "SAME_MACHINE_PM1":

            neighbor_name = str(
                neighbor_row[
                    "machine_name"
                ]
            ).strip()

            if (
                neighbor_name
                != current_name
            ):
                continue

        values.append(
            float(
                neighbor_row[
                    "diff"
                ]
            )
        )

    if not values:

        return 0.0, 0

    return (
        float(
            np.mean(
                values
            )
        ),
        int(
            len(values)
        ),
    )


# ============================================================
# Feature construction
# ============================================================

def build_features(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
    neighbor_mode: str,
) -> pd.DataFrame:
    """
    This intentionally follows the existing 48-series baseline logic
    except for the neighbor feature.

    It keeps the same machine_no history behavior and the same final
    machine_no + machine_name inner merge so that the experiment isolates
    the neighbor rule as much as possible.
    """

    hist = df[
        df["date"] < target_date
    ].copy()

    actual = df[
        df["date"] == target_date
    ][
        [
            "machine_no",
            "machine_name",
            "diff",
        ]
    ].copy()

    if hist.empty or actual.empty:

        return pd.DataFrame()

    target_weekday = (
        target_date.dayofweek
    )

    latest_date = (
        hist["date"].max()
    )

    latest_day = (
        hist[
            hist["date"]
            == latest_date
        ]
        .sort_values(
            "machine_no"
        )
        .drop_duplicates(
            "machine_no",
            keep="last",
        )
        .set_index(
            "machine_no"
        )
    )

    type_stats = (
        hist.groupby(
            "machine_name"
        )["diff"]
        .mean()
        .to_dict()
    )

    rows = []

    for no, m in hist.groupby(
        "machine_no"
    ):

        m = m.sort_values(
            "date"
        )

        if m.empty:
            continue

        name = str(
            m.iloc[-1][
                "machine_name"
            ]
        ).strip()

        avg31 = float(
            m["diff"].mean()
        )

        recent7 = m.tail(7)

        recent7_avg = float(
            recent7[
                "diff"
            ].mean()
        )

        recent7_win = float(
            recent7[
                "win"
            ].mean()
        )

        last_diff = float(
            m.iloc[-1][
                "diff"
            ]
        )

        if len(m) >= 2:

            prev_diff = float(
                m.iloc[-2][
                    "diff"
                ]
            )

        else:

            prev_diff = (
                last_diff
            )

        prev_change = (
            last_diff
            - prev_diff
        )

        wd = m[
            m[
                "date"
            ].dt.dayofweek
            == target_weekday
        ]

        weekday_n = len(
            wd
        )

        if weekday_n:

            weekday_avg_raw = float(
                wd[
                    "diff"
                ].mean()
            )

        else:

            weekday_avg_raw = (
                avg31
            )

        prior_n = 15.0

        wd_weight = (
            weekday_n
            / (
                weekday_n
                + prior_n
            )
        )

        weekday_avg = (
            weekday_avg_raw
            * wd_weight
            + avg31
            * (
                1.0
                - wd_weight
            )
        )

        plus1000_rate = float(
            m[
                "plus1000"
            ].mean()
        )

        plus2000_rate = float(
            m[
                "plus2000"
            ].mean()
        )

        type_avg = float(
            type_stats.get(
                name,
                0.0,
            )
        )

        (
            neighbor_avg,
            neighbor_n,
        ) = calculate_neighbor_avg(
            int(no),
            name,
            latest_day,
            neighbor_mode,
        )

        if last_diff <= -1000:

            bounce_signal = 1.0

        elif last_diff <= -500:

            bounce_signal = 0.5

        elif last_diff >= 1000:

            bounce_signal = -0.25

        else:

            bounce_signal = 0.0

        rows.append(
            {
                "machine_no":
                    int(no),

                "machine_name":
                    name,

                "avg31":
                    avg31,

                "recent7_avg":
                    recent7_avg,

                "recent7_win":
                    recent7_win,

                "last_diff":
                    last_diff,

                "prev_change":
                    prev_change,

                "weekday_avg":
                    weekday_avg,

                "type_avg":
                    type_avg,

                "plus1000_rate":
                    plus1000_rate,

                "plus2000_rate":
                    plus2000_rate,

                "neighbor_avg":
                    neighbor_avg,

                "neighbor_n":
                    neighbor_n,

                "bounce_signal":
                    bounce_signal,
            }
        )

    feat = pd.DataFrame(
        rows
    )

    if feat.empty:

        return feat

    return feat.merge(
        actual,
        on=[
            "machine_no",
            "machine_name",
        ],
        how="inner",
    )


# ============================================================
# Scoring
# ============================================================

def zscore(
    series: pd.Series,
) -> pd.Series:

    s = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(
        0.0
    )

    std = float(
        s.std(
            ddof=0
        )
    )

    if (
        std == 0
        or np.isnan(
            std
        )
    ):

        return pd.Series(
            0.0,
            index=s.index,
        )

    return (
        s - s.mean()
    ) / std


def rank_score(
    df: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:

    x = df.copy()

    score = pd.Series(
        0.0,
        index=x.index,
    )

    for factor, weight in (
        weights.items()
    ):

        if factor not in x.columns:
            continue

        z = zscore(
            x[
                factor
            ]
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

    x["score"] = (
        score
    )

    return x.sort_values(
        [
            "score",
            "machine_no",
        ],
        ascending=[
            False,
            True,
        ],
    )


def evaluate_day(
    panel: pd.DataFrame,
    weights: dict[str, float],
    top_n: int,
) -> dict | None:

    if panel.empty:

        return None

    ranked = rank_score(
        panel,
        weights,
    )

    top = ranked.head(
        min(
            top_n,
            len(ranked),
        )
    )

    d = (
        top[
            "diff"
        ]
        .astype(float)
    )

    return {
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

        "positive":
            int(
                d.sum()
                > 0
            ),

        "total_diff":
            float(
                d.sum()
            ),

        "machines":
            int(
                len(panel)
            ),

        "selected_neighbor_n_mean":
            float(
                top[
                    "neighbor_n"
                ].mean()
            )
            if (
                "neighbor_n"
                in top.columns
            )
            else np.nan,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    print_header(
        "Ana-Slo Ver.4.2 Neighbor Feature Ablation / OOS Test"
    )

    print(
        "Neighbor modes:"
    )

    for mode in NEIGHBOR_MODES:
        print(
            f"  - {mode}"
        )

    print()
    print(
        "CURRENT_PM1       : existing machine_no +/-1 rule"
    )
    print(
        "NO_NEIGHBOR       : neighbor_avg forced to 0"
    )
    print(
        "SAME_MACHINE_PM1  : +/-1 only when latest-day machine_name matches"
    )
    print()
    print(
        "Important: this experiment keeps the 48-series feature logic "
        "otherwise unchanged."
    )

    df = load_data()

    print()
    print(
        f"records = {len(df):,}"
    )
    print(
        f"days    = {df['date'].nunique()}"
    )

    # --------------------------------------------------------
    # Build all daily panels for each mode
    # --------------------------------------------------------

    panels = {}

    all_dates = pd.date_range(
        START + pd.Timedelta(
            days=1
        ),
        END,
    )

    for mode in NEIGHBOR_MODES:

        print_header(
            f"BUILD PANELS: {mode}"
        )

        counts = []

        for target_date in (
            all_dates
        ):

            panel = build_features(
                df,
                target_date,
                mode,
            )

            if panel.empty:
                continue

            panels[
                (
                    mode,
                    target_date,
                )
            ] = panel

            counts.append(
                len(panel)
            )

        print(
            f"panel days   : {len(counts)}"
        )

        if counts:

            print(
                f"min machines : {min(counts)}"
            )
            print(
                f"max machines : {max(counts)}"
            )

    daily_rows = []
    summary_rows = []

    # --------------------------------------------------------
    # Rolling OOS evaluation
    # --------------------------------------------------------

    for (
        split_name,
        train_start,
        train_end,
        test_start,
        test_end,
    ) in ROLLING_SPLITS:

        test_dates = pd.date_range(
            test_start,
            test_end,
        )

        for mode in NEIGHBOR_MODES:

            for (
                model_name,
                weights,
            ) in MODELS.items():

                for top_n in TOP_NS:

                    results = []

                    for target_date in (
                        test_dates
                    ):

                        panel = panels.get(
                            (
                                mode,
                                target_date,
                            )
                        )

                        if (
                            panel is None
                            or panel.empty
                        ):
                            continue

                        result = evaluate_day(
                            panel,
                            weights,
                            top_n,
                        )

                        if result is None:
                            continue

                        result.update(
                            {
                                "split":
                                    split_name,

                                "neighbor_mode":
                                    mode,

                                "model":
                                    model_name,

                                "top_n":
                                    int(
                                        top_n
                                    ),

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

                        results.append(
                            result
                        )

                    if not results:
                        continue

                    rdf = pd.DataFrame(
                        results
                    )

                    summary_rows.append(
                        {
                            "split":
                                split_name,

                            "neighbor_mode":
                                mode,

                            "model":
                                model_name,

                            "top_n":
                                int(
                                    top_n
                                ),

                            "days":
                                int(
                                    len(rdf)
                                ),

                            "avg_diff":
                                float(
                                    rdf[
                                        "avg_diff"
                                    ].mean()
                                ),

                            "median_daily_avg":
                                float(
                                    rdf[
                                        "avg_diff"
                                    ].median()
                                ),

                            "win_rate":
                                float(
                                    rdf[
                                        "win_rate"
                                    ].mean()
                                ),

                            "plus1000_rate":
                                float(
                                    rdf[
                                        "plus1000_rate"
                                    ].mean()
                                ),

                            "plus2000_rate":
                                float(
                                    rdf[
                                        "plus2000_rate"
                                    ].mean()
                                ),

                            "positive_days":
                                float(
                                    rdf[
                                        "positive"
                                    ].mean()
                                    * 100.0
                                ),

                            "total_diff":
                                float(
                                    rdf[
                                        "total_diff"
                                    ].sum()
                                ),
                        }
                    )

    daily_df = pd.DataFrame(
        daily_rows
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    if summary_df.empty:

        raise RuntimeError(
            "No OOS results."
        )

    overall_df = (
        summary_df
        .groupby(
            [
                "neighbor_mode",
                "model",
                "top_n",
            ],
            as_index=False,
        )
        .agg(
            avg_diff=(
                "avg_diff",
                "mean",
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
                "positive_days",
                "mean",
            ),

            total_diff=(
                "total_diff",
                "sum",
            ),

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
                        (
                            s > 0
                        ).mean()
                        * 100.0
                    ),
            ),
        )
    )

    # --------------------------------------------------------
    # Compare every mode with CURRENT_PM1
    # --------------------------------------------------------

    baseline = overall_df[
        overall_df[
            "neighbor_mode"
        ]
        == "CURRENT_PM1"
    ][
        [
            "model",
            "top_n",
            "avg_diff",
            "total_diff",
            "win_rate",
            "positive_days",
        ]
    ].rename(
        columns={
            "avg_diff":
                "baseline_avg_diff",

            "total_diff":
                "baseline_total_diff",

            "win_rate":
                "baseline_win_rate",

            "positive_days":
                "baseline_positive_days",
        }
    )

    comparison_df = (
        overall_df.merge(
            baseline,
            on=[
                "model",
                "top_n",
            ],
            how="left",
        )
    )

    comparison_df[
        "avg_diff_change_vs_current"
    ] = (
        comparison_df[
            "avg_diff"
        ]
        - comparison_df[
            "baseline_avg_diff"
        ]
    )

    comparison_df[
        "total_diff_change_vs_current"
    ] = (
        comparison_df[
            "total_diff"
        ]
        - comparison_df[
            "baseline_total_diff"
        ]
    )

    comparison_df[
        "win_rate_change_vs_current"
    ] = (
        comparison_df[
            "win_rate"
        ]
        - comparison_df[
            "baseline_win_rate"
        ]
    )

    # --------------------------------------------------------
    # Key candidates
    # --------------------------------------------------------

    key_mask = (
        (
            (
                comparison_df[
                    "model"
                ]
                == "V4.2_A"
            )
            & (
                comparison_df[
                    "top_n"
                ]
                == 10
            )
        )
        |
        (
            (
                comparison_df[
                    "model"
                ]
                == "V4.2_C"
            )
            & (
                comparison_df[
                    "top_n"
                ]
                == 5
            )
        )
        |
        (
            (
                comparison_df[
                    "model"
                ]
                == "V4.2_C"
            )
            & (
                comparison_df[
                    "top_n"
                ]
                == 10
            )
        )
    )

    key_df = (
        comparison_df[
            key_mask
        ]
        .copy()
        .sort_values(
            [
                "model",
                "top_n",
                "neighbor_mode",
            ]
        )
    )

    print_header(
        "KEY CANDIDATES"
    )

    print(
        key_df[
            [
                "neighbor_mode",
                "model",
                "top_n",
                "avg_diff",
                "total_diff",
                "win_rate",
                "positive_days",
                "min_split_avg",
                "positive_split_rate",
                "avg_diff_change_vs_current",
                "total_diff_change_vs_current",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Best mode per key candidate (exploratory)
    # --------------------------------------------------------

    best_rows = []

    for (
        model_name,
        top_n,
    ), group in key_df.groupby(
        [
            "model",
            "top_n",
        ]
    ):

        best = (
            group.sort_values(
                [
                    "total_diff",
                    "avg_diff",
                    "min_split_avg",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ]
            )
            .iloc[0]
        )

        best_rows.append(
            {
                "model":
                    model_name,

                "top_n":
                    int(
                        top_n
                    ),

                "best_neighbor_mode":
                    str(
                        best[
                            "neighbor_mode"
                        ]
                    ),

                "avg_diff":
                    float(
                        best[
                            "avg_diff"
                        ]
                    ),

                "total_diff":
                    float(
                        best[
                            "total_diff"
                        ]
                    ),

                "win_rate":
                    float(
                        best[
                            "win_rate"
                        ]
                    ),

                "positive_days":
                    float(
                        best[
                            "positive_days"
                        ]
                    ),

                "min_split_avg":
                    float(
                        best[
                            "min_split_avg"
                        ]
                    ),

                "positive_split_rate":
                    float(
                        best[
                            "positive_split_rate"
                        ]
                    ),
            }
        )

    best_df = pd.DataFrame(
        best_rows
    )

    print_header(
        "BEST NEIGHBOR MODE PER KEY CANDIDATE (EXPLORATORY)"
    )

    print(
        best_df.to_string(
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

    out_daily = (
        OUTPUT_DIR
        / "53_neighbor_ablation_daily.csv"
    )

    out_summary = (
        OUTPUT_DIR
        / "53_neighbor_ablation_summary.csv"
    )

    out_overall = (
        OUTPUT_DIR
        / "53_neighbor_ablation_overall.csv"
    )

    out_comparison = (
        OUTPUT_DIR
        / "53_neighbor_ablation_comparison_vs_current.csv"
    )

    out_key = (
        OUTPUT_DIR
        / "53_neighbor_ablation_key_candidates.csv"
    )

    out_best = (
        OUTPUT_DIR
        / "53_neighbor_ablation_best_exploratory.csv"
    )

    daily_df.to_csv(
        out_daily,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        out_summary,
        index=False,
        encoding="utf-8-sig",
    )

    overall_df.to_csv(
        out_overall,
        index=False,
        encoding="utf-8-sig",
    )

    comparison_df.to_csv(
        out_comparison,
        index=False,
        encoding="utf-8-sig",
    )

    key_df.to_csv(
        out_key,
        index=False,
        encoding="utf-8-sig",
    )

    best_df.to_csv(
        out_best,
        index=False,
        encoding="utf-8-sig",
    )

    print_header(
        "FILES SAVED"
    )

    for path in (
        out_daily,
        out_summary,
        out_overall,
        out_comparison,
        out_key,
        out_best,
    ):
        print(
            path
        )

    print()
    print(
        "Neighbor feature ablation / OOS test complete."
    )

    print(
        "Do not change the production model solely from this 39-day test."
    )


if __name__ == "__main__":
    main()
