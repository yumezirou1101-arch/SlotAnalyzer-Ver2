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
    / "56_Ver4_2_machine_number_position_ablation_oos"
)

START = pd.Timestamp("2026-07-11")
END = pd.Timestamp("2026-08-18")

TOP_N = 10

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
# EXACT EXISTING V4 / V4.2 WEIGHTS
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

V42_C_RAW = V4_WEIGHTS.copy()
V42_C_RAW.pop("recent7_win")
V42_C_RAW.pop("bounce_signal")


def normalize_weights(
    weights: dict[str, float],
) -> dict[str, float]:
    total = sum(weights.values())

    if total <= 0:
        raise ValueError(
            "Weight sum must be positive."
        )

    return {
        k: v / total
        for k, v in weights.items()
    }


V42_C_WEIGHTS = normalize_weights(
    V42_C_RAW
)

# Experimental feature weight.
# Base V4.2_C weights are proportionally reduced so the total remains 1.0.
EXTRA_WEIGHT = 0.10

MODES = {
    "BASE_V4.2_C": None,
    "ADD_MACHINE_HISTORY": "machine_history",
    "ADD_NUMBER_BAND_10": "number_band_10",
    "ADD_NUMBER_BAND_50": "number_band_50",
    "ADD_NUMBER_RUN_EDGE": "number_run_edge",
}

EXPECTED_BASE_TOTAL_DIFF = 130400.0


# ============================================================
# HELPERS
# ============================================================

def print_header(
    title: str,
) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


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


def zscore(
    series: pd.Series,
) -> pd.Series:
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


def build_number_edge_distance(
    machine_nos: list[int],
) -> dict[int, int]:
    nos = sorted(
        set(
            int(x)
            for x in machine_nos
        )
    )

    runs: list[list[int]] = []
    current: list[int] = []

    for no in nos:
        if (
            not current
            or no == current[-1] + 1
        ):
            current.append(no)
        else:
            runs.append(current)
            current = [no]

    if current:
        runs.append(current)

    distance_map: dict[int, int] = {}

    for run in runs:
        run_len = len(run)

        for pos, no in enumerate(
            run
        ):
            left_distance = pos
            right_distance = (
                run_len
                - pos
                - 1
            )

            distance_map[no] = min(
                left_distance,
                right_distance,
            )

    return distance_map


def edge_score(
    distance: int,
) -> float:
    # Diagnostic-only numeric encoding.
    # This is NOT a physical-island corner claim.
    if distance == 0:
        return 1.00
    if distance == 1:
        return 0.75
    if distance == 2:
        return 0.50
    if distance <= 4:
        return 0.25
    return 0.00


def make_weights(
    extra_factor: str | None,
) -> dict[str, float]:
    if extra_factor is None:
        return V42_C_WEIGHTS.copy()

    scale = (
        1.0
        - EXTRA_WEIGHT
    )

    weights = {
        k: v * scale
        for k, v in V42_C_WEIGHTS.items()
    }

    weights[extra_factor] = (
        EXTRA_WEIGHT
    )

    return weights


# ============================================================
# DATA
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
            f"no={no_col}, "
            f"name={name_col}, "
            f"diff={diff_col}"
        )

    df = df.rename(
        columns={
            date_col: "date",
            no_col: "machine_no",
            name_col: "machine_name",
            diff_col: "diff",
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
# FEATURE CONSTRUCTION
# ============================================================

def build_features(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
    edge_distance_map: dict[int, int],
) -> pd.DataFrame:
    """
    Baseline feature construction intentionally mirrors the existing
    48-series rolling script.

    Every experimental historical statistic is based only on:
        df["date"] < target_date

    No future target-day diff is used to create a feature.
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

    if (
        hist.empty
        or actual.empty
    ):
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

    store_avg = float(
        hist["diff"].mean()
    )

    hist_with_bands = (
        hist.copy()
    )

    hist_with_bands[
        "number_band_10_key"
    ] = (
        hist_with_bands[
            "machine_no"
        ]
        // 10
        * 10
    )

    hist_with_bands[
        "number_band_50_key"
    ] = (
        hist_with_bands[
            "machine_no"
        ]
        // 50
        * 50
    )

    band10_stats = (
        hist_with_bands.groupby(
            "number_band_10_key"
        )["diff"]
        .mean()
        .to_dict()
    )

    band50_stats = (
        hist_with_bands.groupby(
            "number_band_50_key"
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
        )

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

        weekday_n = len(wd)

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

        neighbor_values = []

        for n2 in (
            int(no) - 1,
            int(no) + 1,
        ):
            if n2 in latest_day.index:
                value = latest_day.loc[
                    n2,
                    "diff",
                ]

                if isinstance(
                    value,
                    pd.Series,
                ):
                    value = (
                        value.iloc[-1]
                    )

                neighbor_values.append(
                    float(value)
                )

        if neighbor_values:
            neighbor_avg = float(
                np.mean(
                    neighbor_values
                )
            )
        else:
            neighbor_avg = 0.0

        if last_diff <= -1000:
            bounce_signal = 1.0
        elif last_diff <= -500:
            bounce_signal = 0.5
        elif last_diff >= 1000:
            bounce_signal = -0.25
        else:
            bounce_signal = 0.0

        # ----------------------------------------------------
        # Experimental, leakage-safe position/history features
        # ----------------------------------------------------

        machine_prior_n = 30.0

        machine_history = float(
            (
                m["diff"].sum()
                + store_avg
                * machine_prior_n
            )
            / (
                len(m)
                + machine_prior_n
            )
        )

        band10_key = (
            int(no)
            // 10
            * 10
        )

        band50_key = (
            int(no)
            // 50
            * 50
        )

        number_band_10 = float(
            band10_stats.get(
                band10_key,
                store_avg,
            )
        )

        number_band_50 = float(
            band50_stats.get(
                band50_key,
                store_avg,
            )
        )

        distance = int(
            edge_distance_map.get(
                int(no),
                999,
            )
        )

        number_run_edge = float(
            edge_score(
                distance
            )
        )

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

                "bounce_signal":
                    bounce_signal,

                "machine_history":
                    machine_history,

                "number_band_10":
                    number_band_10,

                "number_band_50":
                    number_band_50,

                "number_run_edge":
                    number_run_edge,
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
# SCORING
# ============================================================

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
            raise RuntimeError(
                f"Missing factor: {factor}"
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

    return x.sort_values(
        "score",
        ascending=False,
    )


def evaluate_day(
    panel: pd.DataFrame,
    weights: dict[str, float],
) -> dict | None:
    if panel.empty:
        return None

    ranked = rank_score(
        panel,
        weights,
    )

    top = ranked.head(
        min(
            TOP_N,
            len(ranked),
        )
    )

    d = (
        top["diff"]
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
                * 100
            ),

        "plus1000_rate":
            float(
                (
                    d >= 1000
                ).mean()
                * 100
            ),

        "plus2000_rate":
            float(
                (
                    d >= 2000
                ).mean()
                * 100
            ),

        "positive":
            int(
                d.sum() > 0
            ),

        "total_diff":
            float(
                d.sum()
            ),

        "machines":
            int(
                len(panel)
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print_header(
        "56 - V4.2_C TOP10 Machine-Number / Position Rolling OOS Ablation"
    )

    print(
        "Exact V4.2_C normalized weights:"
    )

    for key, value in (
        V42_C_WEIGHTS.items()
    ):
        print(
            f"{key:<20} "
            f"{value:.12f}"
        )

    print(
        f"weight_sum           : "
        f"{sum(V42_C_WEIGHTS.values()):.12f}"
    )

    df = load_data()

    print()
    print(
        f"records              : "
        f"{len(df):,}"
    )
    print(
        f"days                 : "
        f"{df['date'].nunique()}"
    )
    print(
        f"machines             : "
        f"{df['machine_no'].nunique()}"
    )

    edge_distance_map = (
        build_number_edge_distance(
            df[
                "machine_no"
            ].tolist()
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
            ) in ROLLING_SPLITS
            for d in pd.date_range(
                test_start,
                test_end,
            )
        }
    )

    panels: dict[
        pd.Timestamp,
        pd.DataFrame,
    ] = {}

    print_header(
        "BUILDING PANELS"
    )

    for target_date in all_test_dates:
        panel = build_features(
            df,
            target_date,
            edge_distance_map,
        )

        panels[
            target_date
        ] = panel

        print(
            f"{target_date.date()} "
            f"machines={len(panel)}"
        )

    daily_rows = []

    print_header(
        "ROLLING OOS"
    )

    for (
        split_name,
        train_start,
        train_end,
        test_start,
        test_end,
    ) in ROLLING_SPLITS:
        print(
            f"{split_name}: "
            f"TEST {test_start.date()} "
            f"to {test_end.date()}"
        )

        for mode, extra_factor in (
            MODES.items()
        ):
            weights = make_weights(
                extra_factor
            )

            for target_date in (
                pd.date_range(
                    test_start,
                    test_end,
                )
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
                        "split":
                            split_name,

                        "mode":
                            mode,

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

    summary_df = (
        daily_df.groupby(
            [
                "mode",
                "split",
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

            win_rate=(
                "win_rate",
                "mean",
            ),

            positive_days=(
                "positive",
                "mean",
            ),

            total_diff=(
                "total_diff",
                "sum",
            ),
        )
    )

    summary_df[
        "positive_days"
    ] *= 100.0

    overall_df = (
        daily_df.groupby(
            [
                "mode",
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

            win_rate=(
                "win_rate",
                "mean",
            ),

            positive_days=(
                "positive",
                "mean",
            ),

            total_diff=(
                "total_diff",
                "sum",
            ),
        )
    )

    overall_df[
        "positive_days"
    ] *= 100.0

    split_metrics = (
        summary_df.groupby(
            "mode",
            as_index=False,
        )
        .agg(
            min_split_avg=(
                "avg_diff",
                "min",
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

    overall_df = (
        overall_df.merge(
            split_metrics,
            on="mode",
            how="left",
        )
    )

    base_row = overall_df[
        overall_df["mode"]
        == "BASE_V4.2_C"
    ]

    if base_row.empty:
        raise RuntimeError(
            "BASE_V4.2_C result missing."
        )

    base_total = float(
        base_row.iloc[0][
            "total_diff"
        ]
    )

    print_header(
        "BASELINE SAFETY CHECK"
    )

    print(
        f"56 baseline total_diff : "
        f"{base_total:+.1f}"
    )
    print(
        f"expected 48 total_diff : "
        f"{EXPECTED_BASE_TOTAL_DIFF:+.1f}"
    )

    baseline_ok = bool(
        np.isclose(
            base_total,
            EXPECTED_BASE_TOTAL_DIFF,
            atol=0.01,
        )
    )

    print(
        f"baseline match        : "
        f"{baseline_ok}"
    )

    if not baseline_ok:
        raise RuntimeError(
            "BASELINE MISMATCH: "
            "56 baseline does not reproduce 48 V4.2_C TOP10. "
            "Do not interpret experimental results."
        )

    overall_df[
        "avg_diff_change_vs_base"
    ] = (
        overall_df[
            "avg_diff"
        ]
        - float(
            base_row.iloc[0][
                "avg_diff"
            ]
        )
    )

    overall_df[
        "total_diff_change_vs_base"
    ] = (
        overall_df[
            "total_diff"
        ]
        - base_total
    )

    # --------------------------------------------------------
    # Paired daily comparison
    # --------------------------------------------------------

    base_daily = daily_df[
        daily_df["mode"]
        == "BASE_V4.2_C"
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

    pair_frames = []

    for mode in MODES:
        if mode == "BASE_V4.2_C":
            continue

        mode_daily = daily_df[
            daily_df["mode"]
            == mode
        ][
            [
                "date",
                "split",
                "total_diff",
            ]
        ].rename(
            columns={
                "total_diff":
                    "mode_total_diff",
            }
        )

        pair = base_daily.merge(
            mode_daily,
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
                "mode_total_diff"
            ]
            - pair[
                "base_total_diff"
            ]
        )

        pair_frames.append(
            pair
        )

    paired_df = pd.concat(
        pair_frames,
        ignore_index=True,
    )

    paired_summary_df = (
        paired_df.groupby(
            "mode",
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
                    int(
                        (
                            s > 0
                        ).sum()
                    ),
            ),

            same_days=(
                "change_vs_base",
                lambda s:
                    int(
                        (
                            s == 0
                        ).sum()
                    ),
            ),

            worse_days=(
                "change_vs_base",
                lambda s:
                    int(
                        (
                            s < 0
                        ).sum()
                    ),
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

    # --------------------------------------------------------
    # Split improvement counts
    # --------------------------------------------------------

    base_split = summary_df[
        summary_df["mode"]
        == "BASE_V4.2_C"
    ][
        [
            "split",
            "avg_diff",
        ]
    ].rename(
        columns={
            "avg_diff":
                "base_split_avg",
        }
    )

    assessment_rows = []

    for mode in MODES:
        if mode == "BASE_V4.2_C":
            continue

        overall_row = overall_df[
            overall_df["mode"]
            == mode
        ].iloc[0]

        pair_row = paired_summary_df[
            paired_summary_df["mode"]
            == mode
        ].iloc[0]

        mode_split = summary_df[
            summary_df["mode"]
            == mode
        ][
            [
                "split",
                "avg_diff",
            ]
        ].merge(
            base_split,
            on="split",
            how="inner",
        )

        improved_splits = int(
            (
                mode_split[
                    "avg_diff"
                ]
                > mode_split[
                    "base_split_avg"
                ]
            ).sum()
        )

        if (
            overall_row[
                "total_diff_change_vs_base"
            ] > 0
            and pair_row[
                "median_daily_change"
            ] >= 0
            and pair_row[
                "better_days"
            ] >= pair_row[
                "worse_days"
            ]
            and improved_splits >= 4
        ):
            status = (
                "PROMISING_REQUIRE_ROBUSTNESS_TEST"
            )

        elif (
            overall_row[
                "total_diff_change_vs_base"
            ] > 0
        ):
            status = (
                "POSITIVE_BUT_UNSTABLE"
            )

        else:
            status = (
                "NO_OOS_IMPROVEMENT"
            )

        assessment_rows.append(
            {
                "mode":
                    mode,

                "status":
                    status,

                "total_diff_change_vs_base":
                    float(
                        overall_row[
                            "total_diff_change_vs_base"
                        ]
                    ),

                "avg_diff_change_vs_base":
                    float(
                        overall_row[
                            "avg_diff_change_vs_base"
                        ]
                    ),

                "better_days":
                    int(
                        pair_row[
                            "better_days"
                        ]
                    ),

                "same_days":
                    int(
                        pair_row[
                            "same_days"
                        ]
                    ),

                "worse_days":
                    int(
                        pair_row[
                            "worse_days"
                        ]
                    ),

                "median_daily_change":
                    float(
                        pair_row[
                            "median_daily_change"
                        ]
                    ),

                "improved_splits":
                    improved_splits,

                "production_adopted":
                    False,
            }
        )

    assessment_df = pd.DataFrame(
        assessment_rows
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print_header(
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

    print_header(
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

    print_header(
        "BY SPLIT"
    )

    print(
        summary_df.sort_values(
            [
                "mode",
                "split",
            ]
        ).to_string(
            index=False
        )
    )

    print_header(
        "ASSESSMENT"
    )

    print(
        assessment_df.to_string(
            index=False
        )
    )

    print()
    print(
        "No production feature is adopted by this script."
    )
    print(
        "Only PROMISING candidates should advance to 57 robustness testing."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "56_position_ablation_daily.csv":
            daily_df,

        "56_position_ablation_summary.csv":
            summary_df,

        "56_position_ablation_overall.csv":
            overall_df,

        "56_position_ablation_paired_daily.csv":
            paired_df,

        "56_position_ablation_paired_summary.csv":
            paired_summary_df,

        "56_position_ablation_assessment.csv":
            assessment_df,
    }

    print_header(
        "FILES SAVED"
    )

    for filename, frame in (
        outputs.items()
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

        print(path)

    print()
    print(
        "56 machine-number / position rolling OOS ablation complete."
    )


if __name__ == "__main__":
    main()
