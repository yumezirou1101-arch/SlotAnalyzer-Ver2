from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import re

import numpy as np
import pandas as pd


# ============================================================
# 63 - Champion / Challenger Forward Test
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

LOCKED_BASE_CSV = (
    DATA_DIR
    / "ana_slo_20260711_20260818.csv"
)

OUTPUT_DIR = (
    DATA_DIR
    / "analysis_31days_deep"
    / "63_Ver4_2_forward_champion_challenger"
)

DEVELOPMENT_END = pd.Timestamp("2026-08-18")
FORWARD_START = pd.Timestamp("2026-08-19")

EXPECTED_MACHINES_PER_DAY = 514
ALLOW_NONSTANDARD_MACHINE_COUNT = False

TOP_N = 10

# This is only a review threshold.
# The script NEVER promotes a challenger automatically.
MIN_FORWARD_DAYS_FOR_REVIEW = 21

BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260820


# ============================================================
# FIXED MODEL DEFINITIONS
# ============================================================

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
        k: v / total
        for k, v in weights.items()
    }


def make_models(
    base_weights: dict[str, float],
) -> dict[str, dict[str, float]]:

    champion = base_weights.copy()

    challenger_avg31 = (
        base_weights.copy()
    )
    challenger_avg31[
        "avg31"
    ] *= 0.50
    challenger_avg31 = (
        normalize_weights(
            challenger_avg31
        )
    )

    challenger_plus1000 = (
        base_weights.copy()
    )
    challenger_plus1000[
        "plus1000_rate"
    ] *= 1.50
    challenger_plus1000 = (
        normalize_weights(
            challenger_plus1000
        )
    )

    return {
        "CHAMPION_V4.2_C":
            champion,

        "CHALLENGER_AVG31_X0.50":
            challenger_avg31,

        "CHALLENGER_PLUS1000_X1.50":
            challenger_plus1000,
    }


# ============================================================
# GENERAL HELPERS
# ============================================================

def header(
    title: str,
) -> None:

    print()
    print("=" * 112)
    print(title)
    print("=" * 112)


def load_source_56():

    if not SOURCE_56.exists():
        raise FileNotFoundError(
            f"56 source script not found: {SOURCE_56}"
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            "slotanalyzer_56",
            SOURCE_56,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Could not import 56 source."
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


def canonicalize_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

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

    x = df.rename(
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
    ).copy()

    x["date"] = pd.to_datetime(
        x["date"],
        errors="coerce",
    )

    x["machine_no"] = (
        pd.to_numeric(
            x["machine_no"],
            errors="coerce",
        )
    )

    x["diff"] = (
        x["diff"]
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

    x["diff"] = pd.to_numeric(
        x["diff"],
        errors="coerce",
    )

    x["machine_name"] = (
        x["machine_name"]
        .astype(str)
        .str.strip()
    )

    x = x.dropna(
        subset=[
            "date",
            "machine_no",
            "machine_name",
            "diff",
        ]
    ).copy()

    x["machine_no"] = (
        x["machine_no"]
        .astype(int)
    )

    x["win"] = (
        x["diff"] > 0
    ).astype(int)

    x["plus1000"] = (
        x["diff"] >= 1000
    ).astype(int)

    x["plus2000"] = (
        x["diff"] >= 2000
    ).astype(int)

    return x


def parse_combined_filename(
    path: Path,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:

    match = re.fullmatch(
        r"ana_slo_(\d{8})_(\d{8})\.csv",
        path.name,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    start = pd.to_datetime(
        match.group(1),
        format="%Y%m%d",
        errors="coerce",
    )

    end = pd.to_datetime(
        match.group(2),
        format="%Y%m%d",
        errors="coerce",
    )

    if (
        pd.isna(start)
        or pd.isna(end)
    ):
        return None

    return (
        pd.Timestamp(start),
        pd.Timestamp(end),
    )


def parse_daily_filename(
    path: Path,
) -> pd.Timestamp | None:

    match = re.fullmatch(
        r"ana_slo_(\d{8})\.csv",
        path.name,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    date = pd.to_datetime(
        match.group(1),
        format="%Y%m%d",
        errors="coerce",
    )

    if pd.isna(date):
        return None

    return pd.Timestamp(date)


# ============================================================
# DATA DISCOVERY / ASSEMBLY
# ============================================================

def discover_primary_combined() -> Path:

    if not LOCKED_BASE_CSV.exists():
        raise FileNotFoundError(
            f"Locked development CSV not found: {LOCKED_BASE_CSV}"
        )

    candidates = []

    for path in DATA_DIR.glob(
        "ana_slo_????????_????????.csv"
    ):

        parsed = parse_combined_filename(
            path
        )

        if parsed is None:
            continue

        start, end = parsed

        # Require the full locked development start.
        if start > pd.Timestamp(
            "2026-07-11"
        ):
            continue

        # Must at least cover the locked development end.
        if end < DEVELOPMENT_END:
            continue

        candidates.append(
            (
                end,
                -int(
                    start.strftime(
                        "%Y%m%d"
                    )
                ),
                path,
            )
        )

    if not candidates:
        return LOCKED_BASE_CSV

    candidates.sort(
        key=lambda item:
            (
                item[0],
                item[1],
            ),
        reverse=True,
    )

    return candidates[0][2]


def assemble_dataset() -> tuple[
    pd.DataFrame,
    list[Path],
]:

    primary = (
        discover_primary_combined()
    )

    primary_raw = read_csv_flexible(
        primary
    )

    pieces = [
        canonicalize_data(
            primary_raw
        )
    ]

    sources = [
        primary
    ]

    primary_parsed = (
        parse_combined_filename(
            primary
        )
    )

    if primary_parsed is None:
        primary_end = (
            DEVELOPMENT_END
        )

    else:
        _, primary_end = (
            primary_parsed
        )

    # Append exact daily files later than the primary combined CSV.
    daily_candidates = []

    for path in DATA_DIR.glob(
        "ana_slo_????????.csv"
    ):

        date = parse_daily_filename(
            path
        )

        if date is None:
            continue

        if date <= primary_end:
            continue

        daily_candidates.append(
            (
                date,
                path,
            )
        )

    daily_candidates.sort(
        key=lambda item:
            item[0]
    )

    for _, path in daily_candidates:

        daily_raw = (
            read_csv_flexible(
                path
            )
        )

        pieces.append(
            canonicalize_data(
                daily_raw
            )
        )

        sources.append(
            path
        )

    df = pd.concat(
        pieces,
        ignore_index=True,
    )

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

    return (
        df,
        sources,
    )


# ============================================================
# QUALITY CHECK
# ============================================================

def build_quality_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for date, group in (
        df.groupby(
            "date",
            sort=True,
        )
    ):

        rows.append(
            {
                "date":
                    date,

                "rows":
                    int(
                        len(group)
                    ),

                "machines":
                    int(
                        group[
                            "machine_no"
                        ].nunique()
                    ),

                "duplicates":
                    int(
                        group.duplicated(
                            subset=[
                                "machine_no"
                            ]
                        ).sum()
                    ),

                "missing_name":
                    int(
                        group[
                            "machine_name"
                        ].isna().sum()
                    ),

                "missing_diff":
                    int(
                        group[
                            "diff"
                        ].isna().sum()
                    ),
            }
        )

    q = pd.DataFrame(
        rows
    )

    if q.empty:
        return q

    q["machine_count_ok"] = (
        q["machines"]
        == EXPECTED_MACHINES_PER_DAY
    )

    q["basic_ok"] = (
        (
            q["duplicates"] == 0
        )
        & (
            q["missing_name"] == 0
        )
        & (
            q["missing_diff"] == 0
        )
    )

    q["eligible"] = (
        q["basic_ok"]
        & (
            q["machine_count_ok"]
            | ALLOW_NONSTANDARD_MACHINE_COUNT
        )
    )

    return q


# ============================================================
# MODEL EVALUATION
# ============================================================

def rank_panel(
    m56,
    panel: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:

    return m56.rank_score(
        panel,
        weights,
    )


def evaluate_ranked(
    ranked: pd.DataFrame,
) -> dict:

    selected = ranked.head(
        TOP_N
    ).copy()

    diffs = pd.to_numeric(
        selected["diff"],
        errors="coerce",
    ).dropna()

    selected_nos = tuple(
        int(x)
        for x in selected[
            "machine_no"
        ].tolist()
    )

    selected_names = tuple(
        str(x)
        for x in selected[
            "machine_name"
        ].tolist()
    )

    return {
        "avg_diff":
            float(
                diffs.mean()
            ),

        "median_diff":
            float(
                diffs.median()
            ),

        "win_rate":
            float(
                (
                    diffs > 0
                ).mean()
                * 100.0
            ),

        "plus1000_rate":
            float(
                (
                    diffs >= 1000
                ).mean()
                * 100.0
            ),

        "plus2000_rate":
            float(
                (
                    diffs >= 2000
                ).mean()
                * 100.0
            ),

        "positive":
            int(
                diffs.sum() > 0
            ),

        "total_diff":
            float(
                diffs.sum()
            ),

        "selected_nos":
            selected_nos,

        "selected_names":
            selected_names,
    }


def bootstrap_mean_ci(
    values: np.ndarray,
) -> tuple[
    float,
    float,
]:

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) < 2:
        return (
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    samples = rng.choice(
        x,
        size=(
            BOOTSTRAP_REPS,
            len(x),
        ),
        replace=True,
    )

    means = samples.mean(
        axis=1
    )

    return (
        float(
            np.percentile(
                means,
                2.5,
            )
        ),
        float(
            np.percentile(
                means,
                97.5,
            )
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    header(
        "63 - Champion / Challenger Forward Test"
    )

    m56 = load_source_56()

    models = make_models(
        m56.V42_C_WEIGHTS.copy()
    )

    df, sources = assemble_dataset()

    print(
        f"records              : {len(df):,}"
    )
    print(
        f"days                 : {df['date'].nunique()}"
    )
    print(
        f"date range           : "
        f"{df['date'].min().date()} "
        f"to "
        f"{df['date'].max().date()}"
    )
    print(
        f"development end      : {DEVELOPMENT_END.date()}"
    )
    print(
        f"forward start        : {FORWARD_START.date()}"
    )
    print(
        f"expected machines/day: {EXPECTED_MACHINES_PER_DAY}"
    )

    print()
    print(
        "Data sources:"
    )

    for path in sources:
        print(
            f"  {path}"
        )

    print()
    print(
        "Fixed models:"
    )

    for name, weights in models.items():

        fingerprint_text = "|".join(
            f"{k}:{weights[k]:.15f}"
            for k in sorted(
                weights
            )
        )

        fingerprint = (
            hashlib.sha256(
                fingerprint_text.encode(
                    "utf-8"
                )
            )
            .hexdigest()[:16]
        )

        print(
            f"  {name:<30} "
            f"sum={sum(weights.values()):.12f} "
            f"fingerprint={fingerprint}"
        )

    # --------------------------------------------------------
    # Data quality
    # --------------------------------------------------------

    quality_df = (
        build_quality_table(
            df
        )
    )

    forward_quality = (
        quality_df[
            quality_df["date"]
            >= FORWARD_START
        ].copy()
    )

    header(
        "FORWARD DATA QUALITY"
    )

    if forward_quality.empty:

        print(
            "No forward dates are available yet."
        )

    else:

        print(
            forward_quality.to_string(
                index=False
            )
        )

    eligible_dates = set(
        forward_quality.loc[
            forward_quality[
                "eligible"
            ],
            "date",
        ].tolist()
    )

    all_forward_dates = sorted(
        d
        for d in df["date"].unique()
        if pd.Timestamp(d)
        >= FORWARD_START
    )

    skipped_dates = [
        pd.Timestamp(d)
        for d in all_forward_dates
        if pd.Timestamp(d)
        not in eligible_dates
    ]

    if skipped_dates:

        print()
        print(
            "Skipped forward dates because quality check failed:"
        )

        for date in skipped_dates:
            print(
                f"  {date.date()}"
            )

    # --------------------------------------------------------
    # No forward data yet: create status files and exit cleanly.
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not eligible_dates:

        status_df = pd.DataFrame(
            [
                {
                    "status":
                        "WAITING_FOR_FORWARD_DATA",

                    "development_end":
                        DEVELOPMENT_END.date(),

                    "forward_start":
                        FORWARD_START.date(),

                    "available_forward_days":
                        0,

                    "min_review_days":
                        MIN_FORWARD_DAYS_FOR_REVIEW,

                    "production_model":
                        "CHAMPION_V4.2_C",
                }
            ]
        )

        quality_path = (
            OUTPUT_DIR
            / "63_forward_data_quality.csv"
        )

        status_path = (
            OUTPUT_DIR
            / "63_forward_status.csv"
        )

        quality_df.to_csv(
            quality_path,
            index=False,
            encoding="utf-8-sig",
        )

        status_df.to_csv(
            status_path,
            index=False,
            encoding="utf-8-sig",
        )

        header(
            "STATUS"
        )

        print(
            "WAITING_FOR_FORWARD_DATA"
        )
        print(
            "Add 2026-08-19 or later Ana-Slo daily CSV data, then run 63 again."
        )
        print()
        print(
            quality_path
        )
        print(
            status_path
        )

        return

    # --------------------------------------------------------
    # Build stable number-run map from all currently known machine numbers.
    # It is used only by the inherited feature builder.
    # --------------------------------------------------------

    edge_distance_map = (
        m56.build_number_edge_distance(
            df["machine_no"].tolist()
        )
    )

    daily_rows = []
    pick_rows = []

    header(
        "FORWARD EVALUATION"
    )

    for target_date in sorted(
        eligible_dates
    ):

        target_date = pd.Timestamp(
            target_date
        )

        panel = m56.build_features(
            df,
            target_date,
            edge_distance_map,
        )

        if panel.empty:

            print(
                f"{target_date.date()} "
                f"panel=EMPTY -> skipped"
            )

            continue

        print(
            f"{target_date.date()} "
            f"panel={len(panel)}"
        )

        for model_name, weights in (
            models.items()
        ):

            ranked = rank_panel(
                m56,
                panel,
                weights,
            )

            result = evaluate_ranked(
                ranked
            )

            result.update(
                {
                    "date":
                        target_date,

                    "model":
                        model_name,

                    "panel_machines":
                        int(
                            len(panel)
                        ),

                    "development_locked_through":
                        DEVELOPMENT_END,

                    "forward_start":
                        FORWARD_START,
                }
            )

            daily_rows.append(
                result
            )

            top = ranked.head(
                TOP_N
            ).copy()

            for rank, row in enumerate(
                top.itertuples(
                    index=False
                ),
                start=1,
            ):

                pick_rows.append(
                    {
                        "date":
                            target_date,

                        "model":
                            model_name,

                        "rank":
                            rank,

                        "machine_no":
                            int(
                                row.machine_no
                            ),

                        "machine_name":
                            str(
                                row.machine_name
                            ),

                        "score":
                            float(
                                row.score
                            ),

                        "actual_diff":
                            float(
                                row.diff
                            ),
                    }
                )

    daily_df = pd.DataFrame(
        daily_rows
    )

    picks_df = pd.DataFrame(
        pick_rows
    )

    if daily_df.empty:
        raise RuntimeError(
            "Forward dates existed but no evaluation rows were generated."
        )

    # --------------------------------------------------------
    # Overall model summary
    # --------------------------------------------------------

    overall_df = (
        daily_df.groupby(
            "model",
            as_index=False,
        )
        .agg(
            forward_days=(
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

            plus1000_rate=(
                "plus1000_rate",
                "mean",
            ),

            plus2000_rate=(
                "plus2000_rate",
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

    # --------------------------------------------------------
    # Paired challenger vs champion
    # --------------------------------------------------------

    champion_daily = daily_df[
        daily_df["model"]
        == "CHAMPION_V4.2_C"
    ][
        [
            "date",
            "total_diff",
            "selected_nos",
        ]
    ].rename(
        columns={
            "total_diff":
                "champion_total_diff",

            "selected_nos":
                "champion_selected_nos",
        }
    )

    pair_frames = []

    for challenger in (
        "CHALLENGER_AVG31_X0.50",
        "CHALLENGER_PLUS1000_X1.50",
    ):

        ch = daily_df[
            daily_df["model"]
            == challenger
        ][
            [
                "date",
                "total_diff",
                "selected_nos",
            ]
        ].rename(
            columns={
                "total_diff":
                    "challenger_total_diff",

                "selected_nos":
                    "challenger_selected_nos",
            }
        )

        pair = champion_daily.merge(
            ch,
            on="date",
            how="inner",
            validate="one_to_one",
        )

        pair[
            "challenger"
        ] = challenger

        pair[
            "change_vs_champion"
        ] = (
            pair[
                "challenger_total_diff"
            ]
            - pair[
                "champion_total_diff"
            ]
        )

        overlap = []

        for _, row in (
            pair.iterrows()
        ):

            champion_set = set(
                row[
                    "champion_selected_nos"
                ]
            )

            challenger_set = set(
                row[
                    "challenger_selected_nos"
                ]
            )

            overlap.append(
                len(
                    champion_set
                    & challenger_set
                )
            )

        pair[
            "top10_overlap"
        ] = overlap

        pair[
            "changed_slots"
        ] = (
            TOP_N
            - pair[
                "top10_overlap"
            ]
        )

        pair_frames.append(
            pair
        )

    paired_df = pd.concat(
        pair_frames,
        ignore_index=True,
    )

    summary_rows = []

    for challenger, group in (
        paired_df.groupby(
            "challenger"
        )
    ):

        changes = (
            group[
                "change_vs_champion"
            ]
            .astype(float)
        )

        ci_low, ci_high = (
            bootstrap_mean_ci(
                changes.to_numpy()
            )
        )

        better = int(
            (
                changes > 0
            ).sum()
        )

        same = int(
            (
                changes == 0
            ).sum()
        )

        worse = int(
            (
                changes < 0
            ).sum()
        )

        days = int(
            len(group)
        )

        if (
            days
            < MIN_FORWARD_DAYS_FOR_REVIEW
        ):
            status = (
                "ACCUMULATING_FORWARD_DATA"
            )

        elif (
            changes.sum() > 0
            and better > worse
            and np.isfinite(
                ci_low
            )
            and ci_low > 0
        ):
            status = (
                "STRONG_REVIEW_CANDIDATE"
            )

        elif (
            changes.sum() > 0
        ):
            status = (
                "POSITIVE_BUT_UNCONFIRMED"
            )

        else:
            status = (
                "NO_FORWARD_ADVANTAGE"
            )

        summary_rows.append(
            {
                "challenger":
                    challenger,

                "forward_days":
                    days,

                "better_days":
                    better,

                "same_days":
                    same,

                "worse_days":
                    worse,

                "total_change_vs_champion":
                    float(
                        changes.sum()
                    ),

                "mean_daily_change":
                    float(
                        changes.mean()
                    ),

                "median_daily_change":
                    float(
                        changes.median()
                    ),

                "mean_change_bootstrap_ci95_low":
                    ci_low,

                "mean_change_bootstrap_ci95_high":
                    ci_high,

                "mean_top10_overlap":
                    float(
                        group[
                            "top10_overlap"
                        ].mean()
                    ),

                "mean_changed_slots":
                    float(
                        group[
                            "changed_slots"
                        ].mean()
                    ),

                "review_status":
                    status,

                "production_promoted":
                    False,
            }
        )

    challenger_summary_df = (
        pd.DataFrame(
            summary_rows
        )
    )

    # --------------------------------------------------------
    # Forward status
    # --------------------------------------------------------

    forward_days = int(
        daily_df[
            "date"
        ].nunique()
    )

    if (
        forward_days
        < MIN_FORWARD_DAYS_FOR_REVIEW
    ):
        overall_status = (
            "ACCUMULATING_FORWARD_DATA"
        )
    else:
        overall_status = (
            "READY_FOR_MANUAL_MODEL_REVIEW"
        )

    status_df = pd.DataFrame(
        [
            {
                "status":
                    overall_status,

                "development_end":
                    DEVELOPMENT_END.date(),

                "forward_start":
                    FORWARD_START.date(),

                "available_forward_days":
                    forward_days,

                "min_review_days":
                    MIN_FORWARD_DAYS_FOR_REVIEW,

                "current_production_model":
                    "CHAMPION_V4.2_C",

                "automatic_promotion":
                    False,
            }
        ]
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    header(
        "FORWARD OVERALL RESULTS"
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
        "CHALLENGER VS CHAMPION"
    )

    print(
        challenger_summary_df.to_string(
            index=False
        )
    )

    header(
        "FORWARD STATUS"
    )

    print(
        status_df.to_string(
            index=False
        )
    )

    print()
    print(
        "No model is promoted automatically."
    )
    print(
        "The development period through 2026-08-18 remains locked."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    quality_path = (
        OUTPUT_DIR
        / "63_forward_data_quality.csv"
    )

    daily_path = (
        OUTPUT_DIR
        / "63_forward_daily_results.csv"
    )

    picks_path = (
        OUTPUT_DIR
        / "63_forward_top10_picks.csv"
    )

    overall_path = (
        OUTPUT_DIR
        / "63_forward_overall.csv"
    )

    paired_path = (
        OUTPUT_DIR
        / "63_forward_challenger_vs_champion_daily.csv"
    )

    challenger_summary_path = (
        OUTPUT_DIR
        / "63_forward_challenger_summary.csv"
    )

    status_path = (
        OUTPUT_DIR
        / "63_forward_status.csv"
    )

    quality_df.to_csv(
        quality_path,
        index=False,
        encoding="utf-8-sig",
    )

    daily_df.to_csv(
        daily_path,
        index=False,
        encoding="utf-8-sig",
    )

    picks_df.to_csv(
        picks_path,
        index=False,
        encoding="utf-8-sig",
    )

    overall_df.to_csv(
        overall_path,
        index=False,
        encoding="utf-8-sig",
    )

    paired_df.to_csv(
        paired_path,
        index=False,
        encoding="utf-8-sig",
    )

    challenger_summary_df.to_csv(
        challenger_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    status_df.to_csv(
        status_path,
        index=False,
        encoding="utf-8-sig",
    )

    header(
        "FILES SAVED"
    )

    for path in (
        quality_path,
        daily_path,
        picks_path,
        overall_path,
        paired_path,
        challenger_summary_path,
        status_path,
    ):

        print(path)

    print()
    print(
        "63 Champion / Challenger forward test complete."
    )


if __name__ == "__main__":
    main()
