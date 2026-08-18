# -*- coding: utf-8 -*-

"""
Ana-Slo Prediction Ver.3
Weight Optimization Backtest

Purpose:
- Optimize weights of the 11 Ver.3 factors.
- Use only data before each target date.
- Evaluate TOP1/TOP5/TOP10/TOP20/TOP30.
- Compare against all-machine average.
- Keep weights non-negative and sum to 1.0.

This is an exploratory optimization.
It should not be treated as a final model because the backtest
period is only 16 days.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")

DATA_DIR = (
    BASE
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
)

OUT_DIR = (
    DATA_DIR
    / "analysis_31days_deep"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_0711 = DATA_DIR / "ana_slo_20260711.csv"
CSV_0712_0810 = DATA_DIR / "ana_slo_20260712_20260810.csv"


OUTPUT_FILE = (
    OUT_DIR
    / "09_Ver3_weight_optimization_results.csv"
)

BEST_FILE = (
    OUT_DIR
    / "09_Ver3_weight_optimization_best.txt"
)


# ============================================================
# BACKTEST SETTINGS
# ============================================================

BT_START = pd.Timestamp("2026-07-26")
BT_END = pd.Timestamp("2026-08-10")

RANDOM_PATTERNS = 3000

TOP_N_LIST = [
    1,
    5,
    10,
    20,
    30,
]


# ============================================================
# FACTORS
# ============================================================

FACTORS = [
    "avg31",
    "recent7_avg",
    "recent7_win",
    "last_diff",
    "prev_change",
    "weekday_avg",
    "type_avg",
    "plus1000_rate",
    "plus2000_rate",
    "neighbor_avg",
    "bounce_signal",
]


# Current Ver.3 weights
BASE_WEIGHTS = {
    "avg31": 0.18,
    "recent7_avg": 0.18,
    "recent7_win": 0.08,
    "last_diff": 0.08,
    "prev_change": 0.07,
    "weekday_avg": 0.08,
    "type_avg": 0.08,
    "plus1000_rate": 0.04,
    "plus2000_rate": 0.04,
    "neighbor_avg": 0.04,
    "bounce_signal": 0.03,
}


# ============================================================
# CSV LOADING
# ============================================================

def read_csv(path):
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
        f"Cannot read CSV: {path}"
    )


def load_data():

    frames = []

    for path in (
        CSV_0711,
        CSV_0712_0810,
    ):
        if path.exists():
            frames.append(
                read_csv(path)
            )

    if not frames:
        raise FileNotFoundError(
            "Input CSV files were not found."
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    def find_column(candidates):

        for col in candidates:
            if col in df.columns:
                return col

        return None

    date_col = find_column(
        [
            "date",
            "日付",
        ]
    )

    no_col = find_column(
        [
            "machine_no",
            "台番号",
        ]
    )

    name_col = find_column(
        [
            "machine_name",
            "機種名",
        ]
    )

    diff_col = find_column(
        [
            "diff",
            "差枚",
        ]
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
            "Required columns were not found."
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

    df["machine_name"] = (
        df["machine_name"]
        .astype(str)
        .str.strip()
    )

    df = df.sort_values(
        [
            "date",
            "machine_no",
        ]
    )

    df = df.drop_duplicates(
        [
            "date",
            "machine_no",
        ],
        keep="last",
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

    return df.reset_index(
        drop=True
    )


# ============================================================
# STANDARDIZATION
# ============================================================

def zscore_series(series):

    s = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)

    std = float(
        s.std(
            ddof=0
        )
    )

    if std == 0:
        return pd.Series(
            0.0,
            index=s.index,
        )

    return (
        (s - s.mean())
        / std
    )


# ============================================================
# FEATURE BUILDING
# ============================================================

def build_features(
    df,
    target_date,
):

    hist = df[
        df["date"] < target_date
    ].copy()

    actual = df[
        df["date"] == target_date
    ][
        [
            "machine_no",
            "diff",
        ]
    ].copy()

    if hist.empty:
        return pd.DataFrame()

    if actual.empty:
        return pd.DataFrame()

    target_weekday = (
        target_date.dayofweek
    )

    latest_date = hist["date"].max()

    latest_day = (
        hist[
            hist["date"] == latest_date
        ]
        .set_index("machine_no")
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
            m.iloc[-1]["machine_name"]
        )

        diffs = (
            m["diff"]
            .astype(float)
            .to_numpy()
        )

        recent7 = m.tail(7)

        avg31 = float(
            m["diff"].mean()
        )

        recent7_avg = float(
            recent7["diff"].mean()
        )

        recent7_win = float(
            recent7["win"].mean()
        )

        last_diff = float(
            diffs[-1]
        )

        if len(diffs) >= 2:

            prev_diff = float(
                diffs[-2]
            )

        else:

            prev_diff = last_diff

        prev_change = (
            last_diff
            - prev_diff
        )

        # ----------------------------------------------------
        # Weekday
        # ----------------------------------------------------

        wd = m[
            m["date"].dt.dayofweek
            == target_weekday
        ]

        weekday_n = len(wd)

        if weekday_n > 0:

            weekday_avg_raw = float(
                wd["diff"].mean()
            )

        else:

            weekday_avg_raw = avg31

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
            * (1.0 - wd_weight)
        )

        # ----------------------------------------------------
        # Hit rates
        # ----------------------------------------------------

        plus1000_rate = float(
            m["plus1000"].mean()
        )

        plus2000_rate = float(
            m["plus2000"].mean()
        )

        # ----------------------------------------------------
        # Machine type
        # ----------------------------------------------------

        type_avg = float(
            type_stats.get(
                name,
                0.0,
            )
        )

        # ----------------------------------------------------
        # Neighbor
        # ----------------------------------------------------

        neighbor_values = []

        for n2 in (
            no - 1,
            no + 1,
        ):

            if n2 in latest_day.index:

                value = latest_day.loc[
                    n2,
                    "diff",
                ]

                if not isinstance(
                    value,
                    pd.Series,
                ):
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

        # ----------------------------------------------------
        # Bounce
        # ----------------------------------------------------

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
                "machine_no": int(no),
                "machine_name": name,
                "avg31": avg31,
                "recent7_avg": recent7_avg,
                "recent7_win": recent7_win,
                "last_diff": last_diff,
                "prev_change": prev_change,
                "weekday_avg": weekday_avg,
                "type_avg": type_avg,
                "plus1000_rate": plus1000_rate,
                "plus2000_rate": plus2000_rate,
                "neighbor_avg": neighbor_avg,
                "bounce_signal": bounce_signal,
            }
        )

    feat = pd.DataFrame(
        rows
    )

    feat = feat.merge(
        actual.rename(
            columns={
                "diff": "actual_diff",
            }
        ),
        on="machine_no",
        how="inner",
    )

    return feat


# ============================================================
# SCORE
# ============================================================

def calculate_score(
    feat,
    weights,
):

    score = pd.Series(
        0.0,
        index=feat.index,
    )

    for factor in FACTORS:

        z = zscore_series(
            feat[factor]
        )

        score = (
            score
            + z
            * weights[factor]
        )

    return score


# ============================================================
# DAILY EVALUATION
# ============================================================

def evaluate_daily(
    feat,
    weights,
):

    if feat.empty:
        return {}

    work = feat.copy()

    work["score"] = calculate_score(
        work,
        weights,
    )

    work = work.sort_values(
        "score",
        ascending=False,
    )

    actual = pd.to_numeric(
        work["actual_diff"],
        errors="coerce",
    )

    all_avg = float(
        actual.mean()
    )

    result = {
        "all_avg": all_avg,
    }

    for n in TOP_N_LIST:

        top = work.head(n)

        d = pd.to_numeric(
            top["actual_diff"],
            errors="coerce",
        )

        avg = float(
            d.mean()
        )

        win = float(
            (d > 0).mean()
            * 100.0
        )

        plus2000 = float(
            (d >= 2000).mean()
            * 100.0
        )

        lift = (
            avg
            - all_avg
        )

        result[
            f"top{n}_avg"
        ] = avg

        result[
            f"top{n}_win"
        ] = win

        result[
            f"top{n}_plus2000"
        ] = plus2000

        result[
            f"top{n}_lift"
        ] = lift

    return result


# ============================================================
# WEIGHT GENERATION
# ============================================================

def normalize_weights(
    values
):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = np.maximum(
        values,
        0.0,
    )

    total = values.sum()

    if total <= 0:

        values = np.ones(
            len(values)
        )

        total = values.sum()

    values = (
        values
        / total
    )

    return values


def make_random_weights(
    rng
):

    # Dirichlet distribution.
    # alpha > 1 prevents excessive concentration.
    values = rng.dirichlet(
        np.ones(
            len(FACTORS)
        )
        * 1.5
    )

    values = normalize_weights(
        values
    )

    return dict(
        zip(
            FACTORS,
            values,
        )
    )


def make_base_weights():

    return dict(
        BASE_WEIGHTS
    )


# ============================================================
# PATTERN EVALUATION
# ============================================================

def evaluate_weights(
    panels,
    weights,
):

    daily_results = []

    for date, feat in panels.items():

        r = evaluate_daily(
            feat,
            weights,
        )

        if not r:
            continue

        r["date"] = date

        daily_results.append(
            r
        )

    if not daily_results:
        return None

    daily = pd.DataFrame(
        daily_results
    )

    result = {}

    result[
        "days"
    ] = len(daily)

    for n in TOP_N_LIST:

        avg = float(
            daily[
                f"top{n}_avg"
            ].mean()
        )

        lift = float(
            daily[
                f"top{n}_lift"
            ].mean()
        )

        win = float(
            daily[
                f"top{n}_win"
            ].mean()
        )

        plus2000 = float(
            daily[
                f"top{n}_plus2000"
            ].mean()
        )

        positive_days = float(
            (
                daily[
                    f"top{n}_avg"
                ]
                > 0
            ).mean()
            * 100.0
        )

        result[
            f"top{n}_avg"
        ] = avg

        result[
            f"top{n}_lift"
        ] = lift

        result[
            f"top{n}_win"
        ] = win

        result[
            f"top{n}_plus2000"
        ] = plus2000

        result[
            f"top{n}_positive_days"
        ] = positive_days

    # Main stability metrics
    result[
        "objective"
    ] = (
        result["top10_lift"]
        * 0.40
        + result["top30_lift"]
        * 0.25
        + result["top10_positive_days"]
        * 5.0
        * 0.20
        + result["top30_positive_days"]
        * 5.0
        * 0.15
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "Ana-Slo Ver.3 Weight Optimization"
    )
    print("=" * 70)

    print()

    df = load_data()

    print(
        f"records = {len(df):,}"
    )

    print(
        f"backtest = "
        f"{BT_START.date()} "
        f"to "
        f"{BT_END.date()}"
    )

    print()

    # --------------------------------------------------------
    # Build panels only once.
    # --------------------------------------------------------

    panels = {}

    for target_date in pd.date_range(
        BT_START,
        BT_END,
    ):

        feat = build_features(
            df,
            target_date,
        )

        if feat.empty:
            continue

        panels[
            target_date.strftime(
                "%Y-%m-%d"
            )
        ] = feat

        print(
            target_date.strftime(
                "%Y-%m-%d"
            ),
            f"machines={len(feat)}"
        )

    print()

    if not panels:

        raise RuntimeError(
            "No backtest panels were created."
        )

    # --------------------------------------------------------
    # Candidate weights
    # --------------------------------------------------------

    candidates = []

    # Current Ver.3
    candidates.append(
        (
            "BASE_V3",
            make_base_weights(),
        )
    )

    rng = np.random.default_rng(
        20260816
    )

    for i in range(
        RANDOM_PATTERNS
    ):

        weights = make_random_weights(
            rng
        )

        candidates.append(
            (
                f"RANDOM_{i + 1:04d}",
                weights,
            )
        )

    print(
        f"weight patterns = "
        f"{len(candidates):,}"
    )

    print(
        "Evaluating..."
    )

    rows = []

    for idx, (
        name,
        weights,
    ) in enumerate(
        candidates,
        start=1,
    ):

        metrics = evaluate_weights(
            panels,
            weights,
        )

        if metrics is None:
            continue

        row = {
            "pattern": name,
        }

        row.update(
            metrics
        )

        for factor in FACTORS:

            row[
                f"w_{factor}"
            ] = weights[
                factor
            ]

        rows.append(
            row
        )

        if (
            idx % 500
            == 0
        ):

            print(
                f"progress "
                f"{idx:,}/"
                f"{len(candidates):,}"
            )

    result = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    result = result.sort_values(
        [
            "objective",
            "top10_lift",
            "top30_lift",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    result.insert(
        0,
        "rank",
        np.arange(
            1,
            len(result) + 1,
        ),
    )

    # Save all candidates
    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Best pattern
    # --------------------------------------------------------

    best = result.iloc[0]

    lines = []

    lines.append(
        "Ana-Slo Ver.3 Weight Optimization"
    )

    lines.append(
        ""
    )

    lines.append(
        f"Backtest: "
        f"{BT_START.date()} "
        f"to "
        f"{BT_END.date()}"
    )

    lines.append(
        f"Patterns: "
        f"{len(result):,}"
    )

    lines.append(
        ""
    )

    lines.append(
        "BEST PATTERN"
    )

    lines.append(
        f"rank = {int(best['rank'])}"
    )

    lines.append(
        f"pattern = {best['pattern']}"
    )

    lines.append(
        f"objective = "
        f"{best['objective']:.4f}"
    )

    lines.append(
        ""
    )

    for n in TOP_N_LIST:

        lines.append(
            f"TOP{n}: "
            f"avg={best[f'top{n}_avg']:+.1f}, "
            f"lift={best[f'top{n}_lift']:+.1f}, "
            f"win={best[f'top{n}_win']:.1f}%, "
            f"positive_days="
            f"{best[f'top{n}_positive_days']:.1f}%"
        )

    lines.append(
        ""
    )

    lines.append(
        "WEIGHTS"
    )

    for factor in FACTORS:

        lines.append(
            f"{factor:20s} "
            f"{best[f'w_{factor}'] * 100:6.2f}%"
        )

    lines.append(
        ""
    )

    lines.append(
        "NOTE"
    )

    lines.append(
        "This optimization uses only "
        "the current 16-day backtest period."
    )

    lines.append(
        "Do not treat the best pattern "
        "as a final model yet."
    )

    BEST_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TOP 20 WEIGHT PATTERNS"
    )
    print("=" * 70)

    show_cols = [
        "rank",
        "pattern",
        "objective",
        "top10_avg",
        "top10_lift",
        "top10_win",
        "top10_positive_days",
        "top30_avg",
        "top30_lift",
        "top30_win",
        "top30_positive_days",
    ]

    print(
        result[
            show_cols
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print(
        "BEST WEIGHTS"
    )
    print("=" * 70)

    for factor in FACTORS:

        print(
            f"{factor:20s}: "
            f"{best[f'w_{factor}'] * 100:6.2f}%"
        )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        BEST_FILE
    )

    print()
    print(
        "Optimization complete."
    )


if __name__ == "__main__":
    main()