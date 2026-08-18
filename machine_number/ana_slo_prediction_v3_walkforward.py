from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Ana-Slo Ver.3 Walk-Forward Backtest
# Future data is never used when selecting weights.
# ============================================================

BASE = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
DATA_DIR = BASE / "data" / "maruhan_maebashi" / "machine_number"
OUT_DIR = DATA_DIR / "analysis_31days_deep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV1 = DATA_DIR / "ana_slo_20260711.csv"
CSV2 = DATA_DIR / "ana_slo_20260712_20260810.csv"

START = pd.Timestamp("2026-07-11")

# First prediction day.
# We need enough historical days to optimize weights.
BT_START = pd.Timestamp("2026-07-21")
BT_END = pd.Timestamp("2026-08-10")

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

# Number of candidate weight patterns for each prediction day.
N_PATTERNS = 3001

# Number of machines selected.
TOP_N = [1, 5, 10, 20, 30]

# Fixed random seed makes the experiment reproducible.
RANDOM_SEED = 1476

OUT_DAILY = OUT_DIR / "11_Ver3_walkforward_daily.csv"
OUT_SUMMARY = OUT_DIR / "11_Ver3_walkforward_summary.csv"
OUT_WEIGHTS = OUT_DIR / "11_Ver3_walkforward_weights.csv"
OUT_README = OUT_DIR / "11_Ver3_walkforward_README.txt"


def read_csv(path):
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError("CSV could not be read: " + str(path))


def load_data():

    frames = []

    for p in (CSV1, CSV2):
        if p.exists():
            frames.append(read_csv(p))

    if not frames:
        raise FileNotFoundError("Input CSV not found.")

    df = pd.concat(frames, ignore_index=True)

    def find(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    date_col = find([
        "date",
        "日付",
        "譌･莉・",
    ])

    no_col = find([
        "machine_no",
        "台番号",
        "蜿ｰ逡ｪ蜿ｷ",
    ])

    name_col = find([
        "machine_name",
        "機種名",
        "讖溽ｨｮ蜷・",
    ])

    diff_col = find([
        "diff",
        "差枚",
        "蟾ｮ譫・",
    ])

    if not all([date_col, no_col, name_col, diff_col]):
        raise ValueError(
            f"Required columns not found: "
            f"date={date_col}, no={no_col}, "
            f"name={name_col}, diff={diff_col}"
        )

    df = df.rename(columns={
        date_col: "date",
        no_col: "machine_no",
        name_col: "machine_name",
        diff_col: "diff",
    })

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["machine_no"] = pd.to_numeric(
        df["machine_no"],
        errors="coerce"
    )

    df["diff"] = (
        df["diff"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.strip()
    )

    df["diff"] = pd.to_numeric(
        df["diff"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "date",
            "machine_no",
            "diff",
        ]
    ).copy()

    df["machine_no"] = df["machine_no"].astype(int)

    df["machine_name"] = (
        df["machine_name"]
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["date"] >= START)
        & (df["date"] <= BT_END)
    ].copy()

    df = df.sort_values(
        ["date", "machine_no"]
    )

    df = df.drop_duplicates(
        ["date", "machine_no"],
        keep="last"
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


def make_features(df, target_date):

    hist = df[
        df["date"] < target_date
    ].copy()

    actual = df[
        df["date"] == target_date
    ][
        ["machine_no", "diff"]
    ].copy()

    if hist.empty or actual.empty:
        return pd.DataFrame()

    target_weekday = target_date.dayofweek

    latest_date = hist["date"].max()

    latest_day = (
        hist[
            hist["date"] == latest_date
        ]
        .set_index("machine_no")
    )

    type_stats = (
        hist.groupby("machine_name")["diff"]
        .mean()
        .to_dict()
    )

    rows = []

    for no, m in hist.groupby("machine_no"):

        m = m.sort_values("date")

        if m.empty:
            continue

        name = str(
            m.iloc[-1]["machine_name"]
        )

        avg31 = float(
            m["diff"].mean()
        )

        recent7 = m.tail(7)

        recent7_avg = float(
            recent7["diff"].mean()
        )

        recent7_win = float(
            recent7["win"].mean()
        )

        last_diff = float(
            m.iloc[-1]["diff"]
        )

        if len(m) >= 2:
            prev_diff = float(
                m.iloc[-2]["diff"]
            )
        else:
            prev_diff = last_diff

        prev_change = (
            last_diff - prev_diff
        )

        wd = m[
            m["date"].dt.dayofweek
            == target_weekday
        ]

        weekday_n = len(wd)

        if weekday_n:
            weekday_avg_raw = float(
                wd["diff"].mean()
            )
        else:
            weekday_avg_raw = avg31

        prior_n = 15.0

        wd_weight = (
            weekday_n
            / (weekday_n + prior_n)
        )

        weekday_avg = (
            weekday_avg_raw * wd_weight
            + avg31 * (1.0 - wd_weight)
        )

        plus1000_rate = float(
            m["plus1000"].mean()
        )

        plus2000_rate = float(
            m["plus2000"].mean()
        )

        type_avg = float(
            type_stats.get(name, 0.0)
        )

        neighbor_values = []

        for n2 in (no - 1, no + 1):

            if n2 in latest_day.index:

                neighbor_values.append(
                    float(
                        latest_day.loc[
                            n2,
                            "diff"
                        ]
                    )
                )

        if neighbor_values:
            neighbor_avg = float(
                np.mean(neighbor_values)
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

        rows.append({
            "machine_no": int(no),
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
        })

    feat = pd.DataFrame(rows)

    return feat.merge(
        actual,
        on="machine_no",
        how="inner"
    )


def make_score(df, weights):

    x = df.copy()

    score = pd.Series(
        0.0,
        index=x.index
    )

    for factor in FACTORS:

        s = pd.to_numeric(
            x[factor],
            errors="coerce"
        ).fillna(0.0)

        std = s.std(ddof=0)

        if pd.isna(std) or std == 0:
            z = pd.Series(
                0.0,
                index=s.index
            )
        else:
            z = (
                s - s.mean()
            ) / std

        factor_score = (
            50.0 + z * 12.5
        ).clip(0, 100)

        score += (
            factor_score
            * weights[factor]
        )

    x["score"] = score

    return x.sort_values(
        "score",
        ascending=False
    )


def generate_weight_patterns():

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    patterns = []

    # Include equal weights.
    equal = {
        f: 1.0 / len(FACTORS)
        for f in FACTORS
    }

    patterns.append(
        ("EQUAL_WEIGHT", equal)
    )

    # Generate random Dirichlet weights.
    # Every pattern sums to 1.
    random_weights = rng.dirichlet(
        np.ones(len(FACTORS)),
        size=N_PATTERNS - 1
    )

    for i, arr in enumerate(
        random_weights,
        start=1
    ):

        weights = {
            factor: float(arr[j])
            for j, factor in enumerate(
                FACTORS
            )
        }

        patterns.append(
            (
                f"RANDOM_{i:04d}",
                weights
            )
        )

    return patterns


def evaluate_one_day(
    panel,
    weights
):

    ranked = make_score(
        panel,
        weights
    )

    result = {}

    for n in TOP_N:

        top = ranked.head(n)

        d = top["diff"].astype(float)

        result[n] = {
            "avg": float(d.mean()),
            "win": float(
                (d > 0).mean() * 100
            ),
            "plus2000": float(
                (d >= 2000).mean() * 100
            ),
            "total": float(d.sum()),
        }

    return result


def training_objective(
    panels,
    weights
):

    values = []

    for panel in panels:

        ranked = make_score(
            panel,
            weights
        )

        top = ranked.head(10)

        d = top["diff"].astype(float)

        if len(d):
            values.append(
                float(d.mean())
            )

    if not values:
        return -999999.0

    # Main objective:
    # maximize TOP10 average difference.
    #
    # This is deliberately simple and transparent.
    return float(
        np.mean(values)
    )


def optimize_weights(
    train_panels,
    patterns
):

    best_name = None
    best_weights = None
    best_objective = -999999.0

    for name, weights in patterns:

        objective = training_objective(
            train_panels,
            weights
        )

        if objective > best_objective:

            best_objective = objective
            best_name = name
            best_weights = weights.copy()

    return (
        best_name,
        best_weights,
        best_objective
    )


def main():

    print("=" * 70)
    print(
        "Ana-Slo Ver.3 "
        "Walk-Forward Backtest"
    )
    print("=" * 70)

    df = load_data()

    print()
    print(
        f"records = {len(df):,}"
    )

    print(
        f"backtest = "
        f"{BT_START.date()} "
        f"to {BT_END.date()}"
    )

    patterns = generate_weight_patterns()

    print(
        f"weight patterns per day = "
        f"{len(patterns):,}"
    )

    print()

    # ---------------------------------------------------------
    # Build all daily panels.
    # ---------------------------------------------------------

    panels = {}

    for target_date in pd.date_range(
        BT_START,
        BT_END
    ):

        panel = make_features(
            df,
            target_date
        )

        if not panel.empty:

            panels[target_date] = panel

            print(
                f"{target_date.date()} "
                f"machines={len(panel)}"
            )

    print()
    print(
        "Starting walk-forward evaluation..."
    )
    print()

    daily_rows = []
    weight_rows = []

    dates = sorted(panels.keys())

    for i, target_date in enumerate(
        dates,
        start=1
    ):

        # -----------------------------------------------------
        # Only use panels from dates BEFORE target_date.
        # -----------------------------------------------------

        train_dates = [
            d for d in dates
            if d < target_date
        ]

        train_panels = [
            panels[d]
            for d in train_dates
        ]

        if len(train_panels) < 5:
            continue

        best_name, best_weights, train_score = (
            optimize_weights(
                train_panels,
                patterns
            )
        )

        test_panel = panels[target_date]

        evaluation = evaluate_one_day(
            test_panel,
            best_weights
        )

        row = {
            "date": target_date.date(),
            "train_days": len(train_panels),
            "selected_pattern": best_name,
            "train_objective": train_score,
        }

        for n in TOP_N:

            row[f"top{n}_avg"] = (
                evaluation[n]["avg"]
            )

            row[f"top{n}_win"] = (
                evaluation[n]["win"]
            )

            row[f"top{n}_plus2000"] = (
                evaluation[n]["plus2000"]
            )

            row[f"top{n}_total"] = (
                evaluation[n]["total"]
            )

        daily_rows.append(row)

        weight_row = {
            "date": target_date.date(),
            "train_days": len(train_panels),
            "selected_pattern": best_name,
            "train_objective": train_score,
        }

        for factor in FACTORS:
            weight_row[
                f"w_{factor}"
            ] = best_weights[factor]

        weight_rows.append(
            weight_row
        )

        print(
            f"{target_date.date()} "
            f"train={len(train_panels):2d} "
            f"pattern={best_name:12s} "
            f"TOP10="
            f"{evaluation[10]['avg']:+.0f}"
        )

    daily = pd.DataFrame(
        daily_rows
    )

    weights_df = pd.DataFrame(
        weight_rows
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary_rows = []

    # Full-machine benchmark for each day.
    full_daily = []

    for target_date in dates:

        panel = panels[target_date]

        d = panel["diff"].astype(float)

        full_daily.append({
            "date": target_date.date(),
            "avg": float(d.mean()),
            "win": float(
                (d > 0).mean() * 100
            ),
            "total": float(d.sum()),
        })

    full_df = pd.DataFrame(
        full_daily
    )

    for n in TOP_N:

        avg_col = f"top{n}_avg"
        win_col = f"top{n}_win"
        plus_col = f"top{n}_plus2000"
        total_col = f"top{n}_total"

        if daily.empty:
            continue

        summary_rows.append({
            "target": f"TOP{n}",
            "days": len(daily),
            "avg_diff": float(
                daily[avg_col].mean()
            ),
            "median_daily_avg": float(
                daily[avg_col].median()
            ),
            "win_rate": float(
                daily[win_col].mean()
            ),
            "plus2000_rate": float(
                daily[plus_col].mean()
            ),
            "positive_days": float(
                (
                    daily[avg_col] > 0
                ).mean() * 100
            ),
            "total_diff": float(
                daily[total_col].sum()
            ),
        })

    # Full machine benchmark.
    if not full_df.empty:

        summary_rows.append({
            "target": "ALL",
            "days": len(full_df),
            "avg_diff": float(
                full_df["avg"].mean()
            ),
            "median_daily_avg": float(
                full_df["avg"].median()
            ),
            "win_rate": float(
                full_df["win"].mean()
            ),
            "plus2000_rate": np.nan,
            "positive_days": float(
                (
                    full_df["avg"] > 0
                ).mean() * 100
            ),
            "total_diff": float(
                full_df["total"].sum()
            ),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    # ---------------------------------------------------------
    # Pattern frequency
    # ---------------------------------------------------------

    if not weights_df.empty:

        pattern_frequency = (
            weights_df[
                "selected_pattern"
            ]
            .value_counts()
            .rename_axis(
                "pattern"
            )
            .reset_index(
                name="selected_days"
            )
        )

    else:

        pattern_frequency = pd.DataFrame(
            columns=[
                "pattern",
                "selected_days",
            ]
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    daily.to_csv(
        OUT_DAILY,
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    weights_df.to_csv(
        OUT_WEIGHTS,
        index=False,
        encoding="utf-8-sig"
    )

    readme_lines = [
        "Ana-Slo Ver.3 Walk-Forward Backtest",
        "",
        "Purpose:",
        "Test whether Ver.3 weight optimization works",
        "when weights are selected only from historical data.",
        "",
        f"Data start: {START.date()}",
        f"Backtest start: {BT_START.date()}",
        f"Backtest end: {BT_END.date()}",
        f"Patterns per day: {len(patterns)}",
        f"Random seed: {RANDOM_SEED}",
        "",
        "Method:",
        "For each prediction day, only earlier dates are used",
        "for weight optimization.",
        "The selected weights are then evaluated on the",
        "next unseen day.",
        "",
        "Training objective:",
        "Average TOP10 difference across historical training days.",
        "",
        "Important:",
        "This test is designed to avoid future-data leakage.",
        "",
        "Output files:",
        str(OUT_DAILY),
        str(OUT_SUMMARY),
        str(OUT_WEIGHTS),
    ]

    OUT_README.write_text(
        "\n".join(readme_lines),
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("WALK-FORWARD RESULT")
    print("=" * 70)

    if not summary.empty:
        print(
            summary.to_string(
                index=False
            )
        )

    print()
    print("=" * 70)
    print("SELECTED WEIGHT PATTERN FREQUENCY")
    print("=" * 70)

    print(
        pattern_frequency.to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(OUT_DAILY)
    print(OUT_SUMMARY)
    print(OUT_WEIGHTS)
    print(OUT_README)
    print()
    print(
        "Walk-forward backtest complete."
    )


if __name__ == "__main__":
    main()