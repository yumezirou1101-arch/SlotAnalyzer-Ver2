from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
DATA_DIR = BASE / "data" / "maruhan_maebashi" / "machine_number"
OUT_DIR = DATA_DIR / "analysis_31days_deep"

CSV1 = DATA_DIR / "ana_slo_20260711.csv"
CSV2 = DATA_DIR / "ana_slo_20260712_20260810.csv"

START = pd.Timestamp("2026-07-11")
END = pd.Timestamp("2026-08-10")

# Split 1
TRAIN1_START = pd.Timestamp("2026-07-11")
TRAIN1_END = pd.Timestamp("2026-07-25")
TEST1_START = pd.Timestamp("2026-07-26")
TEST1_END = pd.Timestamp("2026-08-02")

# Split 2
TRAIN2_START = pd.Timestamp("2026-07-11")
TRAIN2_END = pd.Timestamp("2026-08-02")
TEST2_START = pd.Timestamp("2026-08-03")
TEST2_END = pd.Timestamp("2026-08-10")

N_PATTERNS = 3001
RANDOM_SEED = 20260816

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


def read_csv(path):
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError("CSV read failed: " + str(path))


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

    date_col = find(["date", "日付"])
    no_col = find(["machine_no", "台番号"])
    name_col = find(["machine_name", "機種名"])
    diff_col = find(["diff", "差枚"])

    if not all([date_col, no_col, name_col, diff_col]):
        raise ValueError(
            "Required columns not found: "
            + str((date_col, no_col, name_col, diff_col))
        )

    df = df.rename(columns={
        date_col: "date",
        no_col: "machine_no",
        name_col: "machine_name",
        diff_col: "diff",
    })

    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    )

    df["machine_no"] = pd.to_numeric(
        df["machine_no"], errors="coerce"
    )

    df["diff"] = (
        df["diff"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.strip()
    )

    df["diff"] = pd.to_numeric(
        df["diff"], errors="coerce"
    )

    df = df.dropna(
        subset=["date", "machine_no", "diff"]
    ).copy()

    df["machine_no"] = df["machine_no"].astype(int)
    df["machine_name"] = (
        df["machine_name"]
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["date"] >= START)
        & (df["date"] <= END)
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


def zscore_series(s):
    s = pd.to_numeric(
        s,
        errors="coerce"
    ).fillna(0.0)

    std = float(s.std(ddof=0))

    if std == 0 or np.isnan(std):
        return pd.Series(
            0.0,
            index=s.index
        )

    return (
        s - float(s.mean())
    ) / std


def build_features(df, target_date):
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


def make_patterns():
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    patterns = []

    # Equal-weight baseline
    equal = np.ones(
        len(FACTORS)
    ) / len(FACTORS)

    patterns.append(
        ("EQUAL", equal)
    )

    # Random Dirichlet patterns
    for i in range(
        N_PATTERNS - 1
    ):
        weights = rng.dirichlet(
            np.ones(len(FACTORS))
        )

        patterns.append(
            (
                "RANDOM_%04d" % i,
                weights
            )
        )

    return patterns


def score_dataframe(df, weights):
    x = df.copy()

    score = pd.Series(
        0.0,
        index=x.index
    )

    for factor in FACTORS:

        z = zscore_series(
            x[factor]
        )

        component = (
            50.0 + z * 12.5
        ).clip(0, 100)

        score += (
            component
            * weights.get(
                factor,
                0.0
            )
        )

    x["score"] = score

    return x.sort_values(
        "score",
        ascending=False
    )


def evaluate_panel(
    panel,
    weights,
    top_n
):
    if panel.empty:
        return None

    ranked = score_dataframe(
        panel,
        weights
    )

    top = ranked.head(top_n)

    d = top["diff"].astype(float)

    return {
        "avg_diff": float(
            d.mean()
        ),
        "win_rate": float(
            (d > 0).mean() * 100
        ),
        "plus2000_rate": float(
            (d >= 2000).mean() * 100
        ),
        "total_diff": float(
            d.sum()
        ),
        "positive": int(
            d.sum() > 0
        ),
    }


def evaluate_period(
    panels,
    weights,
    start_date,
    end_date
):
    rows = []

    for date, panel in panels.items():

        if (
            date < start_date
            or date > end_date
        ):
            continue

        for top_n in (
            1, 5, 10, 20, 30
        ):

            r = evaluate_panel(
                panel,
                weights,
                top_n
            )

            if r is None:
                continue

            rows.append({
                "date": date.date(),
                "top_n": top_n,
                **r
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def summarize(
    daily,
    label
):
    rows = []

    for top_n in (
        1, 5, 10, 20, 30
    ):

        x = daily[
            daily["top_n"] == top_n
        ].copy()

        if x.empty:
            continue

        rows.append({
            "period": label,
            "top_n": top_n,
            "days": len(x),
            "avg_diff": float(
                x["avg_diff"].mean()
            ),
            "median_daily_avg": float(
                x["avg_diff"].median()
            ),
            "win_rate": float(
                x["win_rate"].mean()
            ),
            "plus2000_rate": float(
                x["plus2000_rate"].mean()
            ),
            "positive_days": float(
                x["positive"].mean()
                * 100
            ),
            "total_diff": float(
                x["total_diff"].sum()
            ),
        })

    return pd.DataFrame(rows)


def optimize_weights(
    panels,
    start_date,
    end_date,
    patterns
):
    training_rows = []

    for date, panel in panels.items():

        if (
            date < start_date
            or date > end_date
        ):
            continue

        training_rows.append(
            panel
        )

    if not training_rows:
        return None

    results = []

    for idx, (
        pattern,
        weights_array
    ) in enumerate(patterns):

        weights = dict(
            zip(
                FACTORS,
                weights_array
            )
        )

        daily_top10 = []

        for panel in training_rows:

            r = evaluate_panel(
                panel,
                weights,
                10
            )

            if r is not None:
                daily_top10.append(
                    r["avg_diff"]
                )

        if not daily_top10:
            continue

        avg = float(
            np.mean(daily_top10)
        )

        positive = float(
            np.mean(
                np.array(
                    daily_top10
                ) > 0
            ) * 100
        )

        # Objective:
        # reward average result,
        # but also reward consistency.
        objective = (
            avg
            + positive * 5.0
        )

        row = {
            "pattern": pattern,
            "objective": objective,
            "train_top10_avg": avg,
            "train_positive_days": positive,
        }

        for factor in FACTORS:
            row[
                "w_" + factor
            ] = weights[factor]

        results.append(row)

    result = pd.DataFrame(
        results
    )

    result = result.sort_values(
        "objective",
        ascending=False
    ).reset_index(
        drop=True
    )

    return result


def get_weights_from_row(row):
    return {
        factor: float(
            row["w_" + factor]
        )
        for factor in FACTORS
    }


def overfit_check(
    train_summary,
    test_summary
):
    rows = []

    for top_n in (
        1, 5, 10, 20, 30
    ):

        tr = train_summary[
            train_summary["top_n"]
            == top_n
        ]

        te = test_summary[
            test_summary["top_n"]
            == top_n
        ]

        if tr.empty or te.empty:
            continue

        train_avg = float(
            tr.iloc[0]["avg_diff"]
        )

        test_avg = float(
            te.iloc[0]["avg_diff"]
        )

        gap = (
            train_avg
            - test_avg
        )

        rows.append({
            "top_n": top_n,
            "train_avg": train_avg,
            "test_avg": test_avg,
            "train_test_gap": gap,
        })

    return pd.DataFrame(rows)


def main():

    print("=" * 70)
    print(
        "Ver.3 Split Backtest "
        "and Overfit Check"
    )
    print("=" * 70)

    df = load_data()

    print(
        "records = %s"
        % format(len(df), ",")
    )

    print()

    print(
        "Building daily feature panels..."
    )

    panels = {}

    for target_date in pd.date_range(
        START,
        END
    ):

        panel = build_features(
            df,
            target_date
        )

        if not panel.empty:
            panels[target_date] = panel

            print(
                "%s machines=%d"
                % (
                    target_date.date(),
                    len(panel)
                )
            )

    print()

    patterns = make_patterns()

    print(
        "weight patterns = %d"
        % len(patterns)
    )

    print()

    all_summary = []
    all_weights = []
    all_overfit = []

    splits = [
        (
            "SPLIT1",
            TRAIN1_START,
            TRAIN1_END,
            TEST1_START,
            TEST1_END
        ),
        (
            "SPLIT2",
            TRAIN2_START,
            TRAIN2_END,
            TEST2_START,
            TEST2_END
        ),
    ]

    for (
        name,
        train_start,
        train_end,
        test_start,
        test_end
    ) in splits:

        print("=" * 70)
        print(name)
        print("=" * 70)

        print(
            "TRAIN: %s to %s"
            % (
                train_start.date(),
                train_end.date()
            )
        )

        print(
            "TEST : %s to %s"
            % (
                test_start.date(),
                test_end.date()
            )
        )

        print()

        print(
            "Optimizing weights "
            "using TRAIN only..."
        )

        opt = optimize_weights(
            panels,
            train_start,
            train_end,
            patterns
        )

        if opt is None or opt.empty:
            print(
                "Optimization failed."
            )
            continue

        best = opt.iloc[0]

        weights = get_weights_from_row(
            best
        )

        print(
            "Selected pattern = %s"
            % best["pattern"]
        )

        print(
            "TRAIN TOP10 avg = %+0.1f"
            % best["train_top10_avg"]
        )

        print(
            "TRAIN positive days = %.1f%%"
            % best["train_positive_days"]
        )

        print()

        print(
            "TEST evaluation "
            "with frozen weights..."
        )

        train_daily = evaluate_period(
            panels,
            weights,
            train_start,
            train_end
        )

        test_daily = evaluate_period(
            panels,
            weights,
            test_start,
            test_end
        )

        train_summary = summarize(
            train_daily,
            name + "_TRAIN"
        )

        test_summary = summarize(
            test_daily,
            name + "_TEST"
        )

        print()
        print(
            "TRAIN RESULT"
        )
        print(
            train_summary.to_string(
                index=False
            )
        )

        print()
        print(
            "TEST RESULT"
        )
        print(
            test_summary.to_string(
                index=False
            )
        )

        overfit = overfit_check(
            train_summary,
            test_summary
        )

        overfit.insert(
            0,
            "split",
            name
        )

        print()
        print(
            "TRAIN vs TEST GAP"
        )

        print(
            overfit.to_string(
                index=False
            )
        )

        # Save selected weights
        for factor in FACTORS:

            all_weights.append({
                "split": name,
                "pattern": best["pattern"],
                "factor": factor,
                "weight": weights[factor],
            })

        train_summary["split"] = name
        test_summary["split"] = name

        all_summary.append(
            train_summary
        )

        all_summary.append(
            test_summary
        )

        all_overfit.append(
            overfit
        )

    if not all_summary:
        raise RuntimeError(
            "No result generated."
        )

    summary_df = pd.concat(
        all_summary,
        ignore_index=True
    )

    weights_df = pd.DataFrame(
        all_weights
    )

    overfit_df = pd.concat(
        all_overfit,
        ignore_index=True
    )

    out_summary = (
        OUT_DIR
        / "12_Ver3_split_overfit_summary.csv"
    )

    out_weights = (
        OUT_DIR
        / "12_Ver3_split_overfit_weights.csv"
    )

    out_overfit = (
        OUT_DIR
        / "12_Ver3_overfit_check.csv"
    )

    summary_df.to_csv(
        out_summary,
        index=False,
        encoding="utf-8-sig"
    )

    weights_df.to_csv(
        out_weights,
        index=False,
        encoding="utf-8-sig"
    )

    overfit_df.to_csv(
        out_overfit,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print(
        "FILES SAVED"
    )
    print("=" * 70)

    print(out_summary)
    print(out_weights)
    print(out_overfit)

    print()
    print(
        "Split backtest complete."
    )


if __name__ == "__main__":
    main()