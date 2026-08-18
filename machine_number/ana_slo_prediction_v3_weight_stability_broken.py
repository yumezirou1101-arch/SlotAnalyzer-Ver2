from pathlib import Path
import pandas as pd
import numpy as np
import math

BASE = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
DATA_DIR = BASE / "data" / "maruhan_maebashi" / "machine_number"
OUT_DIR = DATA_DIR / "analysis_31days_deep"

CSV1 = DATA_DIR / "ana_slo_20260711.csv"
CSV2 = DATA_DIR / "ana_slo_20260712_20260810.csv"

OPT_FILE = OUT_DIR / "09_Ver3_weight_optimization_results.csv"

START = pd.Timestamp("2026-07-11")
BT_START = pd.Timestamp("2026-07-26")
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

    date_col = find(["date", "譌･莉・])
    no_col = find(["machine_no", "蜿ｰ逡ｪ蜿ｷ"])
    name_col = find(["machine_name", "讖溽ｨｮ蜷・])
    diff_col = find(["diff", "蟾ｮ譫・])

    if not all([date_col, no_col, name_col, diff_col]):
        raise ValueError(
            f"Required columns not found: "
            f"date={date_col}, no={no_col}, name={name_col}, diff={diff_col}"
        )

    df = df.rename(columns={
        date_col: "date",
        no_col: "machine_no",
        name_col: "machine_name",
        diff_col: "diff",
    })

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
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
        df["machine_name"].astype(str).str.strip()
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
                            n2, "diff"
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


def rank_score(df, weights):

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

        if std == 0 or pd.isna(std):
            z = pd.Series(
                0.0,
                index=s.index
            )
        else:
            z = (
                s - s.mean()
            ) / std

        score += (
            (50.0 + z * 12.5).clip(
                0, 100
            )
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


def evaluate(df, weights, top_n):

    if df.empty:
        return None

    ranked = rank_score(
        df,
        weights
    )

    top = ranked.head(top_n)

    d = top["diff"].astype(float)

    return {
        "avg_diff": float(d.mean()),
        "win_rate": float(
            (d > 0).mean() * 100
        ),
        "plus1000_rate": float(
            (d >= 1000).mean() * 100
        ),
        "plus2000_rate": float(
            (d >= 2000).mean() * 100
        ),
        "positive": int(
            d.sum() > 0
        ),
        "total_diff": float(
            d.sum()
        ),
    }


def load_patterns():

    if not OPT_FILE.exists():
        raise FileNotFoundError(
            "Optimization result not found: "
            + str(OPT_FILE)
        )

    opt = pd.read_csv(
        OPT_FILE,
        encoding="utf-8-sig"
    )

    patterns = []

    for _, row in opt.head(20).iterrows():

        pattern = str(row["pattern"])
        weights = {}

        for factor in FACTORS:
            col = "w_" + factor

            if col in opt.columns:
                value = row[col]
                weights[factor] = (
                    float(value)
                    if pd.notna(value)
                    else 0.0
                )
            else:
                weights[factor] = 0.0

        patterns.append((pattern, weights))

    return patterns


def main():

    print("=" * 70)
    print("Ver.3 Weight Stability Test")
    print("=" * 70)

    df = load_data()

    print(
        f"records = {len(df):,}"
    )

    patterns = load_patterns()

    print(
        f"patterns loaded = {len(patterns)}"
    )

    print()

    # ---------------------------------------------------------
    # Build daily feature panels
    # ---------------------------------------------------------

    panels = {}

    for target_date in pd.date_range(
        BT_START,
        BT_END
    ):

        panel = build_features(
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

    # ---------------------------------------------------------
    # Full-period stability
    # ---------------------------------------------------------

    stability_rows = []

    for pattern, weights in patterns:

        for top_n in (10, 30):

            results = []

            for date, panel in panels.items():

                r = evaluate(
                    panel,
                    weights,
                    top_n
                )

                if r:
                    results.append(r)

            if not results:
                continue

            avg_diff = np.mean([
                r["avg_diff"]
                for r in results
            ])

            positive_days = (
                np.mean([
                    r["positive"]
                    for r in results
                ])
                * 100
            )

            avg_win = np.mean([
                r["win_rate"]
                for r in results
            ])

            total_diff = np.sum([
                r["total_diff"]
                for r in results
            ])

            stability_rows.append({
                "pattern": pattern,
                "top_n": top_n,
                "avg_diff": avg_diff,
                "avg_win_rate": avg_win,
                "positive_days": positive_days,
                "total_diff": total_diff,
            })

    stability = pd.DataFrame(
        stability_rows
    )

    # ---------------------------------------------------------
    # Train / Test split
    # ---------------------------------------------------------

    train_dates = [
        d for d in panels
        if d <= pd.Timestamp("2026-08-02")
    ]

    test_dates = [
        d for d in panels
        if d >= pd.Timestamp("2026-08-03")
    ]

    print()
    print(
        "TRAIN:",
        train_dates[0].date(),
        "~",
        train_dates[-1].date()
    )

    print(
        "TEST :",
        test_dates[0].date(),
        "~",
        test_dates[-1].date()
    )

    split_rows = []

    for pattern, weights in patterns:

        for period_name, dates in (
            ("TRAIN", train_dates),
            ("TEST", test_dates),
        ):

            for top_n in (10, 30):

                results = []

                for date in dates:

                    r = evaluate(
                        panels[date],
                        weights,
                        top_n
                    )

                    if r:
                        results.append(r)

                if not results:
                    continue

                split_rows.append({
                    "pattern": pattern,
                    "period": period_name,
                    "top_n": top_n,
                    "avg_diff": np.mean([
                        r["avg_diff"]
                        for r in results
                    ]),
                    "win_rate": np.mean([
                        r["win_rate"]
                        for r in results
                    ]),
                    "positive_days": np.mean([
                        r["positive"]
                        for r in results
                    ]) * 100,
                    "total_diff": np.sum([
                        r["total_diff"]
                        for r in results
                    ]),
                })

    split = pd.DataFrame(
        split_rows
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    out1 = (
        OUT_DIR
        / "10_Ver3_weight_stability.csv"
    )

    out2 = (
        OUT_DIR
        / "10_Ver3_weight_train_test.csv"
    )

    stability.to_csv(
        out1,
        index=False,
        encoding="utf-8-sig"
    )

    split.to_csv(
        out2,
        index=False,
        encoding="utf-8-sig"
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("FULL PERIOD STABILITY")
    print("=" * 70)

    print(
        stability
        .sort_values(
            ["top_n", "avg_diff"],
            ascending=[True, False]
        )
        .to_string(index=False)
    )

    print()
    print("=" * 70)
    print("TRAIN / TEST")
    print("=" * 70)

    print(
        split
        .sort_values(
            ["period", "top_n", "avg_diff"],
            ascending=[True, True, False]
        )
        .to_string(index=False)
    )

    print()
    print("Saved:")
    print(out1)
    print(out2)

    print()
    print("Stability test complete.")


if __name__ == "__main__":
    main()
