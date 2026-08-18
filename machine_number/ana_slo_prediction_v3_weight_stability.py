from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
DATA_DIR = BASE / "data" / "maruhan_maebashi" / "machine_number"
OUT_DIR = DATA_DIR / "analysis_31days_deep"

CSV1 = DATA_DIR / "ana_slo_20260711.csv"
CSV2 = DATA_DIR / "ana_slo_20260712_20260810.csv"
OPT_FILE = OUT_DIR / "09_Ver3_weight_optimization_results.csv"

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

    for path in (CSV1, CSV2):
        if path.exists():
            frames.append(read_csv(path))

    if not frames:
        raise FileNotFoundError("Input CSV not found.")

    df = pd.concat(frames, ignore_index=True)

    def find_column(candidates):
        for col in candidates:
            if col in df.columns:
                return col
        return None

    date_col = find_column(["date", "日付"])
    no_col = find_column(["machine_no", "台番号"])
    name_col = find_column(["machine_name", "機種名"])
    diff_col = find_column(["diff", "差枚"])

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
    df["machine_no"] = pd.to_numeric(df["machine_no"], errors="coerce")

    df["diff"] = (
        df["diff"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.strip()
    )

    df["diff"] = pd.to_numeric(df["diff"], errors="coerce")

    df = df.dropna(
        subset=["date", "machine_no", "diff"]
    ).copy()

    df["machine_no"] = df["machine_no"].astype(int)
    df["machine_name"] = df["machine_name"].astype(str).str.strip()

    df = df.sort_values(["date", "machine_no"])
    df = df.drop_duplicates(
        ["date", "machine_no"],
        keep="last"
    )

    df["win"] = (df["diff"] > 0).astype(int)
    df["plus1000"] = (df["diff"] >= 1000).astype(int)
    df["plus2000"] = (df["diff"] >= 2000).astype(int)

    return df


def build_features(df, target_date):
    hist = df[df["date"] < target_date].copy()
    actual = df[df["date"] == target_date][
        ["machine_no", "diff"]
    ].copy()

    if hist.empty or actual.empty:
        return pd.DataFrame()

    target_weekday = target_date.dayofweek
    latest_date = hist["date"].max()

    latest_day = hist[
        hist["date"] == latest_date
    ].set_index("machine_no")

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

        name = str(m.iloc[-1]["machine_name"])

        avg31 = float(m["diff"].mean())

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

        prev_change = last_diff - prev_diff

        wd = m[
            m["date"].dt.dayofweek == target_weekday
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
            weekday_n / (weekday_n + prior_n)
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

        for neighbor_no in (no - 1, no + 1):

            if neighbor_no in latest_day.index:
                neighbor_values.append(
                    float(
                        latest_day.loc[
                            neighbor_no, "diff"
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

    features = pd.DataFrame(rows)

    return features.merge(
        actual,
        on="machine_no",
        how="inner"
    )


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

    required = ["pattern"] + [
        "w_" + factor for factor in FACTORS
    ]

    missing = [
        col for col in required
        if col not in opt.columns
    ]

    if missing:
        raise ValueError(
            "Missing weight columns: "
            + ", ".join(missing)
        )

    patterns = []

    for _, row in opt.head(20).iterrows():

        pattern = str(row["pattern"])

        weights = {}

        for factor in FACTORS:

            col = "w_" + factor

            value = row[col]

            weights[factor] = (
                float(value)
                if pd.notna(value)
                else 0.0
            )

        patterns.append(
            (pattern, weights)
        )

    return patterns


def rank_score(df, weights):

    score = pd.Series(
        0.0,
        index=df.index
    )

    for factor in FACTORS:

        s = pd.to_numeric(
            df[factor],
            errors="coerce"
        ).fillna(0.0)

        std = float(
            s.std(ddof=0)
        )

        if std == 0 or pd.isna(std):
            z = pd.Series(
                0.0,
                index=s.index
            )
        else:
            z = (
                s - float(s.mean())
            ) / std

        normalized = (
            50.0 + z * 12.5
        ).clip(0, 100)

        score += (
            normalized
            * weights.get(factor, 0.0)
        )

    result = df.copy()
    result["score"] = score

    return result.sort_values(
        "score",
        ascending=False
    )


def evaluate(panel, weights, top_n):

    ranked = rank_score(
        panel,
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

    if len(patterns) == 0:
        raise RuntimeError(
            "No optimization patterns found."
        )

    print()

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
    print(
        f"Evaluating {len(patterns)} weight patterns..."
    )

    daily_rows = []
    summary_rows = []

    for pattern, weights in patterns:

        pattern_daily = []

        for target_date, panel in panels.items():

            result10 = evaluate(
                panel,
                weights,
                10
            )

            result30 = evaluate(
                panel,
                weights,
                30
            )

            daily_rows.append({
                "pattern": pattern,
                "date": target_date.date(),
                "top10_avg": result10["avg_diff"],
                "top10_win": result10["win_rate"],
                "top10_plus2000": result10["plus2000_rate"],
                "top10_total": result10["total_diff"],
                "top10_positive": result10["positive"],
                "top30_avg": result30["avg_diff"],
                "top30_win": result30["win_rate"],
                "top30_plus2000": result30["plus2000_rate"],
                "top30_total": result30["total_diff"],
                "top30_positive": result30["positive"],
            })

            pattern_daily.append({
                "date": target_date,
                "top10_avg": result10["avg_diff"],
                "top30_avg": result30["avg_diff"],
            })

        pdf = pd.DataFrame(pattern_daily)

        summary_rows.append({
            "pattern": pattern,
            "days": len(pdf),
            "top10_avg": float(
                pdf["top10_avg"].mean()
            ),
            "top10_median": float(
                pdf["top10_avg"].median()
            ),
            "top10_win": float(
                pdf["top10_avg"].mean()
            ),
            "top10_positive_days": float(
                (pdf["top10_avg"] > 0).mean() * 100
            ),
            "top30_avg": float(
                pdf["top30_avg"].mean()
            ),
            "top30_median": float(
                pdf["top30_avg"].median()
            ),
            "top30_positive_days": float(
                (pdf["top30_avg"] > 0).mean() * 100
            ),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    summary = summary.sort_values(
        "top10_avg",
        ascending=False
    ).reset_index(drop=True)

    summary.insert(
        0,
        "rank",
        np.arange(
            1,
            len(summary) + 1
        )
    )

    daily = pd.DataFrame(
        daily_rows
    )

    out_summary = (
        OUT_DIR
        / "10_Ver3_weight_stability_summary.csv"
    )

    out_daily = (
        OUT_DIR
        / "10_Ver3_weight_stability_daily.csv"
    )

    summary.to_csv(
        out_summary,
        index=False,
        encoding="utf-8-sig"
    )

    daily.to_csv(
        out_daily,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print("STABILITY RESULT")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(out_summary)
    print(out_daily)
    print()
    print("Weight stability test complete.")


if __name__ == "__main__":
    main()
