from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

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

CSV1 = DATA_DIR / "ana_slo_20260711.csv"
CSV2 = DATA_DIR / "ana_slo_20260712_20260810.csv"

OPT_FILE = (
    OUT_DIR
    / "09_Ver3_weight_optimization_results.csv"
)

START = pd.Timestamp("2026-07-11")
END = pd.Timestamp("2026-08-10")

# Walk-forward settings
TRAIN_START = pd.Timestamp("2026-07-11")
FIRST_TEST_DATE = pd.Timestamp("2026-07-26")
TEST_END = pd.Timestamp("2026-08-10")

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

    for enc in (
        "utf-8-sig",
        "utf-8",
        "cp932",
    ):

        try:
            return pd.read_csv(
                path,
                encoding=enc
            )
        except Exception:
            pass

    raise RuntimeError(
        "CSV read failed: " + str(path)
    )


def load_data():

    frames = []

    for p in (
        CSV1,
        CSV2
    ):

        if p.exists():
            frames.append(
                read_csv(p)
            )

    if not frames:
        raise FileNotFoundError(
            "Input CSV not found."
        )

    df = pd.concat(
        frames,
        ignore_index=True
    )

    def find(cols):

        for c in cols:

            if c in df.columns:
                return c

        return None

    date_col = find(
        ["date", "日付"]
    )

    no_col = find(
        ["machine_no", "台番号"]
    )

    name_col = find(
        ["machine_name", "機種名"]
    )

    diff_col = find(
        ["diff", "差枚"]
    )

    if not all([
        date_col,
        no_col,
        name_col,
        diff_col
    ]):

        raise ValueError(
            "Required columns not found."
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
        errors="coerce"
    )

    df["machine_no"] = pd.to_numeric(
        df["machine_no"],
        errors="coerce"
    )

    df["diff"] = (
        df["diff"]
        .astype(str)
        .str.replace(
            ",",
            "",
            regex=False
        )
        .str.replace(
            "+",
            "",
            regex=False
        )
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
            "diff"
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

    df = df[
        (df["date"] >= START)
        & (df["date"] <= END)
    ].copy()

    df = df.sort_values(
        [
            "date",
            "machine_no"
        ]
    )

    df = df.drop_duplicates(
        [
            "date",
            "machine_no"
        ],
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


def build_features(
    df,
    target_date
):

    hist = df[
        df["date"] < target_date
    ].copy()

    actual = df[
        df["date"] == target_date
    ][
        [
            "machine_no",
            "diff"
        ]
    ].copy()

    if hist.empty or actual.empty:
        return pd.DataFrame()

    target_weekday = (
        target_date.dayofweek
    )

    latest_date = hist["date"].max()

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
            last_diff
            - prev_diff
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

        plus1000_rate = float(
            m["plus1000"].mean()
        )

        plus2000_rate = float(
            m["plus2000"].mean()
        )

        type_avg = float(
            type_stats.get(
                name,
                0.0
            )
        )

        neighbor_values = []

        for n2 in (
            no - 1,
            no + 1
        ):

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

        rows.append({

            "machine_no": int(no),

            "avg31": avg31,

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
        })

    feat = pd.DataFrame(
        rows
    )

    return feat.merge(
        actual,
        on="machine_no",
        how="inner"
    )


def zscore(s):

    s = pd.to_numeric(
        s,
        errors="coerce"
    ).fillna(0.0)

    std = float(
        s.std(ddof=0)
    )

    if std == 0 or np.isnan(std):

        return pd.Series(
            0.0,
            index=s.index
        )

    return (
        s - float(s.mean())
    ) / std


def rank_score(
    df,
    weights
):

    x = df.copy()

    score = pd.Series(
        0.0,
        index=x.index
    )

    for factor in FACTORS:

        z = zscore(
            x[factor]
        )

        component = (
            50.0
            + z * 12.5
        ).clip(
            0,
            100
        )

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


def evaluate(
    panel,
    weights,
    top_n
):

    if panel.empty:
        return None

    ranked = rank_score(
        panel,
        weights
    )

    top = ranked.head(
        top_n
    )

    d = top["diff"].astype(
        float
    )

    return {
        "avg_diff": float(
            d.mean()
        ),
        "win_rate": float(
            (d > 0).mean()
            * 100
        ),
        "plus2000_rate": float(
            (d >= 2000).mean()
            * 100
        ),
        "positive": int(
            d.sum() > 0
        ),
        "total_diff": float(
            d.sum()
        ),
    }


def load_robust_weights():

    if not OPT_FILE.exists():

        raise FileNotFoundError(
            "Optimization result not found."
        )

    opt = pd.read_csv(
        OPT_FILE,
        encoding="utf-8-sig"
    )

    groups = {}

    for n, name in [
        (5, "TOP5_MEAN"),
        (10, "TOP10_MEAN"),
        (20, "TOP20_MEAN"),
    ]:

        x = opt.head(n)

        w = {}

        for factor in FACTORS:

            col = (
                "w_"
                + factor
            )

            w[factor] = float(
                x[col].mean()
            )

        total = sum(
            w.values()
        )

        if total > 0:

            w = {
                k: v / total
                for k, v in w.items()
            }

        groups[name] = w

    # Equal baseline
    equal_weight = (
        1.0
        / len(FACTORS)
    )

    groups[
        "EQUAL_BASELINE"
    ] = {
        factor:
            equal_weight
        for factor in FACTORS
    }

    return groups


def evaluate_model(
    panels,
    model_name,
    weights
):

    rows = []

    for date, panel in panels.items():

        if date < FIRST_TEST_DATE:
            continue

        if date > TEST_END:
            continue

        for top_n in [
            1,
            5,
            10,
            20,
            30
        ]:

            r = evaluate(
                panel,
                weights,
                top_n
            )

            if r is None:
                continue

            rows.append({

                "model":
                    model_name,

                "date":
                    date.date(),

                "top_n":
                    top_n,

                **r
            })

    return pd.DataFrame(
        rows
    )


def summarize(
    daily
):

    rows = []

    for model in sorted(
        daily["model"].unique()
    ):

        for top_n in [
            1,
            5,
            10,
            20,
            30
        ]:

            x = daily[
                (daily["model"] == model)
                & (daily["top_n"] == top_n)
            ]

            if x.empty:
                continue

            rows.append({

                "model":
                    model,

                "top_n":
                    top_n,

                "days":
                    len(x),

                "avg_diff":
                    float(
                        x["avg_diff"].mean()
                    ),

                "median_daily_avg":
                    float(
                        x["avg_diff"].median()
                    ),

                "win_rate":
                    float(
                        x["win_rate"].mean()
                    ),

                "plus2000_rate":
                    float(
                        x["plus2000_rate"].mean()
                    ),

                "positive_days":
                    float(
                        x["positive"].mean()
                        * 100
                    ),

                "total_diff":
                    float(
                        x["total_diff"].sum()
                    ),
            })

    return pd.DataFrame(
        rows
    )


def main():

    print("=" * 70)

    print(
        "Ver.3 Robust Weight "
        "Direct Comparison"
    )

    print("=" * 70)

    df = load_data()

    print(
        "records = %s"
        % format(
            len(df),
            ","
        )
    )

    print()

    print(
        "Building daily panels..."
    )

    panels = {}

    for target_date in pd.date_range(
        TRAIN_START,
        TEST_END
    ):

        panel = build_features(
            df,
            target_date
        )

        if not panel.empty:

            panels[
                target_date
            ] = panel

            print(
                "%s machines=%d"
                % (
                    target_date.date(),
                    len(panel)
                )
            )

    print()

    weights = load_robust_weights()

    print(
        "Models = %d"
        % len(weights)
    )

    for name, w in weights.items():

        print(
            "%-18s weight_sum=%.6f"
            % (
                name,
                sum(w.values())
            )
        )

    print()

    all_daily = []

    for name, w in weights.items():

        print(
            "Evaluating %s..."
            % name
        )

        result = evaluate_model(
            panels,
            name,
            w
        )

        if not result.empty:

            all_daily.append(
                result
            )

    daily = pd.concat(
        all_daily,
        ignore_index=True
    )

    summary = summarize(
        daily
    )

    print()
    print(
        "=" * 70
    )
    print(
        "DIRECT COMPARISON RESULT"
    )
    print(
        "=" * 70
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "TOP10 COMPARISON"
    )

    top10 = summary[
        summary["top_n"] == 10
    ].sort_values(
        "avg_diff",
        ascending=False
    )

    print(
        top10.to_string(
            index=False
        )
    )

    print()
    print(
        "TOP10 RANKING BY TOTAL DIFF"
    )

    top10_total = summary[
        summary["top_n"] == 10
    ].sort_values(
        "total_diff",
        ascending=False
    )

    print(
        top10_total[
            [
                "model",
                "days",
                "avg_diff",
                "win_rate",
                "positive_days",
                "total_diff"
            ]
        ].to_string(
            index=False
        )
    )

    # Save
    out_daily = (
        OUT_DIR
        / "14_Ver3_robust_compare_daily.csv"
    )

    out_summary = (
        OUT_DIR
        / "14_Ver3_robust_compare_summary.csv"
    )

    out_weights = (
        OUT_DIR
        / "14_Ver3_robust_compare_weights.csv"
    )

    daily.to_csv(
        out_daily,
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        out_summary,
        index=False,
        encoding="utf-8-sig"
    )

    weight_rows = []

    for model, w in weights.items():

        for factor in FACTORS:

            weight_rows.append({

                "model":
                    model,

                "factor":
                    factor,

                "weight":
                    w[factor]
            })

    pd.DataFrame(
        weight_rows
    ).to_csv(
        out_weights,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        "Saved:"
    )

    print(out_daily)
    print(out_summary)
    print(out_weights)

    print()
    print(
        "Robust comparison complete."
    )


if __name__ == "__main__":
    main()