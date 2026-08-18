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

START = pd.Timestamp("2026-07-11")
TEST_START = pd.Timestamp("2026-07-26")
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

# ============================================================
# Ver.4 固定ウェイト
# TOP20_MEANから算出
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
        "CSV read failed: "
        + str(path)
    )


def load_data():

    frames = []

    for path in (
        CSV1,
        CSV2
    ):

        if path.exists():

            frames.append(
                read_csv(path)
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

        for col in cols:

            if col in df.columns:

                return col

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
        & (df["date"] <= TEST_END)
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

    # 未来データを絶対に使わない
    hist = df[
        df["date"] < target_date
    ].copy()

    actual = df[
        df["date"] == target_date
    ][
        [
            "machine_no",
            "machine_name",
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
        })

    feat = pd.DataFrame(
        rows
    )

    return feat.merge(
        actual,
        on=[
            "machine_no",
            "machine_name"
        ],
        how="inner"
    )


def zscore(series):

    s = pd.to_numeric(
        series,
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


def calculate_score(
    df
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
            * V4_WEIGHTS[factor]
        )

    x["score"] = score

    return x.sort_values(
        "score",
        ascending=False
    )


def evaluate_top(
    panel,
    top_n
):

    ranked = calculate_score(
        panel
    )

    top = ranked.head(
        top_n
    )

    d = top["diff"].astype(
        float
    )

    return {

        "avg_diff":
            float(d.mean()),

        "median_diff":
            float(d.median()),

        "win_rate":
            float(
                (d > 0).mean()
                * 100
            ),

        "plus1000_rate":
            float(
                (d >= 1000).mean()
                * 100
            ),

        "plus2000_rate":
            float(
                (d >= 2000).mean()
                * 100
            ),

        "total_diff":
            float(d.sum()),

        "positive":
            int(d.sum() > 0),
    }


def main():

    print("=" * 70)

    print(
        "Ana-Slo Ver.4 "
        "Fixed Weight OOS Backtest"
    )

    print("=" * 70)

    print()

    print(
        "FIXED WEIGHTS"
    )

    print("-" * 70)

    for factor in FACTORS:

        print(
            "%-18s : %6.2f%%"
            % (
                factor,
                V4_WEIGHTS[factor]
                * 100
            )
        )

    print(
        "weight sum       : %6.2f%%"
        % (
            sum(
                V4_WEIGHTS.values()
            )
            * 100
        )
    )

    print()

    df = load_data()

    print(
        "records = %s"
        % format(
            len(df),
            ","
        )
    )

    print(
        "OOS period = %s to %s"
        % (
            TEST_START.date(),
            TEST_END.date()
        )
    )

    print()

    rows = []

    for target_date in pd.date_range(
        TEST_START,
        TEST_END
    ):

        panel = build_features(
            df,
            target_date
        )

        if panel.empty:

            continue

        print(
            "%s machines=%d"
            % (
                target_date.date(),
                len(panel)
            )
        )

        for top_n in [
            1,
            5,
            10,
            20,
            30
        ]:

            result = evaluate_top(
                panel,
                top_n
            )

            rows.append({

                "date":
                    target_date.date(),

                "top_n":
                    top_n,

                **result
            })

    daily = pd.DataFrame(
        rows
    )

    summary_rows = []

    for top_n in [
        1,
        5,
        10,
        20,
        30
    ]:

        x = daily[
            daily["top_n"]
            == top_n
        ]

        summary_rows.append({

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

            "plus1000_rate":
                float(
                    x["plus1000_rate"].mean()
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

    summary = pd.DataFrame(
        summary_rows
    )

    print()
    print("=" * 70)
    print(
        "VER.4 OOS RESULT"
    )
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "TOP10 DAILY RESULT"
    )

    top10_daily = daily[
        daily["top_n"] == 10
    ][
        [
            "date",
            "avg_diff",
            "median_diff",
            "win_rate",
            "plus1000_rate",
            "plus2000_rate",
            "total_diff"
        ]
    ]

    print(
        top10_daily.to_string(
            index=False
        )
    )

    out_daily = (
        OUT_DIR
        / "15_Ver4_OOS_daily.csv"
    )

    out_summary = (
        OUT_DIR
        / "15_Ver4_OOS_summary.csv"
    )

    out_weights = (
        OUT_DIR
        / "15_Ver4_fixed_weights.csv"
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

    pd.DataFrame(
        [
            {
                "factor":
                    factor,

                "weight":
                    V4_WEIGHTS[
                        factor
                    ]
            }

            for factor in FACTORS
        ]
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
        "Ver.4 OOS backtest complete."
    )


if __name__ == "__main__":
    main()