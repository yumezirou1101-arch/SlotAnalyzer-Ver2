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

RANDOM_TRIALS = 5000
TOP_N = 10
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

# ============================================================
# Ver.4 固定ウェイト
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
        "CSV read failed: " + str(path)
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
        df["machine_no"].astype(int)
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

    feat = pd.DataFrame(rows)

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


def calculate_score(df):

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
            50.0 + z * 12.5
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


def evaluate_selection(
    selected
):

    d = selected["diff"].astype(float)

    return {
        "avg_diff":
            float(d.mean()),

        "win_rate":
            float(
                (d > 0).mean() * 100
            ),

        "plus1000_rate":
            float(
                (d >= 1000).mean() * 100
            ),

        "plus2000_rate":
            float(
                (d >= 2000).mean() * 100
            ),

        "total_diff":
            float(d.sum()),
    }


def percentile_rank(
    value,
    samples
):

    samples = np.asarray(
        samples,
        dtype=float
    )

    return float(
        (samples < value).mean()
        * 100
    )


def main():

    print("=" * 70)
    print(
        "Ver.4 TOP10 "
        "Random Significance Test"
    )
    print("=" * 70)

    print()
    print(
        "random trials per day = %d"
        % RANDOM_TRIALS
    )

    print(
        "random seed = %d"
        % RANDOM_SEED
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

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    daily_rows = []
    random_rows = []

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

        ranked = calculate_score(
            panel
        )

        ver4_top = ranked.head(
            TOP_N
        )

        ver4 = evaluate_selection(
            ver4_top
        )

        random_avgs = np.empty(
            RANDOM_TRIALS
        )

        random_wins = np.empty(
            RANDOM_TRIALS
        )

        random_plus1000 = np.empty(
            RANDOM_TRIALS
        )

        random_plus2000 = np.empty(
            RANDOM_TRIALS
        )

        random_totals = np.empty(
            RANDOM_TRIALS
        )

        n = len(panel)

        diffs = (
            panel["diff"]
            .astype(float)
            .to_numpy()
        )

        for i in range(
            RANDOM_TRIALS
        ):

            idx = rng.choice(
                n,
                size=TOP_N,
                replace=False
            )

            d = diffs[idx]

            random_avgs[i] = d.mean()

            random_wins[i] = (
                (d > 0).mean()
                * 100
            )

            random_plus1000[i] = (
                (d >= 1000).mean()
                * 100
            )

            random_plus2000[i] = (
                (d >= 2000).mean()
                * 100
            )

            random_totals[i] = d.sum()

        daily_rows.append({

            "date":
                target_date.date(),

            "machines":
                len(panel),

            "ver4_avg_diff":
                ver4["avg_diff"],

            "ver4_win_rate":
                ver4["win_rate"],

            "ver4_plus1000_rate":
                ver4["plus1000_rate"],

            "ver4_plus2000_rate":
                ver4["plus2000_rate"],

            "ver4_total_diff":
                ver4["total_diff"],

            "random_mean_avg":
                float(
                    random_avgs.mean()
                ),

            "random_std_avg":
                float(
                    random_avgs.std()
                ),

            "random_p95_avg":
                float(
                    np.percentile(
                        random_avgs,
                        95
                    )
                ),

            "random_p99_avg":
                float(
                    np.percentile(
                        random_avgs,
                        99
                    )
                ),

            "ver4_percentile_avg":
                percentile_rank(
                    ver4["avg_diff"],
                    random_avgs
                ),

            "ver4_percentile_total":
                percentile_rank(
                    ver4["total_diff"],
                    random_totals
                ),

            "random_mean_win":
                float(
                    random_wins.mean()
                ),

            "random_mean_plus1000":
                float(
                    random_plus1000.mean()
                ),

            "random_mean_plus2000":
                float(
                    random_plus2000.mean()
                ),
        })

        for i in range(
            RANDOM_TRIALS
        ):

            random_rows.append({

                "date":
                    target_date.date(),

                "trial":
                    i + 1,

                "avg_diff":
                    random_avgs[i],

                "win_rate":
                    random_wins[i],

                "plus1000_rate":
                    random_plus1000[i],

                "plus2000_rate":
                    random_plus2000[i],

                "total_diff":
                    random_totals[i],
            })

    daily = pd.DataFrame(
        daily_rows
    )

    random_daily = pd.DataFrame(
        random_rows
    )

    print()
    print("=" * 70)
    print(
        "VER.4 vs RANDOM"
    )
    print("=" * 70)

    print(
        daily.to_string(
            index=False
        )
    )

    print()
    print(
        "===== OVERALL ====="
    )

    ver4_avg = float(
        daily[
            "ver4_avg_diff"
        ].mean()
    )

    random_avg = float(
        daily[
            "random_mean_avg"
        ].mean()
    )

    ver4_total = float(
        daily[
            "ver4_total_diff"
        ].sum()
    )

    random_total = float(
        (
            daily[
                "random_mean_avg"
            ] * TOP_N
        ).sum()
    )

    avg_lift = (
        ver4_avg - random_avg
    )

    total_lift = (
        ver4_total
        - random_total
    )

    percentile_avg = float(
        daily[
            "ver4_percentile_avg"
        ].mean()
    )

    percentile_total = float(
        daily[
            "ver4_percentile_total"
        ].mean()
    )

    print(
        "Ver.4 TOP10 avg diff      : "
        "%.2f" % ver4_avg
    )

    print(
        "Random TOP10 avg diff     : "
        "%.2f" % random_avg
    )

    print(
        "Average lift              : "
        "%+.2f" % avg_lift
    )

    print(
        "Ver.4 total diff          : "
        "%+.0f" % ver4_total
    )

    print(
        "Random expected total     : "
        "%+.0f" % random_total
    )

    print(
        "Total lift                : "
        "%+.0f" % total_lift
    )

    print(
        "Mean percentile vs random : "
        "%.2f%%" % percentile_avg
    )

    print(
        "Total percentile vs random: "
        "%.2f%%" % percentile_total
    )

    print()

    if percentile_avg >= 95:

        judgment = (
            "STRONG: "
            "Ver.4 is substantially "
            "better than random."
        )

    elif percentile_avg >= 80:

        judgment = (
            "PROMISING: "
            "Ver.4 appears better "
            "than random."
        )

    elif percentile_avg >= 60:

        judgment = (
            "WEAK: "
            "some evidence, "
            "but not strong."
        )

    else:

        judgment = (
            "NO CLEAR EDGE: "
            "random selection "
            "cannot be rejected."
        )

    print(
        "JUDGMENT:"
    )

    print(judgment)

    out_daily = (
        OUT_DIR
        / "16_Ver4_random_test_daily.csv"
    )

    out_random = (
        OUT_DIR
        / "16_Ver4_random_test_samples.csv"
    )

    daily.to_csv(
        out_daily,
        index=False,
        encoding="utf-8-sig"
    )

    random_daily.to_csv(
        out_random,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        "Saved:"
    )

    print(out_daily)
    print(out_random)

    print()
    print(
        "Random significance test complete."
    )


if __name__ == "__main__":
    main()