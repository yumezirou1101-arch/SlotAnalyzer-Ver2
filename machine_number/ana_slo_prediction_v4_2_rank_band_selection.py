from pathlib import Path
import pandas as pd
import numpy as np


print("=" * 70)
print("Ana-Slo Ver.4.2 Rank Band Selection Test")
print("=" * 70)


# ============================================================
# PATH
# ============================================================

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

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CSV1 = DATA_DIR / "ana_slo_20260711.csv"
CSV2 = DATA_DIR / "ana_slo_20260712_20260810.csv"

START = pd.Timestamp("2026-07-11")
TEST_START = pd.Timestamp("2026-07-21")
TEST_END = pd.Timestamp("2026-08-10")


# ============================================================
# V4.2_C BASE WEIGHTS
# recent7_win / bounce_signal excluded
# ============================================================

BASE_WEIGHTS = {

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


EXCLUDED = {
    "recent7_win",
    "bounce_signal",
}


FACTORS = [
    x
    for x in BASE_WEIGHTS
    if x not in EXCLUDED
]


weight_sum = sum(
    BASE_WEIGHTS[x]
    for x in FACTORS
)


WEIGHTS = {
    x: BASE_WEIGHTS[x] / weight_sum
    for x in FACTORS
}


# ============================================================
# OUTPUT
# ============================================================

OUT_DAILY = (
    OUT_DIR
    / "35_Ver4_2_rank_band_selection_daily.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "35_Ver4_2_rank_band_selection_summary.csv"
)

OUT_COMPARE = (
    OUT_DIR
    / "35_Ver4_2_rank_band_selection_compare.csv"
)

OUT_DIAGNOSTIC = (
    OUT_DIR
    / "35_Ver4_2_rank_band_selection_diagnostic.csv"
)


# ============================================================
# CSV
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
                encoding=enc
            )

        except Exception:
            pass

    raise RuntimeError(
        f"CSV read failed: {path}"
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    frames = []

    for path in (
        CSV1,
        CSV2,
    ):

        if path.exists():

            print(
                f"Loading: {path}"
            )

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
        [
            "date",
            "日付",
        ]
    )

    no_col = find(
        [
            "machine_no",
            "台番号",
        ]
    )

    name_col = find(
        [
            "machine_name",
            "機種名",
        ]
    )

    diff_col = find(
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

    df = df[
        (df["date"] >= START)
        &
        (df["date"] <= TEST_END)
    ].copy()

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

    print(
        f"records = {len(df):,}"
    )

    return df


# ============================================================
# Z SCORE
# ============================================================

def zscore(series):

    s = pd.to_numeric(
        series,
        errors="coerce"
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
            index=s.index
        )

    return (
        s - s.mean()
    ) / std


# ============================================================
# BUILD PANEL
# ============================================================

def build_panel(
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
            "diff",
        ]
    ].copy()

    if hist.empty or actual.empty:
        return pd.DataFrame()

    latest_date = hist["date"].max()

    latest_machine = (
        hist[
            hist["date"] == latest_date
        ][
            [
                "machine_no",
                "machine_name",
            ]
        ]
        .drop_duplicates(
            "machine_no",
            keep="last"
        )
    )

    actual = actual.merge(
        latest_machine,
        on="machine_no",
        how="left",
        suffixes=(
            "_today",
            "_hist",
        )
    )

    actual["machine_change"] = (
        actual["machine_name_hist"].isna()
        |
        (
            actual["machine_name_today"]
            !=
            actual["machine_name_hist"]
        )
    )

    actual = actual[
        ~actual["machine_change"]
    ].copy()

    if actual.empty:
        return pd.DataFrame()

    actual["machine_name"] = (
        actual["machine_name_today"]
    )

    type_stats = (
        hist
        .groupby(
            "machine_name"
        )["diff"]
        .mean()
        .to_dict()
    )

    target_weekday = (
        target_date.dayofweek
    )

    latest_day = (
        hist[
            hist["date"] == latest_date
        ]
        .set_index(
            "machine_no"
        )
    )

    rows = []

    for no, m in hist.groupby(
        "machine_no"
    ):

        m = m.sort_values(
            "date"
        ).copy()

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

            weekday_raw = float(
                wd["diff"].mean()
            )

        else:

            weekday_raw = avg31

        prior_n = 15.0

        wd_weight = (
            weekday_n
            /
            (
                weekday_n
                + prior_n
            )
        )

        weekday_avg = (
            weekday_raw * wd_weight
            +
            avg31
            *
            (
                1.0
                - wd_weight
            )
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
                avg31
            )
        )

        neighbor_values = []

        for n2 in (
            no - 1,
            no + 1,
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
            }
        )

    feat = pd.DataFrame(
        rows
    )

    if feat.empty:
        return pd.DataFrame()

    panel = feat.merge(
        actual[
            [
                "machine_no",
                "machine_name",
                "diff",
            ]
        ],
        on=[
            "machine_no",
            "machine_name",
        ],
        how="inner"
    )

    return panel


# ============================================================
# SCORE
# ============================================================

def calculate_score(panel):

    panel = panel.copy()

    for factor in FACTORS:

        panel[
            factor + "_z"
        ] = zscore(
            panel[factor]
        )

    panel["score"] = 0.0

    for factor in FACTORS:

        panel["score"] += (
            panel[
                factor + "_z"
            ]
            *
            WEIGHTS[factor]
        )

    panel = panel.sort_values(
        [
            "score",
            "machine_no",
        ],
        ascending=[
            False,
            True,
        ]
    ).reset_index(
        drop=True
    )

    panel["rank"] = (
        np.arange(
            len(panel)
        )
        + 1
    )

    panel["win"] = (
        panel["diff"] > 0
    ).astype(int)

    panel["plus1000"] = (
        panel["diff"] >= 1000
    ).astype(int)

    panel["plus2000"] = (
        panel["diff"] >= 2000
    ).astype(int)

    return panel


# ============================================================
# SELECTION RULES
# ============================================================

RULES = {

    "TOP10":
        list(range(1, 11)),

    "TOP4_10":
        list(range(4, 11)),

    "TOP4_5":
        list(range(4, 6)),

    "TOP5_10":
        list(range(5, 11)),

    "TOP6_10":
        list(range(6, 11)),

    "TOP4_8":
        list(range(4, 9)),

    "TOP3_PLUS_TOP4_10":
        list(range(1, 11)),

}


# TOP3_PLUS_TOP4_10 is intentionally equivalent
# to TOP10 and is retained as a control rule.


# ============================================================
# BUILD DAILY PANELS
# ============================================================

df = load_data()

print()
print(
    "MODEL = V4.2_C"
)

print(
    "Excluded = "
    "recent7_win, bounce_signal"
)

print(
    f"OOS = "
    f"{TEST_START.date()} "
    f"to "
    f"{TEST_END.date()}"
)

print()
print(
    "SELECTION RULES"
)

for name, ranks in RULES.items():

    print(
        f"{name:<24} "
        f"ranks={ranks}"
    )


all_daily = []

test_dates = pd.date_range(
    TEST_START,
    TEST_END,
    freq="D"
)

print()
print(
    "Building daily ranking panels..."
)


for target_date in test_dates:

    panel = build_panel(
        df,
        target_date
    )

    if panel.empty:

        print(
            f"{target_date.date()} "
            f"eligible=0"
        )

        continue

    panel = calculate_score(
        panel
    )

    panel["date"] = target_date

    all_daily.append(
        panel
    )

    print(
        f"{target_date.date()} "
        f"eligible={len(panel)}"
    )


daily = pd.concat(
    all_daily,
    ignore_index=True
)


# ============================================================
# DAILY RULE RESULTS
# ============================================================

daily_rows = []


for target_date, day in daily.groupby(
    "date"
):

    day = day.sort_values(
        "rank"
    )

    for rule_name, ranks in RULES.items():

        selected = day[
            day["rank"].isin(ranks)
        ].copy()

        if selected.empty:
            continue

        daily_rows.append(
            {
                "date":
                    target_date,

                "rule":
                    rule_name,

                "machines":
                    len(selected),

                "avg_diff":
                    selected["diff"].mean(),

                "median_diff":
                    selected["diff"].median(),

                "win_rate":
                    selected["win"].mean()
                    * 100,

                "plus1000_rate":
                    selected["plus1000"].mean()
                    * 100,

                "plus2000_rate":
                    selected["plus2000"].mean()
                    * 100,

                "total_diff":
                    selected["diff"].sum(),
            }
        )


daily_result = pd.DataFrame(
    daily_rows
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


for rule_name in RULES:

    sub = daily_result[
        daily_result["rule"]
        == rule_name
    ].copy()

    if sub.empty:
        continue

    avg_series = (
        sub["avg_diff"]
    )

    positive_days = int(
        (
            avg_series > 0
        ).sum()
    )

    negative_days = int(
        (
            avg_series < 0
        ).sum()
    )

    tie_days = int(
        (
            avg_series == 0
        ).sum()
    )

    max_losing = 0
    current_losing = 0

    max_winning = 0
    current_winning = 0

    for value in avg_series:

        if value < 0:
            current_losing += 1
        else:
            current_losing = 0

        max_losing = max(
            max_losing,
            current_losing
        )

        if value > 0:
            current_winning += 1
        else:
            current_winning = 0

        max_winning = max(
            max_winning,
            current_winning
        )

    cumulative = (
        sub["total_diff"]
        .cumsum()
    )

    peak = (
        cumulative
        .cummax()
    )

    drawdown = (
        cumulative
        - peak
    )

    summary_rows.append(
        {
            "rule":
                rule_name,

            "days":
                len(sub),

            "machines_per_day":
                sub["machines"].mean(),

            "avg_diff":
                sub["avg_diff"].mean(),

            "median_daily_avg":
                sub["avg_diff"].median(),

            "std_daily_avg":
                sub["avg_diff"].std(
                    ddof=0
                ),

            "best_day":
                sub["avg_diff"].max(),

            "worst_day":
                sub["avg_diff"].min(),

            "win_rate":
                sub["win_rate"].mean(),

            "plus1000_rate":
                sub["plus1000_rate"].mean(),

            "plus2000_rate":
                sub["plus2000_rate"].mean(),

            "positive_days":
                positive_days,

            "negative_days":
                negative_days,

            "tie_days":
                tie_days,

            "positive_day_rate":
                positive_days
                / len(sub)
                * 100,

            "max_losing_streak":
                max_losing,

            "max_winning_streak":
                max_winning,

            "total_diff":
                sub["total_diff"].sum(),

            "per_machine_avg_diff":
                sub["total_diff"].sum()
                /
                sub["machines"].sum(),

            "max_drawdown":
                drawdown.min(),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# RANKING
# ============================================================

summary["avg_rank"] = (
    summary["avg_diff"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

summary["total_rank"] = (
    summary["total_diff"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

summary["stability_rank"] = (
    summary["positive_day_rate"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


# ============================================================
# VS TOP10
# ============================================================

top10 = summary[
    summary["rule"] == "TOP10"
]

if not top10.empty:

    top10_avg = float(
        top10.iloc[0]["avg_diff"]
    )

    top10_total = float(
        top10.iloc[0]["total_diff"]
    )

else:

    top10_avg = np.nan
    top10_total = np.nan


summary["avg_diff_vs_top10"] = (
    summary["avg_diff"]
    - top10_avg
)

summary["total_diff_vs_top10"] = (
    summary["total_diff"]
    - top10_total
)


# ============================================================
# FIRST / SECOND HALF
# ============================================================

half_rows = []


for rule_name in RULES:

    sub = daily_result[
        daily_result["rule"]
        == rule_name
    ].copy()

    dates = sorted(
        sub["date"].unique()
    )

    midpoint = (
        len(dates)
        // 2
    )

    first = sub[
        sub["date"].isin(
            dates[:midpoint]
        )
    ]

    second = sub[
        sub["date"].isin(
            dates[midpoint:]
        )
    ]

    first_avg = (
        first["avg_diff"].mean()
    )

    second_avg = (
        second["avg_diff"].mean()
    )

    half_rows.append(
        {
            "rule":
                rule_name,

            "first_half_avg":
                first_avg,

            "second_half_avg":
                second_avg,

            "first_second_change":
                second_avg
                - first_avg,

            "first_half_days":
                len(first),

            "second_half_days":
                len(second),
        }
    )


half_df = pd.DataFrame(
    half_rows
)


# ============================================================
# SAVE
# ============================================================

daily_result.to_csv(
    OUT_DAILY,
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)

half_df.to_csv(
    OUT_COMPARE,
    index=False,
    encoding="utf-8-sig"
)


diagnostic = pd.DataFrame(
    {
        "factor": FACTORS,
        "normalized_weight": [
            WEIGHTS[x]
            for x in FACTORS
        ],
    }
)

diagnostic.to_csv(
    OUT_DIAGNOSTIC,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 70)
print("RANK BAND SELECTION RESULT")
print("=" * 70)

print()

display_cols = [
    "rule",
    "machines_per_day",
    "avg_diff",
    "median_daily_avg",
    "best_day",
    "worst_day",
    "win_rate",
    "plus1000_rate",
    "plus2000_rate",
    "positive_day_rate",
    "max_losing_streak",
    "total_diff",
    "per_machine_avg_diff",
    "max_drawdown",
    "avg_rank",
]

print(
    summary[
        display_cols
    ].sort_values(
        "avg_rank"
    ).to_string(
        index=False,
        float_format=lambda x:
            f"{x:.2f}"
    )
)


print()
print("=" * 70)
print("FIRST HALF / SECOND HALF")
print("=" * 70)

print(
    half_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.2f}"
    )
)


print()
print("=" * 70)
print("DIAGNOSTIC")
print("=" * 70)

best = summary.iloc[
    summary["avg_diff"].idxmax()
]

print(
    f"Best rule by average diff : "
    f"{best['rule']}"
)

print(
    f"Average diff              : "
    f"{best['avg_diff']:+.2f}"
)

print(
    f"Total diff                : "
    f"{best['total_diff']:+.0f}"
)

print(
    f"TOP10 average diff        : "
    f"{top10_avg:+.2f}"
)

print(
    f"TOP10 total diff          : "
    f"{top10_total:+.0f}"
)


print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(OUT_DAILY)
print(OUT_SUMMARY)
print(OUT_COMPARE)
print(OUT_DIAGNOSTIC)

print()
print(
    "Ver.4.2 rank band selection test complete."
)