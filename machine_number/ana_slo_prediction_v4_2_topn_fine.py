from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Ana-Slo Ver.4.2 TOP N Fine Optimization
#
# V4.2_C
# exclude = recent7_win, bounce_signal
#
# OOS:
# 2026-07-21 to 2026-08-10
#
# Fine TOP N:
# 1-15, 20
#
# Purpose:
# Determine the practical optimal number of machines.
# ============================================================


print("=" * 70)
print("Ana-Slo Ver.4.2 TOP N Fine Optimization")
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
# V4.2 C WEIGHTS
# ============================================================

BASE_WEIGHTS = {

    "avg31":
        0.0670952025611345,

    "recent7_avg":
        0.05164896703284082,

    "recent7_win":
        0.06602967770818714,

    "last_diff":
        0.12382294629381808,

    "prev_change":
        0.10484738021281044,

    "weekday_avg":
        0.05672674990073483,

    "type_avg":
        0.05843723530102936,

    "plus1000_rate":
        0.17725354845070532,

    "plus2000_rate":
        0.13298938481323394,

    "neighbor_avg":
        0.06161296683628432,

    "bounce_signal":
        0.09953594088922124,
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
    x:
        BASE_WEIGHTS[x] / weight_sum
    for x in FACTORS
}


# ============================================================
# OUTPUT
# ============================================================

OUT_DAILY = (
    OUT_DIR
    / "34_Ver4_2_topn_fine_daily.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "34_Ver4_2_topn_fine_summary.csv"
)

OUT_COMPARE = (
    OUT_DIR
    / "34_Ver4_2_topn_fine_compare.csv"
)

OUT_DIAGNOSTIC = (
    OUT_DIR
    / "34_Ver4_2_topn_fine_diagnostic.csv"
)


# ============================================================
# READ CSV
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
            "譌･莉・",
        ]
    )

    no_col = find(
        [
            "machine_no",
            "台番号",
            "蜿ｰ逡ｪ蜿ｷ",
        ]
    )

    name_col = find(
        [
            "machine_name",
            "機種名",
            "讖溽ｨｮ蜷・",
        ]
    )

    diff_col = find(
        [
            "diff",
            "差枚",
            "蟾ｮ譫・",
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
            date_col:
                "date",

            no_col:
                "machine_no",

            name_col:
                "machine_name",

            diff_col:
                "diff",
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
# Z-SCORE
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
            hist["date"]
            == latest_date
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
        actual["machine_name_hist"]
        .isna()
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
            hist["date"]
            == latest_date
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
            weekday_raw
            * wd_weight
            +
            avg31
            * (
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

def calculate_score(
    panel
):

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
    "TOP N = "
    "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20]"
)

all_daily = []

diagnostics = []

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

    panel["date"] = (
        target_date
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if (
        panel["rank"]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate rank detected."
        )

    expected = set(
        range(
            1,
            len(panel) + 1
        )
    )

    actual_ranks = set(
        panel["rank"].astype(int)
    )

    if expected != actual_ranks:

        raise RuntimeError(
            "Rank coverage error."
        )

    all_daily.append(
        panel
    )

    diagnostics.append(
        {
            "date":
                target_date,

            "eligible":
                len(panel),

            "top1":
                int(
                    (
                        panel["rank"]
                        <= 1
                    ).sum()
                ),

            "top5":
                int(
                    (
                        panel["rank"]
                        <= 5
                    ).sum()
                ),

            "top10":
                int(
                    (
                        panel["rank"]
                        <= 10
                    ).sum()
                ),
        }
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
# DAILY TOP N RESULTS
# ============================================================

TOP_NS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    20,
]


daily_rows = []


for target_date, day in daily.groupby(
    "date"
):

    day = day.sort_values(
        "rank"
    )

    for top_n in TOP_NS:

        selected = day[
            day["rank"]
            <= top_n
        ]

        if selected.empty:

            continue

        daily_rows.append(
            {
                "date":
                    target_date,

                "top_n":
                    top_n,

                "machines":
                    len(selected),

                "avg_diff":
                    selected["diff"]
                    .mean(),

                "median_diff":
                    selected["diff"]
                    .median(),

                "win_rate":
                    selected["win"]
                    .mean()
                    * 100,

                "plus1000_rate":
                    selected["plus1000"]
                    .mean()
                    * 100,

                "plus2000_rate":
                    selected["plus2000"]
                    .mean()
                    * 100,

                "total_diff":
                    selected["diff"]
                    .sum(),
            }
        )


daily_result = pd.DataFrame(
    daily_rows
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


for top_n in TOP_NS:

    sub = daily_result[
        daily_result["top_n"]
        == top_n
    ].copy()

    if sub.empty:

        continue

    daily_avg = (
        sub["avg_diff"]
    )

    positive_days = int(
        (
            daily_avg > 0
        ).sum()
    )

    negative_days = int(
        (
            daily_avg < 0
        ).sum()
    )

    tie_days = int(
        (
            daily_avg == 0
        ).sum()
    )

    # --------------------------------------------------------
    # Losing streak
    # --------------------------------------------------------

    max_losing = 0
    current_losing = 0

    max_winning = 0
    current_winning = 0

    for value in daily_avg:

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

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

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

    max_drawdown = float(
        drawdown.min()
    )

    # --------------------------------------------------------
    # Best / worst day
    # --------------------------------------------------------

    best_day = float(
        daily_avg.max()
    )

    worst_day = float(
        daily_avg.min()
    )

    summary_rows.append(
        {
            "top_n":
                top_n,

            "days":
                len(sub),

            "avg_diff":
                sub["avg_diff"]
                .mean(),

            "median_daily_avg":
                sub["avg_diff"]
                .median(),

            "std_daily_avg":
                sub["avg_diff"]
                .std(
                    ddof=0
                ),

            "best_day":
                best_day,

            "worst_day":
                worst_day,

            "win_rate":
                sub["win_rate"]
                .mean(),

            "plus1000_rate":
                sub["plus1000_rate"]
                .mean(),

            "plus2000_rate":
                sub["plus2000_rate"]
                .mean(),

            "positive_days":
                positive_days,

            "negative_days":
                negative_days,

            "tie_days":
                tie_days,

            "positive_day_rate":
                positive_days
                /
                len(sub)
                * 100,

            "max_losing_streak":
                max_losing,

            "max_winning_streak":
                max_winning,

            "total_diff":
                sub["total_diff"]
                .sum(),

            "per_machine_avg_diff":
                sub["total_diff"]
                .sum()
                /
                sub["machines"]
                .sum(),

            "max_drawdown":
                max_drawdown,
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# COMPARE VS TOP10
# ============================================================

top10_row = summary[
    summary["top_n"] == 10
]

if not top10_row.empty:

    top10_avg = float(
        top10_row.iloc[0]["avg_diff"]
    )

    top10_total = float(
        top10_row.iloc[0]["total_diff"]
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


summary["ranking"] = (
    summary["avg_diff"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


# ============================================================
# FIRST HALF / SECOND HALF
# ============================================================

half_rows = []


for top_n in TOP_NS:

    sub = daily_result[
        daily_result["top_n"]
        == top_n
    ].copy()

    if sub.empty:

        continue

    dates = sorted(
        sub["date"].unique()
    )

    midpoint = (
        len(dates)
        // 2
    )

    first_dates = set(
        dates[:midpoint]
    )

    second_dates = set(
        dates[midpoint:]
    )

    first = sub[
        sub["date"]
        .isin(first_dates)
    ]

    second = sub[
        sub["date"]
        .isin(second_dates)
    ]

    first_avg = float(
        first["avg_diff"].mean()
    )

    second_avg = float(
        second["avg_diff"].mean()
    )

    half_rows.append(
        {
            "top_n":
                top_n,

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
# TOP N PEAK
# ============================================================

best_row = summary.loc[
    summary["avg_diff"].idxmax()
]

best_top_n = int(
    best_row["top_n"]
)

best_avg = float(
    best_row["avg_diff"]
)

best_total = float(
    best_row["total_diff"]
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

pd.DataFrame(
    diagnostics
).to_csv(
    OUT_DIAGNOSTIC,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 70)
print("TOP N FINE OPTIMIZATION RESULT")
print("=" * 70)

print()

display_cols = [
    "top_n",
    "days",
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
    "ranking",
]

print(
    summary[
        display_cols
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.2f}"
    )
)


print()
print("=" * 70)
print("FIRST HALF / SECOND HALF")
print("=" * 70)

print()

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

print(
    f"Best TOP N by average diff : "
    f"TOP{best_top_n}"
)

print(
    f"Average diff               : "
    f"{best_avg:+.2f}"
)

print(
    f"Total diff                 : "
    f"{best_total:+.0f}"
)

print(
    f"TOP10 average diff         : "
    f"{top10_avg:+.2f}"
)

print(
    f"TOP10 total diff           : "
    f"{top10_total:+.0f}"
)


print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    OUT_DAILY
)

print(
    OUT_SUMMARY
)

print(
    OUT_COMPARE
)

print(
    OUT_DIAGNOSTIC
)

print()
print(
    "Ver.4.2 TOP N fine optimization complete."
)