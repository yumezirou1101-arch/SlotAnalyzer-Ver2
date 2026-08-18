from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Ana-Slo Ver.4.2 Score Band Analysis - FIXED
#
# V4.2_C
# exclude = recent7_win, bounce_signal
#
# IMPORTANT:
# Score bands are mutually exclusive.
#
# TOP1
# TOP1-2
# TOP2-5
# TOP5-10
# TOP10-20
# TOP20-30
# TOP30-50
# BOTTOM50
#
# Machine-change machines are excluded from prediction.
# ============================================================


print("=" * 70)
print("Ana-Slo Ver.4.2 Score Band Analysis - FIXED")
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
# V4.2 C
#
# V4 BASE weights
# recent7_win and bounce_signal are removed.
# Remaining weights are re-normalized.
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


EXCLUDED_FACTORS = {
    "recent7_win",
    "bounce_signal",
}


FACTORS = [
    factor
    for factor in BASE_WEIGHTS
    if factor not in EXCLUDED_FACTORS
]


weight_sum = sum(
    BASE_WEIGHTS[factor]
    for factor in FACTORS
)


WEIGHTS = {
    factor:
        BASE_WEIGHTS[factor] / weight_sum
    for factor in FACTORS
}


# ============================================================
# OUTPUT
# ============================================================

OUT_DAILY = (
    OUT_DIR
    / "33_Ver4_2_score_band_fixed_daily.csv"
)

OUT_BAND = (
    OUT_DIR
    / "33_Ver4_2_score_band_fixed_band.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "33_Ver4_2_score_band_fixed_summary.csv"
)

OUT_SELECTION = (
    OUT_DIR
    / "33_Ver4_2_score_band_fixed_selection.csv"
)

OUT_DIAGNOSTIC = (
    OUT_DIR
    / "33_Ver4_2_score_band_fixed_diagnostic.csv"
)


# ============================================================
# CSV READER
# ============================================================

def read_csv(path):

    print(
        f"Loading: {path}"
    )

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "cp932",
    ):

        try:

            return pd.read_csv(
                path,
                encoding=encoding
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

    print(
        f"records = {len(df):,}"
    )

    # --------------------------------------------------------
    # Find columns
    # --------------------------------------------------------

    def find(
        candidates
    ):

        for col in candidates:

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
            "Required columns not found.\n"
            f"columns = {list(df.columns)}"
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

    # --------------------------------------------------------
    # Convert
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Duplicate protection
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        [
            "date",
            "machine_no",
        ],
        keep="last"
    )

    after = len(df)

    if before != after:

        print(
            "duplicates removed = "
            f"{before - after}"
        )

    # --------------------------------------------------------
    # Basic target variables
    # --------------------------------------------------------

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


# ============================================================
# Z-SCORE
# ============================================================

def zscore(series):

    s = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0.0)

    mean = float(
        s.mean()
    )

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
        (s - mean)
        / std
    )


# ============================================================
# BUILD FEATURE PANEL
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

    # --------------------------------------------------------
    # Machine-change detection
    #
    # A machine is eligible only if its latest historical
    # machine name matches today's machine name.
    # --------------------------------------------------------

    latest_hist_date = hist["date"].max()

    latest_machine = (
        hist[
            hist["date"]
            == latest_hist_date
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
        .notna()
        &
        (
            actual["machine_name_today"]
            !=
            actual["machine_name_hist"]
        )
    )

    # If there is no historical machine record,
    # also exclude it.
    actual.loc[
        actual["machine_name_hist"].isna(),
        "machine_change"
    ] = True

    actual["machine_name"] = (
        actual["machine_name_today"]
    )

    actual = actual[
        ~actual["machine_change"]
    ].copy()

    if actual.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Historical type average
    # --------------------------------------------------------

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
            == latest_hist_date
        ]
        .set_index(
            "machine_no"
        )
    )

    rows = []

    # --------------------------------------------------------
    # Feature calculation
    # --------------------------------------------------------

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

        recent7 = (
            m.tail(7)
        )

        recent7_avg = float(
            recent7["diff"].mean()
        )

        # V4.2_C excludes recent7_win,
        # but calculate nothing for the model.

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

        # ----------------------------------------------------
        # Weekday
        # ----------------------------------------------------

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
            /
            (
                weekday_n
                + prior_n
            )
        )

        weekday_avg = (
            weekday_avg_raw
            * wd_weight
            +
            avg31
            * (
                1.0
                - wd_weight
            )
        )

        # ----------------------------------------------------
        # Historical hit rates
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Neighbor
        # ----------------------------------------------------

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

    # --------------------------------------------------------
    # Merge actual
    # --------------------------------------------------------

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

    return panel


# ============================================================
# SCORE BAND ASSIGNMENT
# ============================================================

def assign_score_band(
    panel
):

    panel = panel.copy()

    n = len(panel)

    panel["score_band"] = (
        "BOTTOM50"
    )

    if n == 0:

        return panel

    # --------------------------------------------------------
    # Cumulative boundaries
    # --------------------------------------------------------

    top1_n = max(
        1,
        int(
            np.ceil(
                n * 0.01
            )
        )
    )

    top2_n = max(
        top1_n,
        int(
            np.ceil(
                n * 0.02
            )
        )
    )

    top5_n = max(
        top2_n,
        int(
            np.ceil(
                n * 0.05
            )
        )
    )

    top10_n = max(
        top5_n,
        int(
            np.ceil(
                n * 0.10
            )
        )
    )

    top20_n = max(
        top10_n,
        int(
            np.ceil(
                n * 0.20
            )
        )
    )

    top30_n = max(
        top20_n,
        int(
            np.ceil(
                n * 0.30
            )
        )
    )

    top50_n = max(
        top30_n,
        int(
            np.ceil(
                n * 0.50
            )
        )
    )

    # --------------------------------------------------------
    # MUTUALLY EXCLUSIVE
    # --------------------------------------------------------

    panel.loc[
        panel["rank"] <= top1_n,
        "score_band"
    ] = "TOP1"

    panel.loc[
        (
            panel["rank"] > top1_n
        )
        &
        (
            panel["rank"] <= top2_n
        ),
        "score_band"
    ] = "TOP1-2"

    panel.loc[
        (
            panel["rank"] > top2_n
        )
        &
        (
            panel["rank"] <= top5_n
        ),
        "score_band"
    ] = "TOP2-5"

    panel.loc[
        (
            panel["rank"] > top5_n
        )
        &
        (
            panel["rank"] <= top10_n
        ),
        "score_band"
    ] = "TOP5-10"

    panel.loc[
        (
            panel["rank"] > top10_n
        )
        &
        (
            panel["rank"] <= top20_n
        ),
        "score_band"
    ] = "TOP10-20"

    panel.loc[
        (
            panel["rank"] > top20_n
        )
        &
        (
            panel["rank"] <= top30_n
        ),
        "score_band"
    ] = "TOP20-30"

    panel.loc[
        (
            panel["rank"] > top30_n
        )
        &
        (
            panel["rank"] <= top50_n
        ),
        "score_band"
    ] = "TOP30-50"

    panel.loc[
        panel["rank"] > top50_n,
        "score_band"
    ] = "BOTTOM50"

    return panel


# ============================================================
# BAND ORDER
# ============================================================

BAND_ORDER = [
    "TOP1",
    "TOP1-2",
    "TOP2-5",
    "TOP5-10",
    "TOP10-20",
    "TOP20-30",
    "TOP30-50",
    "BOTTOM50",
]


# ============================================================
# MAIN
# ============================================================

df = load_data()

print()
print("MODEL = V4.2_C")
print(
    "Excluded = recent7_win, bounce_signal"
)
print(
    f"OOS = {TEST_START.date()} "
    f"to {TEST_END.date()}"
)

print()
print("NORMALIZED WEIGHTS")
print("-" * 70)

for factor in FACTORS:

    print(
        f"{factor:18s}: "
        f"{WEIGHTS[factor] * 100:8.3f}%"
    )

print(
    f"weight sum        : "
    f"{sum(WEIGHTS.values()) * 100:8.3f}%"
)


# ============================================================
# DAILY PANELS
# ============================================================

all_daily = []
diagnostics = []


test_dates = pd.date_range(
    TEST_START,
    TEST_END,
    freq="D"
)


print()
print("Building daily score panels...")


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

        diagnostics.append(
            {
                "date":
                    target_date,

                "eligible":
                    0,

                "band_total":
                    0,

                "band_count_ok":
                    False,
            }
        )

        continue

    panel = calculate_score(
        panel
    )

    panel = assign_score_band(
        panel
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
    panel["date"] = (
        target_date
    )

    # --------------------------------------------------------
    # Band count validation
    # --------------------------------------------------------

    band_counts = (
        panel
        .groupby(
            "score_band"
        )
        .size()
        .to_dict()
    )

    band_total = sum(
        band_counts.values()
    )

    count_ok = (
        band_total
        ==
        len(panel)
    )

    print(
        f"{target_date.date()} "
        f"eligible={len(panel)} "
        f"band_total={band_total} "
        f"count_check={count_ok}"
    )

    if not count_ok:

        raise RuntimeError(
            "SCORE BAND COUNT ERROR: "
            f"{target_date.date()} "
            f"eligible={len(panel)} "
            f"band_total={band_total}"
        )

    # --------------------------------------------------------
    # Duplicate rank validation
    # --------------------------------------------------------

    if (
        panel["rank"]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate rank detected."
        )

    # --------------------------------------------------------
    # Rank coverage validation
    # --------------------------------------------------------

    expected_ranks = set(
        range(
            1,
            len(panel) + 1
        )
    )

    actual_ranks = set(
        panel["rank"].astype(int)
    )

    if expected_ranks != actual_ranks:

        raise RuntimeError(
            "Rank coverage error."
        )

    # --------------------------------------------------------
    # Save daily panel
    # --------------------------------------------------------

    all_daily.append(
        panel
    )

    diagnostics.append(
        {
            "date":
                target_date,

            "eligible":
                len(panel),

            "band_total":
                band_total,

            "band_count_ok":
                count_ok,

            "top1":
                band_counts.get(
                    "TOP1",
                    0
                ),

            "top1_2":
                band_counts.get(
                    "TOP1-2",
                    0
                ),

            "top2_5":
                band_counts.get(
                    "TOP2-5",
                    0
                ),

            "top5_10":
                band_counts.get(
                    "TOP5-10",
                    0
                ),

            "top10_20":
                band_counts.get(
                    "TOP10-20",
                    0
                ),

            "top20_30":
                band_counts.get(
                    "TOP20-30",
                    0
                ),

            "top30_50":
                band_counts.get(
                    "TOP30-50",
                    0
                ),

            "bottom50":
                band_counts.get(
                    "BOTTOM50",
                    0
                ),
        }
    )


if not all_daily:

    raise RuntimeError(
        "No daily panels were generated."
    )


daily = pd.concat(
    all_daily,
    ignore_index=True
)


diagnostic_df = pd.DataFrame(
    diagnostics
)


# ============================================================
# BAND PERFORMANCE
# ============================================================

band_rows = []


for band in BAND_ORDER:

    sub = daily[
        daily["score_band"]
        == band
    ].copy()

    if sub.empty:

        continue

    band_rows.append(
        {
            "score_band":
                band,

            "days":
                sub["date"]
                .nunique(),

            "machines":
                len(sub),

            "avg_diff":
                sub["diff"]
                .mean(),

            "median_diff":
                sub["diff"]
                .median(),

            "std_diff":
                sub["diff"]
                .std(
                    ddof=0
                ),

            "total_diff":
                sub["diff"]
                .sum(),

            "win_rate":
                sub["win"]
                .mean()
                * 100,

            "plus1000_rate":
                sub["plus1000"]
                .mean()
                * 100,

            "plus2000_rate":
                sub["plus2000"]
                .mean()
                * 100,

            "positive_machines":
                int(
                    (
                        sub["diff"] > 0
                    ).sum()
                ),

            "negative_machines":
                int(
                    (
                        sub["diff"] < 0
                    ).sum()
                ),
        }
    )


band_summary = pd.DataFrame(
    band_rows
)


# ============================================================
# SELECTION COMPARISON
# ============================================================

selection_rows = []


for top_n in (
    5,
    10,
    20,
    30,
    50,
):

    top = daily[
        daily["rank"] <= top_n
    ].copy()

    if top.empty:

        continue

    daily_stats = (
        top
        .groupby("date")
        ["diff"]
        .mean()
    )

    positive_days = int(
        (
            daily_stats > 0
        ).sum()
    )

    negative_days = int(
        (
            daily_stats < 0
        ).sum()
    )

    tie_days = int(
        (
            daily_stats == 0
        ).sum()
    )

    selection_rows.append(
        {
            "selection":
                f"TOP{top_n}",

            "days":
                len(daily_stats),

            "avg_diff":
                top["diff"].mean(),

            "median_diff":
                top["diff"].median(),

            "total_diff":
                top["diff"].sum(),

            "win_rate":
                top["win"]
                .mean()
                * 100,

            "plus1000_rate":
                top["plus1000"]
                .mean()
                * 100,

            "plus2000_rate":
                top["plus2000"]
                .mean()
                * 100,

            "positive_days":
                positive_days,

            "negative_days":
                negative_days,

            "tie_days":
                tie_days,

            "positive_day_rate":
                positive_days
                /
                len(daily_stats)
                * 100,
        }
    )


selection_summary = pd.DataFrame(
    selection_rows
)


# ============================================================
# SCORE / RANK CORRELATION
# ============================================================

score_corr = float(
    daily[
        [
            "score",
            "diff",
        ]
    ]
    .corr()
    .loc[
        "score",
        "diff"
    ]
)

rank_corr = float(
    daily[
        [
            "rank",
            "diff",
        ]
    ]
    .corr()
    .loc[
        "rank",
        "diff"
    ]
)


# ============================================================
# TOP10 VS BOTTOM50
# ============================================================

top10 = daily[
    daily["rank"] <= 10
]

bottom50 = daily[
    daily["rank"]
    >
    daily.groupby("date")[
        "rank"
    ].transform(
        lambda x:
        int(
            np.ceil(
                len(x)
                * 0.50
            )
        )
    )
]


if not bottom50.empty:

    top10_avg = (
        top10["diff"]
        .mean()
    )

    bottom50_avg = (
        bottom50["diff"]
        .mean()
    )

    spread = (
        top10_avg
        -
        bottom50_avg
    )

else:

    top10_avg = np.nan
    bottom50_avg = np.nan
    spread = np.nan


# ============================================================
# SAVE
# ============================================================

daily.to_csv(
    OUT_DAILY,
    index=False,
    encoding="utf-8-sig"
)

band_summary.to_csv(
    OUT_BAND,
    index=False,
    encoding="utf-8-sig"
)

selection_summary.to_csv(
    OUT_SELECTION,
    index=False,
    encoding="utf-8-sig"
)

diagnostic_df.to_csv(
    OUT_DIAGNOSTIC,
    index=False,
    encoding="utf-8-sig"
)


summary_rows = [
    {
        "model":
            "V4.2_C",

        "excluded":
            "recent7_win,bounce_signal",

        "oos_start":
            TEST_START,

        "oos_end":
            TEST_END,

        "records":
            len(df),

        "test_days":
            len(test_dates),

        "score_diff_correlation":
            score_corr,

        "rank_diff_correlation":
            rank_corr,

        "top10_avg_diff":
            top10_avg,

        "bottom50_avg_diff":
            bottom50_avg,

        "top10_bottom50_spread":
            spread,

        "band_count_validation":
            bool(
                diagnostic_df[
                    "band_count_ok"
                ].all()
            ),
    }
]

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("=" * 70)
print("VER.4.2 FIXED SCORE BAND RESULT")
print("=" * 70)

print()

print(
    band_summary.to_string(
        index=False
    )
)

print()
print("=" * 70)
print("TOP N SELECTION")
print("=" * 70)

print()

print(
    selection_summary.to_string(
        index=False
    )
)

print()
print("=" * 70)
print("DIAGNOSTIC")
print("=" * 70)

print(
    f"Score / Diff correlation : "
    f"{score_corr:+.4f}"
)

print(
    f"Rank / Diff correlation  : "
    f"{rank_corr:+.4f}"
)

print(
    f"TOP10 average diff       : "
    f"{top10_avg:+.2f}"
)

print(
    f"BOTTOM50 average diff    : "
    f"{bottom50_avg:+.2f}"
)

print(
    f"TOP10-BOTTOM50 spread    : "
    f"{spread:+.2f}"
)

print()

print(
    "Band count validation    : "
    +
    str(
        diagnostic_df[
            "band_count_ok"
        ].all()
    )
)


# ============================================================
# FILES
# ============================================================

print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    OUT_DAILY
)

print(
    OUT_BAND
)

print(
    OUT_SUMMARY
)

print(
    OUT_SELECTION
)

print(
    OUT_DIAGNOSTIC
)

print()
print(
    "Ver.4.2 FIXED score band analysis complete."
)