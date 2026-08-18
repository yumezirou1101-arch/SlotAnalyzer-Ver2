from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Ana-Slo Ver.4.2 Score Band Analysis
#
# MODEL:
#   V4.2_C
#
# EXCLUDED:
#   recent7_win
#   bounce_signal
#
# PURPOSE:
#   Verify whether the V4.2_C score itself has predictive power.
#
#   Each OOS day is divided into score/rank bands:
#       TOP 1%
#       TOP 2%
#       TOP 5%
#       TOP 10%
#       TOP 20%
#       TOP 30%
#       TOP 50%
#       BOTTOM 50%
#
#   No future actual data is used for feature calculation.
# ============================================================


from pathlib import Path
import pandas as pd
import numpy as np


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

CSV1 = (
    DATA_DIR
    / "ana_slo_20260711.csv"
)

CSV2 = (
    DATA_DIR
    / "ana_slo_20260712_20260810.csv"
)


OUTPUT_DAILY = (
    OUT_DIR
    / "32_Ver4_2_score_band_daily.csv"
)

OUTPUT_BAND = (
    OUT_DIR
    / "32_Ver4_2_score_band_summary.csv"
)

OUTPUT_DAY = (
    OUT_DIR
    / "32_Ver4_2_score_band_daily_summary.csv"
)

OUTPUT_SUMMARY = (
    OUT_DIR
    / "32_Ver4_2_score_band_diagnostic.csv"
)


START = pd.Timestamp(
    "2026-07-11"
)

TEST_START = pd.Timestamp(
    "2026-07-21"
)

TEST_END = pd.Timestamp(
    "2026-08-10"
)


# ============================================================
# V4.2_C FACTORS
# ============================================================

FACTORS = [
    "avg31",
    "recent7_avg",
    "last_diff",
    "prev_change",
    "weekday_avg",
    "type_avg",
    "plus1000_rate",
    "plus2000_rate",
    "neighbor_avg",
]


# ============================================================
# V4.2_C WEIGHTS
# recent7_win / bounce_signal are intentionally excluded.
# ============================================================

V42C_WEIGHTS = {

    "avg31":
        0.0670952025611345,

    "recent7_avg":
        0.05164896703284082,

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
}


# ============================================================
# SCORE BANDS
# ============================================================

BANDS = [
    ("TOP1", 0.01),
    ("TOP2", 0.02),
    ("TOP5", 0.05),
    ("TOP10", 0.10),
    ("TOP20", 0.20),
    ("TOP30", 0.30),
    ("TOP50", 0.50),
]


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
        "CSV read failed: "
        + str(path)
    )


# ============================================================
# COLUMN FINDER
# ============================================================

def find_column(df, candidates):

    for col in candidates:

        if col in df.columns:

            return col

    return None


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

    print()
    print(
        f"records = {len(df):,}"
    )

    date_col = find_column(
        df,
        [
            "date",
            "日付",
        ]
    )

    no_col = find_column(
        df,
        [
            "machine_no",
            "台番号",
        ]
    )

    name_col = find_column(
        df,
        [
            "machine_name",
            "機種名",
        ]
    )

    diff_col = find_column(
        df,
        [
            "diff",
            "差枚",
        ]
    )

    if not all([
        date_col,
        no_col,
        name_col,
        diff_col,
    ]):

        raise ValueError(
            "Required columns not found.\n"
            f"columns={list(df.columns)}"
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

    if std == 0 or np.isnan(std):

        return pd.Series(
            0.0,
            index=s.index
        )

    return (
        (s - s.mean())
        / std
    )


# ============================================================
# BUILD DAILY PANEL
# ============================================================

def build_panel(
    df,
    target_date
):

    # --------------------------------------------------------
    # Only data BEFORE target_date is allowed.
    # --------------------------------------------------------

    hist = df[
        df["date"]
        < target_date
    ].copy()

    actual = df[
        df["date"]
        == target_date
    ][
        [
            "machine_no",
            "machine_name",
            "diff",
            "win",
            "plus1000",
            "plus2000",
        ]
    ].copy()

    if hist.empty or actual.empty:

        return pd.DataFrame()

    latest_date = (
        hist["date"].max()
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

    target_weekday = (
        target_date.dayofweek
    )

    type_stats = (
        hist.groupby(
            "machine_name"
        )["diff"]
        .mean()
        .to_dict()
    )

    rows = []

    # --------------------------------------------------------
    # Machine history
    # --------------------------------------------------------

    for no, m in hist.groupby(
        "machine_no"
    ):

        m = (
            m.sort_values(
                "date"
            )
        )

        if m.empty:

            continue

        name = str(
            m.iloc[-1][
                "machine_name"
            ]
        )

        # ----------------------------------------------------
        # 31 day average
        # ----------------------------------------------------

        avg31 = float(
            m["diff"].mean()
        )

        # ----------------------------------------------------
        # Recent 7 average
        # ----------------------------------------------------

        recent7 = m.tail(7)

        recent7_avg = float(
            recent7["diff"].mean()
        )

        # ----------------------------------------------------
        # Last diff
        # ----------------------------------------------------

        last_diff = float(
            m.iloc[-1]["diff"]
        )

        # ----------------------------------------------------
        # Previous change
        # ----------------------------------------------------

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
        # Weekday average
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
            / (
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
        # Type average
        # ----------------------------------------------------

        type_avg = float(
            type_stats.get(
                name,
                0.0
            )
        )

        # ----------------------------------------------------
        # +1000 rate
        # ----------------------------------------------------

        plus1000_rate = float(
            m["plus1000"].mean()
        )

        # ----------------------------------------------------
        # +2000 rate
        # ----------------------------------------------------

        plus2000_rate = float(
            m["plus2000"].mean()
        )

        # ----------------------------------------------------
        # Neighbor average
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

        rows.append({

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
        })

    feat = pd.DataFrame(
        rows
    )

    if feat.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Normalize each feature within the day's eligible
    # machines.
    # --------------------------------------------------------

    for factor in FACTORS:

        feat[factor] = zscore(
            feat[factor]
        )

    # --------------------------------------------------------
    # V4.2_C score
    # --------------------------------------------------------

    feat["score"] = 0.0

    for factor, weight in (
        V42C_WEIGHTS.items()
    ):

        feat["score"] += (
            feat[factor]
            * weight
        )

    # --------------------------------------------------------
    # Sort by score
    # --------------------------------------------------------

    feat = feat.sort_values(
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

    feat["rank"] = (
        np.arange(
            1,
            len(feat) + 1
        )
    )

    feat["eligible_machines"] = (
        len(feat)
    )

    # --------------------------------------------------------
    # Actual result
    # --------------------------------------------------------

    feat = feat.merge(
        actual,
        on=[
            "machine_no",
            "machine_name",
        ],
        how="inner"
    )

    return feat


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "Ana-Slo Ver.4.2 Score Band Analysis"
    )

    print("=" * 70)

    print()

    print(
        "MODEL = V4.2_C"
    )

    print(
        "Excluded = recent7_win, bounce_signal"
    )

    print(
        f"OOS = {TEST_START.date()} "
        f"to {TEST_END.date()}"
    )

    print()

    df = load_data()

    all_daily = []

    print()
    print(
        "Building daily score panels..."
    )

    # ========================================================
    # DAILY
    # ========================================================

    for target_date in pd.date_range(
        TEST_START,
        TEST_END,
        freq="D"
    ):

        panel = build_panel(
            df,
            target_date
        )

        if panel.empty:

            print(
                f"{target_date.date()} "
                f"NO DATA"
            )

            continue

        panel["date"] = (
            target_date
        )

        all_daily.append(
            panel
        )

        print(
            f"{target_date.date()} "
            f"eligible={len(panel)}"
        )

    if not all_daily:

        raise RuntimeError(
            "No daily panels generated."
        )

    daily = pd.concat(
        all_daily,
        ignore_index=True
    )

    # ========================================================
    # SCORE BAND ASSIGNMENT
    # ========================================================

    daily["score_band"] = (
        "BOTTOM50"
    )

    for band_name, pct in BANDS:

        daily_count = (
            daily
            .groupby("date")
            .size()
        )

        # Rank threshold calculated separately for each day.
        thresholds = (
            daily_count
            * pct
        )

        threshold_map = (
            thresholds
            .to_dict()
        )

        mask = []

        for _, row in daily.iterrows():

            threshold = (
                threshold_map[
                    row["date"]
                ]
            )

            mask.append(
                row["rank"]
                <= max(
                    1,
                    int(
                        np.ceil(
                            threshold
                        )
                    )
                )
            )

        mask = np.array(
            mask,
            dtype=bool
        )

        # Only overwrite if this is the tighter
        # band than the current assignment.
        daily.loc[
            mask,
            "score_band"
        ] = band_name

    # ========================================================
    # IMPORTANT:
    # BOTTOM50 should mean rank > TOP50.
    # ========================================================

    daily.loc[
        daily["rank"]
        >
        (
            daily["eligible_machines"]
            * 0.50
        ),
        "score_band"
    ] = "BOTTOM50"

    # ========================================================
    # BAND SUMMARY
    # ========================================================

    band_order = [
        "TOP1",
        "TOP2",
        "TOP5",
        "TOP10",
        "TOP20",
        "TOP30",
        "BOTTOM50",
    ]

    band_rows = []

    for band in band_order:

        g = daily[
            daily["score_band"]
            == band
        ].copy()

        if g.empty:

            continue

        days = (
            g["date"]
            .nunique()
        )

        band_rows.append({

            "score_band":
                band,

            "days":
                days,

            "machines":
                len(g),

            "avg_diff":
                float(
                    g["diff"].mean()
                ),

            "median_diff":
                float(
                    g["diff"].median()
                ),

            "std_diff":
                float(
                    g["diff"].std(
                        ddof=0
                    )
                ),

            "total_diff":
                float(
                    g["diff"].sum()
                ),

            "win_rate":
                float(
                    g["win"].mean()
                    * 100.0
                ),

            "plus1000_rate":
                float(
                    g["plus1000"].mean()
                    * 100.0
                ),

            "plus2000_rate":
                float(
                    g["plus2000"].mean()
                    * 100.0
                ),

            "positive_machines":
                int(
                    (
                        g["diff"]
                        > 0
                    ).sum()
                ),

            "negative_machines":
                int(
                    (
                        g["diff"]
                        < 0
                    ).sum()
                ),
        })

    band_summary = pd.DataFrame(
        band_rows
    )

    # ========================================================
    # DAILY BAND PERFORMANCE
    # ========================================================

    daily_band_rows = []

    for (
        date,
        band
    ), g in (
        daily
        .groupby(
            [
                "date",
                "score_band",
            ]
        )
    ):

        daily_band_rows.append({

            "date":
                date,

            "score_band":
                band,

            "machines":
                len(g),

            "avg_diff":
                float(
                    g["diff"].mean()
                ),

            "median_diff":
                float(
                    g["diff"].median()
                ),

            "win_rate":
                float(
                    g["win"].mean()
                    * 100.0
                ),

            "plus1000_rate":
                float(
                    g["plus1000"].mean()
                    * 100.0
                ),

            "plus2000_rate":
                float(
                    g["plus2000"].mean()
                    * 100.0
                ),

            "total_diff":
                float(
                    g["diff"].sum()
                ),
        })

    daily_band = pd.DataFrame(
        daily_band_rows
    )

    # ========================================================
    # DIAGNOSTIC
    # ========================================================

    top10 = daily[
        daily["rank"] <= 10
    ]

    top20 = daily[
        daily["rank"] <= 20
    ]

    top30 = daily[
        daily["rank"] <= 30
    ]

    top50 = daily[
        daily["rank"] <= (
            daily["eligible_machines"]
            * 0.50
        )
    ]

    # --------------------------------------------------------
    # Daily averages of cumulative selections
    # --------------------------------------------------------

    cumulative_rows = []

    for label, g in [
        ("TOP10", top10),
        ("TOP20", top20),
        ("TOP30", top30),
        ("TOP50", top50),
    ]:

        if g.empty:

            continue

        cumulative_rows.append({

            "selection":
                label,

            "days":
                g["date"]
                .nunique(),

            "avg_diff":
                float(
                    g["diff"].mean()
                ),

            "total_diff":
                float(
                    g["diff"].sum()
                ),

            "win_rate":
                float(
                    g["win"].mean()
                    * 100.0
                ),

            "plus1000_rate":
                float(
                    g["plus1000"].mean()
                    * 100.0
                ),

            "plus2000_rate":
                float(
                    g["plus2000"].mean()
                    * 100.0
                ),
        })

    cumulative = pd.DataFrame(
        cumulative_rows
    )

    # ========================================================
    # SCORE / DIFF CORRELATION
    # ========================================================

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

    # ========================================================
    # TOP BAND vs BOTTOM BAND
    # ========================================================

    top10_avg = float(
        top10["diff"].mean()
    )

    bottom50_avg = float(
        daily[
            daily["rank"]
            >
            (
                daily["eligible_machines"]
                * 0.50
            )
        ]["diff"].mean()
    )

    spread = (
        top10_avg
        - bottom50_avg
    )

    diagnostic = pd.DataFrame([
        {

            "model":
                "V4.2_C",

            "oos_start":
                TEST_START.date(),

            "oos_end":
                TEST_END.date(),

            "days":
                daily["date"]
                .nunique(),

            "total_records":
                len(daily),

            "top10_avg_diff":
                top10_avg,

            "bottom50_avg_diff":
                bottom50_avg,

            "top10_vs_bottom50_spread":
                spread,

            "score_diff_correlation":
                score_corr,

            "rank_diff_correlation":
                rank_corr,

            "top10_total_diff":
                float(
                    top10["diff"].sum()
                ),

            "top10_win_rate":
                float(
                    top10["win"].mean()
                    * 100.0
                ),
        }
    ])

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 70)
    print(
        "SCORE BAND RESULT"
    )
    print("=" * 70)

    print(
        band_summary.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.2f}"
        )
    )

    print()
    print("=" * 70)
    print(
        "CUMULATIVE TOP-N RESULT"
    )
    print("=" * 70)

    print(
        cumulative.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.2f}"
        )
    )

    print()
    print("=" * 70)
    print(
        "DIAGNOSTIC"
    )
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

    # ========================================================
    # SAVE
    # ========================================================

    daily.to_csv(
        OUTPUT_DAILY,
        index=False,
        encoding="utf-8-sig"
    )

    band_summary.to_csv(
        OUTPUT_BAND,
        index=False,
        encoding="utf-8-sig"
    )

    daily_band.to_csv(
        OUTPUT_DAY,
        index=False,
        encoding="utf-8-sig"
    )

    diagnostic.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print(
        "FILES SAVED"
    )
    print("=" * 70)

    print(
        OUTPUT_DAILY
    )

    print(
        OUTPUT_BAND
    )

    print(
        OUTPUT_DAY
    )

    print(
        OUTPUT_SUMMARY
    )

    print()
    print(
        "Ver.4.2 score band analysis complete."
    )


if __name__ == "__main__":

    main()