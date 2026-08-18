from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Ana-Slo Ver.4.2 Rank Analysis
#
# V4.2_C:
#   recent7_win を除外
#   bounce_signal を除外
#
# OOS期間:
#   2026-07-21 ～ 2026-08-10
#
# 目的:
#   各日の予測順位 1～10位について、
#   実際の差枚・勝率等を検証する。
#
# ※ 予測時点より後の実績を特徴量に混入させない。
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
    / "31_Ver4_2_rank_analysis_daily.csv"
)

OUTPUT_RANK = (
    OUT_DIR
    / "31_Ver4_2_rank_analysis_rank.csv"
)

OUTPUT_GROUP = (
    OUT_DIR
    / "31_Ver4_2_rank_analysis_group.csv"
)

OUTPUT_SUMMARY = (
    OUT_DIR
    / "31_Ver4_2_rank_analysis_summary.csv"
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


V42C_WEIGHTS = {

    "avg31":
        0.0670952025611345,

    "recent7_avg":
        0.05164896703284082,

    # recent7_win は V4.2_C で除外

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

    # bounce_signal は V4.2_C で除外
}


# ============================================================
# CSV READER
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
# FEATURE PANEL
# ============================================================

def build_panel(
    df,
    target_date
):

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
        # 基本特徴量
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 曜日平均
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
        # 機種平均
        # ----------------------------------------------------

        type_avg = float(
            type_stats.get(
                name,
                0.0
            )
        )

        # ----------------------------------------------------
        # +1000 / +2000
        # ----------------------------------------------------

        plus1000_rate = float(
            m["plus1000"].mean()
        )

        plus2000_rate = float(
            m["plus2000"].mean()
        )

        # ----------------------------------------------------
        # 隣接台
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
    # 正規化
    # --------------------------------------------------------

    for factor in FACTORS:

        feat[
            factor
        ] = zscore(
            feat[factor]
        )

    # --------------------------------------------------------
    # V4.2_C SCORE
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
    # 順位
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

    # --------------------------------------------------------
    # 実績結合
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
# RANK SUMMARY
# ============================================================

def build_rank_summary(
    selected
):

    rows = []

    for rank, g in (
        selected
        .groupby("rank")
    ):

        days = len(g)

        avg_diff = float(
            g["diff"].mean()
        )

        total_diff = float(
            g["diff"].sum()
        )

        win_rate = (
            g["win"].mean()
            * 100.0
        )

        plus1000_rate = (
            g["plus1000"].mean()
            * 100.0
        )

        plus2000_rate = (
            g["plus2000"].mean()
            * 100.0
        )

        rows.append({

            "rank":
                int(rank),

            "days":
                days,

            "avg_diff":
                avg_diff,

            "median_diff":
                float(
                    g["diff"].median()
                ),

            "total_diff":
                total_diff,

            "win_rate":
                win_rate,

            "plus1000_rate":
                plus1000_rate,

            "plus2000_rate":
                plus2000_rate,

            "positive_days":
                int(
                    (
                        g["diff"]
                        > 0
                    ).sum()
                ),

            "negative_days":
                int(
                    (
                        g["diff"]
                        < 0
                    ).sum()
                ),
        })

    return pd.DataFrame(
        rows
    ).sort_values(
        "rank"
    )


# ============================================================
# GROUP SUMMARY
# ============================================================

def build_group_summary(
    selected
):

    selected = selected.copy()

    selected["rank_group"] = np.select(

        [
            selected["rank"]
            <= 3,

            selected["rank"]
            <= 5,

            selected["rank"]
            <= 10,
        ],

        [
            "TOP1-3",
            "TOP4-5",
            "TOP6-10",
        ],

        default="TOP11+",
    )

    groups = [
        "TOP1-3",
        "TOP4-5",
        "TOP6-10",
        "TOP11+",
    ]

    rows = []

    for group in groups:

        g = selected[
            selected[
                "rank_group"
            ]
            == group
        ]

        if g.empty:

            continue

        rows.append({

            "rank_group":
                group,

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

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "Ana-Slo Ver.4.2 Rank Analysis"
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

    daily_rows = []

    print()
    print(
        "Building daily ranking panels..."
    )

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

        # ----------------------------------------------------
        # 8/3の機種変更台は、
        # 同一台履歴が存在しない場合がある。
        #
        # 今回は既存V4.2_Cとの比較整合性を優先し、
        # 予測可能な台だけで順位を作る。
        # ----------------------------------------------------

        panel = panel[
            panel["machine_no"]
            .notna()
        ].copy()

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
                1,
                len(panel) + 1
            )
        )

        top10 = panel[
            panel["rank"] <= 10
        ].copy()

        top10["date"] = (
            target_date
        )

        daily_rows.append(
            top10
        )

        print(
            f"{target_date.date()} "
            f"eligible={len(panel)} "
            f"TOP10={len(top10)}"
        )

    if not daily_rows:

        raise RuntimeError(
            "No ranking data generated."
        )

    daily = pd.concat(
        daily_rows,
        ignore_index=True
    )

    # --------------------------------------------------------
    # 保存：日別TOP10
    # --------------------------------------------------------

    daily.to_csv(
        OUTPUT_DAILY,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 順位別
    # --------------------------------------------------------

    rank_summary = build_rank_summary(
        daily
    )

    print()
    print("=" * 70)
    print(
        "RANK RESULT"
    )
    print("=" * 70)

    print(
        rank_summary.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.2f}"
        )
    )

    rank_summary.to_csv(
        OUTPUT_RANK,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # グループ
    # --------------------------------------------------------

    group_summary = build_group_summary(
        daily
    )

    print()
    print("=" * 70)
    print(
        "RANK GROUP RESULT"
    )
    print("=" * 70)

    print(
        group_summary.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.2f}"
        )
    )

    group_summary.to_csv(
        OUTPUT_GROUP,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # TOP10全体
    # --------------------------------------------------------

    top10_avg = float(
        daily["diff"].mean()
    )

    top10_total = float(
        daily["diff"].sum()
    )

    top10_win = float(
        daily["win"].mean()
        * 100.0
    )

    top10_plus1000 = float(
        daily["plus1000"].mean()
        * 100.0
    )

    top10_plus2000 = float(
        daily["plus2000"].mean()
        * 100.0
    )

    # --------------------------------------------------------
    # 順位相関
    # --------------------------------------------------------

    if len(daily) >= 3:

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

    else:

        rank_corr = np.nan

    summary = pd.DataFrame([
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

            "top_n":
                10,

            "top10_avg_diff":
                top10_avg,

            "top10_total_diff":
                top10_total,

            "top10_win_rate":
                top10_win,

            "top10_plus1000_rate":
                top10_plus1000,

            "top10_plus2000_rate":
                top10_plus2000,

            "rank_diff_correlation":
                rank_corr,
        }
    ])

    print()
    print("=" * 70)
    print(
        "OVERALL"
    )
    print("=" * 70)

    print(
        f"TOP10 avg diff       : "
        f"{top10_avg:+.2f}"
    )

    print(
        f"TOP10 total diff     : "
        f"{top10_total:+.0f}"
    )

    print(
        f"TOP10 win rate       : "
        f"{top10_win:.2f}%"
    )

    print(
        f"TOP10 +1000 rate     : "
        f"{top10_plus1000:.2f}%"
    )

    print(
        f"TOP10 +2000 rate     : "
        f"{top10_plus2000:.2f}%"
    )

    print(
        f"Rank/Diff correlation: "
        f"{rank_corr:+.4f}"
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # FILES
    # --------------------------------------------------------

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
        OUTPUT_RANK
    )

    print(
        OUTPUT_GROUP
    )

    print(
        OUTPUT_SUMMARY
    )

    print()
    print(
        "Ver.4.2 rank analysis complete."
    )


if __name__ == "__main__":

    main()