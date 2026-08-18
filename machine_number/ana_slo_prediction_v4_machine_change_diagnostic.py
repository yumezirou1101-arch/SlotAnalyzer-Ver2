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

DIAG_DATE = pd.Timestamp("2026-08-03")


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
        [
            "date",
            "日付",
            "譌･莉・"
        ]
    )

    no_col = find(
        [
            "machine_no",
            "台番号",
            "蜿ｰ逡ｪ蜿ｷ"
        ]
    )

    name_col = find(
        [
            "machine_name",
            "機種名",
            "讖溽ｨｮ蜷・"
        ]
    )

    diff_col = find(
        [
            "diff",
            "差枚",
            "蟾ｮ譫・"
        ]
    )

    if not all(
        [
            date_col,
            no_col,
            name_col,
            diff_col
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
            "machine_name",
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
        s - s.mean()
    ) / std


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

    for _, actual_row in actual.iterrows():

        no = int(
            actual_row["machine_no"]
        )

        name = str(
            actual_row["machine_name"]
        )

        actual_diff = float(
            actual_row["diff"]
        )

        old_history = hist[
            hist["machine_no"] == no
        ].copy()

        same_history = hist[
            (hist["machine_no"] == no)
            &
            (hist["machine_name"] == name)
        ].copy()

        if not same_history.empty:

            m = same_history.sort_values(
                "date"
            )

            source = "same_machine"

            history_days = len(m)

        else:

            type_history = hist[
                hist["machine_name"] == name
            ].copy()

            if not type_history.empty:

                m = type_history.sort_values(
                    "date"
                )

                source = "type_fallback"

                history_days = 0

            else:

                m = hist.sort_values(
                    "date"
                )

                source = "global_fallback"

                history_days = 0

        avg31 = float(
            m["diff"].mean()
        )

        recent_dates = (
            m["date"]
            .drop_duplicates()
            .sort_values()
            .tail(7)
        )

        recent7 = m[
            m["date"].isin(
                recent_dates
            )
        ]

        recent7_avg = float(
            recent7["diff"].mean()
        )

        recent7_win = float(
            recent7["win"].mean()
        )

        latest_m_date = m["date"].max()

        latest_m = m[
            m["date"] == latest_m_date
        ]

        last_diff = float(
            latest_m["diff"].mean()
        )

        previous_dates = (
            m[
                m["date"]
                < latest_m_date
            ]["date"]
            .drop_duplicates()
            .sort_values()
        )

        if len(previous_dates):

            previous_date = (
                previous_dates.iloc[-1]
            )

            previous_m = m[
                m["date"]
                == previous_date
            ]

            prev_diff = float(
                previous_m["diff"].mean()
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

        weekday_n = (
            wd["date"]
            .nunique()
        )

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
                avg31
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

        old_machine_name = ""

        if not old_history.empty:

            old_machine_name = str(
                old_history.sort_values(
                    "date"
                ).iloc[-1][
                    "machine_name"
                ]
            )

        machine_changed = (
            bool(
                old_machine_name
                and
                old_machine_name != name
            )
        )

        rows.append({

            "machine_no":
                no,

            "machine_name":
                name,

            "old_machine_name":
                old_machine_name,

            "machine_changed":
                machine_changed,

            "feature_source":
                source,

            "history_days":
                history_days,

            "actual_diff":
                actual_diff,

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

    return pd.DataFrame(rows)


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

        factor_score = (
            50.0
            + z * 12.5
        ).clip(
            0,
            100
        )

        score += (
            factor_score
            * weights.get(
                factor,
                0.0
            )
        )

    x["score"] = score

    return x.sort_values(
        "score",
        ascending=False
    ).reset_index(
        drop=True
    )


def group_result(
    df,
    group_name
):

    if df.empty:

        return {

            "group":
                group_name,

            "n":
                0,

            "avg_diff":
                np.nan,

            "median_diff":
                np.nan,

            "win_rate":
                np.nan,

            "plus1000_rate":
                np.nan,

            "plus2000_rate":
                np.nan,

            "positive":
                np.nan,

            "total_diff":
                np.nan,
        }

    d = df[
        "actual_diff"
    ].astype(float)

    return {

        "group":
            group_name,

        "n":
            len(df),

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

        "positive":
            float(
                d.sum() > 0
            ),

        "total_diff":
            float(d.sum()),
    }


def main():

    print("=" * 70)
    print(
        "Ver.4 Machine Change Diagnostic"
    )
    print("=" * 70)

    print()

    df = load_data()

    print(
        f"records = {len(df):,}"
    )

    print(
        f"diagnostic date = "
        f"{DIAG_DATE.date()}"
    )

    print()

    # --------------------------------------------------------
    # 8/3だけを分析
    # --------------------------------------------------------

    panel = build_features(
        df,
        DIAG_DATE
    )

    if panel.empty:

        raise RuntimeError(
            "Diagnostic panel is empty."
        )

    ranked = rank_score(
        panel,
        V4_WEIGHTS
    )

    # --------------------------------------------------------
    # 分類
    # --------------------------------------------------------

    group_a = ranked[
        ~ranked["machine_changed"]
    ].copy()

    group_b = ranked[
        ranked["machine_changed"]
        &
        (
            ranked["feature_source"]
            == "type_fallback"
        )
    ].copy()

    group_c = ranked[
        ranked["machine_changed"]
        &
        (
            ranked["feature_source"]
            == "global_fallback"
        )
    ].copy()

    # --------------------------------------------------------
    # 結果
    # --------------------------------------------------------

    results = [

        group_result(
            group_a,
            "A_same_machine"
        ),

        group_result(
            group_b,
            "B_change_type_history"
        ),

        group_result(
            group_c,
            "C_change_no_type_history"
        ),

        group_result(
            ranked,
            "ALL"
        ),
    ]

    result_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 70)
    print(
        "GROUP RESULT 2026-08-03"
    )
    print("=" * 70)

    print(
        result_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # TOP10 / TOP30への機種変更台の入り込み
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "RANKING DIAGNOSTIC"
    )
    print("=" * 70)

    rank_rows = []

    for top_n in (
        5,
        10,
        20,
        30
    ):

        top = ranked.head(top_n)

        changed = top[
            top["machine_changed"]
        ]

        rank_rows.append({

            "top_n":
                top_n,

            "selected":
                len(top),

            "machine_change":
                len(changed),

            "same_machine":
                len(
                    top[
                        ~top[
                            "machine_changed"
                        ]
                    ]
                ),

            "change_ratio":
                float(
                    len(changed)
                    / len(top)
                    * 100
                ),

            "changed_avg_diff":
                (
                    float(
                        changed[
                            "actual_diff"
                        ].mean()
                    )
                    if not changed.empty
                    else np.nan
                ),

            "same_avg_diff":
                float(
                    top[
                        ~top[
                            "machine_changed"
                        ]
                    ][
                        "actual_diff"
                    ].mean()
                ),
        })

    rank_df = pd.DataFrame(
        rank_rows
    )

    print(
        rank_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 機種変更87台の詳細
    # --------------------------------------------------------

    changed = ranked[
        ranked["machine_changed"]
    ].copy()

    changed = changed.sort_values(
        "score",
        ascending=False
    )

    changed["rank"] = (
        changed.index
        + 1
    )

    detail_columns = [

        "rank",
        "machine_no",
        "old_machine_name",
        "machine_name",
        "feature_source",
        "history_days",
        "score",
        "actual_diff",
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

    detail_df = changed[
        detail_columns
    ].copy()

    print()
    print("=" * 70)
    print(
        "MACHINE CHANGE DETAIL"
    )
    print("=" * 70)

    print(
        detail_df.head(50).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 8/3を除いた15日と8/3の比較
    # --------------------------------------------------------

    daily_rows = []

    for target_date in pd.date_range(
        TEST_START,
        TEST_END
    ):

        p = build_features(
            df,
            target_date
        )

        if p.empty:
            continue

        r = rank_score(
            p,
            V4_WEIGHTS
        )

        top10 = r.head(10)

        d = top10[
            "actual_diff"
        ].astype(float)

        daily_rows.append({

            "date":
                target_date.date(),

            "top10_avg":
                float(d.mean()),

            "top10_total":
                float(d.sum()),

            "top10_win":
                float(
                    (d > 0).mean()
                    * 100
                ),

            "top10_positive":
                float(
                    d.sum() > 0
                ),

            "machine_change_in_top10":
                int(
                    top10[
                        top10[
                            "machine_changed"
                        ]
                    ].shape[0]
                ),
        })

    daily_df = pd.DataFrame(
        daily_rows
    )

    daily_df["period"] = np.where(
        daily_df["date"]
        == DIAG_DATE.date(),
        "2026-08-03",
        "other_days"
    )

    print()
    print("=" * 70)
    print(
        "AUG-03 VS OTHER DAYS"
    )
    print("=" * 70)

    compare_rows = []

    for period, sub in (
        daily_df.groupby("period")
    ):

        compare_rows.append({

            "period":
                period,

            "days":
                len(sub),

            "avg_daily_top10":
                float(
                    sub["top10_avg"].mean()
                ),

            "median_daily_top10":
                float(
                    sub["top10_avg"].median()
                ),

            "total_diff":
                float(
                    sub["top10_total"].sum()
                ),

            "positive_days":
                float(
                    sub["top10_positive"].mean()
                    * 100
                ),

            "avg_machine_change_top10":
                float(
                    sub[
                        "machine_change_in_top10"
                    ].mean()
                ),
        })

    compare_df = pd.DataFrame(
        compare_rows
    )

    print(
        compare_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    out_group = (
        OUT_DIR
        / "19_Ver4_machine_change_group.csv"
    )

    out_rank = (
        OUT_DIR
        / "19_Ver4_machine_change_rank.csv"
    )

    out_detail = (
        OUT_DIR
        / "19_Ver4_machine_change_detail.csv"
    )

    out_daily = (
        OUT_DIR
        / "19_Ver4_machine_change_daily_compare.csv"
    )

    out_compare = (
        OUT_DIR
        / "19_Ver4_machine_change_period_compare.csv"
    )

    result_df.to_csv(
        out_group,
        index=False,
        encoding="utf-8-sig"
    )

    rank_df.to_csv(
        out_rank,
        index=False,
        encoding="utf-8-sig"
    )

    detail_df.to_csv(
        out_detail,
        index=False,
        encoding="utf-8-sig"
    )

    daily_df.to_csv(
        out_daily,
        index=False,
        encoding="utf-8-sig"
    )

    compare_df.to_csv(
        out_compare,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print(
        "FILES SAVED"
    )
    print("=" * 70)

    print(out_group)
    print(out_rank)
    print(out_detail)
    print(out_daily)
    print(out_compare)

    print()
    print(
        "Machine change diagnostic complete."
    )


if __name__ == "__main__":
    main()