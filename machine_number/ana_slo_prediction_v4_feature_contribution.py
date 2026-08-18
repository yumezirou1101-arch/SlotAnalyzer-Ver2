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


# ============================================================
# Ver.4 固定ウェイト
# ============================================================

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


FACTOR_NAMES = {
    "avg31": "31日平均",
    "recent7_avg": "直近7日平均",
    "recent7_win": "直近7日勝率",
    "last_diff": "前日差枚",
    "prev_change": "前々日→前日の変化",
    "weekday_avg": "曜日平均",
    "type_avg": "機種平均",
    "plus1000_rate": "+1000出率",
    "plus2000_rate": "+2000出率",
    "neighbor_avg": "隣接台平均",
    "bounce_signal": "リバウンド信号",
}


# ============================================================
# CSV読み込み
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
# データ読み込み
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
        &
        (df["date"] <= TEST_END)
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


# ============================================================
# 特徴量作成
# Ver.4と同じロジック
# ============================================================

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

        return (
            pd.DataFrame(),
            {
                "machines": len(actual),
                "eligible": 0,
                "excluded_no_history": len(actual)
            }
        )


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

        m = m.sort_values(
            "date"
        )


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
            +
            avg31
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


        rows.append(
            {
                "machine_no": int(no),

                "machine_name": name,

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
            }
        )


    feat = pd.DataFrame(
        rows
    )


    merged = feat.merge(
        actual,
        on=[
            "machine_no",
            "machine_name"
        ],
        how="inner"
    )


    diagnostics = {
        "machines":
            len(actual),

        "eligible":
            len(merged),

        "excluded_no_history":
            len(actual) - len(merged),
    }


    return (
        merged,
        diagnostics
    )


# ============================================================
# 寄与度計算
# ============================================================

def calculate_contributions(
    df,
    weights
):

    x = df.copy()


    total_score = pd.Series(
        0.0,
        index=x.index
    )


    for factor in FACTORS:

        s = pd.to_numeric(
            x[factor],
            errors="coerce"
        ).fillna(0.0)


        std = float(
            s.std(ddof=0)
        )


        if (
            std == 0
            or np.isnan(std)
        ):

            z = pd.Series(
                0.0,
                index=s.index
            )

        else:

            z = (
                s - s.mean()
            ) / std


        normalized = (
            50.0
            + z * 12.5
        ).clip(
            0,
            100
        )


        contribution = (
            normalized
            * weights[factor]
        )


        x[
            f"contrib_{factor}"
        ] = contribution


        x[
            f"z_{factor}"
        ] = z


        total_score += contribution


    x["score"] = total_score


    x = x.sort_values(
        "score",
        ascending=False
    ).reset_index(
        drop=True
    )


    x["rank"] = (
        np.arange(len(x))
        + 1
    )


    return x


# ============================================================
# メイン
# ============================================================

def main():

    print("=" * 70)
    print(
        "Ana-Slo Ver.4 Feature Contribution Analysis"
    )
    print("=" * 70)


    print()
    print("FIXED WEIGHTS")
    print("-" * 70)


    for factor in FACTORS:

        print(
            f"{factor:18s}: "
            f"{V4_WEIGHTS[factor] * 100:7.2f}%"
        )


    print(
        f"weight sum       : "
        f"{sum(V4_WEIGHTS.values()) * 100:7.2f}%"
    )


    df = load_data()


    print()
    print(
        f"records = {len(df):,}"
    )


    print(
        f"OOS period = "
        f"{TEST_START.date()} "
        f"to "
        f"{TEST_END.date()}"
    )


    all_daily = []
    top10_rows = []
    diagnostics_rows = []


    print()
    print(
        "Building daily feature panels..."
    )


    for target_date in pd.date_range(
        TEST_START,
        TEST_END
    ):

        panel, diagnostics = (
            build_features(
                df,
                target_date
            )
        )


        print(
            f"{target_date.date()} "
            f"machines={diagnostics['machines']} "
            f"eligible={diagnostics['eligible']} "
            f"excluded={diagnostics['excluded_no_history']}"
        )


        diagnostics_rows.append(
            {
                "date":
                    target_date.date(),

                "machines":
                    diagnostics["machines"],

                "eligible":
                    diagnostics["eligible"],

                "excluded_no_history":
                    diagnostics[
                        "excluded_no_history"
                    ],
            }
        )


        if panel.empty:

            continue


        ranked = calculate_contributions(
            panel,
            V4_WEIGHTS
        )


        ranked["date"] = (
            target_date.date()
        )


        ranked["selected_top10"] = (
            ranked["rank"] <= 10
        )


        ranked["positive_actual"] = (
            ranked["diff"] > 0
        )


        ranked["top10_result"] = (
            ranked["diff"]
            .where(
                ranked["selected_top10"]
            )
        )


        # --------------------------------------------
        # 全候補台の寄与度データ
        # --------------------------------------------

        all_daily.append(
            ranked
        )


        # --------------------------------------------
        # TOP10
        # --------------------------------------------

        top10 = ranked.head(10).copy()


        for _, row in top10.iterrows():

            record = {
                "date":
                    target_date.date(),

                "machine_no":
                    int(row["machine_no"]),

                "machine_name":
                    row["machine_name"],

                "rank":
                    int(row["rank"]),

                "score":
                    float(row["score"]),

                "diff":
                    float(row["diff"]),

                "win":
                    int(
                        row["diff"] > 0
                    ),
            }


            for factor in FACTORS:

                record[
                    f"contrib_{factor}"
                ] = float(
                    row[
                        f"contrib_{factor}"
                    ]
                )


            top10_rows.append(
                record
            )


    if not all_daily:

        raise RuntimeError(
            "No daily panels were created."
        )


    daily_df = pd.concat(
        all_daily,
        ignore_index=True
    )


    top10_df = pd.DataFrame(
        top10_rows
    )


    diagnostics_df = pd.DataFrame(
        diagnostics_rows
    )


    # ========================================================
    # 因子寄与度サマリー
    # ========================================================

    summary_rows = []


    for factor in FACTORS:

        contrib_col = (
            f"contrib_{factor}"
        )


        selected = daily_df[
            daily_df["selected_top10"]
        ]


        positive_selected = (
            selected[
                selected["diff"] > 0
            ]
        )


        negative_selected = (
            selected[
                selected["diff"] <= 0
            ]
        )


        summary_rows.append(
            {
                "factor":
                    factor,

                "factor_name":
                    FACTOR_NAMES[factor],

                "weight":
                    V4_WEIGHTS[factor],

                "top10_mean_contribution":
                    selected[
                        contrib_col
                    ].mean(),

                "top10_median_contribution":
                    selected[
                        contrib_col
                    ].median(),

                "positive_top10_mean":
                    positive_selected[
                        contrib_col
                    ].mean()
                    if not positive_selected.empty
                    else np.nan,

                "negative_top10_mean":
                    negative_selected[
                        contrib_col
                    ].mean()
                    if not negative_selected.empty
                    else np.nan,

                "positive_minus_negative":
                    (
                        positive_selected[
                            contrib_col
                        ].mean()
                        -
                        negative_selected[
                            contrib_col
                        ].mean()
                    )
                    if (
                        not positive_selected.empty
                        and not negative_selected.empty
                    )
                    else np.nan,

                "contribution_std":
                    selected[
                        contrib_col
                    ].std(ddof=0),

                "high_contribution_rate":
                    (
                        selected[
                            contrib_col
                        ]
                        >=
                        selected[
                            contrib_col
                        ].median()
                    ).mean()
                    * 100,
            }
        )


    factor_summary = pd.DataFrame(
        summary_rows
    )


    # ========================================================
    # 日別TOP10集計
    # ========================================================

    daily_top10_summary = []


    for target_date in pd.date_range(
        TEST_START,
        TEST_END
    ):

        day = daily_df[
            daily_df["date"]
            == target_date.date()
        ]


        if day.empty:

            continue


        top10 = day[
            day["selected_top10"]
        ]


        record = {
            "date":
                target_date.date(),

            "top10_avg_diff":
                top10["diff"].mean(),

            "top10_median_diff":
                top10["diff"].median(),

            "top10_win_rate":
                (
                    top10["diff"] > 0
                ).mean()
                * 100,

            "top10_total_diff":
                top10["diff"].sum(),

            "top10_plus1000_rate":
                (
                    top10["diff"] >= 1000
                ).mean()
                * 100,

            "top10_plus2000_rate":
                (
                    top10["diff"] >= 2000
                ).mean()
                * 100,
        }


        for factor in FACTORS:

            col = (
                f"contrib_{factor}"
            )


            record[
                f"mean_{col}"
            ] = top10[col].mean()


        daily_top10_summary.append(
            record
        )


    daily_top10_df = pd.DataFrame(
        daily_top10_summary
    )


    # ========================================================
    # 寄与度順位
    # ========================================================

    factor_summary = factor_summary.sort_values(
        "positive_minus_negative",
        ascending=False
    ).reset_index(
        drop=True
    )


    factor_summary[
        "contribution_rank"
    ] = (
        np.arange(
            len(factor_summary)
        )
        + 1
    )


    # ========================================================
    # 出力
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    out_daily = (
        OUT_DIR
        / "21_Ver4_feature_contribution_daily.csv"
    )


    out_top10 = (
        OUT_DIR
        / "21_Ver4_feature_contribution_top10.csv"
    )


    out_summary = (
        OUT_DIR
        / "21_Ver4_feature_contribution_summary.csv"
    )


    out_factor = (
        OUT_DIR
        / "21_Ver4_feature_contribution_factor.csv"
    )


    out_diag = (
        OUT_DIR
        / "21_Ver4_feature_contribution_diagnostics.csv"
    )


    daily_df.to_csv(
        out_daily,
        index=False,
        encoding="utf-8-sig"
    )


    top10_df.to_csv(
        out_top10,
        index=False,
        encoding="utf-8-sig"
    )


    daily_top10_df.to_csv(
        out_summary,
        index=False,
        encoding="utf-8-sig"
    )


    factor_summary.to_csv(
        out_factor,
        index=False,
        encoding="utf-8-sig"
    )


    diagnostics_df.to_csv(
        out_diag,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # コンソール表示
    # ========================================================

    print()
    print("=" * 70)
    print(
        "FEATURE CONTRIBUTION RESULT"
    )
    print("=" * 70)


    print()
    print(
        "FACTOR CONTRIBUTION RANKING"
    )


    display_cols = [
        "contribution_rank",
        "factor_name",
        "weight",
        "top10_mean_contribution",
        "positive_top10_mean",
        "negative_top10_mean",
        "positive_minus_negative",
        "contribution_std",
    ]


    print(
        factor_summary[
            display_cols
        ].to_string(
            index=False
        )
    )


    print()
    print("=" * 70)
    print(
        "TOP10 OVERALL"
    )
    print("=" * 70)


    if not daily_top10_df.empty:

        print(
            f"TOP10 avg diff      : "
            f"{daily_top10_df['top10_avg_diff'].mean():.2f}"
        )

        print(
            f"TOP10 total diff    : "
            f"{daily_top10_df['top10_total_diff'].sum():.0f}"
        )

        print(
            f"TOP10 positive days : "
            f"{(daily_top10_df['top10_total_diff'] > 0).mean() * 100:.2f}%"
        )


    print()
    print("=" * 70)
    print(
        "FILES SAVED"
    )
    print("=" * 70)


    print(out_daily)
    print(out_top10)
    print(out_summary)
    print(out_factor)
    print(out_diag)


    print()
    print(
        "Feature contribution analysis complete."
    )


if __name__ == "__main__":

    main()