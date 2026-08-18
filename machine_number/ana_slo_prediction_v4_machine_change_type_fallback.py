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
# TOP20_MEAN
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


# ============================================================
# 機種変更対応特徴量
#
# 重要:
# ・target_date以前のデータだけ使用
# ・同一台の履歴があれば最優先
# ・同一台に履歴がなければ現在機種の過去実績で補完
# ・旧機種の履歴は新機種台には使用しない
# ============================================================

def build_features(df, target_date):

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
            {}
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

    # --------------------------------------------------------
    # 機種別・日別集計
    # --------------------------------------------------------

    type_day = (
        hist.groupby(
            [
                "machine_name",
                "date"
            ]
        )
        .agg(
            diff=("diff", "mean"),
            win=("win", "mean"),
            plus1000=("plus1000", "mean"),
            plus2000=("plus2000", "mean")
        )
        .reset_index()
    )

    type_stats = (
        hist.groupby(
            "machine_name"
        )
        .agg(
            type_avg=("diff", "mean"),
            type_win=("win", "mean"),
            type_plus1000=("plus1000", "mean"),
            type_plus2000=("plus2000", "mean")
        )
    )

    rows = []

    diagnostics = {
        "same_machine_history": 0,
        "type_fallback": 0,
        "global_fallback": 0,
        "machine_change_detected": 0,
    }

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

        # ----------------------------------------------------
        # 現在機種における同一台の履歴
        # ----------------------------------------------------

        machine_hist = hist[
            (hist["machine_no"] == no)
            &
            (hist["machine_name"] == name)
        ].copy()

        # ----------------------------------------------------
        # 旧機種履歴の有無
        # ----------------------------------------------------

        old_machine_hist = hist[
            hist["machine_no"] == no
        ].copy()

        has_old_history = not old_machine_hist.empty

        has_same_machine_history = (
            not machine_hist.empty
        )

        if has_same_machine_history:

            diagnostics[
                "same_machine_history"
            ] += 1

        elif has_old_history:

            diagnostics[
                "machine_change_detected"
            ] += 1

        # ----------------------------------------------------
        # 機種全体の過去履歴
        # ----------------------------------------------------

        type_hist = hist[
            hist["machine_name"] == name
        ].copy()

        # ====================================================
        # 同一台履歴あり
        # ====================================================

        if has_same_machine_history:

            m = machine_hist.sort_values(
                "date"
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
                type_stats.loc[
                    name,
                    "type_avg"
                ]
            ) if name in type_stats.index else 0.0

            source = "same_machine"

            history_days = len(m)

        # ====================================================
        # 同一台履歴なし
        # → 現在機種の過去実績で補完
        # ====================================================

        elif not type_hist.empty:

            diagnostics[
                "type_fallback"
            ] += 1

            th = type_hist.sort_values(
                "date"
            )

            # ----------------------------------------------
            # 機種全体31日平均
            # ----------------------------------------------

            avg31 = float(
                th["diff"].mean()
            )

            # ----------------------------------------------
            # 機種全体の直近7営業日平均
            # ----------------------------------------------

            type_recent_dates = (
                th["date"]
                .drop_duplicates()
                .sort_values()
                .tail(7)
            )

            type_recent7 = th[
                th["date"].isin(
                    type_recent_dates
                )
            ]

            recent7_avg = float(
                type_recent7["diff"].mean()
            )

            recent7_win = float(
                type_recent7["win"].mean()
            )

            # ----------------------------------------------
            # 機種全体の直近日
            # ----------------------------------------------

            latest_type_date = th["date"].max()

            latest_type = th[
                th["date"]
                == latest_type_date
            ]

            last_diff = float(
                latest_type["diff"].mean()
            )

            # ----------------------------------------------
            # 機種全体の前日
            # ----------------------------------------------

            before_latest_dates = (
                th[
                    th["date"]
                    < latest_type_date
                ]["date"]
                .drop_duplicates()
                .sort_values()
            )

            if len(before_latest_dates):

                prev_type_date = (
                    before_latest_dates.iloc[-1]
                )

                prev_type = th[
                    th["date"]
                    == prev_type_date
                ]

                prev_diff = float(
                    prev_type["diff"].mean()
                )

            else:

                prev_diff = last_diff

            prev_change = (
                last_diff
                - prev_diff
            )

            # ----------------------------------------------
            # 曜日平均
            # ----------------------------------------------

            wd = th[
                th["date"].dt.dayofweek
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
                th["plus1000"].mean()
            )

            plus2000_rate = float(
                th["plus2000"].mean()
            )

            type_avg = float(
                th["diff"].mean()
            )

            source = "type_fallback"

            history_days = 0

        # ====================================================
        # 機種履歴もない場合
        # ====================================================

        else:

            diagnostics[
                "global_fallback"
            ] += 1

            avg31 = float(
                hist["diff"].mean()
            )

            recent_dates = (
                hist["date"]
                .drop_duplicates()
                .sort_values()
                .tail(7)
            )

            recent = hist[
                hist["date"].isin(
                    recent_dates
                )
            ]

            recent7_avg = float(
                recent["diff"].mean()
            )

            recent7_win = float(
                recent["win"].mean()
            )

            latest = hist[
                hist["date"] == latest_date
            ]

            last_diff = float(
                latest["diff"].mean()
            )

            before = hist[
                hist["date"] < latest_date
            ]

            if not before.empty:

                prev_date = before["date"].max()

                prev = hist[
                    hist["date"] == prev_date
                ]

                prev_diff = float(
                    prev["diff"].mean()
                )

            else:

                prev_diff = last_diff

            prev_change = (
                last_diff
                - prev_diff
            )

            wd = hist[
                hist["date"].dt.dayofweek
                == target_weekday
            ]

            if not wd.empty:

                weekday_avg_raw = float(
                    wd["diff"].mean()
                )

            else:

                weekday_avg_raw = avg31

            weekday_avg = (
                weekday_avg_raw
                * 0.5
                + avg31
                * 0.5
            )

            plus1000_rate = float(
                hist["plus1000"].mean()
            )

            plus2000_rate = float(
                hist["plus2000"].mean()
            )

            type_avg = avg31

            source = "global_fallback"

            history_days = 0

        # ----------------------------------------------------
        # 隣接台平均
        #
        # これは台番号上の物理的な隣接台なので、
        # 最新日の実績だけを使用。
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

        # ----------------------------------------------------
        # リバウンド信号
        #
        # 同一台履歴がない場合は、
        # 現在機種の直近平均に基づいて評価。
        # ----------------------------------------------------

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
                no,

            "machine_name":
                name,

            "actual_diff":
                actual_diff,

            "history_days":
                history_days,

            "feature_source":
                source,

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

    return (
        pd.DataFrame(rows),
        diagnostics
    )


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
# スコアリング
# ============================================================

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
    )


# ============================================================
# 評価
# ============================================================

def evaluate(
    ranked,
    top_n
):

    if ranked.empty:
        return None

    top = ranked.head(
        min(
            top_n,
            len(ranked)
        )
    )

    d = top[
        "actual_diff"
    ].astype(float)

    return {

        "top_n":
            top_n,

        "avg_diff":
            float(
                d.mean()
            ),

        "median_diff":
            float(
                d.median()
            ),

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
            float(
                d.sum()
            ),
    }


# ============================================================
# メイン
# ============================================================

def main():

    print("=" * 70)
    print(
        "Ana-Slo Ver.4 "
        "Machine Change + Type Fallback OOS Backtest"
    )
    print("=" * 70)

    print()
    print("FIXED WEIGHTS")
    print("-" * 70)

    for factor in FACTORS:

        print(
            f"{factor:<18}: "
            f"{V4_WEIGHTS[factor] * 100:7.2f}%"
        )

    print(
        f"weight sum       : "
        f"{sum(V4_WEIGHTS.values()) * 100:7.2f}%"
    )

    print()

    df = load_data()

    print(
        f"records = {len(df):,}"
    )

    print(
        f"OOS period = "
        f"{TEST_START.date()} "
        f"to "
        f"{TEST_END.date()}"
    )

    print()

    daily_results = []

    diagnostics_results = []

    source_results = []

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

        if panel.empty:

            print(
                target_date.date(),
                "NO DATA"
            )

            continue

        print(
            f"{target_date.date()} "
            f"machines={len(panel)} "
            f"same_machine="
            f"{diagnostics['same_machine_history']} "
            f"type_fallback="
            f"{diagnostics['type_fallback']} "
            f"global_fallback="
            f"{diagnostics['global_fallback']} "
            f"machine_change="
            f"{diagnostics['machine_change_detected']}"
        )

        ranked = rank_score(
            panel,
            V4_WEIGHTS
        )

        # ----------------------------------------------------
        # 日次診断
        # ----------------------------------------------------

        diagnostics_results.append({

            "date":
                target_date.date(),

            "machines":
                len(panel),

            "same_machine_history":
                diagnostics[
                    "same_machine_history"
                ],

            "type_fallback":
                diagnostics[
                    "type_fallback"
                ],

            "global_fallback":
                diagnostics[
                    "global_fallback"
                ],

            "machine_change_detected":
                diagnostics[
                    "machine_change_detected"
                ],
        })

        # ----------------------------------------------------
        # 特徴量ソース
        # ----------------------------------------------------

        source_counts = (
            panel[
                "feature_source"
            ]
            .value_counts()
            .to_dict()
        )

        source_results.append({

            "date":
                target_date.date(),

            "same_machine":
                source_counts.get(
                    "same_machine",
                    0
                ),

            "type_fallback":
                source_counts.get(
                    "type_fallback",
                    0
                ),

            "global_fallback":
                source_counts.get(
                    "global_fallback",
                    0
                ),
        })

        # ----------------------------------------------------
        # TOP10
        # ----------------------------------------------------

        top10 = ranked.head(10)

        d = top10[
            "actual_diff"
        ].astype(float)

        daily_results.append({

            "date":
                target_date.date(),

            "machines":
                len(ranked),

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
        })

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows = []

    # 再構築してTOP別集計
    for target_date in pd.date_range(
        TEST_START,
        TEST_END
    ):

        panel, _ = build_features(
            df,
            target_date
        )

        if panel.empty:
            continue

        ranked = rank_score(
            panel,
            V4_WEIGHTS
        )

        for top_n in (
            1,
            5,
            10,
            20,
            30
        ):

            result = evaluate(
                ranked,
                top_n
            )

            if result is None:
                continue

            result["date"] = (
                target_date.date()
            )

            summary_rows.append(
                result
            )

    summary_daily = pd.DataFrame(
        summary_rows
    )

    summary = []

    for top_n in (
        1,
        5,
        10,
        20,
        30
    ):

        sub = summary_daily[
            summary_daily["top_n"]
            == top_n
        ]

        if sub.empty:
            continue

        summary.append({

            "top_n":
                top_n,

            "days":
                len(sub),

            "avg_diff":
                float(
                    sub["avg_diff"]
                    .mean()
                ),

            "median_daily_avg":
                float(
                    sub["avg_diff"]
                    .median()
                ),

            "win_rate":
                float(
                    (
                        sub["win_rate"]
                        * top_n
                    ).sum()
                    /
                    (
                        top_n
                        * len(sub)
                    )
                ),

            "plus1000_rate":
                float(
                    (
                        sub[
                            "plus1000_rate"
                        ]
                        * top_n
                    ).sum()
                    /
                    (
                        top_n
                        * len(sub)
                    )
                ),

            "plus2000_rate":
                float(
                    (
                        sub[
                            "plus2000_rate"
                        ]
                        * top_n
                    ).sum()
                    /
                    (
                        top_n
                        * len(sub)
                    )
                ),

            "positive_days":
                float(
                    (
                        sub["total_diff"]
                        > 0
                    ).mean()
                    * 100
                ),

            "total_diff":
                float(
                    sub["total_diff"]
                    .sum()
                ),
        })

    summary_df = pd.DataFrame(
        summary
    )

    diagnostics_df = pd.DataFrame(
        diagnostics_results
    )

    source_df = pd.DataFrame(
        source_results
    )

    # ========================================================
    # 保存
    # ========================================================

    out_daily = (
        OUT_DIR
        / "18_Ver4_type_fallback_daily.csv"
    )

    out_summary = (
        OUT_DIR
        / "18_Ver4_type_fallback_summary.csv"
    )

    out_diagnostics = (
        OUT_DIR
        / "18_Ver4_type_fallback_diagnostics.csv"
    )

    out_source = (
        OUT_DIR
        / "18_Ver4_type_fallback_sources.csv"
    )

    out_weights = (
        OUT_DIR
        / "18_Ver4_type_fallback_weights.csv"
    )

    pd.DataFrame(
        daily_results
    ).to_csv(
        out_daily,
        index=False,
        encoding="utf-8-sig"
    )

    summary_df.to_csv(
        out_summary,
        index=False,
        encoding="utf-8-sig"
    )

    diagnostics_df.to_csv(
        out_diagnostics,
        index=False,
        encoding="utf-8-sig"
    )

    source_df.to_csv(
        out_source,
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
                    ],
            }

            for factor in FACTORS
        ]
    ).to_csv(
        out_weights,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 表示
    # ========================================================

    print()
    print("=" * 70)
    print(
        "VER.4 TYPE FALLBACK RESULT"
    )
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "FEATURE SOURCE DIAGNOSTICS"
    )

    print(
        diagnostics_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:"
    )

    print(out_daily)
    print(out_summary)
    print(out_diagnostics)
    print(out_source)
    print(out_weights)

    print()
    print(
        "Ver.4 type fallback "
        "OOS backtest complete."
    )


if __name__ == "__main__":
    main()