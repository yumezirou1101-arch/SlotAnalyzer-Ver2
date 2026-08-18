from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Ana-Slo Ver.4
# 機種変更対応版
#
# 目的：
#   ・台番号が同じでも機種変更した場合、旧機種の履歴を
#     新機種へ引き継がない
#   ・機種変更台も予測対象から除外しない
#   ・既存Ver.4の固定ウェイトを維持
#
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


# ============================================================
# 機種変更対応 feature build
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

        return pd.DataFrame()

    target_weekday = (
        target_date.dayofweek
    )

    # --------------------------------------------------------
    # target日前日の状態
    # --------------------------------------------------------

    latest_date = hist["date"].max()

    latest_day = (
        hist[
            hist["date"] == latest_date
        ]
        .set_index("machine_no")
    )

    # --------------------------------------------------------
    # 機種別平均
    #
    # target_date以前のデータだけを使用。
    # target_date当日のデータは絶対に使用しない。
    # --------------------------------------------------------

    type_stats = (
        hist.groupby(
            "machine_name"
        )["diff"]
        .mean()
        .to_dict()
    )

    rows = []

    # --------------------------------------------------------
    # 現在の実機種を基準に処理
    # --------------------------------------------------------

    for _, actual_row in actual.iterrows():

        no = int(
            actual_row["machine_no"]
        )

        current_name = str(
            actual_row["machine_name"]
        )

        # ----------------------------------------------------
        # 同じ台番号 AND 同じ現在機種だけを履歴として使用
        #
        # これが今回の修正の中心。
        #
        # 例：
        # 931番
        # 旧：からくりサーカス
        # 新：からくりサーカス2
        #
        # → 旧「からくりサーカス」の931番履歴は
        #    新「からくりサーカス2」の台履歴には使わない。
        # ----------------------------------------------------

        m = hist[
            (hist["machine_no"] == no)
            & (
                hist["machine_name"]
                == current_name
            )
        ].copy()

        m = m.sort_values(
            "date"
        )

        history_days = len(m)

        # ----------------------------------------------------
        # 自台の同一機種履歴が存在しない場合
        #
        # 無理に旧機種データを流用しない。
        # featureはNaNにして後段で中立補完する。
        # ----------------------------------------------------

        if history_days == 0:

            avg31 = np.nan
            recent7_avg = np.nan
            recent7_win = np.nan
            last_diff = np.nan
            prev_change = np.nan
            weekday_avg = np.nan
            plus1000_rate = np.nan
            plus2000_rate = np.nan

        else:

            # ------------------------------------------------
            # 31日平均
            # ------------------------------------------------

            avg31 = float(
                m["diff"].mean()
            )

            # ------------------------------------------------
            # 直近7日
            # ------------------------------------------------

            recent7 = m.tail(7)

            recent7_avg = float(
                recent7["diff"].mean()
            )

            recent7_win = float(
                recent7["win"].mean()
            )

            # ------------------------------------------------
            # 前日差枚
            # ------------------------------------------------

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

            # ------------------------------------------------
            # 曜日平均
            # ------------------------------------------------

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

            # ------------------------------------------------
            # 出率
            # ------------------------------------------------

            plus1000_rate = float(
                m["plus1000"].mean()
            )

            plus2000_rate = float(
                m["plus2000"].mean()
            )

        # ----------------------------------------------------
        # 機種平均
        # ----------------------------------------------------

        type_avg = float(
            type_stats.get(
                current_name,
                0.0
            )
        )

        # ----------------------------------------------------
        # 隣接台
        #
        # 機種変更したばかりの台についても、
        # 前日の実際の隣接台データは使用する。
        #
        # ただし隣接台が存在しない場合はNaN。
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

            neighbor_avg = np.nan

        # ----------------------------------------------------
        # リバウンド信号
        #
        # 同一機種の前日データが存在するときだけ計算。
        # ----------------------------------------------------

        if history_days == 0:

            bounce_signal = np.nan

        else:

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
                current_name,

            "history_days":
                history_days,

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

    # --------------------------------------------------------
    # 重要：
    # actualをmachine_noだけで結合。
    #
    # current_nameはactual側を正としているため、
    # 機種変更台も除外されない。
    # --------------------------------------------------------

    return feat.merge(
        actual,
        on="machine_no",
        how="inner",
        suffixes=(
            "",
            "_actual"
        )
    )


# ============================================================
# Z-score
#
# 欠損値は、その日の対象台全体の中央値で補完。
#
# 機種変更直後の台について、
# 旧機種のデータを「0」として扱うことを避ける。
# ============================================================

def zscore(series):

    s = pd.to_numeric(
        series,
        errors="coerce"
    ).copy()

    if s.notna().any():

        median = float(
            s.median()
        )

        s = s.fillna(
            median
        )

    else:

        s = s.fillna(
            0.0
        )

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


# ============================================================
# Score
# ============================================================

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


# ============================================================
# Evaluation
# ============================================================

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


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "Ana-Slo Ver.4 "
        "Machine Change Aware OOS Backtest"
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

    diagnostics = []

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

        # ----------------------------------------------------
        # 診断情報
        # ----------------------------------------------------

        changed_count = int(
            (
                panel["history_days"]
                == 0
            ).sum()
        )

        diagnostics.append({

            "date":
                target_date.date(),

            "machines":
                len(panel),

            "no_same_machine_history":
                changed_count,
        })

        print(
            "%s machines=%d "
            "no_same_machine_history=%d"
            % (
                target_date.date(),
                len(panel),
                changed_count
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

    diagnostics_df = pd.DataFrame(
        diagnostics
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

    # ========================================================
    # 結果表示
    # ========================================================

    print()
    print("=" * 70)

    print(
        "VER.4 MACHINE CHANGE AWARE RESULT"
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

    print()
    print(
        "MACHINE CHANGE DIAGNOSTICS"
    )

    print(
        diagnostics_df.to_string(
            index=False
        )
    )

    # ========================================================
    # 保存
    # ========================================================

    out_daily = (
        OUT_DIR
        / "17_Ver4_machine_change_daily.csv"
    )

    out_summary = (
        OUT_DIR
        / "17_Ver4_machine_change_summary.csv"
    )

    out_diag = (
        OUT_DIR
        / "17_Ver4_machine_change_diagnostics.csv"
    )

    out_weights = (
        OUT_DIR
        / "17_Ver4_machine_change_weights.csv"
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

    diagnostics_df.to_csv(
        out_diag,
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
    print(out_diag)
    print(out_weights)

    print()
    print(
        "Ver.4 Machine Change Aware "
        "OOS backtest complete."
    )


if __name__ == "__main__":
    main()