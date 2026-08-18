# -*- coding: utf-8 -*-

"""
投入パターン予測 V5.1

V5をベースに、要因分析V1.1の結果を反映した改良モデル。

V5.1の重点
------------------------------------------------------------
・台_前回変化       ：最重要
・機種_前日差枚     ：最重要
・台_凹み           ：強化
・台_前日差枚       ：強化
・機種_直近3日平均  ：強化
・台_直近3日平均    ：中程度
・過去平均差枚      ：弱化
・過去プラス率      ：弱化

重要
------------------------------------------------------------
予測対象日の実績は特徴量計算に使用しません。
予測対象は「最新営業日に存在した台」のみです。
"""


import os
import numpy as np
import pandas as pd


# ============================================================
# 基本設定
# ============================================================

BASE_DIR = r"C:\Users\user\Desktop\Documents\SlotAnalyzer"

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "maruhan_maebashi",
    "all_data.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "maruhan_maebashi",
    "machine_number"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "prediction_v5_1.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "prediction_v5_1_summary.csv"
)

TOP_N = 30


# ============================================================
# 共通関数
# ============================================================

def print_line():
    print("=" * 70)


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def normalize_series(series):
    """
    シリーズを0～100に正規化する。
    """

    s = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)

    if len(s) == 0:
        return pd.Series(
            index=series.index,
            dtype=float
        )

    min_value = s.min()
    max_value = s.max()

    if max_value == min_value:
        return pd.Series(
            50.0,
            index=series.index
        )

    return (
        (s - min_value)
        / (max_value - min_value)
        * 100
    )


def score_rank(score):

    if score >= 75:
        return "S"

    elif score >= 65:
        return "A"

    elif score >= 55:
        return "B"

    elif score >= 45:
        return "C"

    elif score >= 30:
        return "D"

    else:
        return "E"


def find_column(df, candidates):

    for col in candidates:

        if col in df.columns:
            return col

    return None


# ============================================================
# データ読み込み
# ============================================================

def load_data():

    print("入力ファイル:")
    print(DATA_FILE)
    print()

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"入力ファイルがありません:\n{DATA_FILE}"
        )

    print("all_data.csv を読み込みます...")

    try:

        df = pd.read_csv(
            DATA_FILE,
            encoding="utf-8-sig"
        )

    except Exception:

        try:

            df = pd.read_csv(
                DATA_FILE,
                encoding="utf-8"
            )

        except Exception:

            df = pd.read_csv(
                DATA_FILE,
                encoding="cp932"
            )

    print(
        f"読み込みデータ: {len(df):,}行"
    )

    print()
    print("必要な列を確認します...")

    date_col = find_column(
        df,
        [
            "日付",
            "date",
            "Date"
        ]
    )

    machine_no_col = find_column(
        df,
        [
            "台番号",
            "台番",
            "machine_number"
        ]
    )

    machine_name_col = find_column(
        df,
        [
            "機種名",
            "機種",
            "機種名称",
            "machine_name"
        ]
    )

    diff_col = find_column(
        df,
        [
            "差枚",
            "差枚数",
            "差枚数（枚）",
            "差枚数(枚)"
        ]
    )

    print(
        f"日付   : {date_col}"
    )

    print(
        f"台番号 : {machine_no_col}"
    )

    print(
        f"機種   : {machine_name_col}"
    )

    print(
        f"差枚   : {diff_col}"
    )

    print()

    if any(
        x is None
        for x in [
            date_col,
            machine_no_col,
            machine_name_col,
            diff_col
        ]
    ):

        raise ValueError(
            "必要な列が見つかりません。"
        )

    print("必要な列: OK")

    df = df.rename(
        columns={
            date_col: "日付",
            machine_no_col: "台番号",
            machine_name_col: "機種",
            diff_col: "差枚"
        }
    )

    df["日付"] = pd.to_datetime(
        df["日付"],
        errors="coerce"
    )

    df["台番号"] = pd.to_numeric(
        df["台番号"],
        errors="coerce"
    )

    df["差枚"] = pd.to_numeric(
        df["差枚"],
        errors="coerce"
    )

    df["機種"] = (
        df["機種"]
        .astype(str)
        .str.strip()
    )

    df = df.dropna(
        subset=[
            "日付",
            "台番号",
            "機種",
            "差枚"
        ]
    ).copy()

    df["台番号"] = (
        df["台番号"]
        .astype(int)
    )

    df["日付"] = (
        df["日付"]
        .dt.normalize()
    )

    df = df.sort_values(
        [
            "日付",
            "台番号"
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"有効データ: {len(df):,}行"
    )

    return df


# ============================================================
# 台番号別特徴量
# ============================================================

def calc_machine_features(
    history,
    candidate_machines
):

    rows = []

    for machine_no in candidate_machines:

        m = history[
            history["台番号"] == machine_no
        ].sort_values(
            "日付"
        )

        if len(m) == 0:
            continue

        # 最新の機種名
        machine_name = str(
            m.iloc[-1]["機種"]
        )

        diffs = (
            pd.to_numeric(
                m["差枚"],
                errors="coerce"
            )
            .dropna()
            .tolist()
        )

        if len(diffs) == 0:
            continue

        last_diff = diffs[-1]

        if len(diffs) >= 2:

            previous_diff = diffs[-2]

            previous_change = (
                last_diff
                - previous_diff
            )

        else:

            previous_diff = last_diff
            previous_change = 0.0

        recent_3 = diffs[-3:]

        recent_3_avg = (
            np.mean(recent_3)
            if recent_3
            else 0.0
        )

        average_diff = np.mean(
            diffs
        )

        positive_rate = (
            np.mean(
                np.array(diffs) > 0
            )
            * 100
        )

        plus_500_rate = (
            np.mean(
                np.array(diffs) >= 500
            )
            * 100
        )

        plus_1000_rate = (
            np.mean(
                np.array(diffs) >= 1000
            )
            * 100
        )

        plus_2000_rate = (
            np.mean(
                np.array(diffs) >= 2000
            )
            * 100
        )

        # ----------------------------------------------------
        # 凹み
        # ----------------------------------------------------

        recent_min = min(
            recent_3
        )

        if recent_min < 0:

            dent = abs(
                recent_min
            )

        else:

            dent = 0.0

        if last_diff < 0:

            dent += (
                abs(last_diff)
                * 0.5
            )

        rows.append(
            {
                "台番号": machine_no,
                "機種": machine_name,
                "台_過去平均差枚": average_diff,
                "台_過去プラス率": positive_rate,
                "台_直近3日平均": recent_3_avg,
                "台_前日差枚": last_diff,
                "台_前回変化": previous_change,
                "台_凹み": dent,
                "台_過去+500率": plus_500_rate,
                "台_過去+1000率": plus_1000_rate,
                "台_過去+2000率": plus_2000_rate,
                "台_履歴日数": len(diffs)
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 機種別特徴量
# ============================================================

def calc_machine_type_features(
    history
):

    rows = []

    for machine_name, group in history.groupby(
        "機種"
    ):

        group = group.sort_values(
            "日付"
        )

        diffs = (
            pd.to_numeric(
                group["差枚"],
                errors="coerce"
            )
            .dropna()
            .tolist()
        )

        if len(diffs) == 0:
            continue

        average_diff = np.mean(
            diffs
        )

        positive_rate = (
            np.mean(
                np.array(diffs) > 0
            )
            * 100
        )

        plus_500_rate = (
            np.mean(
                np.array(diffs) >= 500
            )
            * 100
        )

        plus_1000_rate = (
            np.mean(
                np.array(diffs) >= 1000
            )
            * 100
        )

        plus_2000_rate = (
            np.mean(
                np.array(diffs) >= 2000
            )
            * 100
        )

        # 機種全体の日別平均
        daily = (
            group
            .groupby("日付")["差枚"]
            .mean()
            .sort_index()
        )

        daily_values = (
            daily
            .tolist()
        )

        if daily_values:

            last_day_diff = (
                daily_values[-1]
            )

        else:

            last_day_diff = 0.0

        recent_3 = (
            daily_values[-3:]
        )

        recent_3_avg = (
            np.mean(recent_3)
            if recent_3
            else 0.0
        )

        if len(daily_values) >= 2:

            previous_change = (
                daily_values[-1]
                - daily_values[-2]
            )

        else:

            previous_change = 0.0

        if recent_3:

            recent_min = min(
                recent_3
            )

        else:

            recent_min = 0.0

        if recent_min < 0:

            dent = abs(
                recent_min
            )

        else:

            dent = 0.0

        rows.append(
            {
                "機種": machine_name,
                "機種_過去平均差枚": average_diff,
                "機種_過去プラス率": positive_rate,
                "機種_直近3日平均": recent_3_avg,
                "機種_前日差枚": last_day_diff,
                "機種_前回変化": previous_change,
                "機種_凹み": dent,
                "機種_過去+500率": plus_500_rate,
                "機種_過去+1000率": plus_1000_rate,
                "機種_過去+2000率": plus_2000_rate,
                "機種_履歴日数": len(daily_values)
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# V5.1スコア計算
# ============================================================

def calculate_v51_score(
    df
):

    df = df.copy()

    # --------------------------------------------------------
    # 0～100正規化
    # --------------------------------------------------------

    df["N_台_前回変化"] = normalize_series(
        df["台_前回変化"]
    )

    df["N_台_凹み"] = normalize_series(
        df["台_凹み"]
    )

    df["N_台_前日差枚"] = normalize_series(
        df["台_前日差枚"]
    )

    df["N_台_直近3日平均"] = normalize_series(
        df["台_直近3日平均"]
    )

    df["N_台_過去平均差枚"] = normalize_series(
        df["台_過去平均差枚"]
    )

    df["N_台_過去プラス率"] = normalize_series(
        df["台_過去プラス率"]
    )

    df["N_機種_前日差枚"] = normalize_series(
        df["機種_前日差枚"]
    )

    df["N_機種_直近3日平均"] = normalize_series(
        df["機種_直近3日平均"]
    )

    df["N_機種_凹み"] = normalize_series(
        df["機種_凹み"]
    )

    df["N_機種_過去平均差枚"] = normalize_series(
        df["機種_過去平均差枚"]
    )

    df["N_機種_過去プラス率"] = normalize_series(
        df["機種_過去プラス率"]
    )

    # --------------------------------------------------------
    # V5.1ウェイト
    #
    # 台_前回変化       20%
    # 台_凹み           15%
    # 機種_前日差枚     20%
    # 台_前日差枚       12%
    # 機種_直近3日平均   10%
    # 台_直近3日平均      7%
    # 機種_凹み           5%
    # 機種_過去平均       4%
    # 台_過去平均         2%
    # 台_過去プラス率     3%
    # 機種_過去プラス率   2%
    #
    # 合計100%
    # --------------------------------------------------------

    df["V51スコア_raw"] = (

        df["N_台_前回変化"]
        * 0.20

        + df["N_台_凹み"]
        * 0.15

        + df["N_機種_前日差枚"]
        * 0.20

        + df["N_台_前日差枚"]
        * 0.12

        + df["N_機種_直近3日平均"]
        * 0.10

        + df["N_台_直近3日平均"]
        * 0.07

        + df["N_機種_凹み"]
        * 0.05

        + df["N_機種_過去平均差枚"]
        * 0.04

        + df["N_台_過去平均差枚"]
        * 0.02

        + df["N_台_過去プラス率"]
        * 0.03

        + df["N_機種_過去プラス率"]
        * 0.02
    )

    df["V51スコア"] = (
        df["V51スコア_raw"]
    )

    # --------------------------------------------------------
    # ボーナス
    # --------------------------------------------------------

    # 台_前回変化 +500以上
    df.loc[
        df["台_前回変化"] >= 500,
        "V51スコア"
    ] += 3.0

    # 機種_前日差枚 +500以上
    df.loc[
        df["機種_前日差枚"] >= 500,
        "V51スコア"
    ] += 3.0

    # 台_前日差枚 +500以上
    df.loc[
        df["台_前日差枚"] >= 500,
        "V51スコア"
    ] += 2.0

    # 凹み +500以上
    df.loc[
        df["台_凹み"] >= 500,
        "V51スコア"
    ] += 2.0

    # --------------------------------------------------------
    # ペナルティ
    # --------------------------------------------------------

    # 前日・前回とも1000枚以上
    # →出過ぎ警戒
    df.loc[
        (
            (df["台_前日差枚"] >= 1000)
            &
            (df["台_前回変化"] >= 1000)
        ),
        "V51スコア"
    ] -= 2.0

    # 機種全体が前日-1000枚以下
    df.loc[
        df["機種_前日差枚"] <= -1000,
        "V51スコア"
    ] -= 2.0

    # --------------------------------------------------------
    # 0～100
    # --------------------------------------------------------

    df["V51スコア"] = (
        df["V51スコア"]
        .clip(0, 100)
    )

    df["ランク"] = (
        df["V51スコア"]
        .apply(score_rank)
    )

    return df


# ============================================================
# 重点条件
# ============================================================

def is_focus_candidate(
    row
):

    conditions = 0

    if safe_float(
        row["台_前回変化"]
    ) >= 300:

        conditions += 1

    if safe_float(
        row["台_凹み"]
    ) >= 300:

        conditions += 1

    if safe_float(
        row["機種_前日差枚"]
    ) >= 300:

        conditions += 1

    if safe_float(
        row["台_前日差枚"]
    ) >= 300:

        conditions += 1

    return conditions >= 2


# ============================================================
# 機種別集計
# ============================================================

def create_machine_summary(
    df
):

    summary = (
        df.groupby("機種")
        .agg(
            台数=(
                "台番号",
                "count"
            ),
            平均V51=(
                "V51スコア",
                "mean"
            ),
            最高V51=(
                "V51スコア",
                "max"
            ),
            機種前日=(
                "機種_前日差枚",
                "first"
            ),
            機種直近3日平均=(
                "機種_直近3日平均",
                "first"
            )
        )
        .reset_index()
    )

    summary = summary.sort_values(
        [
            "平均V51",
            "最高V51"
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

    return summary


# ============================================================
# メイン
# ============================================================

def main():

    print_line()
    print("投入パターン予測 V5.1")
    print_line()

    print(
        "V5をベースに要因分析結果を反映した"
        "改良モデルです。"
    )

    print()
    print(
        "予測対象日の実績は特徴量計算に使用しません。"
    )

    print(
        "予測対象は最新営業日に存在する台のみです。"
    )

    print()

    # --------------------------------------------------------
    # 読み込み
    # --------------------------------------------------------

    df = load_data()

    if len(df) == 0:

        raise ValueError(
            "有効データがありません。"
        )

    dates = sorted(
        df["日付"]
        .dt.normalize()
        .unique()
    )

    print()
    print(
        f"収録日数: {len(dates)}"
    )

    print("収録日:")

    print(
        " / ".join(
            pd.Timestamp(d).strftime(
                "%Y-%m-%d"
            )
            for d in dates
        )
    )

    latest_date = pd.Timestamp(
        dates[-1]
    )

    next_date = (
        latest_date
        + pd.Timedelta(days=1)
    )

    print()
    print("最新実績日:")
    print(
        latest_date.strftime(
            "%Y-%m-%d"
        )
    )

    print()
    print("次回予測日:")
    print(
        next_date.strftime(
            "%Y-%m-%d"
        )
    )

    weekday_jp = [
        "月曜日",
        "火曜日",
        "水曜日",
        "木曜日",
        "金曜日",
        "土曜日",
        "日曜日"
    ]

    print(
        "予測曜日: "
        + weekday_jp[
            next_date.weekday()
        ]
    )

    # --------------------------------------------------------
    # 最新営業日以前の履歴
    # --------------------------------------------------------

    history = df[
        df["日付"] <= latest_date
    ].copy()

    # --------------------------------------------------------
    # ★重要
    # 最新営業日に存在した台だけを候補にする
    # --------------------------------------------------------

    latest_day = df[
        df["日付"] == latest_date
    ].copy()

    latest_day = latest_day.drop_duplicates(
        subset=["台番号"],
        keep="last"
    )

    candidate_machines = sorted(
        latest_day["台番号"]
        .dropna()
        .unique()
    )

    print()
    print(
        f"候補台数: "
        f"{len(candidate_machines)}台"
    )

    # --------------------------------------------------------
    # 台特徴量
    # --------------------------------------------------------

    machine_features = (
        calc_machine_features(
            history,
            candidate_machines
        )
    )

    if machine_features.empty:

        raise ValueError(
            "台番号特徴量を作成できませんでした。"
        )

    # --------------------------------------------------------
    # 機種特徴量
    # --------------------------------------------------------

    type_features = (
        calc_machine_type_features(
            history
        )
    )

    if type_features.empty:

        raise ValueError(
            "機種特徴量を作成できませんでした。"
        )

    # --------------------------------------------------------
    # 機種特徴量を結合
    # --------------------------------------------------------

    result = machine_features.merge(
        type_features,
        on="機種",
        how="left"
    )

    # --------------------------------------------------------
    # 欠損処理
    # --------------------------------------------------------

    for col in result.columns:

        if col == "機種":
            continue

        result[col] = pd.to_numeric(
            result[col],
            errors="coerce"
        ).fillna(0)

    # --------------------------------------------------------
    # スコア計算
    # --------------------------------------------------------

    result = calculate_v51_score(
        result
    )

    result["予測日"] = (
        next_date.strftime(
            "%Y-%m-%d"
        )
    )

    result["モデル"] = "V5.1"

    # --------------------------------------------------------
    # 順位
    # --------------------------------------------------------

    result = result.sort_values(
        [
            "V51スコア",
            "台_前回変化",
            "機種_前日差枚",
            "台_前日差枚"
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

    result["予測順位"] = (
        np.arange(
            1,
            len(result) + 1
        )
    )

    # --------------------------------------------------------
    # TOP30
    # --------------------------------------------------------

    print()
    print_line()
    print("【次回おすすめ台 TOP30】")
    print_line()

    top30 = result.head(
        TOP_N
    )

    for _, row in top30.iterrows():

        print(
            f"{int(row['予測順位']):2d}. "
            f"{int(row['台番号'])} "
            f"{row['機種']} / "
            f"V5.1 "
            f"{row['V51スコア']:.1f} / "
            f"{row['ランク']} / "
            f"前回変化 "
            f"{row['台_前回変化']:+.0f}枚 / "
            f"前日 "
            f"{row['台_前日差枚']:+.0f}枚 / "
            f"機種前日 "
            f"{row['機種_前日差枚']:+.0f}枚"
        )

    # --------------------------------------------------------
    # ランク別
    # --------------------------------------------------------

    print()
    print_line()
    print("【ランク別台数】")
    print_line()

    rank_order = [
        "S",
        "A",
        "B",
        "C",
        "D",
        "E"
    ]

    rank_counts = (
        result["ランク"]
        .value_counts()
    )

    for rank in rank_order:

        print(
            f"{rank}: "
            f"{int(rank_counts.get(rank, 0))}台"
        )

    # --------------------------------------------------------
    # 機種別
    # --------------------------------------------------------

    machine_summary = (
        create_machine_summary(
            result
        )
    )

    print()
    print_line()
    print("【機種別おすすめ TOP20】")
    print_line()

    for i, row in machine_summary.head(
        20
    ).iterrows():

        print(
            f"{i + 1:2d}. "
            f"{row['機種']} / "
            f"{int(row['台数'])}台 / "
            f"平均V5.1 "
            f"{row['平均V51']:.1f} / "
            f"最高V5.1 "
            f"{row['最高V51']:.1f} / "
            f"機種前日 "
            f"{row['機種前日']:+.0f}枚"
        )

    # --------------------------------------------------------
    # 重点条件
    # --------------------------------------------------------

    result["重点条件"] = (
        result.apply(
            is_focus_candidate,
            axis=1
        )
    )

    focus = (
        result[
            result["重点条件"]
        ]
        .head(30)
    )

    print()
    print_line()
    print("【V5.1重点条件該当台】")
    print_line()

    if len(focus) == 0:

        print("該当台なし")

    else:

        for _, row in focus.iterrows():

            print(
                f"台番号 "
                f"{int(row['台番号'])} / "
                f"{row['機種']} / "
                f"V5.1 "
                f"{row['V51スコア']:.1f} / "
                f"前回変化 "
                f"{row['台_前回変化']:+.0f}枚 / "
                f"前日 "
                f"{row['台_前日差枚']:+.0f}枚 / "
                f"機種前日 "
                f"{row['機種_前日差枚']:+.0f}枚"
            )

    # --------------------------------------------------------
    # CSV保存
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output_columns = [
        "予測日",
        "モデル",
        "予測順位",
        "台番号",
        "機種",
        "V51スコア",
        "ランク",
        "台_前回変化",
        "台_前日差枚",
        "台_直近3日平均",
        "台_凹み",
        "台_過去平均差枚",
        "台_過去プラス率",
        "機種_前日差枚",
        "機種_直近3日平均",
        "機種_凹み",
        "機種_過去平均差枚",
        "機種_過去プラス率",
        "重点条件"
    ]

    output_df = result[
        output_columns
    ].copy()

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("★ CSV保存成功")
    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_rows = []

    for rank in rank_order:

        subset = result[
            result["ランク"] == rank
        ]

        summary_rows.append(
            {
                "予測日": next_date.strftime(
                    "%Y-%m-%d"
                ),
                "モデル": "V5.1",
                "ランク": rank,
                "台数": len(subset),
                "平均スコア": (
                    subset["V51スコア"].mean()
                    if len(subset)
                    else np.nan
                ),
                "最高スコア": (
                    subset["V51スコア"].max()
                    if len(subset)
                    else np.nan
                )
            }
        )

    # TOP5
    for top_n in [
        5,
        10,
        20,
        30
    ]:

        subset = result.head(
            top_n
        )

        summary_rows.append(
            {
                "予測日": next_date.strftime(
                    "%Y-%m-%d"
                ),
                "モデル": "V5.1",
                "ランク": f"TOP{top_n}",
                "台数": len(subset),
                "平均スコア": (
                    subset["V51スコア"].mean()
                    if len(subset)
                    else np.nan
                ),
                "最高スコア": (
                    subset["V51スコア"].max()
                    if len(subset)
                    else np.nan
                )
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("★ CSV保存成功")
    print(SUMMARY_FILE)

    print()
    print_line()
    print("★★★★★ V5.1予測 完了 ★★★★★")
    print_line()

    print()
    print("保存ファイル:")
    print(OUTPUT_FILE)
    print(SUMMARY_FILE)

    print()
    print("all_data.csv は変更していません。")


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    main()