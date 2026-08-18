# -*- coding: utf-8 -*-

"""
投入パターン解析 V4 バックテスト

目的:
・機種単位の投入傾向
・台番号単位の投入傾向
・直近実績
・凹み/反発
・プラス率
・+1000枚率
・+2000枚率
・データ量による信頼度

を組み合わせて次回狙い機種・台番号を算出し、
過去データを使ってバックテストする。

重要:
バックテストでは対象日のデータを予測計算に使用しない。
対象日前のデータだけでスコアを作る。
"""

import os
import sys
import math
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# ============================================================
# 設定
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    "investment_pattern_v4_backtest.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "investment_pattern_v4_backtest_summary.csv"
)


# ============================================================
# 表示
# ============================================================

def print_header(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


# ============================================================
# CSV読み込み
# ============================================================

def load_data():

    print()
    print("入力ファイル:")
    print(DATA_FILE)

    if not os.path.exists(DATA_FILE):
        print()
        print("ERROR: 入力ファイルが見つかりません。")
        print(DATA_FILE)
        sys.exit(1)

    print()
    print("all_data.csv を読み込みます...")

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp932",
        "shift_jis"
    ]

    df = None

    for enc in encodings:
        try:
            df = pd.read_csv(DATA_FILE, encoding=enc)
            break
        except Exception:
            continue

    if df is None:
        print("ERROR: CSVを読み込めませんでした。")
        sys.exit(1)

    print(f"読み込みデータ: {len(df):,}行")

    print("必要な列を確認します...")

    # --------------------------------------------------------
    # 列名の候補
    # --------------------------------------------------------

    date_candidates = [
        "日付",
        "DATE",
        "date",
        "年月日",
        "営業日"
    ]

    machine_candidates = [
        "機種",
        "機種名",
        "機種名称",
        "machine",
        "machine_name"
    ]

    number_candidates = [
        "台番号",
        "台番",
        "台No",
        "台ＮＯ",
        "台NO",
        "number",
        "machine_number"
    ]

    diff_candidates = [
        "差枚",
        "差枚数",
        "差枚数（枚）",
        "差枚（枚）",
        "差枚数枚",
        "差枚数(枚)"
    ]

    def find_column(candidates):

        for col in candidates:
            if col in df.columns:
                return col

        # 部分一致
        for col in df.columns:
            col_str = str(col)

            for candidate in candidates:
                if candidate in col_str:
                    return col

        return None

    date_col = find_column(date_candidates)
    machine_col = find_column(machine_candidates)
    number_col = find_column(number_candidates)
    diff_col = find_column(diff_candidates)

    if date_col is None:
        print("ERROR: 日付列が見つかりません。")
        print(list(df.columns))
        sys.exit(1)

    if machine_col is None:
        print("ERROR: 機種列が見つかりません。")
        print(list(df.columns))
        sys.exit(1)

    if number_col is None:
        print("ERROR: 台番号列が見つかりません。")
        print(list(df.columns))
        sys.exit(1)

    if diff_col is None:
        print("ERROR: 差枚列が見つかりません。")
        print(list(df.columns))
        sys.exit(1)

    print("必要な列: OK")

    # --------------------------------------------------------
    # 標準列名へ統一
    # --------------------------------------------------------

    df = df.rename(
        columns={
            date_col: "日付",
            machine_col: "機種",
            number_col: "台番号",
            diff_col: "差枚"
        }
    )

    # --------------------------------------------------------
    # データ整形
    # --------------------------------------------------------

    df["日付"] = pd.to_datetime(
        df["日付"],
        errors="coerce"
    )

    df["差枚"] = pd.to_numeric(
        df["差枚"],
        errors="coerce"
    )

    df["台番号"] = pd.to_numeric(
        df["台番号"],
        errors="coerce"
    )

    df["機種"] = df["機種"].astype(str).str.strip()

    df = df.dropna(
        subset=["日付", "機種", "台番号", "差枚"]
    ).copy()

    df["台番号"] = df["台番号"].astype(int)

    df = df.sort_values(
        ["日付", "台番号"]
    ).reset_index(drop=True)

    return df


# ============================================================
# 信頼度
# ============================================================

def reliability_score(n, max_n=50):

    """
    データ件数が少ない機種・台番号を過大評価しないための信頼度。

    n=1付近では低く、
    nが増えるほど上昇。
    """

    if n <= 0:
        return 0.0

    score = 100.0 * (
        1.0 - math.exp(-n / 12.0)
    )

    return min(score, 100.0)


# ============================================================
# 0～100へ変換
# ============================================================

def normalize(value, low, high):

    if pd.isna(value):
        return 50.0

    if high <= low:
        return 50.0

    score = (
        (value - low)
        / (high - low)
        * 100.0
    )

    return max(0.0, min(100.0, score))


# ============================================================
# 機種スコア
# ============================================================

def calc_machine_scores(history):

    if history.empty:
        return pd.DataFrame()

    results = []

    for machine_name, group in history.groupby("機種", sort=False):

        group = group.sort_values("日付")

        n = len(group)

        mean_diff = group["差枚"].mean()

        plus_rate = (
            (group["差枚"] > 0).mean()
            * 100
        )

        plus1000_rate = (
            (group["差枚"] >= 1000).mean()
            * 100
        )

        plus2000_rate = (
            (group["差枚"] >= 2000).mean()
            * 100
        )

        # ----------------------------------------------------
        # 直近3日
        # ----------------------------------------------------

        recent_dates = (
            group["日付"]
            .drop_duplicates()
            .sort_values()
            .tail(3)
        )

        recent3 = group[
            group["日付"].isin(recent_dates)
        ]

        recent3_mean = (
            recent3["差枚"].mean()
            if len(recent3)
            else 0
        )

        # ----------------------------------------------------
        # 前回営業日
        # ----------------------------------------------------

        last_date = (
            group["日付"]
            .drop_duplicates()
            .sort_values()
            .iloc[-1]
        )

        previous = group[
            group["日付"] == last_date
        ]

        previous_mean = (
            previous["差枚"].mean()
            if len(previous)
            else 0
        )

        # ----------------------------------------------------
        # 直近と全期間の変化
        # ----------------------------------------------------

        change = recent3_mean - mean_diff

        # ----------------------------------------------------
        # 凹み
        # ----------------------------------------------------

        drawdown = (
            recent3_mean - previous_mean
        )

        # ----------------------------------------------------
        # トレンド
        # ----------------------------------------------------

        if len(recent_dates) >= 2:

            daily_means = (
                group
                .groupby("日付")["差枚"]
                .mean()
                .reindex(recent_dates)
            )

            x = np.arange(len(daily_means))

            try:
                trend = np.polyfit(
                    x,
                    daily_means.values,
                    1
                )[0]
            except Exception:
                trend = 0.0

        else:
            trend = 0.0

        # ----------------------------------------------------
        # 信頼度
        # ----------------------------------------------------

        reliability = reliability_score(n)

        results.append(
            {
                "機種": machine_name,
                "台日数": n,
                "平均差枚": mean_diff,
                "プラス率": plus_rate,
                "+1000率": plus1000_rate,
                "+2000率": plus2000_rate,
                "直近3日平均": recent3_mean,
                "前回平均": previous_mean,
                "直近変化": change,
                "凹み": drawdown,
                "トレンド": trend,
                "信頼度": reliability
            }
        )

    result = pd.DataFrame(results)

    if result.empty:
        return result

    # --------------------------------------------------------
    # 各指標を0～100へ正規化
    # --------------------------------------------------------

    result["平均スコア"] = result["平均差枚"].apply(
        lambda x: normalize(
            x,
            result["平均差枚"].quantile(0.05),
            result["平均差枚"].quantile(0.95)
        )
    )

    result["直近スコア"] = result["直近3日平均"].apply(
        lambda x: normalize(
            x,
            result["直近3日平均"].quantile(0.05),
            result["直近3日平均"].quantile(0.95)
        )
    )

    result["凹みスコア"] = result["凹み"].apply(
        lambda x: normalize(
            -x,
            (-result["凹み"]).quantile(0.05),
            (-result["凹み"]).quantile(0.95)
        )
    )

    # --------------------------------------------------------
    # V4機種スコア
    #
    # 長期実績        25%
    # プラス率        15%
    # +1000率         10%
    # +2000率          5%
    # 直近3日         15%
    # 凹み/反発        10%
    # トレンド         5%
    # 信頼度           15%
    # --------------------------------------------------------

    result["V4機種スコア"] = (
        result["平均スコア"] * 0.25
        + result["プラス率"] * 0.15
        + result["+1000率"] * 0.10
        + result["+2000率"] * 0.05
        + result["直近スコア"] * 0.15
        + result["凹みスコア"] * 0.10
        + result["信頼度"] * 0.15
        + 50.0 * 0.05
    )

    # --------------------------------------------------------
    # ランク
    # --------------------------------------------------------

    def rank(score):

        if score >= 80:
            return "S"

        if score >= 70:
            return "A"

        if score >= 60:
            return "B"

        if score >= 50:
            return "C"

        if score >= 40:
            return "D"

        return "E"

    result["ランク"] = result["V4機種スコア"].apply(rank)

    return result.sort_values(
        "V4機種スコア",
        ascending=False
    ).reset_index(drop=True)


# ============================================================
# 台番号スコア
# ============================================================

def calc_number_scores(history):

    if history.empty:
        return pd.DataFrame()

    results = []

    for machine_number, group in history.groupby(
        "台番号",
        sort=False
    ):

        group = group.sort_values("日付")

        n = len(group)

        mean_diff = group["差枚"].mean()

        plus_rate = (
            (group["差枚"] > 0).mean()
            * 100
        )

        plus1000_rate = (
            (group["差枚"] >= 1000).mean()
            * 100
        )

        plus2000_rate = (
            (group["差枚"] >= 2000).mean()
            * 100
        )

        # ----------------------------------------------------
        # 機種名
        # ----------------------------------------------------

        latest_machine = (
            group
            .sort_values("日付")
            .iloc[-1]["機種"]
        )

        # ----------------------------------------------------
        # 直近3日
        # ----------------------------------------------------

        recent_dates = (
            group["日付"]
            .drop_duplicates()
            .sort_values()
            .tail(3)
        )

        recent3 = group[
            group["日付"].isin(recent_dates)
        ]

        recent3_mean = (
            recent3["差枚"].mean()
            if len(recent3)
            else 0
        )

        # ----------------------------------------------------
        # 前回
        # ----------------------------------------------------

        last_date = (
            group["日付"]
            .drop_duplicates()
            .sort_values()
            .iloc[-1]
        )

        previous = group[
            group["日付"] == last_date
        ]

        previous_mean = (
            previous["差枚"].mean()
            if len(previous)
            else 0
        )

        # ----------------------------------------------------
        # 凹み
        # ----------------------------------------------------

        drawdown = (
            recent3_mean - previous_mean
        )

        # ----------------------------------------------------
        # 直近変化
        # ----------------------------------------------------

        recent_change = (
            recent3_mean - mean_diff
        )

        reliability = reliability_score(n)

        results.append(
            {
                "台番号": int(machine_number),
                "機種": latest_machine,
                "台日数": n,
                "平均差枚": mean_diff,
                "プラス率": plus_rate,
                "+1000率": plus1000_rate,
                "+2000率": plus2000_rate,
                "直近3日平均": recent3_mean,
                "前回差枚": previous_mean,
                "直近変化": recent_change,
                "凹み": drawdown,
                "信頼度": reliability
            }
        )

    result = pd.DataFrame(results)

    if result.empty:
        return result

    # --------------------------------------------------------
    # 正規化
    # --------------------------------------------------------

    result["平均スコア"] = result["平均差枚"].apply(
        lambda x: normalize(
            x,
            result["平均差枚"].quantile(0.05),
            result["平均差枚"].quantile(0.95)
        )
    )

    result["直近スコア"] = result["直近3日平均"].apply(
        lambda x: normalize(
            x,
            result["直近3日平均"].quantile(0.05),
            result["直近3日平均"].quantile(0.95)
        )
    )

    result["凹みスコア"] = result["凹み"].apply(
        lambda x: normalize(
            -x,
            (-result["凹み"]).quantile(0.05),
            (-result["凹み"]).quantile(0.95)
        )
    )

    # --------------------------------------------------------
    # 台番号V4スコア
    #
    # 長期実績       25%
    # プラス率       10%
    # +1000率        10%
    # +2000率         5%
    # 直近3日        15%
    # 凹み            15%
    # 信頼度          20%
    # --------------------------------------------------------

    result["V4台番号スコア"] = (
        result["平均スコア"] * 0.25
        + result["プラス率"] * 0.10
        + result["+1000率"] * 0.10
        + result["+2000率"] * 0.05
        + result["直近スコア"] * 0.15
        + result["凹みスコア"] * 0.15
        + result["信頼度"] * 0.20
    )

    return result.sort_values(
        "V4台番号スコア",
        ascending=False
    ).reset_index(drop=True)


# ============================================================
# 機種＋台番号 複合スコア
# ============================================================

def combine_scores(machine_scores, number_scores):

    if machine_scores.empty or number_scores.empty:
        return pd.DataFrame()

    result = number_scores.merge(
        machine_scores[
            [
                "機種",
                "V4機種スコア",
                "ランク"
            ]
        ],
        on="機種",
        how="left"
    )

    result["V4機種スコア"] = (
        result["V4機種スコア"]
        .fillna(50)
    )

    # --------------------------------------------------------
    # 最終V4
    #
    # 機種傾向 45%
    # 台番号傾向 55%
    # --------------------------------------------------------

    result["V4総合スコア"] = (
        result["V4機種スコア"] * 0.45
        + result["V4台番号スコア"] * 0.55
    )

    def rank(score):

        if score >= 80:
            return "S"

        if score >= 70:
            return "A"

        if score >= 60:
            return "B"

        if score >= 50:
            return "C"

        if score >= 40:
            return "D"

        return "E"

    result["総合ランク"] = (
        result["V4総合スコア"]
        .apply(rank)
    )

    return result.sort_values(
        "V4総合スコア",
        ascending=False
    ).reset_index(drop=True)


# ============================================================
# 対象日の実績
# ============================================================

def calc_actual_machine(actual_df):

    if actual_df.empty:
        return pd.DataFrame()

    result = (
        actual_df
        .groupby("機種")
        .agg(
            実績平均差枚=("差枚", "mean"),
            実績プラス率=(
                "差枚",
                lambda x: (x > 0).mean() * 100
            ),
            実績台数=("差枚", "count"),
            実績_1000枚以上=(
                "差枚",
                lambda x: (x >= 1000).sum()
            ),
            実績_2000枚以上=(
                "差枚",
                lambda x: (x >= 2000).sum()
            )
        )
        .reset_index()
    )

    result["実績+1000率"] = (
        result["実績_1000枚以上"]
        / result["実績台数"]
        * 100
    )

    result["実績+2000率"] = (
        result["実績_2000枚以上"]
        / result["実績台数"]
        * 100
    )

    return result


# ============================================================
# 推奨機種の実績
# ============================================================

def evaluate_machine_recommendation(
    ranking,
    actual_df,
    top_n
):

    top = ranking.head(top_n).copy()

    if top.empty:
        return None

    actual = calc_actual_machine(
        actual_df
    )

    merged = top.merge(
        actual,
        on="機種",
        how="left"
    )

    merged["実績平均差枚"] = (
        merged["実績平均差枚"]
        .fillna(0)
    )

    merged["実績プラス率"] = (
        merged["実績プラス率"]
        .fillna(0)
    )

    merged["実績+1000率"] = (
        merged["実績+1000率"]
        .fillna(0)
    )

    merged["実績+2000率"] = (
        merged["実績+2000率"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # 機種勝率
    #
    # 推奨した機種のうち、
    # 当日平均差枚がプラスだった機種の割合
    # --------------------------------------------------------

    machine_win_rate = (
        (
            merged["実績平均差枚"] > 0
        ).mean()
        * 100
    )

    return {
        "TOP": top_n,
        "実績平均差枚": merged["実績平均差枚"].mean(),
        "平均プラス率": merged["実績プラス率"].mean(),
        "+1000率": merged["実績+1000率"].mean(),
        "+2000率": merged["実績+2000率"].mean(),
        "機種勝率": machine_win_rate
    }


# ============================================================
# メイン
# ============================================================

def main():

    print_header(
        "投入パターン解析 V4 バックテスト"
    )

    df = load_data()

    all_dates = (
        df["日付"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    print()
    print(f"解析日数: {len(all_dates)}日")

    if len(all_dates) < 4:
        print("ERROR: バックテストに必要な日数が不足しています。")
        return

    print(
        "解析期間: "
        f"{all_dates[0].strftime('%Y-%m-%d')}"
        " ～ "
        f"{all_dates[-1].strftime('%Y-%m-%d')}"
    )

    # --------------------------------------------------------
    # 最低3日分の履歴を確保
    # --------------------------------------------------------

    target_dates = all_dates[3:]

    all_results = []

    summary_results = []

    # ========================================================
    # 日付ごとにバックテスト
    # ========================================================

    for target_date in target_dates:

        history_dates = [
            d for d in all_dates
            if d < target_date
        ]

        history = df[
            df["日付"].isin(history_dates)
        ].copy()

        actual = df[
            df["日付"] == target_date
        ].copy()

        if history.empty or actual.empty:
            continue

        print_header(
            f"【V4バックテスト】"
            f"{target_date.strftime('%Y-%m-%d')}"
        )

        print(
            "使用履歴: "
            f"{history_dates[0].strftime('%Y-%m-%d')}"
            " ～ "
            f"{history_dates[-1].strftime('%Y-%m-%d')}"
        )

        # ----------------------------------------------------
        # 機種スコア
        # ----------------------------------------------------

        machine_scores = calc_machine_scores(
            history
        )

        # ----------------------------------------------------
        # 台番号スコア
        # ----------------------------------------------------

        number_scores = calc_number_scores(
            history
        )

        # ----------------------------------------------------
        # 複合
        # ----------------------------------------------------

        ranking = combine_scores(
            machine_scores,
            number_scores
        )

        if ranking.empty:
            continue

        # ----------------------------------------------------
        # 当日の台番号実績
        # ----------------------------------------------------

        actual_number = (
            actual[
                [
                    "台番号",
                    "機種",
                    "差枚"
                ]
            ]
            .groupby(
                ["台番号", "機種"],
                as_index=False
            )
            .agg(
                当日差枚=("差枚", "mean")
            )
        )

        ranking_with_actual = ranking.merge(
            actual_number,
            on=["台番号", "機種"],
            how="left"
        )

        ranking_with_actual["当日差枚"] = (
            ranking_with_actual["当日差枚"]
            .fillna(0)
        )

        # ----------------------------------------------------
        # TOP表示
        # ----------------------------------------------------

        for top_n in [5, 10, 20, 30]:

            evaluation = evaluate_machine_recommendation(
                machine_scores,
                actual,
                top_n
            )

            if evaluation is None:
                continue

            print(
                f"TOP{top_n} / "
                f"実績平均差枚 "
                f"{evaluation['実績平均差枚']:+.1f}枚 / "
                f"プラス率 "
                f"{evaluation['平均プラス率']:.1f}% / "
                f"+1000率 "
                f"{evaluation['+1000率']:.1f}% / "
                f"+2000率 "
                f"{evaluation['+2000率']:.1f}% / "
                f"機種勝率 "
                f"{evaluation['機種勝率']:.1f}%"
            )

            summary_results.append(
                {
                    "対象日": target_date.strftime(
                        "%Y-%m-%d"
                    ),
                    "TOP": top_n,
                    "実績平均差枚":
                        evaluation["実績平均差枚"],
                    "平均プラス率":
                        evaluation["平均プラス率"],
                    "+1000率":
                        evaluation["+1000率"],
                    "+2000率":
                        evaluation["+2000率"],
                    "機種勝率":
                        evaluation["機種勝率"]
                }
            )

        # ----------------------------------------------------
        # 台番号TOP30を保存
        # ----------------------------------------------------

        top30 = ranking_with_actual.head(30).copy()

        top30.insert(
            0,
            "対象日",
            target_date.strftime("%Y-%m-%d")
        )

        top30["予測順位"] = (
            np.arange(len(top30)) + 1
        )

        all_results.append(
            top30
        )

    # ========================================================
    # 総合結果
    # ========================================================

    print_header(
        "【V4バックテスト総合結果】"
    )

    summary_df = pd.DataFrame(
        summary_results
    )

    if summary_df.empty:
        print("バックテスト結果がありません。")
        return

    total_summary = []

    for top_n in [5, 10, 20, 30]:

        sub = summary_df[
            summary_df["TOP"] == top_n
        ]

        if sub.empty:
            continue

        result = {
            "TOP": top_n,
            "平均差枚": sub["実績平均差枚"].mean(),
            "プラス率": sub["平均プラス率"].mean(),
            "+1000率": sub["+1000率"].mean(),
            "+2000率": sub["+2000率"].mean(),
            "機種勝率": sub["機種勝率"].mean()
        }

        total_summary.append(
            result
        )

        print()
        print(
            f"TOP{top_n} / "
            f"平均差枚 "
            f"{result['平均差枚']:+.1f}枚 / "
            f"プラス率 "
            f"{result['プラス率']:.1f}% / "
            f"+1000率 "
            f"{result['+1000率']:.1f}% / "
            f"+2000率 "
            f"{result['+2000率']:.1f}% / "
            f"機種勝率 "
            f"{result['機種勝率']:.1f}%"
        )

    # ========================================================
    # ベストTOP
    # ========================================================

    if total_summary:

        best = max(
            total_summary,
            key=lambda x: x["平均差枚"]
        )

        print_header(
            "【V4 ベストTOP】"
        )

        print(
            f"TOP{best['TOP']}"
        )

        print(
            f"平均差枚: "
            f"{best['平均差枚']:+.1f}枚"
        )

        print(
            f"プラス率: "
            f"{best['プラス率']:.1f}%"
        )

        print(
            f"+1000率: "
            f"{best['+1000率']:.1f}%"
        )

        print(
            f"+2000率: "
            f"{best['+2000率']:.1f}%"
        )

        print(
            f"機種勝率: "
            f"{best['機種勝率']:.1f}%"
        )

    # ========================================================
    # CSV保存
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    if all_results:

        result_df = pd.concat(
            all_results,
            ignore_index=True
        )

        result_df.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print()
        print("★ CSV保存成功")
        print(OUTPUT_FILE)

    summary_output = pd.DataFrame(
        total_summary
    )

    summary_output.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("★ CSV保存成功")
    print(SUMMARY_FILE)

    print()
    print("=" * 70)
    print(
        "★★★★★ 投入パターン V4 "
        "バックテスト完了 ★★★★★"
    )
    print("=" * 70)

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