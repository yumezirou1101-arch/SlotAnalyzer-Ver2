# -*- coding: utf-8 -*-

"""
投入パターン予測 V5.1 バックテスト

V5.1モデルを過去データに対して再計算し、
翌日の実績差枚と比較する。

重要:
・予測対象日の実績は特徴量計算に使用しない
・予測対象日は、その前日までのデータだけで予測する
・最新営業日に存在する台を候補とする
・TOP5 / TOP10 / TOP20 / TOP30を評価する
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "maruhan_maebashi"
    / "all_data.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
)

OUTPUT_FILE = OUTPUT_DIR / "prediction_v5_1_backtest.csv"
SUMMARY_FILE = OUTPUT_DIR / "prediction_v5_1_backtest_summary.csv"

TOP_LIST = [5, 10, 20, 30]


# ============================================================
# 表示
# ============================================================

def print_line():
    print("=" * 70)


# ============================================================
# 数値変換
# ============================================================

def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


# ============================================================
# データ読み込み
# ============================================================

def load_data():

    print("入力ファイル:")
    print(DATA_FILE)
    print()

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"入力ファイルがありません:\n{DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

    print(f"読み込みデータ: {len(df):,}行")
    print()

    required = ["日付", "台番号", "機種名", "差枚"]

    print("必要な列を確認します...")

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"必要な列がありません: {col}"
            )

        print(f"{col:<4}: OK")

    print()

    df["日付"] = pd.to_datetime(
        df["日付"],
        errors="coerce"
    )

    df["台番号"] = to_numeric(df["台番号"])

    df["差枚"] = to_numeric(df["差枚"])

    df["機種名"] = (
        df["機種名"]
        .astype(str)
        .str.strip()
    )

    df = df.dropna(
        subset=["日付", "台番号", "機種名", "差枚"]
    ).copy()

    df["台番号"] = df["台番号"].astype(int)

    df = df.sort_values(
        ["日付", "台番号"]
    ).reset_index(drop=True)

    print(f"有効データ: {len(df):,}行")

    return df


# ============================================================
# 過去実績から特徴量を作成
# ============================================================

def build_features(history, target_date):

    if history.empty:
        return pd.DataFrame()

    latest_date = history["日付"].max()

    # --------------------------------------------------------
    # 最新営業日に存在する台だけを候補にする
    # --------------------------------------------------------

    latest = history[
        history["日付"] == latest_date
    ].copy()

    if latest.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # 台番号別特徴量
    # --------------------------------------------------------

    machine_rows = []

    for machine_no, g in history.groupby("台番号"):

        g = g.sort_values("日付")

        if g.empty:
            continue

        latest_row = g.iloc[-1]

        # 過去平均
        past_mean = g["差枚"].mean()

        # 過去プラス率
        past_plus_rate = (
            (g["差枚"] > 0).mean() * 100
        )

        # +1000率
        past_1000_rate = (
            (g["差枚"] >= 1000).mean() * 100
        )

        # +2000率
        past_2000_rate = (
            (g["差枚"] >= 2000).mean() * 100
        )

        # 前日差枚
        previous_diff = latest_row["差枚"]

        # 直近3日平均
        recent3 = g.tail(3)

        recent3_mean = recent3["差枚"].mean()

        # 前回変化
        if len(g) >= 2:
            previous_change = (
                g.iloc[-1]["差枚"]
                - g.iloc[-2]["差枚"]
            )
        else:
            previous_change = 0.0

        # 凹み
        # 過去平均に対する直近値のマイナス幅
        depression = (
            past_mean - recent3_mean
        )

        machine_rows.append({
            "台番号": int(machine_no),
            "台_過去平均差枚": past_mean,
            "台_過去プラス率": past_plus_rate,
            "台_過去+1000率": past_1000_rate,
            "台_過去+2000率": past_2000_rate,
            "台_前日差枚": previous_diff,
            "台_直近3日平均": recent3_mean,
            "台_前回変化": previous_change,
            "台_凹み": depression,
        })

    machine_features = pd.DataFrame(machine_rows)

    if machine_features.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # 最新営業日の機種を紐付け
    # --------------------------------------------------------

    latest_machine = latest[
        ["台番号", "機種名"]
    ].drop_duplicates(
        subset=["台番号"]
    )

    machine_features = machine_features.merge(
        latest_machine,
        on="台番号",
        how="inner"
    )

    # --------------------------------------------------------
    # 機種別特徴量
    # --------------------------------------------------------

    type_rows = []

    for type_name, g in history.groupby("機種名"):

        g = g.sort_values("日付")

        if g.empty:
            continue

        type_past_mean = g["差枚"].mean()

        type_plus_rate = (
            (g["差枚"] > 0).mean() * 100
        )

        type_2000_rate = (
            (g["差枚"] >= 2000).mean() * 100
        )

        recent3_dates = (
            g["日付"]
            .drop_duplicates()
            .sort_values()
            .tail(3)
        )

        recent3_data = g[
            g["日付"].isin(recent3_dates)
        ]

        type_recent3_mean = (
            recent3_data["差枚"].mean()
            if not recent3_data.empty
            else type_past_mean
        )

        # 機種前日差枚
        type_latest = g[
            g["日付"] == latest_date
        ]

        if not type_latest.empty:
            type_previous_diff = (
                type_latest["差枚"].mean()
            )
        else:
            type_previous_diff = type_past_mean

        # 機種前回変化
        unique_dates = sorted(
            g["日付"].drop_duplicates()
        )

        if len(unique_dates) >= 2:

            d1 = unique_dates[-2]
            d2 = unique_dates[-1]

            m1 = g[
                g["日付"] == d1
            ]["差枚"].mean()

            m2 = g[
                g["日付"] == d2
            ]["差枚"].mean()

            type_change = m2 - m1

        else:
            type_change = 0.0

        # 機種凹み
        type_depression = (
            type_past_mean
            - type_recent3_mean
        )

        type_rows.append({
            "機種名": type_name,
            "機種_過去平均差枚": type_past_mean,
            "機種_過去プラス率": type_plus_rate,
            "機種_過去+2000率": type_2000_rate,
            "機種_直近3日平均": type_recent3_mean,
            "機種_前日差枚": type_previous_diff,
            "機種_前回変化": type_change,
            "機種_凹み": type_depression,
        })

    type_features = pd.DataFrame(type_rows)

    if type_features.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # 台＋機種特徴量
    # --------------------------------------------------------

    result = machine_features.merge(
        type_features,
        on="機種名",
        how="left"
    )

    return result


# ============================================================
# 0～100へ正規化
# ============================================================

def rank_score(series, higher_is_better=True):

    s = pd.to_numeric(
        series,
        errors="coerce"
    )

    if s.notna().sum() <= 1:
        return pd.Series(
            50.0,
            index=series.index
        )

    ranks = s.rank(
        method="average",
        pct=True
    )

    if higher_is_better:
        return ranks * 100.0

    return (1.0 - ranks) * 100.0


# ============================================================
# V5.1スコア計算
# ============================================================

def calculate_v51_score(df):

    result = df.copy()

    # --------------------------------------------------------
    # 各要因を0～100へ変換
    # --------------------------------------------------------

    result["S_台前回変化"] = rank_score(
        result["台_前回変化"]
    )

    result["S_台凹み"] = rank_score(
        result["台_凹み"]
    )

    result["S_機種前日差枚"] = rank_score(
        result["機種_前日差枚"]
    )

    result["S_機種直近3日"] = rank_score(
        result["機種_直近3日平均"]
    )

    result["S_台前日差枚"] = rank_score(
        result["台_前日差枚"]
    )

    result["S_機種凹み"] = rank_score(
        result["機種_凹み"]
    )

    result["S_機種過去平均"] = rank_score(
        result["機種_過去平均差枚"]
    )

    result["S_台過去プラス率"] = rank_score(
        result["台_過去プラス率"]
    )

    result["S_台直近3日"] = rank_score(
        result["台_直近3日平均"]
    )

    result["S_機種過去プラス率"] = rank_score(
        result["機種_過去プラス率"]
    )

    # --------------------------------------------------------
    # V5.1 重み
    #
    # 要因分析結果を反映
    # --------------------------------------------------------

    score = (
        result["S_台前回変化"] * 0.20
        + result["S_台凹み"] * 0.15
        + result["S_機種前日差枚"] * 0.15
        + result["S_機種直近3日"] * 0.10
        + result["S_台前日差枚"] * 0.10
        + result["S_機種凹み"] * 0.08
        + result["S_機種過去平均"] * 0.08
        + result["S_台過去プラス率"] * 0.05
        + result["S_台直近3日"] * 0.04
        + result["S_機種過去プラス率"] * 0.03
        + rank_score(result["機種_前回変化"]) * 0.02
    )

    result["V5.1スコア"] = score

    # --------------------------------------------------------
    # ランク
    # --------------------------------------------------------

    def get_rank(x):

        if x >= 75:
            return "S"

        if x >= 65:
            return "A"

        if x >= 55:
            return "B"

        if x >= 45:
            return "C"

        if x >= 35:
            return "D"

        return "E"

    result["ランク"] = result[
        "V5.1スコア"
    ].apply(get_rank)

    # --------------------------------------------------------
    # 予測順位
    # --------------------------------------------------------

    result = result.sort_values(
        "V5.1スコア",
        ascending=False
    ).reset_index(drop=True)

    result["予測順位"] = (
        np.arange(len(result)) + 1
    )

    return result


# ============================================================
# 当日実績を結合
# ============================================================

def attach_actual(
    prediction,
    actual,
    target_date
):

    actual_day = actual[
        actual["日付"] == target_date
    ][
        ["台番号", "差枚"]
    ].copy()

    actual_day = actual_day.rename(
        columns={
            "差枚": "当日差枚"
        }
    )

    result = prediction.merge(
        actual_day,
        on="台番号",
        how="left"
    )

    result = result.dropna(
        subset=["当日差枚"]
    ).copy()

    return result


# ============================================================
# TOP評価
# ============================================================

def evaluate_top(df, top_n):

    if df.empty:
        return None

    selected = (
        df.sort_values(
            "予測順位"
        )
        .head(top_n)
        .copy()
    )

    if selected.empty:
        return None

    actual = selected["当日差枚"]

    avg_diff = actual.mean()

    plus_rate = (
        (actual > 0).mean() * 100
    )

    plus500_rate = (
        (actual >= 500).mean() * 100
    )

    plus1000_rate = (
        (actual >= 1000).mean() * 100
    )

    plus2000_rate = (
        (actual >= 2000).mean() * 100
    )

    plus3000_rate = (
        (actual >= 3000).mean() * 100
    )

    # 機種勝率
    selected = selected.copy()

    type_result = (
        selected.groupby("機種名")[
            "当日差枚"
        ]
        .mean()
    )

    if len(type_result) > 0:

        type_win_rate = (
            (type_result > 0).mean()
            * 100
        )

    else:
        type_win_rate = np.nan

    return {
        "平均差枚": avg_diff,
        "プラス率": plus_rate,
        "+500率": plus500_rate,
        "+1000率": plus1000_rate,
        "+2000率": plus2000_rate,
        "+3000率": plus3000_rate,
        "機種勝率": type_win_rate,
        "選択台数": len(selected),
    }


# ============================================================
# バックテスト
# ============================================================

def run_backtest(df):

    dates = sorted(
        df["日付"].drop_duplicates()
    )

    print()
    print(f"解析日数: {len(dates)}日")

    # 最低1日前が必要
    if len(dates) < 2:
        raise ValueError(
            "バックテストに必要な営業日数がありません。"
        )

    results = []

    # --------------------------------------------------------
    # 各営業日を予測対象にする
    # --------------------------------------------------------

    for i in range(1, len(dates)):

        target_date = dates[i]

        history_dates = dates[:i]

        history = df[
            df["日付"].isin(history_dates)
        ].copy()

        print()
        print_line()
        print(
            f"【V5.1バックテスト】"
            f"{target_date.strftime('%Y-%m-%d')}"
        )

        print(
            f"使用履歴: "
            f"{history_dates[0].strftime('%Y-%m-%d')}"
            f" ～ "
            f"{history_dates[-1].strftime('%Y-%m-%d')}"
        )

        # ----------------------------------------------------
        # 特徴量
        # ----------------------------------------------------

        features = build_features(
            history,
            target_date
        )

        if features.empty:
            print("特徴量を作成できませんでした。")
            continue

        # ----------------------------------------------------
        # V5.1スコア
        # ----------------------------------------------------

        prediction = calculate_v51_score(
            features
        )

        # ----------------------------------------------------
        # 当日実績
        # ----------------------------------------------------

        evaluated = attach_actual(
            prediction,
            df,
            target_date
        )

        if evaluated.empty:
            print("当日実績がありません。")
            continue

        # ----------------------------------------------------
        # TOP評価
        # ----------------------------------------------------

        for top_n in TOP_LIST:

            metric = evaluate_top(
                evaluated,
                top_n
            )

            if metric is None:
                continue

            print(
                f"TOP{top_n} / "
                f"実績平均差枚 "
                f"{metric['平均差枚']:+.1f}枚 / "
                f"プラス率 "
                f"{metric['プラス率']:.1f}% / "
                f"+500率 "
                f"{metric['+500率']:.1f}% / "
                f"+1000率 "
                f"{metric['+1000率']:.1f}% / "
                f"+2000率 "
                f"{metric['+2000率']:.1f}% / "
                f"+3000率 "
                f"{metric['+3000率']:.1f}% / "
                f"機種勝率 "
                f"{metric['機種勝率']:.1f}%"
            )

            results.append({
                "予測日": target_date.strftime(
                    "%Y-%m-%d"
                ),
                "TOP": top_n,
                "選択台数": metric["選択台数"],
                "実績平均差枚": metric["平均差枚"],
                "実績プラス率": metric["プラス率"],
                "+500率": metric["+500率"],
                "+1000率": metric["+1000率"],
                "+2000率": metric["+2000率"],
                "+3000率": metric["+3000率"],
                "機種勝率": metric["機種勝率"],
            })

    return pd.DataFrame(results)


# ============================================================
# 総合集計
# ============================================================

def make_summary(results):

    summary_rows = []

    for top_n in TOP_LIST:

        g = results[
            results["TOP"] == top_n
        ].copy()

        if g.empty:
            continue

        summary_rows.append({
            "TOP": top_n,
            "評価日数": len(g),
            "平均差枚": g["実績平均差枚"].mean(),
            "プラス率": g["実績プラス率"].mean(),
            "+500率": g["+500率"].mean(),
            "+1000率": g["+1000率"].mean(),
            "+2000率": g["+2000率"].mean(),
            "+3000率": g["+3000率"].mean(),
            "機種勝率": g["機種勝率"].mean(),
        })

    return pd.DataFrame(summary_rows)


# ============================================================
# ベストTOP判定
# ============================================================

def choose_best(summary):

    if summary.empty:
        return None

    # 平均差枚を最優先
    best = summary.sort_values(
        [
            "平均差枚",
            "プラス率",
            "+500率"
        ],
        ascending=False
    ).iloc[0]

    return best


# ============================================================
# メイン
# ============================================================

def main():

    print_line()
    print(
        "投入パターン予測 V5.1 バックテスト"
    )
    print_line()

    print(
        "V5.1を過去データに対して再計算し、"
    )
    print(
        "実際の翌日差枚と比較します。"
    )
    print()

    print(
        "重要:"
    )
    print(
        "予測対象日の実績は特徴量計算に使用しません。"
    )
    print()

    # --------------------------------------------------------
    # データ
    # --------------------------------------------------------

    df = load_data()

    dates = sorted(
        df["日付"].drop_duplicates()
    )

    print()
    print(
        f"解析日数: {len(dates)}日"
    )

    if len(dates) >= 2:
        print(
            f"バックテスト日数: "
            f"{len(dates) - 1}日"
        )

    # --------------------------------------------------------
    # バックテスト
    # --------------------------------------------------------

    results = run_backtest(df)

    if results.empty:

        print()
        print_line()
        print(
            "バックテスト結果がありません。"
        )
        print_line()

        return

    # --------------------------------------------------------
    # 総合結果
    # --------------------------------------------------------

    summary = make_summary(
        results
    )

    print()
    print_line()
    print(
        "【V5.1バックテスト総合結果】"
    )
    print_line()

    for _, row in summary.iterrows():

        print()

        print(
            f"TOP{int(row['TOP'])} / "
            f"評価日数 {int(row['評価日数'])}日 / "
            f"平均差枚 "
            f"{row['平均差枚']:+.1f}枚 / "
            f"プラス率 "
            f"{row['プラス率']:.1f}% / "
            f"+500率 "
            f"{row['+500率']:.1f}% / "
            f"+1000率 "
            f"{row['+1000率']:.1f}% / "
            f"+2000率 "
            f"{row['+2000率']:.1f}% / "
            f"+3000率 "
            f"{row['+3000率']:.1f}% / "
            f"機種勝率 "
            f"{row['機種勝率']:.1f}%"
        )

    # --------------------------------------------------------
    # ベストTOP
    # --------------------------------------------------------

    best = choose_best(
        summary
    )

    print()
    print_line()
    print(
        "【V5.1 ベストTOP】"
    )
    print_line()

    if best is not None:

        print(
            f"TOP{int(best['TOP'])}"
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
            f"+500率: "
            f"{best['+500率']:.1f}%"
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
            f"+3000率: "
            f"{best['+3000率']:.1f}%"
        )

        print(
            f"機種勝率: "
            f"{best['機種勝率']:.1f}%"
        )

    # --------------------------------------------------------
    # CSV保存
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("★ CSV保存成功")
    print(OUTPUT_FILE)

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("★ CSV保存成功")
    print(SUMMARY_FILE)

    print()
    print_line()
    print(
        "★★★★★ V5.1 バックテスト完了 ★★★★★"
    )
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