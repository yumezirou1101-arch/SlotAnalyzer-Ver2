# -*- coding: utf-8 -*-

"""
======================================================================
V3・V4・V5 モデル完全比較 V1
======================================================================

V3 / V4 / V5 のバックテスト結果を、
可能な限り同一条件で比較する。

比較項目:
・TOP5
・TOP10
・TOP20
・TOP30

評価項目:
・平均差枚
・プラス率
・+500率
・+1000率
・+2000率
・+3000率
・機種勝率

さらに、
・共通予測日のみの公平比較
・モデル別総合スコア
・最良TOP
・最良モデル
を算出する。

重要:
元のバックテストCSVは変更しない。
======================================================================
"""

import os
import math
import pandas as pd
import numpy as np


# ======================================================================
# 設定
# ======================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "maruhan_maebashi",
    "machine_number"
)

V3_FILE = os.path.join(
    DATA_DIR,
    "investment_pattern_v3_backtest.csv"
)

V4_FILE = os.path.join(
    DATA_DIR,
    "investment_pattern_v4_backtest.csv"
)

V5_FILE = os.path.join(
    DATA_DIR,
    "prediction_v5_backtest.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "prediction_model_comparison_v1.csv"
)

SUMMARY_FILE = os.path.join(
    DATA_DIR,
    "prediction_model_comparison_v1_summary.csv"
)


TOP_LIST = [5, 10, 20, 30]


# ======================================================================
# 表示
# ======================================================================

def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def fmt(value, digits=1):
    if value is None:
        return "N/A"

    try:
        value = float(value)
    except Exception:
        return "N/A"

    if math.isnan(value):
        return "N/A"

    return f"{value:.{digits}f}"


# ======================================================================
# CSV読み込み
# ======================================================================

def read_csv_file(path, model_name):

    print()
    print(f"{model_name}バックテスト:")
    print(f"ファイル: {path}")

    if not os.path.exists(path):
        print("  → ファイルがありません")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(path, encoding="cp932")
        except Exception as e:
            print(f"  → 読み込み失敗: {e}")
            return pd.DataFrame()

    print(f"読み込み: {len(df)}行")

    return df


# ======================================================================
# 列名の正規化
# ======================================================================

def normalize_columns(df):

    if df.empty:
        return df

    rename_map = {}

    for col in df.columns:

        name = str(col).strip()

        if name in ["対象日", "予測日", "日付"]:
            rename_map[col] = "予測日"

        elif name in ["TOP", "top", "順位"]:
            rename_map[col] = "TOP"

        elif name in ["実績平均差枚", "平均差枚"]:
            rename_map[col] = "平均差枚"

        elif name in ["実績プラス率", "プラス率"]:
            rename_map[col] = "プラス率"

        elif name in ["+500率"]:
            rename_map[col] = "+500率"

        elif name in ["+1000率"]:
            rename_map[col] = "+1000率"

        elif name in ["+2000率"]:
            rename_map[col] = "+2000率"

        elif name in ["+3000率"]:
            rename_map[col] = "+3000率"

        elif name in ["機種勝率"]:
            rename_map[col] = "機種勝率"

        elif name in ["予測順位"]:
            rename_map[col] = "予測順位"

        elif name in ["当日差枚"]:
            rename_map[col] = "当日差枚"

    df = df.rename(columns=rename_map)

    if "予測日" in df.columns:
        df["予測日"] = pd.to_datetime(
            df["予測日"],
            errors="coerce"
        ).dt.date

    return df


# ======================================================================
# TOP別集計
# ======================================================================

def aggregate_detailed(df, model_name):

    """
    詳細CSVからTOP別結果を作る。

    例えばV4/V5のように

        予測日
        予測順位
        当日差枚

    が存在する場合に使用する。
    """

    if df.empty:
        return pd.DataFrame()

    required = [
        "予測日",
        "予測順位",
        "当日差枚"
    ]

    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    work = df.copy()

    work["予測順位"] = pd.to_numeric(
        work["予測順位"],
        errors="coerce"
    )

    work["当日差枚"] = pd.to_numeric(
        work["当日差枚"],
        errors="coerce"
    )

    work = work.dropna(
        subset=[
            "予測日",
            "予測順位",
            "当日差枚"
        ]
    )

    results = []

    for target_date, day_df in work.groupby("予測日"):

        for top in TOP_LIST:

            selected = day_df[
                day_df["予測順位"] <= top
            ].copy()

            if selected.empty:
                continue

            diff = selected["当日差枚"]

            result = {
                "モデル": model_name,
                "予測日": target_date,
                "TOP": top,
                "平均差枚": diff.mean(),
                "プラス率": (diff > 0).mean() * 100,
                "+500率": (diff >= 500).mean() * 100,
                "+1000率": (diff >= 1000).mean() * 100,
                "+2000率": (diff >= 2000).mean() * 100,
                "+3000率": (diff >= 3000).mean() * 100,
            }

            results.append(result)

    return pd.DataFrame(results)


# ======================================================================
# 既にTOP別集計されているCSVの処理
# ======================================================================

def normalize_summary(df, model_name):

    if df.empty:
        return pd.DataFrame()

    df = normalize_columns(df)

    required = [
        "予測日",
        "TOP",
        "平均差枚",
        "プラス率"
    ]

    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    work = df.copy()

    work["TOP"] = pd.to_numeric(
        work["TOP"],
        errors="coerce"
    )

    work["平均差枚"] = pd.to_numeric(
        work["平均差枚"],
        errors="coerce"
    )

    work["プラス率"] = pd.to_numeric(
        work["プラス率"],
        errors="coerce"
    )

    for col in [
        "+500率",
        "+1000率",
        "+2000率",
        "+3000率",
        "機種勝率"
    ]:

        if col not in work.columns:
            work[col] = np.nan

        work[col] = pd.to_numeric(
            work[col],
            errors="coerce"
        )

    work = work[
        work["TOP"].isin(TOP_LIST)
    ].copy()

    work["モデル"] = model_name

    return work[
        [
            "モデル",
            "予測日",
            "TOP",
            "平均差枚",
            "プラス率",
            "+500率",
            "+1000率",
            "+2000率",
            "+3000率",
            "機種勝率"
        ]
    ]


# ======================================================================
# モデルデータを統一
# ======================================================================

def prepare_model(df, model_name):

    if df.empty:
        return pd.DataFrame()

    df = normalize_columns(df)

    # --------------------------------------------------------------
    # まずTOP別集計形式を試す
    # --------------------------------------------------------------

    summary = normalize_summary(
        df,
        model_name
    )

    if not summary.empty:

        # TOP別に複数行存在する場合はそのまま利用
        if len(summary) > 0:
            return summary

    # --------------------------------------------------------------
    # 詳細形式を試す
    # --------------------------------------------------------------

    detailed = aggregate_detailed(
        df,
        model_name
    )

    if not detailed.empty:
        return detailed

    return pd.DataFrame()


# ======================================================================
# 共通日抽出
# ======================================================================

def get_common_dates(*dfs):

    valid_dates = []

    for df in dfs:

        if df.empty or "予測日" not in df.columns:
            return set()

        dates = set(
            df["予測日"]
            .dropna()
            .unique()
        )

        valid_dates.append(dates)

    if not valid_dates:
        return set()

    common = valid_dates[0]

    for dates in valid_dates[1:]:
        common = common.intersection(dates)

    return common


# ======================================================================
# 共通日だけ抽出
# ======================================================================

def filter_common_dates(df, common_dates):

    if df.empty:
        return df.copy()

    return df[
        df["予測日"].isin(common_dates)
    ].copy()


# ======================================================================
# TOP別総合評価
# ======================================================================

def summarize(df, model_name):

    results = []

    for top in TOP_LIST:

        sub = df[
            df["TOP"] == top
        ].copy()

        if sub.empty:
            continue

        # 日ごとの結果を平均
        metrics = [
            "平均差枚",
            "プラス率",
            "+500率",
            "+1000率",
            "+2000率",
            "+3000率",
            "機種勝率"
        ]

        row = {
            "モデル": model_name,
            "TOP": top,
            "評価日数": sub["予測日"].nunique()
        }

        for metric in metrics:

            if metric not in sub.columns:
                row[metric] = np.nan
            else:
                values = pd.to_numeric(
                    sub[metric],
                    errors="coerce"
                )

                row[metric] = values.mean()

        results.append(row)

    return pd.DataFrame(results)


# ======================================================================
# 総合モデルスコア
# ======================================================================

def model_score(summary_df):

    if summary_df.empty:
        return np.nan

    scores = []

    for top in TOP_LIST:

        row = summary_df[
            summary_df["TOP"] == top
        ]

        if row.empty:
            continue

        row = row.iloc[0]

        avg_diff = safe_float(
            row["平均差枚"]
        )

        plus_rate = safe_float(
            row["プラス率"]
        )

        plus500 = safe_float(
            row["+500率"]
        )

        machine_win = safe_float(
            row["機種勝率"]
        )

        # ----------------------------------------------------------
        # スコア
        #
        # 平均差枚を最重要
        # プラス率を次点
        # +500率を補助
        # 機種勝率を補助
        # ----------------------------------------------------------

        if np.isnan(avg_diff):
            continue

        score = (
            avg_diff * 0.45
            + (plus_rate if not np.isnan(plus_rate) else 0) * 2.0
            + (plus500 if not np.isnan(plus500) else 0) * 1.0
            + (machine_win if not np.isnan(machine_win) else 0) * 1.0
        )

        scores.append(
            {
                "TOP": top,
                "score": score
            }
        )

    if not scores:
        return np.nan

    # TOP5を若干重視
    weights = {
        5: 1.30,
        10: 1.10,
        20: 1.00,
        30: 0.90
    }

    weighted_scores = []

    for item in scores:

        weighted_scores.append(
            item["score"]
            * weights.get(
                item["TOP"],
                1.0
            )
        )

    return float(
        np.mean(weighted_scores)
    )


# ======================================================================
# 結果表示
# ======================================================================

def print_summary(title, summary):

    header(title)

    if summary.empty:
        print("評価データがありません。")
        return

    for _, row in summary.iterrows():

        print(
            f"TOP{int(row['TOP'])} / "
            f"評価日数 {int(row['評価日数'])}日 / "
            f"平均差枚 {fmt(row['平均差枚']):>7}枚 / "
            f"プラス率 {fmt(row['プラス率']):>5}% / "
            f"+500率 {fmt(row['+500率']):>5}% / "
            f"+1000率 {fmt(row['+1000率']):>5}% / "
            f"+2000率 {fmt(row['+2000率']):>5}% / "
            f"+3000率 {fmt(row['+3000率']):>5}% / "
            f"機種勝率 {fmt(row['機種勝率']):>5}%"
        )


# ======================================================================
# 比較表示
# ======================================================================

def print_comparison(
    v3_summary,
    v4_summary,
    v5_summary
):

    header("【V3・V4・V5 公平比較】")

    for top in TOP_LIST:

        print()
        print(f"--- TOP{top} ---")

        for model_name, summary in [
            ("V3", v3_summary),
            ("V4", v4_summary),
            ("V5", v5_summary)
        ]:

            row = summary[
                summary["TOP"] == top
            ]

            if row.empty:
                print(
                    f"{model_name}: 評価データなし"
                )
                continue

            row = row.iloc[0]

            print(
                f"{model_name}: "
                f"平均差枚 {fmt(row['平均差枚']):>7}枚 / "
                f"プラス率 {fmt(row['プラス率']):>5}% / "
                f"+500率 {fmt(row['+500率']):>5}% / "
                f"+1000率 {fmt(row['+1000率']):>5}% / "
                f"機種勝率 {fmt(row['機種勝率']):>5}%"
            )


# ======================================================================
# メイン
# ======================================================================

def main():

    header("V3・V4・V5 モデル完全比較 V1")

    print(
        "V3・V4・V5を同じ条件で再評価します。"
    )

    print()
    print("データフォルダ:")
    print(DATA_DIR)

    # --------------------------------------------------------------
    # 読み込み
    # --------------------------------------------------------------

    v3_raw = read_csv_file(
        V3_FILE,
        "V3"
    )

    v4_raw = read_csv_file(
        V4_FILE,
        "V4"
    )

    v5_raw = read_csv_file(
        V5_FILE,
        "V5"
    )

    # --------------------------------------------------------------
    # 統一
    # --------------------------------------------------------------

    v3 = prepare_model(
        v3_raw,
        "V3"
    )

    v4 = prepare_model(
        v4_raw,
        "V4"
    )

    v5 = prepare_model(
        v5_raw,
        "V5"
    )

    print()
    print(
        f"統一後データ件数:"
    )
    print(
        f"V3: {len(v3)}行"
    )
    print(
        f"V4: {len(v4)}行"
    )
    print(
        f"V5: {len(v5)}行"
    )

    # --------------------------------------------------------------
    # 通常評価
    # --------------------------------------------------------------

    v3_summary = summarize(
        v3,
        "V3"
    )

    v4_summary = summarize(
        v4,
        "V4"
    )

    v5_summary = summarize(
        v5,
        "V5"
    )

    print_summary(
        "【V3 総合評価】",
        v3_summary
    )

    print_summary(
        "【V4 総合評価】",
        v4_summary
    )

    print_summary(
        "【V5 総合評価】",
        v5_summary
    )

    # --------------------------------------------------------------
    # 共通日
    # --------------------------------------------------------------

    common_dates = get_common_dates(
        v3,
        v4,
        v5
    )

    header(
        "【V3・V4・V5 共通日による公平比較】"
    )

    if common_dates:

        common_dates = sorted(
            common_dates
        )

        print(
            "共通評価日:"
        )

        print(
            " / ".join(
                str(x)
                for x in common_dates
            )
        )

    else:

        print(
            "V3・V4・V5すべてに共通する"
            "予測日がありません。"
        )

    # --------------------------------------------------------------
    # 共通日データ
    # --------------------------------------------------------------

    v3_common = filter_common_dates(
        v3,
        common_dates
    )

    v4_common = filter_common_dates(
        v4,
        common_dates
    )

    v5_common = filter_common_dates(
        v5,
        common_dates
    )

    v3_common_summary = summarize(
        v3_common,
        "V3"
    )

    v4_common_summary = summarize(
        v4_common,
        "V4"
    )

    v5_common_summary = summarize(
        v5_common,
        "V5"
    )

    print_summary(
        "【V3 共通日評価】",
        v3_common_summary
    )

    print_summary(
        "【V4 共通日評価】",
        v4_common_summary
    )

    print_summary(
        "【V5 共通日評価】",
        v5_common_summary
    )

    # --------------------------------------------------------------
    # 公平比較
    # --------------------------------------------------------------

    print_comparison(
        v3_common_summary,
        v4_common_summary,
        v5_common_summary
    )

    # --------------------------------------------------------------
    # モデルスコア
    # --------------------------------------------------------------

    header(
        "【モデル総合評価】"
    )

    scores = []

    for model_name, summary in [
        ("V3", v3_common_summary),
        ("V4", v4_common_summary),
        ("V5", v5_common_summary)
    ]:

        score = model_score(
            summary
        )

        scores.append(
            {
                "モデル": model_name,
                "総合スコア": score
            }
        )

        if np.isnan(score):

            print(
                f"{model_name} / "
                f"総合評価 N/A"
            )

        else:

            print(
                f"{model_name} / "
                f"総合評価 {score:.1f}"
            )

    # --------------------------------------------------------------
    # ベストモデル
    # --------------------------------------------------------------

    valid_scores = [
        x for x in scores
        if not np.isnan(x["総合スコア"])
    ]

    if valid_scores:

        best = max(
            valid_scores,
            key=lambda x: x["総合スコア"]
        )

        print()
        print(
            f"★ 現時点のベストモデル: "
            f"{best['モデル']}"
        )

    else:

        print()
        print(
            "V3・V4・V5とも評価できませんでした。"
        )

    # --------------------------------------------------------------
    # TOP別ベスト
    # --------------------------------------------------------------

    header(
        "【TOP別ベストモデル】"
    )

    best_rows = []

    for top in TOP_LIST:

        candidates = []

        for model_name, summary in [
            ("V3", v3_common_summary),
            ("V4", v4_common_summary),
            ("V5", v5_common_summary)
        ]:

            row = summary[
                summary["TOP"] == top
            ]

            if row.empty:
                continue

            row = row.iloc[0]

            avg_diff = safe_float(
                row["平均差枚"]
            )

            if np.isnan(avg_diff):
                continue

            candidates.append(
                {
                    "モデル": model_name,
                    "TOP": top,
                    "平均差枚": avg_diff,
                    "プラス率": safe_float(
                        row["プラス率"]
                    ),
                    "+500率": safe_float(
                        row["+500率"]
                    ),
                    "機種勝率": safe_float(
                        row["機種勝率"]
                    )
                }
            )

        if candidates:

            best_top = max(
                candidates,
                key=lambda x: x["平均差枚"]
            )

            best_rows.append(
                best_top
            )

            print(
                f"TOP{top}: "
                f"{best_top['モデル']} / "
                f"平均差枚 "
                f"{best_top['平均差枚']:+.1f}枚 / "
                f"プラス率 "
                f"{fmt(best_top['プラス率'])}%"
            )

    # --------------------------------------------------------------
    # CSV作成
    # --------------------------------------------------------------

    output_rows = []

    for model_name, summary, evaluation_type in [
        (
            "V3",
            v3_summary,
            "全評価日"
        ),
        (
            "V4",
            v4_summary,
            "全評価日"
        ),
        (
            "V5",
            v5_summary,
            "全評価日"
        ),
        (
            "V3",
            v3_common_summary,
            "共通日"
        ),
        (
            "V4",
            v4_common_summary,
            "共通日"
        ),
        (
            "V5",
            v5_common_summary,
            "共通日"
        )
    ]:

        if summary.empty:
            continue

        for _, row in summary.iterrows():

            output_rows.append(
                {
                    "モデル": model_name,
                    "評価種類": evaluation_type,
                    "TOP": int(row["TOP"]),
                    "評価日数": int(
                        row["評価日数"]
                    ),
                    "平均差枚": row["平均差枚"],
                    "プラス率": row["プラス率"],
                    "+500率": row["+500率"],
                    "+1000率": row["+1000率"],
                    "+2000率": row["+2000率"],
                    "+3000率": row["+3000率"],
                    "機種勝率": row["機種勝率"]
                }
            )

    output_df = pd.DataFrame(
        output_rows
    )

    try:

        output_df.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print()
        print(
            "★ CSV保存成功"
        )
        print(
            OUTPUT_FILE
        )

    except Exception as e:

        print()
        print(
            f"CSV保存エラー: {e}"
        )

    # --------------------------------------------------------------
    # サマリーCSV
    # --------------------------------------------------------------

    summary_rows = []

    for item in scores:

        summary_rows.append(
            {
                "モデル": item["モデル"],
                "総合スコア": item["総合スコア"],
                "共通評価日数": len(
                    common_dates
                )
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    try:

        summary_df.to_csv(
            SUMMARY_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print()
        print(
            "★ CSV保存成功"
        )
        print(
            SUMMARY_FILE
        )

    except Exception as e:

        print()
        print(
            f"CSV保存エラー: {e}"
        )

    # --------------------------------------------------------------
    # 完了
    # --------------------------------------------------------------

    header(
        "★★★★★ V3・V4・V5 モデル比較完了 ★★★★★"
    )

    print()
    print(
        "保存ファイル:"
    )
    print(
        OUTPUT_FILE
    )
    print(
        SUMMARY_FILE
    )

    print()
    print(
        "元のバックテストCSVは変更していません。"
    )


# ======================================================================
# 実行
# ======================================================================

if __name__ == "__main__":
    main()