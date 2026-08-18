import os
import sys
import pandas as pd
import numpy as np


# ============================================================
# 設定
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "maruhan_maebashi",
    "machine_number"
)

V5_FILE = os.path.join(
    DATA_DIR,
    "prediction_v5_backtest.csv"
)

V51_FILE = os.path.join(
    DATA_DIR,
    "prediction_v5_1_backtest.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "prediction_v5_v5_1_comparison.csv"
)

SUMMARY_FILE = os.path.join(
    DATA_DIR,
    "prediction_v5_v5_1_comparison_summary.csv"
)

TOPS = [5, 10, 20, 30]


# ============================================================
# 共通関数
# ============================================================

def print_line():
    print("=" * 70)


def safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def format_metric(value, digits=1):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


def find_column(df, candidates):
    """
    CSVの列名の違いに対応する。
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


# ============================================================
# CSV読み込み
# ============================================================

def load_backtest(path, model_name):
    if not os.path.exists(path):
        print()
        print(f"【エラー】{model_name}バックテストCSVがありません。")
        print(path)
        return None

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(path, encoding="cp932")
        except Exception as e:
            print()
            print(f"【エラー】{model_name}CSVを読み込めません。")
            print(e)
            return None

    print()
    print(f"{model_name}バックテスト:")
    print(f"ファイル: {path}")
    print(f"読み込み: {len(df)}行")

    if "対象日" in df.columns:
        date_col = "対象日"
    elif "予測日" in df.columns:
        date_col = "予測日"
    else:
        date_col = None

    if date_col is None:
        print(f"【エラー】{model_name}に日付列がありません。")
        print("列名:", list(df.columns))
        return None

    df = df.copy()

    df["評価日"] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df["評価日"] = df["評価日"].dt.strftime("%Y-%m-%d")

    top_col = find_column(
        df,
        ["TOP", "top", "予測TOP"]
    )

    if top_col is not None:
        df["TOP"] = pd.to_numeric(
            df[top_col],
            errors="coerce"
        )

    else:
        # V5 / V5.1のバックテストでは、
        # TOPごとに複数行ある形式を想定する。
        df["TOP"] = np.nan

    return df


# ============================================================
# TOP別集計
# ============================================================

def summarize_top(df, top):
    if df is None or len(df) == 0:
        return None

    work = df.copy()

    # TOP列が存在する場合
    if "TOP" in work.columns and work["TOP"].notna().any():

        work["TOP"] = pd.to_numeric(
            work["TOP"],
            errors="coerce"
        )

        work = work[work["TOP"] == top].copy()

    else:
        return None

    if len(work) == 0:
        return None

    # 必要な数値列
    numeric_candidates = [
        "実績平均差枚",
        "平均差枚",
        "実績プラス率",
        "プラス率",
        "+500率",
        "+1000率",
        "+2000率",
        "+3000率",
        "機種勝率"
    ]

    for col in numeric_candidates:
        if col in work.columns:
            work[col] = pd.to_numeric(
                work[col],
                errors="coerce"
            )

    result = {}

    result["TOP"] = top

    if "評価日" in work.columns:
        result["評価日数"] = work["評価日"].nunique()
    else:
        result["評価日数"] = np.nan

    # 平均差枚
    diff_col = find_column(
        work,
        ["実績平均差枚", "平均差枚"]
    )

    if diff_col:
        result["平均差枚"] = work[diff_col].mean()
    else:
        result["平均差枚"] = np.nan

    # プラス率
    plus_col = find_column(
        work,
        ["実績プラス率", "プラス率"]
    )

    if plus_col:
        result["プラス率"] = work[plus_col].mean()
    else:
        result["プラス率"] = np.nan

    # 各到達率
    for metric in [
        "+500率",
        "+1000率",
        "+2000率",
        "+3000率",
        "機種勝率"
    ]:
        if metric in work.columns:
            result[metric] = work[metric].mean()
        else:
            result[metric] = np.nan

    return result


# ============================================================
# 共通評価日の抽出
# ============================================================

def get_common_dates(v5, v51):

    if v5 is None or v51 is None:
        return []

    if "評価日" not in v5.columns:
        return []

    if "評価日" not in v51.columns:
        return []

    dates_v5 = set(
        v5["評価日"]
        .dropna()
        .astype(str)
        .unique()
    )

    dates_v51 = set(
        v51["評価日"]
        .dropna()
        .astype(str)
        .unique()
    )

    common = sorted(
        dates_v5.intersection(dates_v51)
    )

    return common


# ============================================================
# 共通日TOP別集計
# ============================================================

def summarize_common_dates(df, dates, top):

    if df is None:
        return None

    if not dates:
        return None

    work = df[
        df["評価日"].isin(dates)
    ].copy()

    return summarize_top(
        work,
        top
    )


# ============================================================
# 差分計算
# ============================================================

def calc_difference(v5_value, v51_value):

    if pd.isna(v5_value) or pd.isna(v51_value):
        return np.nan

    return v51_value - v5_value


# ============================================================
# 総合評価
# ============================================================

def calculate_model_score(summary_list):

    valid = [
        x for x in summary_list
        if x is not None
        and not pd.isna(x.get("平均差枚", np.nan))
    ]

    if not valid:
        return np.nan

    avg_diff = np.mean([
        x["平均差枚"]
        for x in valid
    ])

    plus_rate = np.mean([
        x["プラス率"]
        for x in valid
        if not pd.isna(x["プラス率"])
    ])

    plus500_values = [
        x["+500率"]
        for x in valid
        if not pd.isna(x["+500率"])
    ]

    if plus500_values:
        plus500 = np.mean(plus500_values)
    else:
        plus500 = 0.0

    machine_win_values = [
        x["機種勝率"]
        for x in valid
        if not pd.isna(x["機種勝率"])
    ]

    if machine_win_values:
        machine_win = np.mean(machine_win_values)
    else:
        machine_win = 0.0

    # --------------------------------------------------------
    # 総合評価
    #
    # 平均差枚を中心に、
    # プラス率・+500率・機種勝率を加味する。
    #
    # あくまで比較用の相対スコア。
    # --------------------------------------------------------

    score = 50.0

    # 平均差枚
    score += avg_diff / 20.0

    # プラス率
    score += (plus_rate - 50.0) * 0.5

    # +500率
    score += plus500 * 0.2

    # 機種勝率
    score += (machine_win - 50.0) * 0.2

    return max(0.0, min(100.0, score))


# ============================================================
# メイン
# ============================================================

def main():

    print_line()
    print("V5・V5.1 モデル比較 V1")
    print_line()

    print("V5とV5.1を同じ条件で比較します。")
    print("共通評価日を使用した公平比較も行います。")
    print()

    print("データフォルダ:")
    print(DATA_DIR)

    v5 = load_backtest(
        V5_FILE,
        "V5"
    )

    v51 = load_backtest(
        V51_FILE,
        "V5.1"
    )

    if v5 is None or v51 is None:
        print()
        print("必要なCSVがありません。")
        return

    print()
    print("統一後データ件数:")
    print(f"V5 : {len(v5)}行")
    print(f"V5.1: {len(v51)}行")

    # ========================================================
    # 通常評価
    # ========================================================

    v5_results = {}
    v51_results = {}

    print_line()
    print("【V5 総合評価】")
    print_line()

    for top in TOPS:

        result = summarize_top(
            v5,
            top
        )

        v5_results[top] = result

        if result is None:
            print(
                f"TOP{top} / 評価データなし"
            )
            continue

        print(
            f"TOP{top} / "
            f"評価日数 {result['評価日数']}日 / "
            f"平均差枚 {format_metric(result['平均差枚']):>7}枚 / "
            f"プラス率 {format_metric(result['プラス率']):>5}% / "
            f"+500率 {format_metric(result['+500率']):>5}% / "
            f"+1000率 {format_metric(result['+1000率']):>5}% / "
            f"+2000率 {format_metric(result['+2000率']):>5}% / "
            f"+3000率 {format_metric(result['+3000率']):>5}% / "
            f"機種勝率 {format_metric(result['機種勝率']):>5}%"
        )

    print_line()
    print("【V5.1 総合評価】")
    print_line()

    for top in TOPS:

        result = summarize_top(
            v51,
            top
        )

        v51_results[top] = result

        if result is None:
            print(
                f"TOP{top} / 評価データなし"
            )
            continue

        print(
            f"TOP{top} / "
            f"評価日数 {result['評価日数']}日 / "
            f"平均差枚 {format_metric(result['平均差枚']):>7}枚 / "
            f"プラス率 {format_metric(result['プラス率']):>5}% / "
            f"+500率 {format_metric(result['+500率']):>5}% / "
            f"+1000率 {format_metric(result['+1000率']):>5}% / "
            f"+2000率 {format_metric(result['+2000率']):>5}% / "
            f"+3000率 {format_metric(result['+3000率']):>5}% / "
            f"機種勝率 {format_metric(result['機種勝率']):>5}%"
        )

    # ========================================================
    # 共通日
    # ========================================================

    common_dates = get_common_dates(
        v5,
        v51
    )

    print_line()
    print("【V5・V5.1 共通日による公平比較】")
    print_line()

    if common_dates:

        print(
            "共通評価日:"
        )

        print(
            " / ".join(common_dates)
        )

    else:

        print("共通評価日はありません。")

    common_v5_results = {}
    common_v51_results = {}

    if common_dates:

        print_line()
        print("【V5 共通日評価】")
        print_line()

        for top in TOPS:

            result = summarize_common_dates(
                v5,
                common_dates,
                top
            )

            common_v5_results[top] = result

            if result is None:
                print(
                    f"TOP{top} / 評価データなし"
                )
                continue

            print(
                f"TOP{top} / "
                f"評価日数 {result['評価日数']}日 / "
                f"平均差枚 {format_metric(result['平均差枚']):>7}枚 / "
                f"プラス率 {format_metric(result['プラス率']):>5}% / "
                f"+500率 {format_metric(result['+500率']):>5}% / "
                f"+1000率 {format_metric(result['+1000率']):>5}% / "
                f"+2000率 {format_metric(result['+2000率']):>5}% / "
                f"+3000率 {format_metric(result['+3000率']):>5}% / "
                f"機種勝率 {format_metric(result['機種勝率']):>5}%"
            )

        print_line()
        print("【V5.1 共通日評価】")
        print_line()

        for top in TOPS:

            result = summarize_common_dates(
                v51,
                common_dates,
                top
            )

            common_v51_results[top] = result

            if result is None:
                print(
                    f"TOP{top} / 評価データなし"
                )
                continue

            print(
                f"TOP{top} / "
                f"評価日数 {result['評価日数']}日 / "
                f"平均差枚 {format_metric(result['平均差枚']):>7}枚 / "
                f"プラス率 {format_metric(result['プラス率']):>5}% / "
                f"+500率 {format_metric(result['+500率']):>5}% / "
                f"+1000率 {format_metric(result['+1000率']):>5}% / "
                f"+2000率 {format_metric(result['+2000率']):>5}% / "
                f"+3000率 {format_metric(result['+3000率']):>5}% / "
                f"機種勝率 {format_metric(result['機種勝率']):>5}%"
            )

    # ========================================================
    # V5 → V5.1 改善・悪化
    # ========================================================

    print_line()
    print("【V5 → V5.1 改善・悪化】")
    print_line()

    comparison_rows = []

    for top in TOPS:

        v5_result = common_v5_results.get(top)
        v51_result = common_v51_results.get(top)

        if v5_result is None or v51_result is None:
            print()
            print(f"--- TOP{top} ---")
            print("比較データなし")
            continue

        avg_diff_change = calc_difference(
            v5_result["平均差枚"],
            v51_result["平均差枚"]
        )

        plus_change = calc_difference(
            v5_result["プラス率"],
            v51_result["プラス率"]
        )

        plus500_change = calc_difference(
            v5_result["+500率"],
            v51_result["+500率"]
        )

        plus1000_change = calc_difference(
            v5_result["+1000率"],
            v51_result["+1000率"]
        )

        machine_change = calc_difference(
            v5_result["機種勝率"],
            v51_result["機種勝率"]
        )

        print()
        print(f"--- TOP{top} ---")

        print(
            f"平均差枚: "
            f"{format_metric(v5_result['平均差枚'])} → "
            f"{format_metric(v51_result['平均差枚'])} "
            f"({avg_diff_change:+.1f}枚)"
        )

        print(
            f"プラス率: "
            f"{format_metric(v5_result['プラス率'])}% → "
            f"{format_metric(v51_result['プラス率'])}% "
            f"({plus_change:+.1f}pt)"
        )

        print(
            f"+500率: "
            f"{format_metric(v5_result['+500率'])}% → "
            f"{format_metric(v51_result['+500率'])}% "
            f"({plus500_change:+.1f}pt)"
        )

        print(
            f"+1000率: "
            f"{format_metric(v5_result['+1000率'])}% → "
            f"{format_metric(v51_result['+1000率'])}% "
            f"({plus1000_change:+.1f}pt)"
        )

        print(
            f"機種勝率: "
            f"{format_metric(v5_result['機種勝率'])}% → "
            f"{format_metric(v51_result['機種勝率'])}% "
            f"({machine_change:+.1f}pt)"
        )

        comparison_rows.append({
            "モデル": "V5_vs_V5.1",
            "TOP": top,
            "評価日数": v51_result["評価日数"],
            "V5平均差枚": v5_result["平均差枚"],
            "V5.1平均差枚": v51_result["平均差枚"],
            "平均差枚変化": avg_diff_change,
            "V5プラス率": v5_result["プラス率"],
            "V5.1プラス率": v51_result["プラス率"],
            "プラス率変化": plus_change,
            "V5+500率": v5_result["+500率"],
            "V5.1+500率": v51_result["+500率"],
            "+500率変化": plus500_change,
            "V5+1000率": v5_result["+1000率"],
            "V5.1+1000率": v51_result["+1000率"],
            "+1000率変化": plus1000_change,
            "V5機種勝率": v5_result["機種勝率"],
            "V5.1機種勝率": v51_result["機種勝率"],
            "機種勝率変化": machine_change
        })

    # ========================================================
    # TOP別ベスト
    # ========================================================

    print_line()
    print("【TOP別ベストモデル】")
    print_line()

    for top in TOPS:

        v5_result = common_v5_results.get(top)
        v51_result = common_v51_results.get(top)

        if v5_result is None or v51_result is None:
            print(
                f"TOP{top}: 比較不能"
            )
            continue

        v5_avg = v5_result["平均差枚"]
        v51_avg = v51_result["平均差枚"]

        if pd.isna(v5_avg) and pd.isna(v51_avg):
            print(
                f"TOP{top}: 比較不能"
            )

        elif pd.isna(v5_avg):
            print(
                f"TOP{top}: V5.1"
            )

        elif pd.isna(v51_avg):
            print(
                f"TOP{top}: V5"
            )

        elif v51_avg > v5_avg:
            print(
                f"TOP{top}: V5.1 / "
                f"平均差枚 {v51_avg:+.1f}枚 / "
                f"プラス率 {v51_result['プラス率']:.1f}%"
            )

        else:
            print(
                f"TOP{top}: V5 / "
                f"平均差枚 {v5_avg:+.1f}枚 / "
                f"プラス率 {v5_result['プラス率']:.1f}%"
            )

    # ========================================================
    # 総合評価
    # ========================================================

    v5_score = calculate_model_score(
        list(common_v5_results.values())
    )

    v51_score = calculate_model_score(
        list(common_v51_results.values())
    )

    print_line()
    print("【モデル総合評価】")
    print_line()

    print(
        f"V5   / 総合評価 "
        f"{format_metric(v5_score)}"
    )

    print(
        f"V5.1 / 総合評価 "
        f"{format_metric(v51_score)}"
    )

    if not pd.isna(v5_score) and not pd.isna(v51_score):

        if v51_score > v5_score:
            print()
            print("★ 現時点のベストモデル: V5.1")

        elif v5_score > v51_score:
            print()
            print("★ 現時点のベストモデル: V5")

        else:
            print()
            print("★ V5とV5.1は同点")

    else:

        print()
        print("V5・V5.1の両方を評価できませんでした。")

    # ========================================================
    # CSV保存
    # ========================================================

    if comparison_rows:

        result_df = pd.DataFrame(
            comparison_rows
        )

        result_df.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print()
        print("★ CSV保存成功")
        print(OUTPUT_FILE)

    summary_rows = []

    for top in TOPS:

        v5_result = common_v5_results.get(top)
        v51_result = common_v51_results.get(top)

        summary_rows.append({
            "TOP": top,
            "V5評価日数": (
                v5_result["評価日数"]
                if v5_result else np.nan
            ),
            "V5平均差枚": (
                v5_result["平均差枚"]
                if v5_result else np.nan
            ),
            "V5プラス率": (
                v5_result["プラス率"]
                if v5_result else np.nan
            ),
            "V5+500率": (
                v5_result["+500率"]
                if v5_result else np.nan
            ),
            "V5+1000率": (
                v5_result["+1000率"]
                if v5_result else np.nan
            ),
            "V5機種勝率": (
                v5_result["機種勝率"]
                if v5_result else np.nan
            ),
            "V5.1評価日数": (
                v51_result["評価日数"]
                if v51_result else np.nan
            ),
            "V5.1平均差枚": (
                v51_result["平均差枚"]
                if v51_result else np.nan
            ),
            "V5.1プラス率": (
                v51_result["プラス率"]
                if v51_result else np.nan
            ),
            "V5.1+500率": (
                v51_result["+500率"]
                if v51_result else np.nan
            ),
            "V5.1+1000率": (
                v51_result["+1000率"]
                if v51_result else np.nan
            ),
            "V5.1機種勝率": (
                v51_result["機種勝率"]
                if v51_result else np.nan
            )
        })

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

    print_line()
    print("★★★★★ V5・V5.1 モデル比較完了 ★★★★★")
    print_line()

    print()
    print("保存ファイル:")
    print(OUTPUT_FILE)
    print(SUMMARY_FILE)

    print()
    print("元のバックテストCSVは変更していません。")


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    main()