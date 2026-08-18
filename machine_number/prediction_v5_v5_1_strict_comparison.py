import os
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
    "prediction_v5_v5_1_strict_comparison.csv"
)

SUMMARY_FILE = os.path.join(
    DATA_DIR,
    "prediction_v5_v5_1_strict_comparison_summary.csv"
)


TOPS = [5, 10, 20, 30]


# ============================================================
# 表示
# ============================================================

def print_header(title):
    print("=" * 70)
    print(title)
    print("=" * 70)


def fmt(value, digits=1):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


def fmt_signed(value, digits=1):
    if pd.isna(value):
        return "N/A"
    return f"{value:+.{digits}f}"


# ============================================================
# 列名を統一
# ============================================================

def normalize_columns(df):
    rename_map = {}

    for col in df.columns:
        c = str(col).strip()

        if c in ["対象日", "予測日", "日付"]:
            rename_map[col] = "対象日"

        elif c in ["TOP", "top", "Top"]:
            rename_map[col] = "TOP"

        elif c in ["実績平均差枚", "平均差枚"]:
            rename_map[col] = "実績平均差枚"

        elif c in ["実績プラス率", "プラス率"]:
            rename_map[col] = "実績プラス率"

        elif c in ["+500率", "実績+500率"]:
            rename_map[col] = "+500率"

        elif c in ["+1000率", "実績+1000率"]:
            rename_map[col] = "+1000率"

        elif c in ["+2000率", "実績+2000率"]:
            rename_map[col] = "+2000率"

        elif c in ["+3000率", "実績+3000率"]:
            rename_map[col] = "+3000率"

        elif c in ["機種勝率", "実績機種勝率"]:
            rename_map[col] = "機種勝率"

    df = df.rename(columns=rename_map)

    return df


# ============================================================
# CSV読み込み
# ============================================================

def load_backtest(path, model_name):

    print(f"{model_name}バックテスト:")
    print(f"ファイル: {path}")

    if not os.path.exists(path):
        print("ファイルがありません。")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path, encoding="cp932")

    print(f"読み込み: {len(df)}行")

    df = normalize_columns(df)

    required = [
        "対象日",
        "TOP",
        "実績平均差枚",
        "実績プラス率"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"必要列がありません: {missing}")
        print("実際の列:")
        print(list(df.columns))
        return pd.DataFrame()

    df["対象日"] = pd.to_datetime(
        df["対象日"],
        errors="coerce"
    )

    df["TOP"] = pd.to_numeric(
        df["TOP"],
        errors="coerce"
    )

    numeric_cols = [
        "実績平均差枚",
        "実績プラス率",
        "+500率",
        "+1000率",
        "+2000率",
        "+3000率",
        "機種勝率"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df = df.dropna(
        subset=["対象日", "TOP"]
    )

    df["TOP"] = df["TOP"].astype(int)

    return df


# ============================================================
# 日付×TOPを完全一致させる
# ============================================================

def create_strict_comparison(v5, v51):

    print()
    print_header("【日付×TOP 完全一致による比較】")

    v5_keys = set(
        zip(
            v5["対象日"],
            v5["TOP"]
        )
    )

    v51_keys = set(
        zip(
            v51["対象日"],
            v51["TOP"]
        )
    )

    common_keys = sorted(
        v5_keys & v51_keys
    )

    print(f"V5 日付×TOP件数   : {len(v5_keys)}")
    print(f"V5.1 日付×TOP件数 : {len(v51_keys)}")
    print(f"完全一致件数       : {len(common_keys)}")

    if not common_keys:
        print("共通データがありません。")
        return pd.DataFrame()

    rows = []

    for date, top in common_keys:

        v5_row = v5[
            (v5["対象日"] == date) &
            (v5["TOP"] == top)
        ]

        v51_row = v51[
            (v51["対象日"] == date) &
            (v51["TOP"] == top)
        ]

        if len(v5_row) == 0 or len(v51_row) == 0:
            continue

        v5_row = v5_row.iloc[0]
        v51_row = v51_row.iloc[0]

        row = {
            "対象日": date.strftime("%Y-%m-%d"),
            "TOP": top,

            "V5_平均差枚": v5_row["実績平均差枚"],
            "V5_プラス率": v5_row["実績プラス率"],
            "V5_+500率": (
                v5_row["+500率"]
                if "+500率" in v5_row.index
                else np.nan
            ),
            "V5_+1000率": (
                v5_row["+1000率"]
                if "+1000率" in v5_row.index
                else np.nan
            ),
            "V5_+2000率": (
                v5_row["+2000率"]
                if "+2000率" in v5_row.index
                else np.nan
            ),
            "V5_+3000率": (
                v5_row["+3000率"]
                if "+3000率" in v5_row.index
                else np.nan
            ),
            "V5_機種勝率": (
                v5_row["機種勝率"]
                if "機種勝率" in v5_row.index
                else np.nan
            ),

            "V5.1_平均差枚": v51_row["実績平均差枚"],
            "V5.1_プラス率": v51_row["実績プラス率"],
            "V5.1_+500率": (
                v51_row["+500率"]
                if "+500率" in v51_row.index
                else np.nan
            ),
            "V5.1_+1000率": (
                v51_row["+1000率"]
                if "+1000率" in v51_row.index
                else np.nan
            ),
            "V5.1_+2000率": (
                v51_row["+2000率"]
                if "+2000率" in v51_row.index
                else np.nan
            ),
            "V5.1_+3000率": (
                v51_row["+3000率"]
                if "+3000率" in v51_row.index
                else np.nan
            ),
            "V5.1_機種勝率": (
                v51_row["機種勝率"]
                if "機種勝率" in v51_row.index
                else np.nan
            )
        }

        row["差_平均差枚"] = (
            row["V5.1_平均差枚"]
            - row["V5_平均差枚"]
        )

        row["差_プラス率"] = (
            row["V5.1_プラス率"]
            - row["V5_プラス率"]
        )

        row["差_+500率"] = (
            row["V5.1_+500率"]
            - row["V5_+500率"]
            if not pd.isna(row["V5.1_+500率"])
            and not pd.isna(row["V5_+500率"])
            else np.nan
        )

        row["差_+1000率"] = (
            row["V5.1_+1000率"]
            - row["V5_+1000率"]
            if not pd.isna(row["V5.1_+1000率"])
            and not pd.isna(row["V5_+1000率"])
            else np.nan
        )

        row["差_機種勝率"] = (
            row["V5.1_機種勝率"]
            - row["V5_機種勝率"]
            if not pd.isna(row["V5.1_機種勝率"])
            and not pd.isna(row["V5_機種勝率"])
            else np.nan
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    return result


# ============================================================
# TOP別集計
# ============================================================

def summarize_by_top(df, top):

    data = df[df["TOP"] == top].copy()

    if data.empty:
        return None

    result = {
        "評価日数": len(data),

        "平均差枚": data["V5_平均差枚"].mean(),
        "V5.1平均差枚": data["V5.1_平均差枚"].mean(),

        "プラス率": data["V5_プラス率"].mean(),
        "V5.1プラス率": data["V5.1_プラス率"].mean(),

        "平均差枚改善": data["差_平均差枚"].mean(),
        "プラス率改善": data["差_プラス率"].mean()
    }

    for col in ["V5_+500率", "V5.1_+500率",
                "V5_+1000率", "V5.1_+1000率",
                "V5_機種勝率", "V5.1_機種勝率"]:

        if col in data.columns:
            result[col] = data[col].mean()

    return result


# ============================================================
# TOP別表示
# ============================================================

def print_top_summary(df):

    print()
    print_header("【完全一致データによるTOP別比較】")

    summary_rows = []

    for top in TOPS:

        s = summarize_by_top(df, top)

        if s is None:
            print(f"TOP{top}: データなし")
            continue

        print()
        print(f"--- TOP{top} ---")

        print(
            f"評価日数: {s['評価日数']}日"
        )

        print(
            f"V5   平均差枚: "
            f"{fmt_signed(s['平均差枚'])}枚"
        )

        print(
            f"V5.1 平均差枚: "
            f"{fmt_signed(s['V5.1平均差枚'])}枚"
        )

        print(
            f"改善幅: "
            f"{fmt_signed(s['平均差枚改善'])}枚"
        )

        print(
            f"V5   プラス率: "
            f"{fmt(s['プラス率'])}%"
        )

        print(
            f"V5.1 プラス率: "
            f"{fmt(s['V5.1プラス率'])}%"
        )

        print(
            f"プラス率改善: "
            f"{fmt_signed(s['プラス率改善'])}pt"
        )

        summary_rows.append({
            "TOP": top,
            **s
        })

    return pd.DataFrame(summary_rows)


# ============================================================
# 日別比較
# ============================================================

def print_daily_comparison(df):

    print()
    print_header("【日別 V5 → V5.1 比較】")

    for date in sorted(df["対象日"].unique()):

        day = df[df["対象日"] == date]

        print()
        print(
            f"--- {date} ---"
        )

        for _, row in day.sort_values("TOP").iterrows():

            print(
                f"TOP{int(row['TOP']):2d} : "
                f"V5 {row['V5_平均差枚']:+.1f}枚 "
                f"→ "
                f"V5.1 {row['V5.1_平均差枚']:+.1f}枚 "
                f"("
                f"{row['差_平均差枚']:+.1f}"
                f")"
            )


# ============================================================
# V5.1が改善した日数
# ============================================================

def calculate_wins(df):

    print()
    print_header("【V5.1 改善日数分析】")

    for top in TOPS:

        data = df[df["TOP"] == top]

        if data.empty:
            continue

        avg_diff = data["差_平均差枚"]

        improved = (avg_diff > 0).sum()
        worsened = (avg_diff < 0).sum()
        equal = (avg_diff == 0).sum()

        print(
            f"TOP{top}: "
            f"改善 {improved}日 / "
            f"悪化 {worsened}日 / "
            f"同値 {equal}日"
        )


# ============================================================
# モデル判定
# ============================================================

def model_judgement(summary):

    print()
    print_header("【V5・V5.1 最終判定】")

    if summary.empty:
        print("判定できるデータがありません。")
        return

    v5_avg = summary["平均差枚"].mean()
    v51_avg = summary["V5.1平均差枚"].mean()

    v5_plus = summary["プラス率"].mean()
    v51_plus = summary["V5.1プラス率"].mean()

    print(
        f"V5   全TOP平均差枚: "
        f"{v5_avg:+.1f}枚"
    )

    print(
        f"V5.1 全TOP平均差枚: "
        f"{v51_avg:+.1f}枚"
    )

    print()

    print(
        f"V5   全TOP平均プラス率: "
        f"{v5_plus:.1f}%"
    )

    print(
        f"V5.1 全TOP平均プラス率: "
        f"{v51_plus:.1f}%"
    )

    print()

    if v51_avg > v5_avg and v51_plus >= v5_plus:
        print("★ V5.1を採用候補とします。")

    elif v5_avg > v51_avg and v5_plus >= v51_plus:
        print("★ V5を採用候補とします。")

    else:
        print(
            "★ 平均差枚とプラス率で結果が分かれています。"
        )
        print(
            "  まだV5.1への変更は行わず、追加データを蓄積します。"
        )


# ============================================================
# メイン
# ============================================================

def main():

    print_header(
        "V5・V5.1 厳密モデル比較 V2"
    )

    print(
        "同じ予測日 × 同じTOPだけを比較します。"
    )

    print(
        "評価期間の違いによる偏りを排除します。"
    )

    print()

    print(f"データフォルダ:")
    print(DATA_DIR)
    print()

    v5 = load_backtest(
        V5_FILE,
        "V5"
    )

    print()

    v51 = load_backtest(
        V51_FILE,
        "V5.1"
    )

    if v5.empty or v51.empty:
        print()
        print("必要なバックテストデータがありません。")
        return

    comparison = create_strict_comparison(
        v5,
        v51
    )

    if comparison.empty:
        print()
        print("比較可能なデータがありません。")
        return

    print()
    print_header("【完全一致評価日】")

    dates = sorted(
        comparison["対象日"].unique()
    )

    print(
        " / ".join(dates)
    )

    print(
        f"\n共通予測日数: {len(dates)}日"
    )

    summary = print_top_summary(
        comparison
    )

    print_daily_comparison(
        comparison
    )

    calculate_wins(
        comparison
    )

    model_judgement(
        summary
    )

    # ========================================================
    # TOP別ベスト
    # ========================================================

    print()
    print_header("【TOP別 平均差枚ベスト】")

    for top in TOPS:

        data = comparison[
            comparison["TOP"] == top
        ]

        if data.empty:
            continue

        v5_mean = data["V5_平均差枚"].mean()
        v51_mean = data["V5.1_平均差枚"].mean()

        if v5_mean > v51_mean:
            best = "V5"
            best_value = v5_mean
        else:
            best = "V5.1"
            best_value = v51_mean

        print(
            f"TOP{top}: "
            f"{best} / "
            f"平均差枚 {best_value:+.1f}枚"
        )

    # ========================================================
    # 保存
    # ========================================================

    comparison.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("★ CSV保存成功")
    print(OUTPUT_FILE)

    print()
    print("★ CSV保存成功")
    print(SUMMARY_FILE)

    print()
    print_header(
        "★★★★★ V5・V5.1 厳密比較 完了 ★★★★★"
    )

    print()
    print("元のバックテストCSVは変更していません。")


if __name__ == "__main__":
    main()