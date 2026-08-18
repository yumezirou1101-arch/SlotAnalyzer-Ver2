# -*- coding: utf-8 -*-

"""
予測結果・バックテスト評価 V2.5

V3とV4を同じ基準で比較する。

V3:
investment_pattern_v3_backtest.csv
    予測日
    TOP
    予測機種数
    実績平均差枚
    実績プラス率
    +1000率
    +2000率
    機種勝率

V4:
investment_pattern_v4_backtest.csv
    対象日
    台番号
    機種
    ...
    V4総合スコア
    総合ランク
    当日差枚
    予測順位

V4は「予測順位」を使用してTOP5/TOP10/TOP20/TOP30を作成し、
当日差枚から実績を再集計する。

重要:
予測結果そのものは変更しない。
バックテストCSVも変更しない。
"""

import os
import math
import pandas as pd


# ============================================================
# 設定
# ============================================================

BASE_DIR = r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
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

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "prediction_evaluation_v2.csv"
)

SUMMARY_FILE = os.path.join(
    DATA_DIR,
    "prediction_evaluation_v2_summary.csv"
)

TOP_LIST = [5, 10, 20, 30]


# ============================================================
# 共通関数
# ============================================================

def print_line(char="=", length=70):
    print(char * length)


def print_header(text):
    print()
    print_line("=")
    print(text)
    print_line("=")


def safe_float(value):
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def fmt(value, digits=1):
    """
    NaNをN/Aとして表示。
    """
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def fmt_signed(value, digits=1):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):+.{digits}f}"
    except Exception:
        return "N/A"


def mean_safe(values):
    values = pd.Series(values, dtype="float64")
    values = values.dropna()

    if len(values) == 0:
        return float("nan")

    return values.mean()


# ============================================================
# CSV読み込み
# ============================================================

def load_v3():
    print()
    print("V3バックテスト:")
    print(f"ファイル: {V3_FILE}")

    if not os.path.exists(V3_FILE):
        print("V3ファイルがありません。")
        return None

    df = pd.read_csv(V3_FILE)

    print(f"読み込み: {len(df)}行")

    required = [
        "予測日",
        "TOP",
        "予測機種数",
        "実績平均差枚",
        "実績プラス率",
        "+1000率",
        "+2000率",
        "機種勝率",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        print("V3に不足している列:")
        for c in missing:
            print(f"  {c}")
        return None

    df["予測日"] = pd.to_datetime(
        df["予測日"],
        errors="coerce"
    )

    for col in [
        "TOP",
        "予測機種数",
        "実績平均差枚",
        "実績プラス率",
        "+1000率",
        "+2000率",
        "機種勝率",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df


def load_v4():
    print()
    print("V4バックテスト:")
    print(f"ファイル: {V4_FILE}")

    if not os.path.exists(V4_FILE):
        print("V4ファイルがありません。")
        return None

    df = pd.read_csv(V4_FILE)

    print(f"読み込み: {len(df)}行")

    required = [
        "対象日",
        "台番号",
        "機種",
        "V4総合スコア",
        "当日差枚",
        "予測順位",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        print("V4に不足している列:")
        for c in missing:
            print(f"  {c}")
        return None

    df["対象日"] = pd.to_datetime(
        df["対象日"],
        errors="coerce"
    )

    for col in [
        "台番号",
        "V4総合スコア",
        "当日差枚",
        "予測順位",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df


# ============================================================
# V3評価
# ============================================================

def evaluate_v3(v3):
    results = []

    if v3 is None or v3.empty:
        return pd.DataFrame()

    for top in TOP_LIST:

        part = v3[
            v3["TOP"] == top
        ].copy()

        if part.empty:
            continue

        results.append({
            "モデル": "V3",
            "TOP": top,
            "評価日数": part["予測日"].nunique(),
            "平均差枚": mean_safe(
                part["実績平均差枚"]
            ),
            "プラス率": mean_safe(
                part["実績プラス率"]
            ),
            "+1000率": mean_safe(
                part["+1000率"]
            ),
            "+2000率": mean_safe(
                part["+2000率"]
            ),
            "機種勝率": mean_safe(
                part["機種勝率"]
            ),
        })

    return pd.DataFrame(results)


# ============================================================
# V4評価
# ============================================================

def evaluate_v4(v4):
    results = []

    if v4 is None or v4.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # 予測順位を数値化
    # --------------------------------------------------------

    v4 = v4.copy()

    v4["予測順位"] = pd.to_numeric(
        v4["予測順位"],
        errors="coerce"
    )

    v4["当日差枚"] = pd.to_numeric(
        v4["当日差枚"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # 日付ごとに評価
    # --------------------------------------------------------

    dates = sorted(
        v4["対象日"]
        .dropna()
        .unique()
    )

    for top in TOP_LIST:

        daily_rows = []

        for target_date in dates:

            day = v4[
                v4["対象日"] == target_date
            ].copy()

            day = day.sort_values(
                "予測順位"
            )

            selected = day[
                day["予測順位"] <= top
            ].copy()

            if selected.empty:
                continue

            actual = selected["当日差枚"].dropna()

            if actual.empty:
                continue

            avg_diff = actual.mean()

            plus_rate = (
                (actual > 0).mean() * 100
            )

            rate1000 = (
                (actual >= 1000).mean() * 100
            )

            rate2000 = (
                (actual >= 2000).mean() * 100
            )

            # V4は台単位の予測なので、
            # 「機種勝率」は予測台のうち
            # 当日プラスになった台の割合として扱う。
            machine_win_rate = plus_rate

            daily_rows.append({
                "予測日": target_date,
                "平均差枚": avg_diff,
                "プラス率": plus_rate,
                "+1000率": rate1000,
                "+2000率": rate2000,
                "機種勝率": machine_win_rate,
                "予測台数": len(actual),
            })

        if not daily_rows:
            continue

        daily = pd.DataFrame(
            daily_rows
        )

        results.append({
            "モデル": "V4",
            "TOP": top,
            "評価日数": len(daily),
            "平均差枚": daily["平均差枚"].mean(),
            "プラス率": daily["プラス率"].mean(),
            "+1000率": daily["+1000率"].mean(),
            "+2000率": daily["+2000率"].mean(),
            "機種勝率": daily["機種勝率"].mean(),
        })

    return pd.DataFrame(results)


# ============================================================
# 公平比較
# ============================================================

def evaluate_v3_common_dates(v3, v4):
    """
    V3とV4の両方に存在する日だけでV3を再評価。

    これにより、
    「V3は8日、V4は6日」
    のような評価日数の違いによる偏りを防ぐ。
    """

    if v3 is None or v4 is None:
        return pd.DataFrame()

    v3_dates = set(
        v3["予測日"]
        .dropna()
        .dt.date
    )

    v4_dates = set(
        v4["対象日"]
        .dropna()
        .dt.date
    )

    common_dates = v3_dates & v4_dates

    if not common_dates:
        return pd.DataFrame()

    temp = v3[
        v3["予測日"].dt.date.isin(common_dates)
    ].copy()

    results = []

    for top in TOP_LIST:

        part = temp[
            temp["TOP"] == top
        ].copy()

        if part.empty:
            continue

        results.append({
            "モデル": "V3",
            "TOP": top,
            "評価日数": part["予測日"].nunique(),
            "平均差枚": mean_safe(
                part["実績平均差枚"]
            ),
            "プラス率": mean_safe(
                part["実績プラス率"]
            ),
            "+1000率": mean_safe(
                part["+1000率"]
            ),
            "+2000率": mean_safe(
                part["+2000率"]
            ),
            "機種勝率": mean_safe(
                part["機種勝率"]
            ),
        })

    return pd.DataFrame(results)


# ============================================================
# 総合評価
# ============================================================

def calculate_model_score(df):
    """
    総合評価。

    評価項目:
    ・平均差枚
    ・プラス率
    ・+1000率
    ・+2000率

    各TOPの結果を平均してモデル評価する。

    注意:
    現在のデータでは+1000率/+2000率が0%でも
    正常なデータとして扱う。
    """

    if df is None or df.empty:
        return float("nan")

    avg_diff = mean_safe(
        df["平均差枚"]
    )

    plus_rate = mean_safe(
        df["プラス率"]
    )

    rate1000 = mean_safe(
        df["+1000率"]
    )

    rate2000 = mean_safe(
        df["+2000率"]
    )

    if pd.isna(avg_diff):
        return float("nan")

    # --------------------------------------------------------
    # 平均差枚スコア
    # -1000枚 → 0
    # +1000枚 → 100
    # --------------------------------------------------------

    diff_score = max(
        0,
        min(
            100,
            (avg_diff + 1000) / 20
        )
    )

    # --------------------------------------------------------
    # プラス率スコア
    # 50%を基準に100点満点へ
    # --------------------------------------------------------

    plus_score = max(
        0,
        min(
            100,
            plus_rate * 2
        )
    )

    # --------------------------------------------------------
    # +1000率
    # --------------------------------------------------------

    score1000 = max(
        0,
        min(
            100,
            rate1000 * 2
        )
    )

    # --------------------------------------------------------
    # +2000率
    # --------------------------------------------------------

    score2000 = max(
        0,
        min(
            100,
            rate2000 * 2
        )
    )

    score = (
        diff_score * 0.40
        + plus_score * 0.30
        + score1000 * 0.20
        + score2000 * 0.10
    )

    return score


# ============================================================
# 表示
# ============================================================

def print_summary(title, df):

    print_header(title)

    if df is None or df.empty:
        print("評価データがありません。")
        return

    for _, row in df.iterrows():

        print(
            f"TOP{int(row['TOP'])} / "
            f"評価日数 {int(row['評価日数'])}日 / "
            f"平均差枚 {fmt_signed(row['平均差枚'])}枚 / "
            f"プラス率 {fmt(row['プラス率'])}% / "
            f"+1000率 {fmt(row['+1000率'])}% / "
            f"+2000率 {fmt(row['+2000率'])}% / "
            f"機種勝率 {fmt(row['機種勝率'])}%"
        )


def print_comparison(v3, v4):

    print_header("【V3・V4 公平比較】")

    for top in TOP_LIST:

        print()
        print(f"--- TOP{top} ---")

        v3_row = v3[
            v3["TOP"] == top
        ] if v3 is not None and not v3.empty else pd.DataFrame()

        v4_row = v4[
            v4["TOP"] == top
        ] if v4 is not None and not v4.empty else pd.DataFrame()

        if not v3_row.empty:

            r = v3_row.iloc[0]

            print(
                f"V3: "
                f"評価日数 {int(r['評価日数'])}日 / "
                f"平均差枚 {fmt_signed(r['平均差枚'])}枚 / "
                f"プラス率 {fmt(r['プラス率'])}% / "
                f"+1000率 {fmt(r['+1000率'])}% / "
                f"+2000率 {fmt(r['+2000率'])}% / "
                f"機種勝率 {fmt(r['機種勝率'])}%"
            )

        else:

            print("V3: 評価データなし")

        if not v4_row.empty:

            r = v4_row.iloc[0]

            print(
                f"V4: "
                f"評価日数 {int(r['評価日数'])}日 / "
                f"平均差枚 {fmt_signed(r['平均差枚'])}枚 / "
                f"プラス率 {fmt(r['プラス率'])}% / "
                f"+1000率 {fmt(r['+1000率'])}% / "
                f"+2000率 {fmt(r['+2000率'])}% / "
                f"機種勝率 {fmt(r['機種勝率'])}%"
            )

        else:

            print("V4: 評価データなし")


# ============================================================
# メイン
# ============================================================

def main():

    print()
    print_line("=")
    print("予測結果・バックテスト評価 V2.5")
    print_line("=")

    print()
    print("V3とV4を同じ基準で比較します。")
    print("V4は予測順位からTOP5/TOP10/TOP20/TOP30を再集計します。")
    print("V3・V4共通日による公平比較も行います。")

    print()
    print("データフォルダ:")
    print(DATA_DIR)

    # --------------------------------------------------------
    # 読み込み
    # --------------------------------------------------------

    v3 = load_v3()
    v4 = load_v4()

    # --------------------------------------------------------
    # V3通常評価
    # --------------------------------------------------------

    v3_all = evaluate_v3(v3)

    # --------------------------------------------------------
    # V4再集計評価
    # --------------------------------------------------------

    v4_all = evaluate_v4(v4)

    # --------------------------------------------------------
    # 共通日V3
    # --------------------------------------------------------

    v3_common = evaluate_v3_common_dates(
        v3,
        v4
    )

    # --------------------------------------------------------
    # V4は共通日に限定
    # --------------------------------------------------------

    v4_common = pd.DataFrame()

    if v4 is not None and not v4.empty:

        if v3 is not None and not v3.empty:

            v3_dates = set(
                v3["予測日"]
                .dropna()
                .dt.date
            )

            v4_common_raw = v4[
                v4["対象日"].dt.date.isin(
                    v3_dates
                )
            ].copy()

            v4_common = evaluate_v4(
                v4_common_raw
            )

    # --------------------------------------------------------
    # 表示
    # --------------------------------------------------------

    print_summary(
        "【V3 総合評価】",
        v3_all
    )

    print_summary(
        "【V4 総合評価】",
        v4_all
    )

    # --------------------------------------------------------
    # 公平比較
    # --------------------------------------------------------

    print_header(
        "【V3・V4 共通日による公平比較】"
    )

    print(
        "V3・V4の両方に存在する予測日のみを使用します。"
    )

    if v3_common is not None and not v3_common.empty:

        common_dates_v3 = set(
            v3[
                v3["予測日"].dt.date.isin(
                    set(
                        v4["対象日"]
                        .dropna()
                        .dt.date
                    )
                )
            ]["予測日"]
            .dt.date
        )

        print(
            "共通評価日: "
            + " / ".join(
                sorted(
                    str(x)
                    for x in common_dates_v3
                )
            )
        )

    print_summary(
        "【V3 共通日評価】",
        v3_common
    )

    print_summary(
        "【V4 共通日評価】",
        v4_common
    )

    print_comparison(
        v3_common,
        v4_common
    )

    # --------------------------------------------------------
    # モデル総合評価
    # --------------------------------------------------------

    print_header(
        "【モデル総合評価】"
    )

    v3_score = calculate_model_score(
        v3_common
    )

    v4_score = calculate_model_score(
        v4_common
    )

    print(
        f"V3 / 総合評価 "
        f"{fmt(v3_score)}"
    )

    print(
        f"V4 / 総合評価 "
        f"{fmt(v4_score)}"
    )

    if not pd.isna(v3_score) and not pd.isna(v4_score):

        if v4_score > v3_score:

            print()
            print(
                "★ 現時点のベストモデル: V4"
            )

        elif v3_score > v4_score:

            print()
            print(
                "★ 現時点のベストモデル: V3"
            )

        else:

            print()
            print(
                "★ V3・V4は同点です。"
            )

    else:

        print()
        print(
            "V3・V4の両方を評価できないため、"
            "ベストモデル判定は行いません。"
        )

    # --------------------------------------------------------
    # CSV保存
    # --------------------------------------------------------

    output_rows = []

    for source_name, df in [
        ("V3", v3_common),
        ("V4", v4_common),
    ]:

        if df is None or df.empty:
            continue

        for _, row in df.iterrows():

            output_rows.append({
                "モデル": source_name,
                "TOP": row["TOP"],
                "評価日数": row["評価日数"],
                "平均差枚": row["平均差枚"],
                "プラス率": row["プラス率"],
                "+1000率": row["+1000率"],
                "+2000率": row["+2000率"],
                "機種勝率": row["機種勝率"],
                "総合評価": calculate_model_score(
                    pd.DataFrame([row])
                ),
            })

    result_df = pd.DataFrame(
        output_rows
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # モデルサマリー
    # --------------------------------------------------------

    summary_rows = []

    for model_name, df in [
        ("V3", v3_common),
        ("V4", v4_common),
    ]:

        if df is None or df.empty:

            summary_rows.append({
                "モデル": model_name,
                "評価日数": 0,
                "平均差枚": float("nan"),
                "プラス率": float("nan"),
                "+1000率": float("nan"),
                "+2000率": float("nan"),
                "機種勝率": float("nan"),
                "総合評価": float("nan"),
            })

            continue

        summary_rows.append({
            "モデル": model_name,
            "評価日数": int(
                df["評価日数"].max()
            ),
            "平均差枚": mean_safe(
                df["平均差枚"]
            ),
            "プラス率": mean_safe(
                df["プラス率"]
            ),
            "+1000率": mean_safe(
                df["+1000率"]
            ),
            "+2000率": mean_safe(
                df["+2000率"]
            ),
            "機種勝率": mean_safe(
                df["機種勝率"]
            ),
            "総合評価": calculate_model_score(
                df
            ),
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
    print(OUTPUT_FILE)

    print()
    print("★ CSV保存成功")
    print(SUMMARY_FILE)

    print()
    print_line("=")
    print("★★★★★ 予測評価 V2.5 完了 ★★★★★")
    print_line("=")

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