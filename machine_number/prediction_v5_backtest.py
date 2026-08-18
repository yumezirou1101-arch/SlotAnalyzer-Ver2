# -*- coding: utf-8 -*-

"""
SlotAnalyzer
投入パターン予測 V5 バックテスト

V5と同じロジックを過去の日付ごとに再計算し、
翌日の実績と比較する。

重要:
・予測対象日の実績はスコア計算に使用しない
・各予測日は、その前日までのデータだけを使用
・TOP5 / TOP10 / TOP20 / TOP30を評価
・日別結果と総合結果をCSV保存
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 設定
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "maruhan_maebashi"
)

INPUT_FILE = os.path.join(
    DATA_DIR,
    "all_data.csv"
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "machine_number"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "prediction_v5_backtest.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "prediction_v5_backtest_summary.csv"
)


# ============================================================
# 共通関数
# ============================================================

def safe_float(value, default=0.0):

    try:
        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def normalize_0_100(value, low, high):

    if high == low:
        return 50.0

    score = (
        (value - low)
        / (high - low)
        * 100
    )

    return float(
        np.clip(
            score,
            0,
            100
        )
    )


def find_column(df, candidates):

    for col in candidates:

        if col in df.columns:
            return col

    return None


def number_band(number):

    number = int(number)

    if 500 <= number <= 599:
        return "500～599"

    if 600 <= number <= 699:
        return "600～699"

    if 700 <= number <= 799:
        return "700～799"

    if 800 <= number <= 899:
        return "800～899"

    if 900 <= number <= 999:
        return "900～999"

    if 1000 <= number <= 1099:
        return "1000～1099"

    return "その他"


# ============================================================
# V5スコア計算
# ============================================================

def calculate_v5_scores(
    history,
    candidate_df,
    target_date
):

    machine_groups = {}

    for machine_name, g in history.groupby("機種"):

        machine_groups[machine_name] = (
            g.sort_values("日付")
        )

    number_groups = {}

    for number, g in history.groupby("台番号"):

        number_groups[number] = (
            g.sort_values("日付")
        )

    # --------------------------------------------------------
    # 曜日実績
    # --------------------------------------------------------

    weekday_stats = {}

    for date_value, g in history.groupby(
        history["日付"].dt.normalize()
    ):

        wd = pd.Timestamp(
            date_value
        ).weekday()

        avg_diff = safe_float(
            g["差枚"].mean()
        )

        plus_rate = safe_float(
            (g["差枚"] > 0).mean() * 100
        )

        weekday_stats.setdefault(
            wd,
            []
        )

        weekday_stats[wd].append(
            {
                "avg": avg_diff,
                "plus": plus_rate
            }
        )

    weekday_score = {}

    for wd in range(7):

        rows = weekday_stats.get(
            wd,
            []
        )

        if not rows:

            weekday_score[wd] = 50.0
            continue

        avg = np.mean(
            [r["avg"] for r in rows]
        )

        plus = np.mean(
            [r["plus"] for r in rows]
        )

        weekday_score[wd] = (
            normalize_0_100(
                avg,
                -300,
                100
            ) * 0.7
            +
            normalize_0_100(
                plus,
                20,
                45
            ) * 0.3
        )

    # --------------------------------------------------------
    # 特徴量
    # --------------------------------------------------------

    results = []

    for _, row in candidate_df.iterrows():

        number = int(
            row["台番号"]
        )

        machine_name = str(
            row["機種"]
        )

        ng = number_groups.get(
            number
        )

        mg = machine_groups.get(
            machine_name
        )

        # ----------------------------------------------------
        # 台データ
        # ----------------------------------------------------

        if ng is None or len(ng) == 0:

            number_values = pd.Series(
                [0.0]
            )

        else:

            number_values = (
                ng["差枚"]
                .astype(float)
            )

        number_last = safe_float(
            number_values.iloc[-1]
        )

        number_previous = (

            safe_float(
                number_values.iloc[-2]
            )
            if len(number_values) >= 2
            else number_last
        )

        number_avg = safe_float(
            number_values.mean()
        )

        number_recent3 = safe_float(
            number_values.tail(3).mean()
        )

        number_plus_rate = safe_float(
            (number_values > 0).mean()
            * 100,
            50.0
        )

        number_rate1000 = safe_float(
            (number_values >= 1000).mean()
            * 100
        )

        number_rate2000 = safe_float(
            (number_values >= 2000).mean()
            * 100
        )

        number_change = (
            number_last
            - number_previous
        )

        number_drawdown = (
            number_avg
            - number_last
        )

        # ----------------------------------------------------
        # 機種データ
        # ----------------------------------------------------

        if mg is None or len(mg) == 0:

            machine_values = pd.Series(
                [0.0]
            )

        else:

            machine_values = (
                mg["差枚"]
                .astype(float)
            )

        machine_last = safe_float(
            machine_values.iloc[-1]
        )

        machine_previous = (

            safe_float(
                machine_values.iloc[-2]
            )
            if len(machine_values) >= 2
            else machine_last
        )

        machine_avg = safe_float(
            machine_values.mean()
        )

        machine_recent3 = safe_float(
            machine_values.tail(3).mean()
        )

        machine_plus_rate = safe_float(
            (machine_values > 0).mean()
            * 100,
            50.0
        )

        machine_rate1000 = safe_float(
            (machine_values >= 1000).mean()
            * 100
        )

        machine_rate2000 = safe_float(
            (machine_values >= 2000).mean()
            * 100
        )

        machine_change = (
            machine_last
            - machine_previous
        )

        machine_drawdown = (
            machine_avg
            - machine_last
        )

        # ----------------------------------------------------
        # 台_前回変化スコア
        # ----------------------------------------------------

        if (
            0
            <= number_change
            <= 1000
        ):

            score_change = 90.0

        elif (
            -500
            <= number_change
            < 0
        ):

            score_change = 60.0

        elif (
            1000
            < number_change
            <= 2000
        ):

            score_change = 55.0

        elif number_change < -1000:

            score_change = 25.0

        else:

            score_change = 45.0

        # ----------------------------------------------------
        # 台_前日差枚スコア
        # ----------------------------------------------------

        if (
            0
            <= number_last
            <= 1000
        ):

            score_yesterday = 90.0

        elif (
            -500
            <= number_last
            < 0
        ):

            score_yesterday = 60.0

        elif (
            1000
            < number_last
            <= 2000
        ):

            score_yesterday = 55.0

        elif number_last < -1000:

            score_yesterday = 25.0

        else:

            score_yesterday = 45.0

        # ----------------------------------------------------
        # 台_凹みスコア
        # ----------------------------------------------------

        if (
            -1000
            <= number_drawdown
            <= 0
        ):

            score_drawdown = 75.0

        elif number_drawdown < -1000:

            score_drawdown = 50.0

        elif (
            0
            < number_drawdown
            <= 1000
        ):

            score_drawdown = 40.0

        else:

            score_drawdown = 25.0

        # ----------------------------------------------------
        # 機種_前日差枚スコア
        # ----------------------------------------------------

        if (
            0
            <= machine_last
            <= 1000
        ):

            score_machine_yesterday = 90.0

        elif (
            -500
            <= machine_last
            < 0
        ):

            score_machine_yesterday = 60.0

        elif (
            1000
            < machine_last
            <= 2000
        ):

            score_machine_yesterday = 55.0

        elif machine_last < -1000:

            score_machine_yesterday = 25.0

        else:

            score_machine_yesterday = 45.0

        # ----------------------------------------------------
        # 機種_直近3日
        # ----------------------------------------------------

        score_recent3 = normalize_0_100(
            machine_recent3,
            -1000,
            1000
        )

        # ----------------------------------------------------
        # 機種_過去平均
        # ----------------------------------------------------

        score_machine_avg = normalize_0_100(
            machine_avg,
            -1000,
            1000
        )

        # ----------------------------------------------------
        # 曜日
        # ----------------------------------------------------

        score_weekday = weekday_score.get(
            target_date.weekday(),
            50.0
        )

        # ----------------------------------------------------
        # 台番号帯
        # ----------------------------------------------------

        band = number_band(
            number
        )

        band_bonus = {

            "500～599": 35.0,
            "600～699": 45.0,
            "700～799": 58.0,
            "800～899": 60.0,
            "900～999": 35.0,
            "1000～1099": 30.0,
            "その他": 40.0,
        }

        score_band = band_bonus.get(
            band,
            40.0
        )

        # ----------------------------------------------------
        # 奇偶
        # ----------------------------------------------------

        if number % 2 == 1:

            score_parity = 55.0

        else:

            score_parity = 45.0

        # ----------------------------------------------------
        # V5総合スコア
        # ----------------------------------------------------

        score = (

            score_change * 0.22

            + score_yesterday * 0.18

            + score_drawdown * 0.12

            + score_machine_yesterday * 0.18

            + score_recent3 * 0.10

            + score_machine_avg * 0.06

            + score_weekday * 0.06

            + score_band * 0.05

            + score_parity * 0.03
        )

        score = float(
            np.clip(
                score,
                0,
                100
            )
        )

        if score >= 75:

            rank = "S"

        elif score >= 65:

            rank = "A"

        elif score >= 55:

            rank = "B"

        elif score >= 45:

            rank = "C"

        elif score >= 35:

            rank = "D"

        else:

            rank = "E"

        results.append({

            "予測日":
                target_date.strftime(
                    "%Y-%m-%d"
                ),

            "台番号":
                number,

            "機種":
                machine_name,

            "V5総合スコア":
                score,

            "ランク":
                rank,

            "台_前回変化":
                number_change,

            "台_前日差枚":
                number_last,

            "台_凹み":
                number_drawdown,

            "機種_前日差枚":
                machine_last,

            "機種_直近3日平均":
                machine_recent3,

            "機種_過去平均差枚":
                machine_avg,

            "曜日":
                target_date.strftime(
                    "%A"
                ),

            "台番号帯":
                band
        })

    result = pd.DataFrame(
        results
    )

    if result.empty:
        return result

    result = result.sort_values(
        [
            "V5総合スコア",
            "台_前回変化",
            "台_前日差枚"
        ],
        ascending=[
            False,
            False,
            False
        ]
    ).reset_index(
        drop=True
    )

    result["予測順位"] = (
        result.index + 1
    )

    return result


# ============================================================
# 実績評価
# ============================================================

def evaluate_top(
    prediction,
    actual,
    top_n
):

    top = prediction.head(
        top_n
    ).copy()

    if top.empty:

        return None

    actual_map = (
        actual[
            [
                "台番号",
                "差枚"
            ]
        ]
        .drop_duplicates(
            subset=["台番号"],
            keep="last"
        )
        .set_index("台番号")["差枚"]
    )

    top["当日差枚"] = (
        top["台番号"]
        .map(actual_map)
    )

    top = top.dropna(
        subset=["当日差枚"]
    )

    if top.empty:

        return None

    diffs = top[
        "当日差枚"
    ].astype(float)

    avg_diff = safe_float(
        diffs.mean()
    )

    plus_rate = safe_float(
        (diffs > 0).mean() * 100
    )

    rate500 = safe_float(
        (diffs >= 500).mean() * 100
    )

    rate1000 = safe_float(
        (diffs >= 1000).mean() * 100
    )

    rate2000 = safe_float(
        (diffs >= 2000).mean() * 100
    )

    rate3000 = safe_float(
        (diffs >= 3000).mean() * 100
    )

    # 機種勝率
    machine_result = (
        top.groupby("機種")[
            "当日差枚"
        ]
        .mean()
    )

    machine_win_rate = safe_float(
        (machine_result > 0).mean()
        * 100
    )

    return {

        "TOP": top_n,

        "予測台数":
            len(prediction.head(top_n)),

        "実績台数":
            len(top),

        "実績平均差枚":
            avg_diff,

        "実績プラス率":
            plus_rate,

        "+500率":
            rate500,

        "+1000率":
            rate1000,

        "+2000率":
            rate2000,

        "+3000率":
            rate3000,

        "機種勝率":
            machine_win_rate
    }


# ============================================================
# 表示
# ============================================================

def print_result(
    label,
    r
):

    print(
        f"{label} / "
        f"実績平均差枚 "
        f"{r['実績平均差枚']:+.1f}枚 / "
        f"プラス率 "
        f"{r['実績プラス率']:.1f}% / "
        f"+500率 "
        f"{r['+500率']:.1f}% / "
        f"+1000率 "
        f"{r['+1000率']:.1f}% / "
        f"+2000率 "
        f"{r['+2000率']:.1f}% / "
        f"+3000率 "
        f"{r['+3000率']:.1f}% / "
        f"機種勝率 "
        f"{r['機種勝率']:.1f}%"
    )


# ============================================================
# メイン
# ============================================================

def main():

    print("=" * 70)
    print("投入パターン予測 V5 バックテスト")
    print("=" * 70)

    print()
    print("V5を過去データに対して再計算し、")
    print("実際の翌日差枚と比較します。")

    print()
    print("重要:")
    print("予測対象日の実績は予測スコア計算に使用しません。")

    print()
    print("入力ファイル:")
    print(INPUT_FILE)

    if not os.path.exists(INPUT_FILE):

        print()
        print("ERROR: all_data.csv がありません。")
        input("Enterキーで終了...")
        return

    # --------------------------------------------------------
    # 読み込み
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            INPUT_FILE,
            encoding="utf-8-sig"
        )

    except Exception:

        try:

            df = pd.read_csv(
                INPUT_FILE,
                encoding="cp932"
            )

        except Exception as e:

            print()
            print("CSV読み込みエラー:")
            print(e)

            input("Enterキーで終了...")
            return

    print()
    print(
        f"読み込みデータ: {len(df):,}行"
    )

    # --------------------------------------------------------
    # 列
    # --------------------------------------------------------

    date_col = find_column(
        df,
        [
            "日付",
            "Date",
            "date"
        ]
    )

    number_col = find_column(
        df,
        [
            "台番号",
            "台番",
            "台No",
            "台NO",
            "台"
        ]
    )

    machine_col = find_column(
        df,
        [
            "機種名",
            "機種",
            "machine"
        ]
    )

    diff_col = find_column(
        df,
        [
            "差枚",
            "差枚数",
            "出玉",
            "差玉"
        ]
    )

    if not all(
        [
            date_col,
            number_col,
            machine_col,
            diff_col
        ]
    ):

        print()
        print("ERROR: 必要な列がありません。")

        input("Enterキーで終了...")
        return

    work = df[
        [
            date_col,
            number_col,
            machine_col,
            diff_col
        ]
    ].copy()

    work.columns = [
        "日付",
        "台番号",
        "機種",
        "差枚"
    ]

    work["日付"] = pd.to_datetime(
        work["日付"],
        errors="coerce"
    )

    work["台番号"] = pd.to_numeric(
        work["台番号"],
        errors="coerce"
    )

    work["差枚"] = pd.to_numeric(
        work["差枚"],
        errors="coerce"
    )

    work["機種"] = (
        work["機種"]
        .astype(str)
        .str.strip()
    )

    work = work.dropna(
        subset=[
            "日付",
            "台番号",
            "機種",
            "差枚"
        ]
    )

    work["台番号"] = (
        work["台番号"]
        .astype(int)
    )

    work["日付"] = (
        work["日付"]
        .dt.normalize()
    )

    work = work.sort_values(
        [
            "日付",
            "台番号"
        ]
    ).reset_index(
        drop=True
    )

    print()
    print(
        f"有効データ: {len(work):,}行"
    )

    dates = sorted(
        work["日付"].unique()
    )

    print()
    print(
        f"解析日数: {len(dates)}日"
    )

    if len(dates) < 2:

        print()
        print("ERROR: 日数不足です。")
        input("Enterキーで終了...")
        return

    # ========================================================
    # バックテスト
    # ========================================================

    all_results = []

    top_values = [
        5,
        10,
        20,
        30
    ]

    # 最初の日は予測不能
    target_dates = dates[1:]

    print()
    print(
        f"バックテスト日数: "
        f"{len(target_dates)}日"
    )

    for target_date_raw in target_dates:

        target_date = pd.Timestamp(
            target_date_raw
        )

        previous_date = (
            target_date
            - pd.Timedelta(days=1)
        )

        # ----------------------------------------------------
        # 予測対象日の前日まで
        # ----------------------------------------------------

        history = work[
            work["日付"] < target_date
        ].copy()

        actual = work[
            work["日付"] == target_date
        ].copy()

        if history.empty or actual.empty:

            continue

        # ----------------------------------------------------
        # 前日に存在していた台を候補とする
        # ----------------------------------------------------

        previous_day = history[
            history["日付"]
            == history["日付"].max()
        ]

        candidates = (
            previous_day[
                [
                    "台番号",
                    "機種"
                ]
            ]
            .drop_duplicates(
                subset=["台番号"]
            )
        )

        if candidates.empty:

            continue

        # ----------------------------------------------------
        # V5再計算
        # ----------------------------------------------------

        prediction = calculate_v5_scores(
            history,
            candidates,
            target_date
        )

        if prediction.empty:

            continue

        print()
        print("=" * 70)
        print(
            f"【V5バックテスト】"
            f"{target_date.strftime('%Y-%m-%d')}"
        )
        print(
            f"使用履歴: "
            f"{history['日付'].min().strftime('%Y-%m-%d')}"
            f" ～ "
            f"{history['日付'].max().strftime('%Y-%m-%d')}"
        )
        print("=" * 70)

        for top_n in top_values:

            r = evaluate_top(
                prediction,
                actual,
                top_n
            )

            if r is None:

                continue

            r["予測日"] = (
                target_date.strftime(
                    "%Y-%m-%d"
                )
            )

            all_results.append(r)

            print_result(
                f"TOP{top_n}",
                r
            )

    # ========================================================
    # 結果確認
    # ========================================================

    if not all_results:

        print()
        print("バックテスト結果がありません。")
        input("Enterキーで終了...")
        return

    result_df = pd.DataFrame(
        all_results
    )

    # ========================================================
    # 総合結果
    # ========================================================

    print()
    print("=" * 70)
    print("【V5バックテスト総合結果】")
    print("=" * 70)

    summary_rows = []

    for top_n in top_values:

        g = result_df[
            result_df["TOP"] == top_n
        ]

        if g.empty:
            continue

        avg_diff = safe_float(
            g["実績平均差枚"].mean()
        )

        plus_rate = safe_float(
            g["実績プラス率"].mean()
        )

        rate500 = safe_float(
            g["+500率"].mean()
        )

        rate1000 = safe_float(
            g["+1000率"].mean()
        )

        rate2000 = safe_float(
            g["+2000率"].mean()
        )

        rate3000 = safe_float(
            g["+3000率"].mean()
        )

        machine_win = safe_float(
            g["機種勝率"].mean()
        )

        print()

        print(
            f"TOP{top_n} / "
            f"評価日数 {len(g)}日 / "
            f"平均差枚 "
            f"{avg_diff:+.1f}枚 / "
            f"プラス率 "
            f"{plus_rate:.1f}% / "
            f"+500率 "
            f"{rate500:.1f}% / "
            f"+1000率 "
            f"{rate1000:.1f}% / "
            f"+2000率 "
            f"{rate2000:.1f}% / "
            f"+3000率 "
            f"{rate3000:.1f}% / "
            f"機種勝率 "
            f"{machine_win:.1f}%"
        )

        summary_rows.append({

            "TOP":
                top_n,

            "評価日数":
                len(g),

            "平均差枚":
                avg_diff,

            "プラス率":
                plus_rate,

            "+500率":
                rate500,

            "+1000率":
                rate1000,

            "+2000率":
                rate2000,

            "+3000率":
                rate3000,

            "機種勝率":
                machine_win
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # ベストTOP
    # ========================================================

    if not summary_df.empty:

        # 平均差枚を最優先
        best_row = (
            summary_df
            .sort_values(
                [
                    "平均差枚",
                    "+1000率",
                    "プラス率"
                ],
                ascending=False
            )
            .iloc[0]
        )

        print()
        print("=" * 70)
        print("【V5 ベストTOP】")
        print("=" * 70)

        print(
            f"TOP{int(best_row['TOP'])}"
        )

        print(
            f"平均差枚: "
            f"{best_row['平均差枚']:+.1f}枚"
        )

        print(
            f"プラス率: "
            f"{best_row['プラス率']:.1f}%"
        )

        print(
            f"+500率: "
            f"{best_row['+500率']:.1f}%"
        )

        print(
            f"+1000率: "
            f"{best_row['+1000率']:.1f}%"
        )

        print(
            f"+2000率: "
            f"{best_row['+2000率']:.1f}%"
        )

        print(
            f"+3000率: "
            f"{best_row['+3000率']:.1f}%"
        )

        print(
            f"機種勝率: "
            f"{best_row['機種勝率']:.1f}%"
        )

    # ========================================================
    # CSV保存
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    try:

        result_df.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print()
        print("★ CSV保存成功")
        print(OUTPUT_FILE)

    except Exception as e:

        print()
        print("CSV保存エラー:")
        print(e)

    try:

        summary_df.to_csv(
            SUMMARY_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print()
        print("★ CSV保存成功")
        print(SUMMARY_FILE)

    except Exception as e:

        print()
        print("サマリー保存エラー:")
        print(e)

    # ========================================================
    # 完了
    # ========================================================

    print()
    print("=" * 70)
    print("★★★★★ 投入パターン V5 バックテスト完了 ★★★★★")
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