# -*- coding: utf-8 -*-

"""
SlotAnalyzer
投入パターン予測 V5

V3・V4・要因分析の結果を反映した新予測モデル。

主な評価要因
1. 台_前回変化
2. 台_前日差枚
3. 台_凹み
4. 機種_前日差枚
5. 機種_直近3日平均
6. 機種_過去平均差枚
7. 曜日
8. 台番号帯
9. 台番号偶奇

重要:
予測対象日の実績はスコア計算に使用しない。
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 設定
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    "prediction_v5.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "prediction_v5_summary.csv"
)


# ============================================================
# 表示
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


# ============================================================
# 列名確認
# ============================================================

def find_column(df, candidates):

    for col in candidates:
        if col in df.columns:
            return col

    return None


# ============================================================
# 正規化
# ============================================================

def normalize_0_100(value, low, high):

    if high == low:
        return 50.0

    score = (value - low) / (high - low) * 100

    return float(np.clip(score, 0, 100))


# ============================================================
# メイン
# ============================================================

def main():

    print_line()
    print("投入パターン予測 V5")
    print_line()

    print()
    print("V3・V4・要因分析の結果を反映したV5モデルです。")
    print()
    print("予測対象日の実績は特徴量計算に使用しません。")
    print()

    print("入力ファイル:")
    print(INPUT_FILE)

    if not os.path.exists(INPUT_FILE):

        print()
        print("ERROR: all_data.csv が見つかりません。")
        input("Enterキーで終了...")
        return

    # ========================================================
    # データ読み込み
    # ========================================================

    print()
    print("all_data.csv を読み込みます...")

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

    print(f"読み込みデータ: {len(df):,}行")

    # ========================================================
    # 列を探す
    # ========================================================

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
            "機種名 ",
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

    print()
    print("必要な列を確認します...")

    print(
        f"日付   : "
        f"{date_col if date_col else '見つかりません'}"
    )

    print(
        f"台番号 : "
        f"{number_col if number_col else '見つかりません'}"
    )

    print(
        f"機種   : "
        f"{machine_col if machine_col else '見つかりません'}"
    )

    print(
        f"差枚   : "
        f"{diff_col if diff_col else '見つかりません'}"
    )

    if not all([
        date_col,
        number_col,
        machine_col,
        diff_col
    ]):

        print()
        print("ERROR: 必要な列が見つかりません。")
        input("Enterキーで終了...")
        return

    print()
    print("必要な列: OK")

    # ========================================================
    # 必要データ整理
    # ========================================================

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

    work["台番号"] = work["台番号"].astype(int)

    work = work.sort_values(
        [
            "日付",
            "台番号"
        ]
    ).reset_index(drop=True)

    print(
        f"有効データ: {len(work):,}行"
    )

    if len(work) == 0:

        print()
        print("ERROR: 有効なデータがありません。")
        input("Enterキーで終了...")
        return

    # ========================================================
    # 収録日
    # ========================================================

    dates = sorted(
        work["日付"].dt.normalize().unique()
    )

    print()
    print(f"収録日数: {len(dates)}")

    print(
        "収録日:"
    )

    print(
        " / ".join(
            pd.Timestamp(d).strftime("%Y-%m-%d")
            for d in dates
        )
    )

    if len(dates) < 2:

        print()
        print("ERROR: 予測に必要な日数が不足しています。")
        input("Enterキーで終了...")
        return

    # ========================================================
    # 次回予測日
    # ========================================================

    latest_date = pd.Timestamp(dates[-1])

    prediction_date = latest_date + pd.Timedelta(days=1)

    print()
    print("最新実績日:")
    print(
        latest_date.strftime("%Y-%m-%d")
    )

    print()
    print("次回予測日:")
    print(
        prediction_date.strftime("%Y-%m-%d")
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

    target_weekday = weekday_jp[
        prediction_date.weekday()
    ]

    print(
        f"予測曜日: {target_weekday}"
    )

    # ========================================================
    # 最新日までの履歴だけを使用
    # ========================================================

    history = work[
        work["日付"] <= latest_date
    ].copy()

    # ========================================================
    # 台別・機種別特徴量
    # ========================================================

    machine_groups = {}

    for machine_name, g in history.groupby("機種"):

        machine_groups[machine_name] = g.sort_values(
            "日付"
        )

    machine_number_groups = {}

    for number, g in history.groupby("台番号"):

        machine_number_groups[number] = g.sort_values(
            "日付"
        )

    # ========================================================
    # 直近データ取得関数
    # ========================================================

    def get_machine_history(machine_name):

        if machine_name not in machine_groups:

            return pd.DataFrame(
                columns=history.columns
            )

        return machine_groups[machine_name]

    def get_number_history(number):

        if number not in machine_number_groups:

            return pd.DataFrame(
                columns=history.columns
            )

        return machine_number_groups[number]

    # ========================================================
    # 台の特徴量
    # ========================================================

    def machine_number_features(number):

        g = get_number_history(number)

        if len(g) == 0:

            return {
                "台_過去平均差枚": 0.0,
                "台_過去プラス率": 50.0,
                "台_直近3日平均": 0.0,
                "台_前日差枚": 0.0,
                "台_前回差枚": 0.0,
                "台_前回変化": 0.0,
                "台_凹み": 0.0,
                "台_過去+1000率": 0.0,
                "台_過去+2000率": 0.0,
            }

        values = g["差枚"].astype(float)

        last = safe_float(
            values.iloc[-1]
        )

        previous = (
            safe_float(values.iloc[-2])
            if len(values) >= 2
            else last
        )

        avg = safe_float(
            values.mean()
        )

        recent3 = safe_float(
            values.tail(3).mean()
        )

        plus_rate = safe_float(
            (values > 0).mean() * 100,
            50.0
        )

        rate1000 = safe_float(
            (values >= 1000).mean() * 100
        )

        rate2000 = safe_float(
            (values >= 2000).mean() * 100
        )

        change = last - previous

        # 過去平均との差
        # プラスほど「前日が過去平均より下」
        drawdown = avg - last

        return {
            "台_過去平均差枚": avg,
            "台_過去プラス率": plus_rate,
            "台_直近3日平均": recent3,
            "台_前日差枚": last,
            "台_前回差枚": previous,
            "台_前回変化": change,
            "台_凹み": drawdown,
            "台_過去+1000率": rate1000,
            "台_過去+2000率": rate2000,
        }

    # ========================================================
    # 機種の特徴量
    # ========================================================

    def machine_features(machine_name):

        g = get_machine_history(machine_name)

        if len(g) == 0:

            return {
                "機種_過去平均差枚": 0.0,
                "機種_過去プラス率": 50.0,
                "機種_直近3日平均": 0.0,
                "機種_前日差枚": 0.0,
                "機種_前回差枚": 0.0,
                "機種_前回変化": 0.0,
                "機種_凹み": 0.0,
                "機種_過去+1000率": 0.0,
                "機種_過去+2000率": 0.0,
            }

        values = g["差枚"].astype(float)

        last = safe_float(
            values.iloc[-1]
        )

        previous = (
            safe_float(values.iloc[-2])
            if len(values) >= 2
            else last
        )

        avg = safe_float(
            values.mean()
        )

        recent3 = safe_float(
            values.tail(3).mean()
        )

        plus_rate = safe_float(
            (values > 0).mean() * 100,
            50.0
        )

        rate1000 = safe_float(
            (values >= 1000).mean() * 100
        )

        rate2000 = safe_float(
            (values >= 2000).mean() * 100
        )

        change = last - previous

        drawdown = avg - last

        return {
            "機種_過去平均差枚": avg,
            "機種_過去プラス率": plus_rate,
            "機種_直近3日平均": recent3,
            "機種_前日差枚": last,
            "機種_前回差枚": previous,
            "機種_前回変化": change,
            "機種_凹み": drawdown,
            "機種_過去+1000率": rate1000,
            "機種_過去+2000率": rate2000,
        }

    # ========================================================
    # 曜日実績
    # ========================================================

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
            [
                r["avg"]
                for r in rows
            ]
        )

        plus = np.mean(
            [
                r["plus"]
                for r in rows
            ]
        )

        # 平均差枚を中心に評価
        # 過去全体の平均を基準に±150枚程度で正規化
        weekday_score[wd] = normalize_0_100(
            avg,
            -300,
            100
        ) * 0.7 + normalize_0_100(
            plus,
            20,
            45
        ) * 0.3

    # ========================================================
    # 台番号帯
    # ========================================================

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

    # 今回の分析結果を反映
    band_bonus = {
        "500～599": 35.0,
        "600～699": 45.0,
        "700～799": 58.0,
        "800～899": 60.0,
        "900～999": 35.0,
        "1000～1099": 30.0,
        "その他": 40.0,
    }

    # ========================================================
    # 候補台
    # ========================================================

    # 最新実績日に存在した台を候補とする
    latest_rows = history[
        history["日付"] == latest_date
    ].copy()

    candidates = (
        latest_rows[
            [
                "台番号",
                "機種"
            ]
        ]
        .drop_duplicates(
            subset=["台番号"]
        )
        .sort_values("台番号")
    )

    print()
    print(
        f"候補台数: {len(candidates):,}台"
    )

    # ========================================================
    # スコア計算
    # ========================================================

    results = []

    for _, row in candidates.iterrows():

        number = int(
            row["台番号"]
        )

        machine_name = str(
            row["機種"]
        )

        nf = machine_number_features(
            number
        )

        mf = machine_features(
            machine_name
        )

        # ----------------------------------------------------
        # ① 台_前回変化
        # ----------------------------------------------------
        # 分析結果:
        # 0～+1000枚が最も良好
        # ----------------------------------------------------

        number_change = nf[
            "台_前回変化"
        ]

        if 0 <= number_change <= 1000:
            score_change = 90.0
        elif -500 <= number_change < 0:
            score_change = 60.0
        elif 1000 < number_change <= 2000:
            score_change = 55.0
        elif number_change < -1000:
            score_change = 25.0
        else:
            score_change = 45.0

        # ----------------------------------------------------
        # ② 台_前日差枚
        # ----------------------------------------------------

        yesterday = nf[
            "台_前日差枚"
        ]

        if 0 <= yesterday <= 1000:
            score_yesterday = 90.0
        elif -500 <= yesterday < 0:
            score_yesterday = 60.0
        elif 1000 < yesterday <= 2000:
            score_yesterday = 55.0
        elif yesterday < -1000:
            score_yesterday = 25.0
        else:
            score_yesterday = 45.0

        # ----------------------------------------------------
        # ③ 台_凹み
        # ----------------------------------------------------
        # 今回の相関はマイナス。
        # 過度な凹みを狙い打ちしない。
        # ----------------------------------------------------

        drawdown = nf[
            "台_凹み"
        ]

        if -1000 <= drawdown <= 0:
            score_drawdown = 75.0
        elif drawdown < -1000:
            score_drawdown = 50.0
        elif 0 < drawdown <= 1000:
            score_drawdown = 40.0
        else:
            score_drawdown = 25.0

        # ----------------------------------------------------
        # ④ 機種_前日差枚
        # ----------------------------------------------------

        machine_yesterday = mf[
            "機種_前日差枚"
        ]

        if 0 <= machine_yesterday <= 1000:
            score_machine_yesterday = 90.0
        elif -500 <= machine_yesterday < 0:
            score_machine_yesterday = 60.0
        elif 1000 < machine_yesterday <= 2000:
            score_machine_yesterday = 55.0
        elif machine_yesterday < -1000:
            score_machine_yesterday = 25.0
        else:
            score_machine_yesterday = 45.0

        # ----------------------------------------------------
        # ⑤ 機種_直近3日平均
        # ----------------------------------------------------

        machine_recent3 = mf[
            "機種_直近3日平均"
        ]

        score_recent3 = normalize_0_100(
            machine_recent3,
            -1000,
            1000
        )

        # ----------------------------------------------------
        # ⑥ 機種_過去平均差枚
        # ----------------------------------------------------

        machine_avg = mf[
            "機種_過去平均差枚"
        ]

        score_machine_avg = normalize_0_100(
            machine_avg,
            -1000,
            1000
        )

        # ----------------------------------------------------
        # ⑦ 曜日
        # ----------------------------------------------------

        score_weekday = weekday_score.get(
            prediction_date.weekday(),
            50.0
        )

        # ----------------------------------------------------
        # ⑧ 台番号帯
        # ----------------------------------------------------

        band = number_band(
            number
        )

        score_band = band_bonus.get(
            band,
            40.0
        )

        # ----------------------------------------------------
        # ⑨ 奇数・偶数
        # ----------------------------------------------------

        if number % 2 == 1:
            score_parity = 55.0
        else:
            score_parity = 45.0

        # ====================================================
        # V5総合スコア
        # ====================================================

        score = (

            # 最重要
            score_change * 0.22

            # 台前日差枚
            + score_yesterday * 0.18

            # 台凹み
            + score_drawdown * 0.12

            # 機種前日差枚
            + score_machine_yesterday * 0.18

            # 機種直近
            + score_recent3 * 0.10

            # 機種過去平均
            + score_machine_avg * 0.06

            # 曜日
            + score_weekday * 0.06

            # 台番号帯
            + score_band * 0.05

            # 奇偶
            + score_parity * 0.03
        )

        score = float(
            np.clip(
                score,
                0,
                100
            )
        )

        # ----------------------------------------------------
        # ランク
        # ----------------------------------------------------

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
                prediction_date.strftime(
                    "%Y-%m-%d"
                ),

            "台番号":
                number,

            "機種":
                machine_name,

            "台_過去平均差枚":
                nf["台_過去平均差枚"],

            "台_過去プラス率":
                nf["台_過去プラス率"],

            "台_直近3日平均":
                nf["台_直近3日平均"],

            "台_前日差枚":
                nf["台_前日差枚"],

            "台_前回差枚":
                nf["台_前回差枚"],

            "台_前回変化":
                nf["台_前回変化"],

            "台_凹み":
                nf["台_凹み"],

            "機種_過去平均差枚":
                mf["機種_過去平均差枚"],

            "機種_過去プラス率":
                mf["機種_過去プラス率"],

            "機種_直近3日平均":
                mf["機種_直近3日平均"],

            "機種_前日差枚":
                mf["機種_前日差枚"],

            "機種_前回変化":
                mf["機種_前回変化"],

            "機種_凹み":
                mf["機種_凹み"],

            "曜日":
                target_weekday,

            "台番号帯":
                band,

            "台番号偶奇":
                "奇数" if number % 2 else "偶数",

            "前回変化スコア":
                score_change,

            "前日差枚スコア":
                score_yesterday,

            "台凹みスコア":
                score_drawdown,

            "機種前日差枚スコア":
                score_machine_yesterday,

            "機種直近3日スコア":
                score_recent3,

            "機種過去平均スコア":
                score_machine_avg,

            "曜日スコア":
                score_weekday,

            "台番号帯スコア":
                score_band,

            "奇偶スコア":
                score_parity,

            "V5総合スコア":
                score,

            "ランク":
                rank
        })

    # ========================================================
    # DataFrame
    # ========================================================

    result = pd.DataFrame(
        results
    )

    if result.empty:

        print()
        print("ERROR: 予測結果が作成できませんでした。")
        input("Enterキーで終了...")
        return

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

    # ========================================================
    # TOP30表示
    # ========================================================

    print()
    print_line()
    print("【次回おすすめ台 TOP30】")
    print_line()

    top30 = result.head(30)

    for _, r in top30.iterrows():

        print(
            f"{int(r['予測順位']):2d}. "
            f"{int(r['台番号'])} "
            f"{r['機種']} / "
            f"V5 {r['V5総合スコア']:.1f} / "
            f"{r['ランク']} / "
            f"前回変化 "
            f"{r['台_前回変化']:+.0f}枚 / "
            f"前日 "
            f"{r['台_前日差枚']:+.0f}枚 / "
            f"機種前日 "
            f"{r['機種_前日差枚']:+.0f}枚"
        )

    # ========================================================
    # ランク別
    # ========================================================

    print()
    print_line()
    print("【ランク別台数】")
    print_line()

    for rank in [
        "S",
        "A",
        "B",
        "C",
        "D",
        "E"
    ]:

        count = int(
            (result["ランク"] == rank).sum()
        )

        print(
            f"{rank}: {count}台"
        )

    # ========================================================
    # 機種別TOP20
    # ========================================================

    print()
    print_line()
    print("【機種別おすすめ TOP20】")
    print_line()

    machine_result = (
        result
        .groupby("機種")
        .agg(
            台数=("台番号", "count"),
            平均スコア=("V5総合スコア", "mean"),
            最高スコア=("V5総合スコア", "max"),
            平均前日差枚=("機種_前日差枚", "mean"),
            平均直近3日=("機種_直近3日平均", "mean")
        )
        .reset_index()
        .sort_values(
            [
                "平均スコア",
                "最高スコア"
            ],
            ascending=False
        )
        .reset_index(drop=True)
    )

    for i, (_, r) in enumerate(
        machine_result.head(20).iterrows(),
        start=1
    ):

        print(
            f"{i:2d}. "
            f"{r['機種']} / "
            f"{int(r['台数'])}台 / "
            f"平均V5 "
            f"{r['平均スコア']:.1f} / "
            f"最高V5 "
            f"{r['最高スコア']:.1f} / "
            f"機種前日 "
            f"{r['平均前日差枚']:+.0f}枚"
        )

    # ========================================================
    # 特に注目する条件
    # ========================================================

    print()
    print_line()
    print("【V5重点条件該当台】")
    print_line()

    focus = result[
        (
            (result["台_前回変化"] >= 0)
            &
            (result["台_前回変化"] <= 1000)
        )
        &
        (
            (result["台_前日差枚"] >= 0)
            &
            (result["台_前日差枚"] <= 1000)
        )
        &
        (
            (result["機種_前日差枚"] >= 0)
            &
            (result["機種_前日差枚"] <= 1000)
        )
    ]

    if len(focus) == 0:

        print(
            "3条件すべてを満たす台はありません。"
        )

    else:

        focus = focus.sort_values(
            "V5総合スコア",
            ascending=False
        )

        for _, r in focus.head(20).iterrows():

            print(
                f"台番号 {int(r['台番号'])} / "
                f"{r['機種']} / "
                f"V5 {r['V5総合スコア']:.1f} / "
                f"前回変化 {r['台_前回変化']:+.0f}枚 / "
                f"前日 {r['台_前日差枚']:+.0f}枚 / "
                f"機種前日 {r['機種_前日差枚']:+.0f}枚"
            )

    # ========================================================
    # CSV保存
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    try:

        result.to_csv(
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

    # ========================================================
    # サマリー
    # ========================================================

    summary_rows = [

        {
            "項目": "予測日",
            "値": prediction_date.strftime(
                "%Y-%m-%d"
            )
        },

        {
            "項目": "予測曜日",
            "値": target_weekday
        },

        {
            "項目": "候補台数",
            "値": len(result)
        },

        {
            "項目": "TOP5平均V5",
            "値": result.head(5)[
                "V5総合スコア"
            ].mean()
        },

        {
            "項目": "TOP10平均V5",
            "値": result.head(10)[
                "V5総合スコア"
            ].mean()
        },

        {
            "項目": "TOP20平均V5",
            "値": result.head(20)[
                "V5総合スコア"
            ].mean()
        },

        {
            "項目": "TOP30平均V5",
            "値": result.head(30)[
                "V5総合スコア"
            ].mean()
        },

        {
            "項目": "Sランク台数",
            "値": int(
                (result["ランク"] == "S").sum()
            )
        },

        {
            "項目": "Aランク台数",
            "値": int(
                (result["ランク"] == "A").sum()
            )
        },

        {
            "項目": "Bランク台数",
            "値": int(
                (result["ランク"] == "B").sum()
            )
        },

        {
            "項目": "Cランク台数",
            "値": int(
                (result["ランク"] == "C").sum()
            )
        },

        {
            "項目": "重点条件該当台数",
            "値": len(focus)
        }
    ]

    summary = pd.DataFrame(
        summary_rows
    )

    try:

        summary.to_csv(
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
    print_line()
    print("★★★★★ V5予測 完了 ★★★★★")
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