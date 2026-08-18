# -*- coding: utf-8 -*-

"""
============================================================
投入パターン分析 V1
============================================================

目的:
V5を作る前に、店舗の翌日投入パターンを分析する。

分析項目:
・曜日
・前日差枚
・前日大幅プラス/マイナス
・台番号の偶奇
・台番号下一桁
・台番号帯
・機種
・機種前日差枚
・機種直近3日平均
・台の凹み
・台の前回変化

重要:
予測対象日の実績は、その日の特徴量計算には使用しない。
============================================================
"""

import os
import numpy as np
import pandas as pd


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
    "prediction_pattern_analysis.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "prediction_pattern_analysis_summary.csv"
)


# ============================================================
# 表示関数
# ============================================================

def print_line():
    print("=" * 70)


def print_title(text):
    print_line()
    print(text)
    print_line()


def safe_mean(series):
    if series is None:
        return np.nan

    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) == 0:
        return np.nan

    return float(s.mean())


def safe_rate(series, condition):
    if series is None:
        return np.nan

    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) == 0:
        return np.nan

    return float(condition(s).mean() * 100)


def format_num(value):
    if pd.isna(value):
        return "N/A"

    return f"{value:+.1f}"


def format_rate(value):
    if pd.isna(value):
        return "N/A"

    return f"{value:.1f}%"


# ============================================================
# CSV読み込み
# ============================================================

print_title("投入パターン分析 V1")

print("V5を作る前に、店舗の翌日投入パターンを分析します。")
print()
print("※予測対象日の実績は特徴量計算には使用しません。")
print()

print("入力ファイル:")
print(DATA_FILE)
print()

if not os.path.exists(DATA_FILE):
    print("ERROR: all_data.csv が見つかりません。")
    input("Enterキーで終了...")
    raise SystemExit


df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

print(f"読み込みデータ: {len(df):,}行")
print()


# ============================================================
# 列確認
# ============================================================

print("必要な列を確認します...")

required_columns = [
    "日付",
    "台番号",
    "機種名",
    "差枚"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print()
    print("ERROR: 必要な列がありません。")
    print("不足列:")
    for col in missing_columns:
        print(f"  {col}")

    print()
    print("現在の列:")
    for col in df.columns:
        print(f"  {col}")

    input("Enterキーで終了...")
    raise SystemExit


print("日付   : OK")
print("台番号 : OK")
print("機種   : OK")
print("差枚   : OK")
print()
print("必要な列: OK")


# ============================================================
# データ整形
# ============================================================

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

df["機種名"] = df["機種名"].astype(str).str.strip()

df = df.dropna(
    subset=[
        "日付",
        "台番号",
        "差枚",
        "機種名"
    ]
).copy()

df["台番号"] = df["台番号"].astype(int)

df = df.sort_values(
    ["日付", "台番号"]
).reset_index(drop=True)

print(f"有効データ: {len(df):,}行")


# ============================================================
# 基本情報
# ============================================================

dates = sorted(df["日付"].unique())

print()
print(f"収録日数: {len(dates)}")

print(
    "収録日: "
    + " / ".join(
        pd.Timestamp(d).strftime("%Y-%m-%d")
        for d in dates
    )
)

if len(dates) < 4:
    print()
    print("ERROR: 分析に必要な日数が不足しています。")
    input("Enterキーで終了...")
    raise SystemExit


# ============================================================
# 曜日マップ
# ============================================================

weekday_names = {
    0: "月曜日",
    1: "火曜日",
    2: "水曜日",
    3: "木曜日",
    4: "金曜日",
    5: "土曜日",
    6: "日曜日"
}


# ============================================================
# 前日データを作成
# ============================================================

print()
print_title("翌日予測用特徴量作成")

records = []


# ============================================================
# 日付ごとに特徴量を作る
# ============================================================

for target_index in range(1, len(dates)):

    target_date = pd.Timestamp(dates[target_index])

    previous_dates = [
        pd.Timestamp(x)
        for x in dates
        if pd.Timestamp(x) < target_date
    ]

    if len(previous_dates) == 0:
        continue

    previous_date = previous_dates[-1]

    target_df = df[
        df["日付"] == target_date
    ].copy()

    history_df = df[
        df["日付"] < target_date
    ].copy()

    previous_df = df[
        df["日付"] == previous_date
    ].copy()

    # --------------------------------------------------------
    # 台ごとの特徴量
    # --------------------------------------------------------

    machine_groups = history_df.groupby("台番号")

    machine_stats = machine_groups["差枚"].agg(
        台_過去平均差枚="mean",
        台_過去データ数="count"
    ).reset_index()

    machine_positive = (
        history_df
        .groupby("台番号")["差枚"]
        .apply(lambda x: (x > 0).mean() * 100)
        .reset_index(name="台_過去プラス率")
    )

    machine_stats = machine_stats.merge(
        machine_positive,
        on="台番号",
        how="left"
    )

    # --------------------------------------------------------
    # 台の前日差枚
    # --------------------------------------------------------

    previous_machine = previous_df[
        ["台番号", "差枚"]
    ].rename(
        columns={
            "差枚": "台_前日差枚"
        }
    )

    machine_stats = machine_stats.merge(
        previous_machine,
        on="台番号",
        how="left"
    )

    # --------------------------------------------------------
    # 台の直近3日平均
    # --------------------------------------------------------

    recent_machine_records = []

    for machine_no, group in history_df.groupby("台番号"):

        group = group.sort_values("日付")

        recent = group.tail(3)

        if len(recent) > 0:
            recent_mean = recent["差枚"].mean()
        else:
            recent_mean = np.nan

        recent_machine_records.append(
            {
                "台番号": machine_no,
                "台_直近3日平均": recent_mean
            }
        )

    recent_machine_df = pd.DataFrame(
        recent_machine_records
    )

    machine_stats = machine_stats.merge(
        recent_machine_df,
        on="台番号",
        how="left"
    )

    # --------------------------------------------------------
    # 台の前回変化
    #
    # 「前回の差枚」と「その1つ前の差枚」の変化
    # --------------------------------------------------------

    change_records = []

    for machine_no, group in history_df.groupby("台番号"):

        group = group.sort_values("日付")

        values = group["差枚"].dropna().tolist()

        if len(values) >= 2:

            previous_value = values[-1]
            before_value = values[-2]

            change = previous_value - before_value

        else:
            change = np.nan

        change_records.append(
            {
                "台番号": machine_no,
                "台_前回変化": change
            }
        )

    change_df = pd.DataFrame(change_records)

    machine_stats = machine_stats.merge(
        change_df,
        on="台番号",
        how="left"
    )

    # --------------------------------------------------------
    # 台の凹み
    #
    # 過去平均 - 前日差枚
    # --------------------------------------------------------

    machine_stats["台_凹み"] = (
        machine_stats["台_過去平均差枚"]
        - machine_stats["台_前日差枚"]
    )

    # --------------------------------------------------------
    # 台番号特徴
    # --------------------------------------------------------

    machine_stats["台_偶奇"] = (
        machine_stats["台番号"] % 2
    )

    machine_stats["台_下一桁"] = (
        machine_stats["台番号"] % 10
    )

    machine_stats["台_番号帯"] = (
        machine_stats["台番号"] // 100
    )

    # --------------------------------------------------------
    # 機種ごとの特徴
    # --------------------------------------------------------

    model_stats = (
        history_df
        .groupby("機種名")["差枚"]
        .agg(
            機種_過去平均差枚="mean",
            機種_過去データ数="count"
        )
        .reset_index()
    )

    model_positive = (
        history_df
        .groupby("機種名")["差枚"]
        .apply(lambda x: (x > 0).mean() * 100)
        .reset_index(
            name="機種_過去プラス率"
        )
    )

    model_stats = model_stats.merge(
        model_positive,
        on="機種名",
        how="left"
    )

    # --------------------------------------------------------
    # 機種の前日差枚
    # --------------------------------------------------------

    previous_model = (
        previous_df
        .groupby("機種名")["差枚"]
        .mean()
        .reset_index(
            name="機種_前日差枚"
        )
    )

    model_stats = model_stats.merge(
        previous_model,
        on="機種名",
        how="left"
    )

    # --------------------------------------------------------
    # 機種直近3日平均
    # --------------------------------------------------------

    model_recent_records = []

    for model_name, group in history_df.groupby("機種名"):

        group = group.sort_values("日付")

        recent_dates = (
            group["日付"]
            .drop_duplicates()
            .sort_values()
            .tail(3)
        )

        recent = group[
            group["日付"].isin(recent_dates)
        ]

        if len(recent) > 0:
            recent_mean = recent["差枚"].mean()
        else:
            recent_mean = np.nan

        model_recent_records.append(
            {
                "機種名": model_name,
                "機種_直近3日平均": recent_mean
            }
        )

    model_recent_df = pd.DataFrame(
        model_recent_records
    )

    model_stats = model_stats.merge(
        model_recent_df,
        on="機種名",
        how="left"
    )

    # --------------------------------------------------------
    # 機種の前回変化
    # --------------------------------------------------------

    model_change_records = []

    for model_name, group in history_df.groupby("機種名"):

        daily = (
            group
            .groupby("日付")["差枚"]
            .mean()
            .sort_index()
        )

        values = daily.dropna().tolist()

        if len(values) >= 2:

            change = (
                values[-1]
                - values[-2]
            )

        else:
            change = np.nan

        model_change_records.append(
            {
                "機種名": model_name,
                "機種_前回変化": change
            }
        )

    model_change_df = pd.DataFrame(
        model_change_records
    )

    model_stats = model_stats.merge(
        model_change_df,
        on="機種名",
        how="left"
    )

    # --------------------------------------------------------
    # 機種の凹み
    # --------------------------------------------------------

    model_stats["機種_凹み"] = (
        model_stats["機種_過去平均差枚"]
        - model_stats["機種_前日差枚"]
    )

    # --------------------------------------------------------
    # 対象日の実績を結合
    #
    # ここで初めて当日差枚を追加する。
    # 特徴量には使用しない。
    # --------------------------------------------------------

    target_df = target_df[
        [
            "日付",
            "台番号",
            "機種名",
            "差枚"
        ]
    ].rename(
        columns={
            "差枚": "当日差枚"
        }
    )

    feature_df = target_df.merge(
        machine_stats,
        on="台番号",
        how="left"
    )

    feature_df = feature_df.merge(
        model_stats,
        on="機種名",
        how="left",
        suffixes=("", "_model")
    )

    feature_df["予測日"] = target_date.strftime(
        "%Y-%m-%d"
    )

    feature_df["曜日"] = (
        weekday_names[target_date.weekday()]
    )

    feature_df["前日"] = previous_date.strftime(
        "%Y-%m-%d"
    )

    records.append(feature_df)


    print(
        f"特徴量作成: "
        f"{target_date.strftime('%Y-%m-%d')}"
    )


# ============================================================
# 結合
# ============================================================

if len(records) == 0:

    print()
    print("特徴量を作成できませんでした。")
    input("Enterキーで終了...")
    raise SystemExit


analysis_df = pd.concat(
    records,
    ignore_index=True
)


# ============================================================
# 数値列
# ============================================================

numeric_columns = [
    "当日差枚",
    "台_過去平均差枚",
    "台_過去プラス率",
    "台_前日差枚",
    "台_直近3日平均",
    "台_前回変化",
    "台_凹み",
    "台_偶奇",
    "台_下一桁",
    "台_番号帯",
    "機種_過去平均差枚",
    "機種_過去プラス率",
    "機種_前日差枚",
    "機種_直近3日平均",
    "機種_前回変化",
    "機種_凹み"
]

for col in numeric_columns:

    if col in analysis_df.columns:

        analysis_df[col] = pd.to_numeric(
            analysis_df[col],
            errors="coerce"
        )


# ============================================================
# 有効データ
# ============================================================

analysis_df = analysis_df.dropna(
    subset=["当日差枚"]
).copy()

print()
print(f"特徴量データ: {len(analysis_df):,}行")
print(
    f"予測日数: "
    f"{analysis_df['予測日'].nunique()}日"
)


# ============================================================
# ① 曜日別分析
# ============================================================

print_title("曜日別 投入傾向")

weekday_rows = []

weekday_order = [
    "月曜日",
    "火曜日",
    "水曜日",
    "木曜日",
    "金曜日",
    "土曜日",
    "日曜日"
]

for weekday in weekday_order:

    sub = analysis_df[
        analysis_df["曜日"] == weekday
    ]

    if len(sub) == 0:
        continue

    avg = safe_mean(
        sub["当日差枚"]
    )

    positive = safe_rate(
        sub["当日差枚"],
        lambda x: x > 0
    )

    rate500 = safe_rate(
        sub["当日差枚"],
        lambda x: x >= 500
    )

    rate1000 = safe_rate(
        sub["当日差枚"],
        lambda x: x >= 1000
    )

    rate2000 = safe_rate(
        sub["当日差枚"],
        lambda x: x >= 2000
    )

    print(
        f"{weekday} / "
        f"n={len(sub)} / "
        f"平均差枚 {format_num(avg)}枚 / "
        f"プラス率 {format_rate(positive)} / "
        f"+500率 {format_rate(rate500)} / "
        f"+1000率 {format_rate(rate1000)} / "
        f"+2000率 {format_rate(rate2000)}"
    )

    weekday_rows.append(
        {
            "分析種別": "曜日",
            "条件": weekday,
            "n": len(sub),
            "平均差枚": avg,
            "プラス率": positive,
            "+500率": rate500,
            "+1000率": rate1000,
            "+2000率": rate2000
        }
    )


# ============================================================
# 条件分析関数
# ============================================================

def analyze_condition(
    name,
    condition,
    category="条件"
):

    sub = analysis_df[
        condition
    ].copy()

    if len(sub) == 0:
        return None

    avg = safe_mean(
        sub["当日差枚"]
    )

    positive = safe_rate(
        sub["当日差枚"],
        lambda x: x > 0
    )

    rate500 = safe_rate(
        sub["当日差枚"],
        lambda x: x >= 500
    )

    rate1000 = safe_rate(
        sub["当日差枚"],
        lambda x: x >= 1000
    )

    rate2000 = safe_rate(
        sub["当日差枚"],
        lambda x: x >= 2000
    )

    return {
        "分析種別": category,
        "条件": name,
        "n": len(sub),
        "平均差枚": avg,
        "プラス率": positive,
        "+500率": rate500,
        "+1000率": rate1000,
        "+2000率": rate2000
    }


# ============================================================
# ② 前日差枚分析
# ============================================================

print_title("前日差枚別 翌日実績")

previous_conditions = [
    (
        "前日 -3000枚以下",
        analysis_df["台_前日差枚"] <= -3000
    ),
    (
        "前日 -2000～-3000枚",
        (
            (analysis_df["台_前日差枚"] > -3000)
            & (analysis_df["台_前日差枚"] <= -2000)
        )
    ),
    (
        "前日 -1000～-2000枚",
        (
            (analysis_df["台_前日差枚"] > -2000)
            & (analysis_df["台_前日差枚"] <= -1000)
        )
    ),
    (
        "前日 -1000～0枚",
        (
            (analysis_df["台_前日差枚"] > -1000)
            & (analysis_df["台_前日差枚"] <= 0)
        )
    ),
    (
        "前日 0～+1000枚",
        (
            (analysis_df["台_前日差枚"] > 0)
            & (analysis_df["台_前日差枚"] <= 1000)
        )
    ),
    (
        "前日 +1000～+2000枚",
        (
            (analysis_df["台_前日差枚"] > 1000)
            & (analysis_df["台_前日差枚"] <= 2000)
        )
    ),
    (
        "前日 +2000～+3000枚",
        (
            (analysis_df["台_前日差枚"] > 2000)
            & (analysis_df["台_前日差枚"] <= 3000)
        )
    ),
    (
        "前日 +3000枚以上",
        analysis_df["台_前日差枚"] > 3000
    )
]

condition_rows = []

for name, condition in previous_conditions:

    result = analyze_condition(
        name,
        condition,
        "前日差枚"
    )

    if result is None:
        continue

    condition_rows.append(result)

    print(
        f"{name} / "
        f"n={result['n']} / "
        f"平均差枚 {format_num(result['平均差枚'])}枚 / "
        f"プラス率 {format_rate(result['プラス率'])} / "
        f"+1000率 {format_rate(result['+1000率'])} / "
        f"+2000率 {format_rate(result['+2000率'])}"
    )


# ============================================================
# ③ 前日大幅プラス・マイナス
# ============================================================

print_title("前日大幅出玉・大幅凹み分析")

special_conditions = [
    (
        "前日 +2000枚以上",
        analysis_df["台_前日差枚"] >= 2000
    ),
    (
        "前日 +3000枚以上",
        analysis_df["台_前日差枚"] >= 3000
    ),
    (
        "前日 +5000枚以上",
        analysis_df["台_前日差枚"] >= 5000
    ),
    (
        "前日 -2000枚以下",
        analysis_df["台_前日差枚"] <= -2000
    ),
    (
        "前日 -3000枚以下",
        analysis_df["台_前日差枚"] <= -3000
    ),
    (
        "前日 -5000枚以下",
        analysis_df["台_前日差枚"] <= -5000
    )
]

for name, condition in special_conditions:

    result = analyze_condition(
        name,
        condition,
        "前日極端値"
    )

    if result is None:
        continue

    condition_rows.append(result)

    print(
        f"{name} / "
        f"n={result['n']} / "
        f"平均差枚 {format_num(result['平均差枚'])}枚 / "
        f"プラス率 {format_rate(result['プラス率'])} / "
        f"+1000率 {format_rate(result['+1000率'])}"
    )


# ============================================================
# ④ 台番号 偶奇
# ============================================================

print_title("台番号 偶奇分析")

odd_condition = (
    analysis_df["台_偶奇"] == 1
)

even_condition = (
    analysis_df["台_偶奇"] == 0
)

for name, condition in [
    ("奇数台", odd_condition),
    ("偶数台", even_condition)
]:

    result = analyze_condition(
        name,
        condition,
        "台番号偶奇"
    )

    if result is None:
        continue

    condition_rows.append(result)

    print(
        f"{name} / "
        f"n={result['n']} / "
        f"平均差枚 {format_num(result['平均差枚'])}枚 / "
        f"プラス率 {format_rate(result['プラス率'])} / "
        f"+1000率 {format_rate(result['+1000率'])}"
    )


# ============================================================
# ⑤ 下一桁
# ============================================================

print_title("台番号 下一桁分析")

last_digit_rows = []

for digit in range(10):

    sub = analysis_df[
        analysis_df["台_下一桁"] == digit
    ]

    if len(sub) == 0:
        continue

    avg = safe_mean(
        sub["当日差枚"]
    )

    positive = safe_rate(
        sub["当日差枚"],
        lambda x: x > 0
    )

    rate1000 = safe_rate(
        sub["当日差枚"],
        lambda x: x >= 1000
    )

    print(
        f"下一桁 {digit} / "
        f"n={len(sub)} / "
        f"平均差枚 {format_num(avg)}枚 / "
        f"プラス率 {format_rate(positive)} / "
        f"+1000率 {format_rate(rate1000)}"
    )

    last_digit_rows.append(
        {
            "分析種別": "台番号下一桁",
            "条件": f"下一桁 {digit}",
            "n": len(sub),
            "平均差枚": avg,
            "プラス率": positive,
            "+500率": safe_rate(
                sub["当日差枚"],
                lambda x: x >= 500
            ),
            "+1000率": rate1000,
            "+2000率": safe_rate(
                sub["当日差枚"],
                lambda x: x >= 2000
            )
        }
    )


# ============================================================
# ⑥ 台番号帯
# ============================================================

print_title("台番号帯分析")

band_rows = []

bands = [
    (500, 599),
    (600, 699),
    (700, 799),
    (800, 899),
    (900, 999),
    (1000, 1099)
]

for start, end in bands:

    condition = (
        (analysis_df["台番号"] >= start)
        & (analysis_df["台番号"] <= end)
    )

    result = analyze_condition(
        f"{start}～{end}",
        condition,
        "台番号帯"
    )

    if result is None:
        continue

    band_rows.append(result)
    condition_rows.append(result)

    print(
        f"{start}～{end} / "
        f"n={result['n']} / "
        f"平均差枚 {format_num(result['平均差枚'])}枚 / "
        f"プラス率 {format_rate(result['プラス率'])} / "
        f"+1000率 {format_rate(result['+1000率'])}"
    )


# ============================================================
# ⑦ 台の前回変化
# ============================================================

print_title("台・前回変化分析")

change_conditions = [
    (
        "前回変化 -2000枚以下",
        analysis_df["台_前回変化"] <= -2000
    ),
    (
        "前回変化 -1000～-2000枚",
        (
            (analysis_df["台_前回変化"] > -2000)
            & (analysis_df["台_前回変化"] <= -1000)
        )
    ),
    (
        "前回変化 -1000～0枚",
        (
            (analysis_df["台_前回変化"] > -1000)
            & (analysis_df["台_前回変化"] <= 0)
        )
    ),
    (
        "前回変化 0～+1000枚",
        (
            (analysis_df["台_前回変化"] > 0)
            & (analysis_df["台_前回変化"] <= 1000)
        )
    ),
    (
        "前回変化 +1000～+2000枚",
        (
            (analysis_df["台_前回変化"] > 1000)
            & (analysis_df["台_前回変化"] <= 2000)
        )
    ),
    (
        "前回変化 +2000枚以上",
        analysis_df["台_前回変化"] > 2000
    )
]

for name, condition in change_conditions:

    result = analyze_condition(
        name,
        condition,
        "台前回変化"
    )

    if result is None:
        continue

    condition_rows.append(result)

    print(
        f"{name} / "
        f"n={result['n']} / "
        f"平均差枚 {format_num(result['平均差枚'])}枚 / "
        f"プラス率 {format_rate(result['プラス率'])} / "
        f"+1000率 {format_rate(result['+1000率'])}"
    )


# ============================================================
# ⑧ 台の凹み
# ============================================================

print_title("台・凹み分析")

drawdown_conditions = [
    (
        "凹み -2000枚以下",
        analysis_df["台_凹み"] <= -2000
    ),
    (
        "凹み -1000～-2000枚",
        (
            (analysis_df["台_凹み"] > -2000)
            & (analysis_df["台_凹み"] <= -1000)
        )
    ),
    (
        "凹み -1000～0枚",
        (
            (analysis_df["台_凹み"] > -1000)
            & (analysis_df["台_凹み"] <= 0)
        )
    ),
    (
        "凹み 0～+1000枚",
        (
            (analysis_df["台_凹み"] > 0)
            & (analysis_df["台_凹み"] <= 1000)
        )
    ),
    (
        "凹み +1000～+2000枚",
        (
            (analysis_df["台_凹み"] > 1000)
            & (analysis_df["台_凹み"] <= 2000)
        )
    ),
    (
        "凹み +2000枚以上",
        analysis_df["台_凹み"] > 2000
    )
]

for name, condition in drawdown_conditions:

    result = analyze_condition(
        name,
        condition,
        "台凹み"
    )

    if result is None:
        continue

    condition_rows.append(result)

    print(
        f"{name} / "
        f"n={result['n']} / "
        f"平均差枚 {format_num(result['平均差枚'])}枚 / "
        f"プラス率 {format_rate(result['プラス率'])} / "
        f"+1000率 {format_rate(result['+1000率'])}"
    )


# ============================================================
# ⑨ 機種前日差枚
# ============================================================

print_title("機種・前日差枚分析")

model_previous_conditions = [
    (
        "機種前日 -2000枚以下",
        analysis_df["機種_前日差枚"] <= -2000
    ),
    (
        "機種前日 -1000～-2000枚",
        (
            (analysis_df["機種_前日差枚"] > -2000)
            & (analysis_df["機種_前日差枚"] <= -1000)
        )
    ),
    (
        "機種前日 -1000～0枚",
        (
            (analysis_df["機種_前日差枚"] > -1000)
            & (analysis_df["機種_前日差枚"] <= 0)
        )
    ),
    (
        "機種前日 0～+1000枚",
        (
            (analysis_df["機種_前日差枚"] > 0)
            & (analysis_df["機種_前日差枚"] <= 1000)
        )
    ),
    (
        "機種前日 +1000～+2000枚",
        (
            (analysis_df["機種_前日差枚"] > 1000)
            & (analysis_df["機種_前日差枚"] <= 2000)
        )
    ),
    (
        "機種前日 +2000枚以上",
        analysis_df["機種_前日差枚"] > 2000
    )
]

for name, condition in model_previous_conditions:

    result = analyze_condition(
        name,
        condition,
        "機種前日差枚"
    )

    if result is None:
        continue

    condition_rows.append(result)

    print(
        f"{name} / "
        f"n={result['n']} / "
        f"平均差枚 {format_num(result['平均差枚'])}枚 / "
        f"プラス率 {format_rate(result['プラス率'])} / "
        f"+1000率 {format_rate(result['+1000率'])}"
    )


# ============================================================
# ⑩ 機種凹み
# ============================================================

print_title("機種・凹み分析")

model_drawdown_conditions = [
    (
        "機種凹み -2000枚以下",
        analysis_df["機種_凹み"] <= -2000
    ),
    (
        "機種凹み -1000～-2000枚",
        (
            (analysis_df["機種_凹み"] > -2000)
            & (analysis_df["機種_凹み"] <= -1000)
        )
    ),
    (
        "機種凹み -1000～0枚",
        (
            (analysis_df["機種_凹み"] > -1000)
            & (analysis_df["機種_凹み"] <= 0)
        )
    ),
    (
        "機種凹み 0～+1000枚",
        (
            (analysis_df["機種_凹み"] > 0)
            & (analysis_df["機種_凹み"] <= 1000)
        )
    ),
    (
        "機種凹み +1000～+2000枚",
        (
            (analysis_df["機種_凹み"] > 1000)
            & (analysis_df["機種_凹み"] <= 2000)
        )
    ),
    (
        "機種凹み +2000枚以上",
        analysis_df["機種_凹み"] > 2000
    )
]

for name, condition in model_drawdown_conditions:

    result = analyze_condition(
        name,
        condition,
        "機種凹み"
    )

    if result is None:
        continue

    condition_rows.append(result)

    print(
        f"{name} / "
        f"n={result['n']} / "
        f"平均差枚 {format_num(result['平均差枚'])}枚 / "
        f"プラス率 {format_rate(result['プラス率'])} / "
        f"+1000率 {format_rate(result['+1000率'])}"
    )


# ============================================================
# ⑪ 要因別相関
# ============================================================

print_title("要因別 相関分析")

factor_columns = [
    "台_過去平均差枚",
    "台_過去プラス率",
    "台_前日差枚",
    "台_直近3日平均",
    "台_前回変化",
    "台_凹み",
    "機種_過去平均差枚",
    "機種_過去プラス率",
    "機種_前日差枚",
    "機種_直近3日平均",
    "機種_前回変化",
    "機種_凹み"
]

correlation_rows = []

for factor in factor_columns:

    temp = analysis_df[
        [factor, "当日差枚"]
    ].dropna()

    if len(temp) < 10:
        corr = np.nan
    else:
        corr = temp[factor].corr(
            temp["当日差枚"]
        )

    correlation_rows.append(
        {
            "要因": factor,
            "相関": corr,
            "n": len(temp)
        }
    )


correlation_df = pd.DataFrame(
    correlation_rows
)

correlation_df["絶対相関"] = (
    correlation_df["相関"].abs()
)

correlation_df = (
    correlation_df
    .sort_values(
        "絶対相関",
        ascending=False
    )
    .reset_index(drop=True)
)

print()

for i, row in correlation_df.head(12).iterrows():

    corr = row["相関"]

    if pd.isna(corr):
        corr_text = "N/A"
    else:
        corr_text = f"{corr:+.4f}"

    print(
        f"{i + 1:2d}. "
        f"{row['要因']} / "
        f"相関 {corr_text} / "
        f"n={int(row['n'])}"
    )


# ============================================================
# ⑫ 機種別実績
# ============================================================

print_title("機種別 翌日実績")

model_rows = []

for model_name, sub in analysis_df.groupby("機種名"):

    if len(sub) < 3:
        continue

    avg = safe_mean(
        sub["当日差枚"]
    )

    positive = safe_rate(
        sub["当日差枚"],
        lambda x: x > 0
    )

    rate500 = safe_rate(
        sub["当日差枚"],
        lambda x: x >= 500
    )

    rate1000 = safe_rate(
        sub["当日差枚"],
        lambda x: x >= 1000
    )

    rate2000 = safe_rate(
        sub["当日差枚"],
        lambda x: x >= 2000
    )

    model_rows.append(
        {
            "機種名": model_name,
            "n": len(sub),
            "平均差枚": avg,
            "プラス率": positive,
            "+500率": rate500,
            "+1000率": rate1000,
            "+2000率": rate2000
        }
    )


model_df = pd.DataFrame(
    model_rows
)

if len(model_df) > 0:

    model_df = model_df.sort_values(
        ["平均差枚", "プラス率"],
        ascending=False
    )

    print()

    for _, row in model_df.head(20).iterrows():

        print(
            f"{row['機種名']} / "
            f"n={int(row['n'])} / "
            f"平均差枚 {format_num(row['平均差枚'])}枚 / "
            f"プラス率 {format_rate(row['プラス率'])} / "
            f"+1000率 {format_rate(row['+1000率'])}"
        )


# ============================================================
# ⑬ 条件ランキング
# ============================================================

print_title("投入パターン候補ランキング")

condition_df = pd.DataFrame(
    condition_rows
)

if len(condition_df) > 0:

    # 少数サンプルによる極端値を抑制する。
    #
    # nが10未満の場合はランキング評価を下げる。
    condition_df["信頼補正"] = np.minimum(
        condition_df["n"] / 30,
        1.0
    )

    condition_df["条件スコア"] = (
        condition_df["平均差枚"].fillna(0) * 0.50
        + condition_df["プラス率"].fillna(0) * 2.0
        + condition_df["+1000率"].fillna(0) * 1.0
    )

    condition_df["補正スコア"] = (
        condition_df["条件スコア"]
        * (
            0.5
            + 0.5 * condition_df["信頼補正"]
        )
    )

    condition_df = condition_df.sort_values(
        "補正スコア",
        ascending=False
    ).reset_index(drop=True)

    for i, row in condition_df.head(30).iterrows():

        print(
            f"{i + 1:2d}. "
            f"{row['分析種別']} / "
            f"{row['条件']} / "
            f"n={int(row['n'])} / "
            f"平均差枚 {format_num(row['平均差枚'])}枚 / "
            f"プラス率 {format_rate(row['プラス率'])} / "
            f"+1000率 {format_rate(row['+1000率'])} / "
            f"補正スコア {row['補正スコア']:.1f}"
        )


# ============================================================
# ⑭ 総合サマリー
# ============================================================

print_title("分析結果サマリー")

overall_avg = safe_mean(
    analysis_df["当日差枚"]
)

overall_positive = safe_rate(
    analysis_df["当日差枚"],
    lambda x: x > 0
)

overall_1000 = safe_rate(
    analysis_df["当日差枚"],
    lambda x: x >= 1000
)

print(
    f"総サンプル数 : "
    f"{len(analysis_df):,}"
)

print(
    f"予測日数 : "
    f"{analysis_df['予測日'].nunique()}"
)

print(
    f"平均当日差枚 : "
    f"{overall_avg:.2f}"
)

print(
    f"当日プラス率 : "
    f"{overall_positive:.2f}%"
)

print(
    f"当日+1000率 : "
    f"{overall_1000:.2f}%"
)

if len(correlation_df) > 0:

    best_factor = correlation_df.iloc[0]

    print(
        f"最も相関が強い要因 : "
        f"{best_factor['要因']}"
    )

    if pd.isna(best_factor["相関"]):
        print(
            "最も相関が強い要因_相関 : N/A"
        )
    else:
        print(
            f"最も相関が強い要因_相関 : "
            f"{best_factor['相関']:.4f}"
        )


if len(condition_df) > 0:

    best_condition = condition_df.iloc[0]

    print(
        f"最有力投入条件 : "
        f"{best_condition['分析種別']} / "
        f"{best_condition['条件']}"
    )

    print(
        f"最有力投入条件_平均差枚 : "
        f"{best_condition['平均差枚']:.1f}"
    )

    print(
        f"最有力投入条件_プラス率 : "
        f"{best_condition['プラス率']:.1f}%"
    )


# ============================================================
# CSV保存
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# メイン分析結果
analysis_output = analysis_df.copy()

analysis_output.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print()
print("★ CSV保存成功")
print(OUTPUT_FILE)


# ============================================================
# サマリーCSV
# ============================================================

summary_rows = []

# 全体
summary_rows.append(
    {
        "分析種別": "全体",
        "条件": "全データ",
        "n": len(analysis_df),
        "平均差枚": overall_avg,
        "プラス率": overall_positive,
        "+500率": safe_rate(
            analysis_df["当日差枚"],
            lambda x: x >= 500
        ),
        "+1000率": overall_1000,
        "+2000率": safe_rate(
            analysis_df["当日差枚"],
            lambda x: x >= 2000
        )
    }
)

# 曜日
summary_rows.extend(
    weekday_rows
)

# 条件
summary_rows.extend(
    condition_rows
)

# 下一桁
summary_rows.extend(
    last_digit_rows
)

# 相関
for _, row in correlation_df.iterrows():

    summary_rows.append(
        {
            "分析種別": "相関",
            "条件": row["要因"],
            "n": int(row["n"]),
            "平均差枚": np.nan,
            "プラス率": np.nan,
            "+500率": np.nan,
            "+1000率": np.nan,
            "+2000率": np.nan,
            "相関": row["相関"]
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


# ============================================================
# 完了
# ============================================================

print()
print_title("投入パターン分析 完了")

print("保存ファイル:")
print(OUTPUT_FILE)
print(SUMMARY_FILE)

print()
print("all_data.csv は変更していません。")

input("Enterキーで終了...")