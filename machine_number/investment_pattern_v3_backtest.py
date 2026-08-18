from pathlib import Path
import pandas as pd
import numpy as np


print("=" * 70)
print("投入パターン解析 V3 バックテスト")
print("=" * 70)


# ============================================================
# パス
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "maruhan_maebashi"

INPUT_FILE = DATA_DIR / "all_data.csv"

OUTPUT_FILE = (
    DATA_DIR
    / "machine_number"
    / "investment_pattern_v3_backtest.csv"
)

SUMMARY_FILE = (
    DATA_DIR
    / "machine_number"
    / "investment_pattern_v3_backtest_summary.csv"
)


print()
print("入力ファイル:")
print(INPUT_FILE)


# ============================================================
# CSV読み込み
# ============================================================

print()
print("all_data.csv を読み込みます...")

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print(f"読み込みデータ: {len(df):,}行")


required_columns = [
    "日付",
    "機種名",
    "台番号",
    "差枚",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print()
    print("[エラー]")
    print("必要な列がありません:")

    for col in missing_columns:
        print(col)

    raise SystemExit(1)

print("必要な列: OK")


# ============================================================
# データ整理
# ============================================================

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

df = df.dropna(
    subset=[
        "日付",
        "機種名",
        "台番号",
        "差枚",
    ]
).copy()

df = df.sort_values(
    [
        "日付",
        "台番号",
    ]
).reset_index(drop=True)


dates = sorted(
    df["日付"].dt.date.unique()
)


print()
print(f"解析日数: {len(dates)}日")

if len(dates) < 3:
    print("[エラー] データ日数が不足しています。")
    raise SystemExit(1)

print(
    f"解析期間: {dates[0]} ～ {dates[-1]}"
)


# ============================================================
# V3スコア計算
# ============================================================

def calculate_v3(train_data):

    rows = []

    grouped = train_data.groupby("機種名")

    for machine_name, machine_data in grouped:

        machine_data = machine_data.sort_values("日付")

        data_count = len(machine_data)

        if data_count == 0:
            continue

        # ----------------------------------------------------
        # 過去平均
        # ----------------------------------------------------

        avg_diff = machine_data["差枚"].mean()

        # ----------------------------------------------------
        # プラス率
        # ----------------------------------------------------

        positive_rate = (
            (machine_data["差枚"] > 0).mean()
            * 100
        )

        # ----------------------------------------------------
        # 日別平均
        # ----------------------------------------------------

        daily_avg = (
            machine_data
            .groupby("日付")["差枚"]
            .mean()
            .sort_index()
        )

        # ----------------------------------------------------
        # 直近3日
        # ----------------------------------------------------

        recent_3 = daily_avg.tail(3)

        if len(recent_3) > 0:
            recent_3_avg = recent_3.mean()
        else:
            recent_3_avg = 0

        # ----------------------------------------------------
        # 前回
        # ----------------------------------------------------

        if len(daily_avg) > 0:
            previous_avg = daily_avg.iloc[-1]
        else:
            previous_avg = 0

        # ----------------------------------------------------
        # 直近変化
        # ----------------------------------------------------

        recent_change = (
            previous_avg - avg_diff
        )

        # ----------------------------------------------------
        # 凹み
        # ----------------------------------------------------

        drawdown = (
            avg_diff - previous_avg
        )

        # ----------------------------------------------------
        # 直近3日変化
        # ----------------------------------------------------

        recent_3_change = (
            recent_3_avg - avg_diff
        )

        # ----------------------------------------------------
        # 信頼度
        # ----------------------------------------------------

        confidence = min(
            100,
            np.sqrt(data_count / 50) * 100
        )

        # ----------------------------------------------------
        # 各スコア
        # ----------------------------------------------------

        avg_score = np.clip(
            50 + avg_diff / 100,
            0,
            100
        )

        positive_score = positive_rate

        recent_score = np.clip(
            50 + recent_3_change / 100,
            0,
            100
        )

        drawdown_score = np.clip(
            50 + drawdown / 50,
            0,
            100
        )

        # ----------------------------------------------------
        # V3総合スコア
        # ----------------------------------------------------

        v3_score = (
            avg_score * 0.35
            + positive_score * 0.20
            + recent_score * 0.20
            + drawdown_score * 0.15
            + confidence * 0.10
        )

        # ----------------------------------------------------
        # ランク
        # ----------------------------------------------------

        if v3_score >= 75:
            rank = "S"

        elif v3_score >= 65:
            rank = "A"

        elif v3_score >= 55:
            rank = "B"

        elif v3_score >= 45:
            rank = "C"

        elif v3_score >= 35:
            rank = "D"

        else:
            rank = "E"

        rows.append(
            {
                "機種名": machine_name,
                "データ数": data_count,
                "平均差枚": avg_diff,
                "プラス率": positive_rate,
                "直近3日平均": recent_3_avg,
                "前回平均": previous_avg,
                "直近変化": recent_change,
                "凹み": drawdown,
                "信頼度": confidence,
                "V3スコア": v3_score,
                "ランク": rank,
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result = result.sort_values(
        "V3スコア",
        ascending=False
    ).reset_index(drop=True)

    result["順位"] = (
        result.index + 1
    )

    return result


# ============================================================
# バックテスト
# ============================================================

backtest_rows = []


# 最初の日は予測できないため、
# 2日目からバックテストする

test_dates = dates[1:]


for target_date in test_dates:

    print()
    print("=" * 70)
    print(f"【バックテスト】{target_date}")
    print("=" * 70)

    # --------------------------------------------------------
    # 予測対象日の前日まで
    # --------------------------------------------------------

    train_data = df[
        df["日付"].dt.date < target_date
    ].copy()

    # --------------------------------------------------------
    # 実際の対象日
    # --------------------------------------------------------

    actual_data = df[
        df["日付"].dt.date == target_date
    ].copy()

    if train_data.empty:
        continue

    if actual_data.empty:
        continue

    # --------------------------------------------------------
    # V3予測
    # --------------------------------------------------------

    prediction = calculate_v3(
        train_data
    )

    if prediction.empty:
        continue

    # --------------------------------------------------------
    # 対象日の実績を機種別集計
    # --------------------------------------------------------

    actual = (
        actual_data
        .groupby("機種名")
        .agg(
            actual_avg_diff=("差枚", "mean"),
            actual_positive_rate=(
                "差枚",
                lambda x:
                (x > 0).mean() * 100
            ),
            actual_machine_count=(
                "差枚",
                "count"
            ),
            thousand_count=(
                "差枚",
                lambda x:
                (x >= 1000).sum()
            ),
            two_thousand_count=(
                "差枚",
                lambda x:
                (x >= 2000).sum()
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # 実績率
    # --------------------------------------------------------

    actual["thousand_rate"] = (
        actual["thousand_count"]
        / actual["actual_machine_count"]
        * 100
    )

    actual["two_thousand_rate"] = (
        actual["two_thousand_count"]
        / actual["actual_machine_count"]
        * 100
    )

    # --------------------------------------------------------
    # 予測と実績を結合
    # --------------------------------------------------------

    merged = prediction.merge(
        actual,
        on="機種名",
        how="inner"
    )

    if merged.empty:
        continue

    # --------------------------------------------------------
    # TOP5 / TOP10 / TOP20 / TOP30
    # --------------------------------------------------------

    for top_n in [5, 10, 20, 30]:

        top_data = merged.head(top_n)

        if top_data.empty:
            continue

        result_avg_diff = (
            top_data["actual_avg_diff"].mean()
        )

        result_positive_rate = (
            top_data["actual_positive_rate"].mean()
        )

        result_thousand_rate = (
            top_data["thousand_rate"].mean()
        )

        result_two_thousand_rate = (
            top_data["two_thousand_rate"].mean()
        )

        machine_win_rate = (
            (
                top_data["actual_avg_diff"]
                > 0
            ).mean()
            * 100
        )

        backtest_rows.append(
            {
                "予測日": target_date,
                "TOP": top_n,
                "予測機種数": len(top_data),
                "実績平均差枚": result_avg_diff,
                "実績プラス率": result_positive_rate,
                "+1000率": result_thousand_rate,
                "+2000率": result_two_thousand_rate,
                "機種勝率": machine_win_rate,
            }
        )

        print(
            f"TOP{top_n:>2} / "
            f"実績平均差枚 "
            f"{result_avg_diff:+.1f}枚 / "
            f"プラス率 "
            f"{result_positive_rate:.1f}% / "
            f"+1000率 "
            f"{result_thousand_rate:.1f}% / "
            f"+2000率 "
            f"{result_two_thousand_rate:.1f}% / "
            f"機種勝率 "
            f"{machine_win_rate:.1f}%"
        )


# ============================================================
# バックテスト結果
# ============================================================

result_df = pd.DataFrame(
    backtest_rows
)

if result_df.empty:

    print()
    print("[エラー]")
    print("バックテスト結果がありません。")

    raise SystemExit(1)


# ============================================================
# 総合結果
# ============================================================

print()
print("=" * 70)
print("【V3 バックテスト総合結果】")
print("=" * 70)


summary_rows = []


for top_n in [5, 10, 20, 30]:

    temp = result_df[
        result_df["TOP"] == top_n
    ]

    if temp.empty:
        continue

    avg_diff = (
        temp["実績平均差枚"].mean()
    )

    positive_rate = (
        temp["実績プラス率"].mean()
    )

    thousand_rate = (
        temp["+1000率"].mean()
    )

    two_thousand_rate = (
        temp["+2000率"].mean()
    )

    machine_win_rate = (
        temp["機種勝率"].mean()
    )

    print()
    print(
        f"TOP{top_n:>2} / "
        f"平均差枚 {avg_diff:+.1f}枚 / "
        f"プラス率 {positive_rate:.1f}% / "
        f"+1000率 {thousand_rate:.1f}% / "
        f"+2000率 {two_thousand_rate:.1f}% / "
        f"機種勝率 {machine_win_rate:.1f}%"
    )

    summary_rows.append(
        {
            "TOP": top_n,
            "平均差枚": avg_diff,
            "プラス率": positive_rate,
            "+1000率": thousand_rate,
            "+2000率": two_thousand_rate,
            "機種勝率": machine_win_rate,
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# ベストTOP
# ============================================================

if not summary_df.empty:

    best_index = (
        summary_df["平均差枚"].idxmax()
    )

    best = summary_df.loc[
        best_index
    ]

    print()
    print("=" * 70)
    print("【V3 ベストTOP】")
    print("=" * 70)

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


# ============================================================
# 保存
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
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
print("=" * 70)
print("★★★★★ 投入パターン V3 バックテスト完了 ★★★★★")
print("=" * 70)

print()
print("保存ファイル:")
print(OUTPUT_FILE)
print(SUMMARY_FILE)

print()
print("all_data.csv は変更していません。")