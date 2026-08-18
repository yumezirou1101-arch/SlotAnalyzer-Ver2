# -*- coding: utf-8 -*-

"""
======================================================================
予測要因別・的中分析 V1.1
======================================================================

目的:
V5を作る前に、どの予測要因が翌日の実績に効いているかを分析する。

重要:
予測対象日の実績は、予測スコア計算には使用しない。

分析要因:
・台番号の過去平均差枚
・台番号の過去プラス率
・台番号の過去+1000率
・台番号の過去+2000率
・台番号の直近3日平均
・台番号の前日差枚
・台番号の前回変化
・台番号の凹み
・機種の過去平均差枚
・機種の過去プラス率
・機種の過去+1000率
・機種の過去+2000率
・機種の直近3日平均
・機種の前日差枚
・機種の前回変化
・機種の凹み

出力:
prediction_factor_analysis.csv
prediction_factor_analysis_summary.csv

all_data.csv は変更しない。
======================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ======================================================================
# 設定
# ======================================================================

BASE_DIR = r"C:\Users\user\Desktop\Documents\SlotAnalyzer"

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
    "prediction_factor_analysis.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "prediction_factor_analysis_summary.csv"
)


# ======================================================================
# 表示
# ======================================================================

def print_line():
    print("=" * 70)


def print_header(text):
    print_line()
    print(text)
    print_line()


# ======================================================================
# 列名検索
# ======================================================================

def find_column(df, candidates):

    for col in candidates:
        if col in df.columns:
            return col

    normalized = {}

    for col in df.columns:
        key = (
            str(col)
            .replace(" ", "")
            .replace("_", "")
            .lower()
        )
        normalized[key] = col

    for candidate in candidates:

        key = (
            str(candidate)
            .replace(" ", "")
            .replace("_", "")
            .lower()
        )

        if key in normalized:
            return normalized[key]

    return None


# ======================================================================
# CSV読み込み
# ======================================================================

def load_data():

    print()
    print("入力ファイル:")
    print(DATA_FILE)
    print()

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"入力ファイルがありません:\n{DATA_FILE}"
        )

    encodings = [
        "utf-8-sig",
        "cp932",
        "utf-8"
    ]

    df = None

    for encoding in encodings:

        try:

            df = pd.read_csv(
                DATA_FILE,
                encoding=encoding
            )

            break

        except UnicodeDecodeError:
            continue

    if df is None:

        raise ValueError(
            "CSVの文字コードを読み込めませんでした。"
        )

    print(
        f"読み込みデータ: {len(df):,}行"
    )

    return df


# ======================================================================
# データ整形
# ======================================================================

def prepare_columns(df):

    date_col = find_column(
        df,
        [
            "日付",
            "DATE",
            "date",
            "年月日"
        ]
    )

    machine_number_col = find_column(
        df,
        [
            "台番号",
            "台No",
            "台NO",
            "台№",
            "台番",
            "machine_number"
        ]
    )

    machine_name_col = find_column(
        df,
        [
            "機種",
            "機種名",
            "機種名称",
            "machine",
            "machine_name"
        ]
    )

    diff_col = find_column(
        df,
        [
            "差枚",
            "差枚数",
            "差玉",
            "差枚数(枚)",
            "差枚数（枚）",
            "diff"
        ]
    )

    print()
    print("必要な列を確認します...")

    print(f"日付   : {date_col}")
    print(f"台番号 : {machine_number_col}")
    print(f"機種   : {machine_name_col}")
    print(f"差枚   : {diff_col}")

    if any(
        x is None
        for x in [
            date_col,
            machine_number_col,
            machine_name_col,
            diff_col
        ]
    ):

        print()
        print("現在のCSV列:")

        for col in df.columns:
            print(f"  {col}")

        raise ValueError(
            "必要な列を自動判定できませんでした。"
        )

    work = df[
        [
            date_col,
            machine_number_col,
            machine_name_col,
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

    work["機種"] = work["機種"].astype(str)

    work = work.dropna(
        subset=[
            "日付",
            "台番号",
            "机種" if "机種" in work.columns else "機種",
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

    print()
    print("必要な列: OK")
    print(
        f"有効データ: {len(work):,}行"
    )

    return work


# ======================================================================
# 履歴から特徴量
# ======================================================================

def calc_features(history):

    history = history.sort_values(
        "日付"
    ).copy()

    if history.empty:
        return None

    diffs = history["差枚"].astype(float)

    avg_diff = diffs.mean()

    plus_rate = (
        (diffs > 0).mean() * 100
    )

    plus1000_rate = (
        (diffs >= 1000).mean() * 100
    )

    plus2000_rate = (
        (diffs >= 2000).mean() * 100
    )

    recent3 = history.tail(3)

    recent3_avg = (
        recent3["差枚"].mean()
        if not recent3.empty
        else 0
    )

    previous_diff = (
        float(history.iloc[-1]["差枚"])
    )

    if len(history) >= 2:

        previous_previous_diff = float(
            history.iloc[-2]["差枚"]
        )

        previous_change = (
            previous_diff
            -
            previous_previous_diff
        )

    else:

        previous_change = 0

    dip = (
        recent3_avg
        -
        avg_diff
    )

    return {
        "過去平均差枚": avg_diff,
        "過去プラス率": plus_rate,
        "過去+1000率": plus1000_rate,
        "過去+2000率": plus2000_rate,
        "直近3日平均": recent3_avg,
        "前日差枚": previous_diff,
        "前回変化": previous_change,
        "凹み": dip,
        "履歴日数": len(history)
    }


# ======================================================================
# 特徴量作成
# ======================================================================

def create_feature_data(df):

    dates = sorted(
        df["日付"]
        .dt
        .normalize()
        .unique()
    )

    print()
    print(
        f"収録日数: {len(dates)}日"
    )

    print(
        "収録日:",
        " / ".join(
            pd.Timestamp(d).strftime(
                "%Y-%m-%d"
            )
            for d in dates
        )
    )

    rows = []

    for target_date in dates:

        target_date = pd.Timestamp(
            target_date
        ).normalize()

        history = df[
            df["日付"] < target_date
        ].copy()

        actual = df[
            df["日付"] == target_date
        ].copy()

        if history.empty:
            continue

        print(
            f"\r特徴量作成中: "
            f"{target_date.strftime('%Y-%m-%d')}",
            end=""
        )

        # ----------------------------------------------------------
        # 台番号
        # ----------------------------------------------------------

        machine_feature_map = {}

        for machine_number, group in history.groupby(
            "台番号"
        ):

            features = calc_features(
                group
            )

            if features is not None:

                machine_feature_map[
                    machine_number
                ] = features

        # ----------------------------------------------------------
        # 機種
        # ----------------------------------------------------------

        type_feature_map = {}

        for machine_name, group in history.groupby(
            "機種"
        ):

            features = calc_features(
                group
            )

            if features is not None:

                type_feature_map[
                    machine_name
                ] = features

        # ----------------------------------------------------------
        # 当日の実績を結合
        # ----------------------------------------------------------

        for _, actual_row in actual.iterrows():

            machine_number = int(
                actual_row["台番号"]
            )

            machine_name = actual_row[
                "機種"
            ]

            actual_diff = float(
                actual_row["差枚"]
            )

            machine_features = (
                machine_feature_map.get(
                    machine_number
                )
            )

            type_features = (
                type_feature_map.get(
                    machine_name
                )
            )

            if machine_features is None:
                continue

            if type_features is None:
                continue

            row = {

                "予測日":
                    target_date,

                "台番号":
                    machine_number,

                "機種":
                    machine_name,

                "当日差枚":
                    actual_diff,

                # 台番号
                "台_過去平均差枚":
                    machine_features[
                        "過去平均差枚"
                    ],

                "台_過去プラス率":
                    machine_features[
                        "過去プラス率"
                    ],

                "台_過去+1000率":
                    machine_features[
                        "過去+1000率"
                    ],

                "台_過去+2000率":
                    machine_features[
                        "過去+2000率"
                    ],

                "台_直近3日平均":
                    machine_features[
                        "直近3日平均"
                    ],

                "台_前日差枚":
                    machine_features[
                        "前日差枚"
                    ],

                "台_前回変化":
                    machine_features[
                        "前回変化"
                    ],

                "台_凹み":
                    machine_features[
                        "凹み"
                    ],

                "台_履歴日数":
                    machine_features[
                        "履歴日数"
                    ],

                # 機種
                "機種_過去平均差枚":
                    type_features[
                        "過去平均差枚"
                    ],

                "機種_過去プラス率":
                    type_features[
                        "過去プラス率"
                    ],

                "機種_過去+1000率":
                    type_features[
                        "過去+1000率"
                    ],

                "機種_過去+2000率":
                    type_features[
                        "過去+2000率"
                    ],

                "機種_直近3日平均":
                    type_features[
                        "直近3日平均"
                    ],

                "機種_前日差枚":
                    type_features[
                        "前日差枚"
                    ],

                "機種_前回変化":
                    type_features[
                        "前回変化"
                    ],

                "機種_凹み":
                    type_features[
                        "凹み"
                    ],

                "機種_履歴日数":
                    type_features[
                        "履歴日数"
                    ],

                # 実績
                "当日プラス":
                    1
                    if actual_diff > 0
                    else 0,

                "当日+500":
                    1
                    if actual_diff >= 500
                    else 0,

                "当日+1000":
                    1
                    if actual_diff >= 1000
                    else 0,

                "当日+2000":
                    1
                    if actual_diff >= 2000
                    else 0,

                "当日+3000":
                    1
                    if actual_diff >= 3000
                    else 0
            }

            rows.append(row)

    print()

    return pd.DataFrame(rows)


# ======================================================================
# 相関分析
# ======================================================================

def correlation_analysis(feature_df):

    print_header(
        "【要因別 相関分析】"
    )

    features = [

        "台_過去平均差枚",
        "台_過去プラス率",
        "台_過去+1000率",
        "台_過去+2000率",
        "台_直近3日平均",
        "台_前日差枚",
        "台_前回変化",
        "台_凹み",

        "機種_過去平均差枚",
        "機種_過去プラス率",
        "機種_過去+1000率",
        "機種_過去+2000率",
        "機種_直近3日平均",
        "機種_前日差枚",
        "機種_前回変化",
        "機種_凹み"
    ]

    targets = [
        "当日差枚",
        "当日プラス",
        "当日+500",
        "当日+1000",
        "当日+2000",
        "当日+3000"
    ]

    rows = []

    for feature in features:

        for target in targets:

            temp = feature_df[
                [
                    feature,
                    target
                ]
            ].dropna()

            if len(temp) >= 10:

                corr = temp[
                    feature
                ].corr(
                    temp[target]
                )

            else:

                corr = np.nan

            rows.append({

                "要因":
                    feature,

                "評価対象":
                    target,

                "相関係数":
                    corr,

                "サンプル数":
                    len(temp)
            })

    result = pd.DataFrame(rows)

    diff_result = result[
        result["評価対象"] ==
        "当日差枚"
    ].copy()

    diff_result["絶対相関"] = (
        diff_result["相関係数"]
        .abs()
    )

    diff_result = diff_result.sort_values(
        "絶対相関",
        ascending=False
    )

    print()
    print(
        "【当日差枚との相関 TOP10】"
    )

    for i, (_, row) in enumerate(
        diff_result.head(10).iterrows(),
        1
    ):

        corr = row["相関係数"]

        if pd.isna(corr):

            corr_text = "N/A"

        else:

            corr_text = (
                f"{corr:+.4f}"
            )

        print(
            f"{i:2d}. "
            f"{row['要因']} / "
            f"相関 {corr_text} / "
            f"n={int(row['サンプル数'])}"
        )

    return result


# ======================================================================
# 5分位分析
# ======================================================================

def quintile_analysis(
    feature_df,
    feature
):

    columns = [
        feature,
        "当日差枚",
        "当日プラス",
        "当日+500",
        "当日+1000",
        "当日+2000",
        "当日+3000"
    ]

    data = feature_df[
        columns
    ].dropna().copy()

    if len(data) < 25:
        return None

    try:

        data["分位"] = pd.qcut(
            data[feature].rank(
                method="first"
            ),
            5,
            labels=[
                "Q1(最低)",
                "Q2",
                "Q3",
                "Q4",
                "Q5(最高)"
            ]
        )

    except Exception:

        return None

    # +で始まる列名を直接aggに書かない
    # 辞書形式で安全に指定する
    agg_dict = {

        "サンプル数":
            ("当日差枚", "count"),

        "平均差枚":
            ("当日差枚", "mean"),

        "プラス率":
            ("当日プラス", "mean"),

        "+500率":
            ("当日+500", "mean"),

        "+1000率":
            ("当日+1000", "mean"),

        "+2000率":
            ("当日+2000", "mean"),

        "+3000率":
            ("当日+3000", "mean")
    }

    result = (
        data
        .groupby(
            "分位",
            observed=False
        )
        .agg(
            **agg_dict
        )
        .reset_index()
    )

    result["要因"] = feature

    rate_columns = [
        "プラス率",
        "+500率",
        "+1000率",
        "+2000率",
        "+3000率"
    ]

    for col in rate_columns:

        result[col] = (
            result[col] * 100
        )

    return result[
        [
            "要因",
            "分位",
            "サンプル数",
            "平均差枚",
            "プラス率",
            "+500率",
            "+1000率",
            "+2000率",
            "+3000率"
        ]
    ]


# ======================================================================
# 全5分位分析
# ======================================================================

def all_quintile_analysis(
    feature_df
):

    print_header(
        "【要因別 5分位分析】"
    )

    features = [

        "台_過去平均差枚",
        "台_過去プラス率",
        "台_直近3日平均",
        "台_前日差枚",
        "台_前回変化",
        "台_凹み",

        "機種_過去平均差枚",
        "機種_過去プラス率",
        "機種_直近3日平均",
        "機種_前日差枚",
        "機種_前回変化",
        "機種_凹み"
    ]

    results = []

    for feature in features:

        result = quintile_analysis(
            feature_df,
            feature
        )

        if result is None:
            continue

        results.append(result)

        q1 = result[
            result["分位"] ==
            "Q1(最低)"
        ]

        q5 = result[
            result["分位"] ==
            "Q5(最高)"
        ]

        if q1.empty or q5.empty:
            continue

        q1_avg = float(
            q1.iloc[0]["平均差枚"]
        )

        q5_avg = float(
            q5.iloc[0]["平均差枚"]
        )

        q1_plus = float(
            q1.iloc[0]["プラス率"]
        )

        q5_plus = float(
            q5.iloc[0]["プラス率"]
        )

        print()
        print(
            f"【{feature}】"
        )

        print(
            f"Q1 平均差枚 "
            f"{q1_avg:+.1f}枚 / "
            f"プラス率 {q1_plus:.1f}%"
        )

        print(
            f"Q5 平均差枚 "
            f"{q5_avg:+.1f}枚 / "
            f"プラス率 {q5_plus:.1f}%"
        )

        print(
            f"Q5-Q1 "
            f"{q5_avg - q1_avg:+.1f}枚"
        )

    if not results:
        return pd.DataFrame()

    return pd.concat(
        results,
        ignore_index=True
    )


# ======================================================================
# 要因ランキング
# ======================================================================

def factor_ranking(
    correlation_df,
    quintile_df
):

    print_header(
        "【予測要因ランキング】"
    )

    features = sorted(
        correlation_df["要因"]
        .unique()
    )

    rows = []

    for feature in features:

        corr_row = correlation_df[
            (
                correlation_df["要因"]
                == feature
            )
            &
            (
                correlation_df["評価対象"]
                == "当日差枚"
            )
        ]

        if corr_row.empty:
            continue

        corr = corr_row.iloc[0][
            "相関係数"
        ]

        spread = np.nan

        if not quintile_df.empty:

            q = quintile_df[
                quintile_df["要因"]
                == feature
            ]

            q1 = q[
                q["分位"] ==
                "Q1(最低)"
            ]

            q5 = q[
                q["分位"] ==
                "Q5(最高)"
            ]

            if (
                not q1.empty
                and
                not q5.empty
            ):

                spread = (
                    float(
                        q5.iloc[0]["平均差枚"]
                    )
                    -
                    float(
                        q1.iloc[0]["平均差枚"]
                    )
                )

        if pd.isna(corr):

            corr_score = 0

        else:

            corr_score = (
                abs(float(corr))
                * 100
            )

        if pd.isna(spread):

            spread_score = 0

        else:

            spread_score = min(
                abs(float(spread)) / 10,
                100
            )

        strength = (
            corr_score * 0.5
            +
            spread_score * 0.5
        )

        rows.append({

            "要因":
                feature,

            "相関係数":
                corr,

            "Q5-Q1平均差枚":
                spread,

            "要因強度スコア":
                strength
        })

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result = result.sort_values(
        "要因強度スコア",
        ascending=False
    ).reset_index(
        drop=True
    )

    print()

    for i, (_, row) in enumerate(
        result.head(15).iterrows(),
        1
    ):

        corr = row[
            "相関係数"
        ]

        spread = row[
            "Q5-Q1平均差枚"
        ]

        score = row[
            "要因強度スコア"
        ]

        corr_text = (
            "N/A"
            if pd.isna(corr)
            else f"{corr:+.4f}"
        )

        spread_text = (
            "N/A"
            if pd.isna(spread)
            else f"{spread:+.1f}枚"
        )

        print(
            f"{i:2d}. "
            f"{row['要因']} / "
            f"相関 {corr_text} / "
            f"Q5-Q1 {spread_text} / "
            f"強度 {score:.1f}"
        )

    return result


# ======================================================================
# 複合要因分析
# ======================================================================

def combined_factor_analysis(
    feature_df
):

    print_header(
        "【複合要因分析】"
    )

    data = feature_df.copy()

    base_features = [

        "台_過去平均差枚",
        "台_過去プラス率",
        "台_直近3日平均",
        "台_前日差枚",
        "台_凹み",

        "機種_過去平均差枚",
        "機種_過去プラス率",
        "機種_直近3日平均"
    ]

    for feature in base_features:

        rank_column = (
            feature
            + "_順位"
        )

        data[rank_column] = (
            data
            .groupby("予測日")[feature]
            .rank(pct=True)
        )

    combinations = {

        "平均差枚重視": [
            "台_過去平均差枚_順位",
            "機種_過去平均差枚_順位"
        ],

        "直近重視": [
            "台_直近3日平均_順位",
            "機種_直近3日平均_順位"
        ],

        "プラス率重視": [
            "台_過去プラス率_順位",
            "機種_過去プラス率_順位"
        ],

        "台番号重視": [
            "台_過去平均差枚_順位",
            "台_過去プラス率_順位",
            "台_直近3日平均_順位"
        ],

        "機種重視": [
            "機種_過去平均差枚_順位",
            "機種_過去プラス率_順位",
            "機種_直近3日平均_順位"
        ],

        "バランス型": [
            "台_過去平均差枚_順位",
            "台_過去プラス率_順位",
            "台_直近3日平均_順位",
            "機種_過去平均差枚_順位",
            "機種_過去プラス率_順位",
            "機種_直近3日平均_順位"
        ]
    }

    results = []

    for name, columns in combinations.items():

        data["複合スコア"] = (
            data[columns]
            .mean(axis=1)
        )

        threshold = (
            data
            .groupby("予測日")[
                "複合スコア"
            ]
            .transform(
                lambda x:
                x.quantile(0.8)
            )
        )

        selected = data[
            data["複合スコア"]
            >= threshold
        ]

        if selected.empty:
            continue

        results.append({

            "複合モデル":
                name,

            "サンプル数":
                len(selected),

            "平均差枚":
                selected["当日差枚"]
                .mean(),

            "プラス率":
                selected["当日プラス"]
                .mean() * 100,

            "+500率":
                selected["当日+500"]
                .mean() * 100,

            "+1000率":
                selected["当日+1000"]
                .mean() * 100,

            "+2000率":
                selected["当日+2000"]
                .mean() * 100,

            "+3000率":
                selected["当日+3000"]
                .mean() * 100
        })

    result = pd.DataFrame(
        results
    )

    if result.empty:
        return result

    result = result.sort_values(
        "平均差枚",
        ascending=False
    ).reset_index(
        drop=True
    )

    print()

    for _, row in result.iterrows():

        print(
            f"{row['複合モデル']} / "
            f"平均差枚 "
            f"{row['平均差枚']:+.1f}枚 / "
            f"プラス率 "
            f"{row['プラス率']:.1f}% / "
            f"+1000率 "
            f"{row['+1000率']:.1f}%"
        )

    return result


# ======================================================================
# サマリー
# ======================================================================

def create_summary(
    feature_df,
    factor_df,
    combined_df
):

    rows = []

    rows.append({
        "分析項目":
            "総サンプル数",
        "値":
            len(feature_df)
    })

    rows.append({
        "分析項目":
            "予測日数",
        "値":
            feature_df[
                "予測日"
            ].nunique()
    })

    rows.append({
        "分析項目":
            "平均当日差枚",
        "値":
            feature_df[
                "当日差枚"
            ].mean()
    })

    rows.append({
        "分析項目":
            "当日プラス率",
        "値":
            feature_df[
                "当日プラス"
            ].mean() * 100
    })

    rows.append({
        "分析項目":
            "当日+1000率",
        "値":
            feature_df[
                "当日+1000"
            ].mean() * 100
    })

    if not factor_df.empty:

        best = factor_df.iloc[0]

        rows.append({
            "分析項目":
                "最有力要因",
            "値":
                best["要因"]
        })

        rows.append({
            "分析項目":
                "最有力要因_相関",
            "値":
                best["相関係数"]
        })

        rows.append({
            "分析項目":
                "最有力要因_Q5-Q1",
            "値":
                best["Q5-Q1平均差枚"]
        })

    if not combined_df.empty:

        best = combined_df.iloc[0]

        rows.append({
            "分析項目":
                "最良複合モデル",
            "値":
                best["複合モデル"]
        })

        rows.append({
            "分析項目":
                "最良複合モデル_平均差枚",
            "値":
                best["平均差枚"]
        })

        rows.append({
            "分析項目":
                "最良複合モデル_プラス率",
            "値":
                best["プラス率"]
        })

    return pd.DataFrame(
        rows
    )


# ======================================================================
# メイン
# ======================================================================

def main():

    print_header(
        "予測要因別・的中分析 V1.1"
    )

    print()
    print(
        "V5を作る前に、"
        "どの要因が翌日の実績に効いているかを分析します。"
    )

    print()
    print(
        "※予測対象日の実績は"
        "特徴量計算には使用しません。"
    )

    # ----------------------------------------------------------
    # 読み込み
    # ----------------------------------------------------------

    raw_df = load_data()

    df = prepare_columns(
        raw_df
    )

    # ----------------------------------------------------------
    # 特徴量
    # ----------------------------------------------------------

    print_header(
        "【予測特徴量作成】"
    )

    feature_df = create_feature_data(
        df
    )

    if feature_df.empty:

        raise ValueError(
            "分析可能なデータが作成できませんでした。"
        )

    print()
    print(
        f"特徴量データ: "
        f"{len(feature_df):,}行"
    )

    print(
        f"予測日数: "
        f"{feature_df['予測日'].nunique()}日"
    )

    # ----------------------------------------------------------
    # 相関
    # ----------------------------------------------------------

    correlation_df = (
        correlation_analysis(
            feature_df
        )
    )

    # ----------------------------------------------------------
    # 5分位
    # ----------------------------------------------------------

    quintile_df = (
        all_quintile_analysis(
            feature_df
        )
    )

    # ----------------------------------------------------------
    # 要因ランキング
    # ----------------------------------------------------------

    factor_df = factor_ranking(
        correlation_df,
        quintile_df
    )

    # ----------------------------------------------------------
    # 複合要因
    # ----------------------------------------------------------

    combined_df = (
        combined_factor_analysis(
            feature_df
        )
    )

    # ----------------------------------------------------------
    # サマリー
    # ----------------------------------------------------------

    summary_df = create_summary(
        feature_df,
        factor_df,
        combined_df
    )

    # ----------------------------------------------------------
    # 保存
    # ----------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    feature_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    summary_parts = []

    if not correlation_df.empty:

        temp = correlation_df.copy()

        temp["分析種類"] = "相関"

        summary_parts.append(
            temp
        )

    if not quintile_df.empty:

        temp = quintile_df.copy()

        temp["分析種類"] = "5分位"

        summary_parts.append(
            temp
        )

    if not factor_df.empty:

        temp = factor_df.copy()

        temp["分析種類"] = (
            "要因ランキング"
        )

        summary_parts.append(
            temp
        )

    if not combined_df.empty:

        temp = combined_df.copy()

        temp["分析種類"] = (
            "複合要因"
        )

        summary_parts.append(
            temp
        )

    if summary_parts:

        final_summary = pd.concat(
            summary_parts,
            ignore_index=True,
            sort=False
        )

    else:

        final_summary = summary_df

    final_summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ----------------------------------------------------------
    # 最終表示
    # ----------------------------------------------------------

    print_header(
        "【分析結果サマリー】"
    )

    for _, row in summary_df.iterrows():

        value = row["値"]

        if isinstance(
            value,
            (float, np.floating)
        ):

            if pd.isna(value):

                text = "N/A"

            else:

                text = f"{value:.2f}"

        else:

            text = str(value)

        print(
            f"{row['分析項目']} : {text}"
        )

    print()

    print(
        "★ CSV保存成功"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        "★ CSV保存成功"
    )

    print(
        SUMMARY_FILE
    )

    print()

    print_header(
        "★★★★★ 予測要因別・的中分析 完了 ★★★★★"
    )

    print()
    print(
        "all_data.csv は変更していません。"
    )


# ======================================================================
# 実行
# ======================================================================

if __name__ == "__main__":
    main()