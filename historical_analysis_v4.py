import pandas as pd
import os
from collections import defaultdict

# ========================================
# SlotAnalyzer 過去データ分析 v4
# 複数店舗対応版
# ========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ----------------------------------------
# 店舗設定
# ----------------------------------------

STORES = {
    "1": {
        "name": "マルハンメガシティ前橋インター",
        "folder": "maruhan_maebashi"
    },
    "2": {
        "name": "ビックマーチ高崎大八木店",
        "folder": "bigmarch_takasaki_oyagi"
    },
    "3": {
        "name": "ビックつばめ高崎店",
        "folder": "bigtsubame_takasaki"
    },
    "4": {
        "name": "やすだ前橋店",
        "folder": "yasuda_maebashi"
    }
}


# ----------------------------------------
# 店舗選択
# ----------------------------------------

print()
print("========================================")
print(" SlotAnalyzer 過去データ分析 v4")
print("========================================")
print()

print("店舗を選択してください。")
print()

for number, store in STORES.items():
    print(f"{number}. {store['name']}")

print()

while True:
    choice = input("店舗番号: ").strip()

    if choice in STORES:
        break

    print("1～4の番号を入力してください。")

store = STORES[choice]
store_name = store["name"]
folder_name = store["folder"]

store_dir = os.path.join(DATA_DIR, folder_name)

print()
print(f"選択店舗: {store_name}")
print(f"データフォルダ: {store_dir}")
print()


# ----------------------------------------
# フォルダ確認
# ----------------------------------------

if not os.path.isdir(store_dir):
    print("データフォルダが存在しません。")
    print()
    print(f"確認してください:")
    print(store_dir)
    print()

    input("Enterキーで終了します...")
    exit()


# ----------------------------------------
# CSV読み込み
# ----------------------------------------

csv_files = []

for filename in os.listdir(store_dir):
    if filename.lower().endswith(".csv") and filename != "all_data.csv":
        csv_files.append(filename)

csv_files.sort()

if not csv_files:
    print("日付別CSVがありません。")
    print()
    print(f"フォルダ: {store_dir}")
    print()

    input("Enterキーで終了します...")
    exit()


print(f"日付別CSV: {len(csv_files)} ファイル")
print()


dataframes = []

for filename in csv_files:

    filepath = os.path.join(store_dir, filename)

    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding="utf-8")
        except Exception as e:
            print(f"読み込み失敗: {filename}")
            print(e)
            continue
    except Exception as e:
        print(f"読み込み失敗: {filename}")
        print(e)
        continue

    if df.empty:
        print(f"空ファイル: {filename}")
        continue

    print(f"読み込み: {filename}")
    print(f"→ {len(df)} 台")

    dataframes.append(df)


if not dataframes:
    print()
    print("読み込めるCSVがありません。")
    input("Enterキーで終了します...")
    exit()


# ----------------------------------------
# データ統合
# ----------------------------------------

df_all = pd.concat(dataframes, ignore_index=True)

print()
print("========================================")
print(" データ概要")
print("========================================")
print()

print(f"店舗: {store_name}")
print(f"総台データ: {len(df_all):,}")


# ----------------------------------------
# 列名確認
# ----------------------------------------

required_columns = [
    "日付",
    "台番号",
    "機種名",
    "G数",
    "差枚"
]

missing_columns = [
    col for col in required_columns
    if col not in df_all.columns
]

if missing_columns:

    print()
    print("必要な列が見つかりません。")
    print()

    print("不足している列:")
    for col in missing_columns:
        print(f"  {col}")

    print()
    print("現在の列:")
    print(list(df_all.columns))

    input("Enterキーで終了します...")
    exit()


# ----------------------------------------
# 数値変換
# ----------------------------------------

df_all["台番号"] = pd.to_numeric(
    df_all["台番号"],
    errors="coerce"
)

df_all["G数"] = pd.to_numeric(
    df_all["G数"],
    errors="coerce"
)

df_all["差枚"] = (
    df_all["差枚"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("+", "", regex=False)
)

df_all["差枚"] = pd.to_numeric(
    df_all["差枚"],
    errors="coerce"
)

df_all["日付"] = pd.to_datetime(
    df_all["日付"],
    errors="coerce"
)


# ----------------------------------------
# 不正データ除外
# ----------------------------------------

df_all = df_all.dropna(
    subset=["日付", "台番号", "差枚"]
)

df_all = df_all.sort_values(
    ["日付", "台番号"]
).reset_index(drop=True)


# ----------------------------------------
# 基本情報
# ----------------------------------------

dates = sorted(df_all["日付"].dt.date.unique())

print(f"収録日数: {len(dates)}")

if dates:
    print(
        f"収録期間: {dates[0]} ～ {dates[-1]}"
    )


# ----------------------------------------
# 日付別基本統計
# ----------------------------------------

print()
print("========================================")
print(" 【日付別】")
print("========================================")
print()

for date in dates:

    temp = df_all[
        df_all["日付"].dt.date == date
    ]

    count = len(temp)

    avg_games = temp["G数"].mean()
    avg_diff = temp["差枚"].mean()

    plus_rate = (
        (temp["差枚"] > 0).mean() * 100
    )

    over_1000 = (
        (temp["差枚"] >= 1000).mean() * 100
    )

    over_3000 = (
        (temp["差枚"] >= 3000).mean() * 100
    )

    print(
        f"{date} "
        f"台数:{count} "
        f"平均G:{avg_games:.1f} "
        f"平均差枚:{avg_diff:.1f} "
        f"プラス率:{plus_rate:.1f}% "
        f"1000枚以上率:{over_1000:.1f}% "
        f"3000枚以上率:{over_3000:.1f}%"
    )


# ----------------------------------------
# 機種別分析
# ----------------------------------------

print()
print("========================================")
print(" 【機種別】")
print("========================================")
print()

machine_stats = []

for machine, temp in df_all.groupby("機種名"):

    count = len(temp)

    if count < 2:
        continue

    avg_games = temp["G数"].mean()
    avg_diff = temp["差枚"].mean()

    plus_rate = (
        (temp["差枚"] > 0).mean() * 100
    )

    over_1000 = (
        (temp["差枚"] >= 1000).mean() * 100
    )

    over_3000 = (
        (temp["差枚"] >= 3000).mean() * 100
    )

    machine_stats.append(
        (
            machine,
            count,
            avg_games,
            avg_diff,
            plus_rate,
            over_1000,
            over_3000
        )
    )


machine_stats.sort(
    key=lambda x: x[3],
    reverse=True
)


for item in machine_stats:

    (
        machine,
        count,
        avg_games,
        avg_diff,
        plus_rate,
        over_1000,
        over_3000
    ) = item

    print(
        f"{machine} "
        f"台数:{count} "
        f"平均G:{avg_games:.1f} "
        f"平均差枚:{avg_diff:.1f} "
        f"プラス率:{plus_rate:.1f}% "
        f"1000枚以上率:{over_1000:.1f}% "
        f"3000枚以上率:{over_3000:.1f}%"
    )


# ----------------------------------------
# 台番号別分析
# ----------------------------------------

print()
print("========================================")
print(" 【台番号別】")
print("========================================")
print()

machine_number_stats = []

for number, temp in df_all.groupby("台番号"):

    count = len(temp)

    if count < 2:
        continue

    avg_diff = temp["差枚"].mean()
    median_diff = temp["差枚"].median()

    plus_rate = (
        (temp["差枚"] > 0).mean() * 100
    )

    machine_name = temp["機種名"].iloc[-1]

    machine_number_stats.append(
        (
            number,
            machine_name,
            count,
            avg_diff,
            median_diff,
            plus_rate
        )
    )


machine_number_stats.sort(
    key=lambda x: x[3],
    reverse=True
)


for i, item in enumerate(
    machine_number_stats[:30],
    start=1
):

    (
        number,
        machine_name,
        count,
        avg_diff,
        median_diff,
        plus_rate
    ) = item

    print(
        f"{i}. "
        f"{int(number)} "
        f"{machine_name} "
        f"データ:{count} "
        f"平均差枚:{avg_diff:.1f} "
        f"中央値:{median_diff:.1f} "
        f"プラス率:{plus_rate:.1f}%"
    )


# ----------------------------------------
# 前日 → 翌日分析
# ----------------------------------------

print()
print("========================================")
print(" 【前日差枚別 → 翌日】")
print("========================================")
print()

if len(dates) >= 2:

    previous = df_all.copy()

    previous["日付"] = previous["日付"].dt.date

    next_day = previous.copy()

    previous = previous.rename(
        columns={
            "差枚": "前日差枚"
        }
    )

    next_day = next_day.rename(
        columns={
            "差枚": "翌日差枚"
        }
    )

    previous["翌日"] = previous["日付"].apply(
        lambda x: pd.Timestamp(x) + pd.Timedelta(days=1)
    )

    previous["翌日"] = previous["翌日"].dt.date

    next_day["日付"] = next_day["日付"].apply(
        lambda x: x
    )

    merged = pd.merge(
        previous[
            ["台番号", "翌日", "前日差枚"]
        ],
        next_day[
            ["台番号", "日付", "翌日差枚"]
        ],
        left_on=["台番号", "翌日"],
        right_on=["台番号", "日付"],
        how="inner"
    )

    print(
        f"比較できる台データ: {len(merged)}"
    )
    print()

    def analyze_range(
        name,
        condition
    ):

        temp = merged[condition].copy()

        if len(temp) == 0:
            print(f"{name}")
            print("件数: 0")
            print()
            return

        avg = temp["翌日差枚"].mean()

        plus_rate = (
            (temp["翌日差枚"] > 0).mean()
            * 100
        )

        over_3000 = (
            (temp["翌日差枚"] >= 3000).mean()
            * 100
        )

        print(name)
        print(
            f"件数: {len(temp)} "
            f"翌日平均差枚: {avg:.1f} "
            f"翌日プラス率: {plus_rate:.1f}% "
            f"翌日3000枚以上率: {over_3000:.1f}%"
        )
        print()

    analyze_range(
        "前日3000枚以上",
        merged["前日差枚"] >= 3000
    )

    analyze_range(
        "前日1000～2999枚",
        (
            (merged["前日差枚"] >= 1000)
            &
            (merged["前日差枚"] < 3000)
        )
    )

    analyze_range(
        "前日0～999枚",
        (
            (merged["前日差枚"] >= 0)
            &
            (merged["前日差枚"] < 1000)
        )
    )

    analyze_range(
        "前日-1～-999枚",
        (
            (merged["前日差枚"] < 0)
            &
            (merged["前日差枚"] > -1000)
        )
    )

    analyze_range(
        "前日-1000～-2999枚",
        (
            (merged["前日差枚"] <= -1000)
            &
            (merged["前日差枚"] > -3000)
        )
    )

    analyze_range(
        "前日-3000枚以下",
        merged["前日差枚"] <= -3000
    )

else:

    print("前日→翌日比較には2日以上のデータが必要です。")


# ----------------------------------------
# 終了
# ----------------------------------------

print()
print("========================================")
print(" 分析完了")
print("========================================")
print()

print(f"店舗: {store_name}")
print(f"収録日数: {len(dates)}")
print(f"総台データ: {len(df_all):,}")

print()

if len(dates) < 7:

    print(
        "現在はデータが少ないため、"
        "傾向判断は暫定です。"
    )

elif len(dates) < 30:

    print(
        "データが蓄積されてきました。"
        "曜日・機種・台番号の傾向分析が可能になってきています。"
    )

else:

    print(
        "十分なデータが蓄積されています。"
        "本格的な傾向分析が可能です。"
    )

print()

input("Enterキーで終了します...")