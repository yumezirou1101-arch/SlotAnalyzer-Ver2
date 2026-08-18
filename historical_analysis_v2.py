import csv
import os
from collections import defaultdict


# ==========================================
# 設定
# ==========================================

INPUT_FILE = "all_data.csv"

# 前日差枚の区分
BIG_PLUS = 3000
BIG_MINUS = -3000


# ==========================================
# CSV読み込み
# ==========================================

if not os.path.exists(INPUT_FILE):

    print()
    print("all_data.csv が見つかりません。")
    print()
    print("先に merge_all_data.py を実行してください。")
    exit()


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)
    rows = list(reader)


if not rows:

    print()
    print("all_data.csv にデータがありません。")
    exit()


# ==========================================
# 数値変換
# ==========================================

def to_int(value):

    try:
        return int(
            str(value)
            .replace(",", "")
            .strip()
        )
    except:
        return 0


for row in rows:

    row["_G数"] = to_int(
        row.get("G数", 0)
    )

    row["_差枚"] = to_int(
        row.get("差枚", 0)
    )


# ==========================================
# 日付順に並べる
# ==========================================

rows.sort(
    key=lambda x: (
        x.get("日付", ""),
        to_int(x.get("台番号", 0))
    )
)


# ==========================================
# 基本統計
# ==========================================

def calc_stats(data):

    if not data:

        return {
            "台数": 0,
            "平均G": 0,
            "平均差枚": 0,
            "中央値": 0,
            "プラス率": 0,
            "1000枚以上率": 0,
            "3000枚以上率": 0
        }

    diffs = sorted(
        x["_差枚"]
        for x in data
    )

    count = len(data)

    total_g = sum(
        x["_G数"]
        for x in data
    )

    total_diff = sum(
        x["_差枚"]
        for x in data
    )

    plus_count = sum(
        x["_差枚"] > 0
        for x in data
    )

    plus1000_count = sum(
        x["_差枚"] >= 1000
        for x in data
    )

    plus3000_count = sum(
        x["_差枚"] >= 3000
        for x in data
    )

    middle = count // 2

    if count % 2 == 0:

        median = (
            diffs[middle - 1]
            + diffs[middle]
        ) / 2

    else:

        median = diffs[middle]

    return {
        "台数": count,
        "平均G": total_g / count,
        "平均差枚": total_diff / count,
        "中央値": median,
        "プラス率": plus_count / count * 100,
        "1000枚以上率": plus1000_count / count * 100,
        "3000枚以上率": plus3000_count / count * 100
    }


# ==========================================
# 日付一覧
# ==========================================

dates = sorted(
    set(
        row.get("日付", "")
        for row in rows
    )
)


# ==========================================
# 台番号ごとの日別データ
# ==========================================

machine_history = defaultdict(list)

for row in rows:

    machine_number = row.get(
        "台番号",
        ""
    )

    machine_history[
        machine_number
    ].append(row)


# 日付順にする
for machine_number in machine_history:

    machine_history[machine_number].sort(
        key=lambda x: x.get("日付", "")
    )


# ==========================================
# 開始
# ==========================================

print()
print("========================================")
print(" SlotAnalyzer 過去データ分析 V2")
print("========================================")
print()

print(
    "読み込みデータ:",
    INPUT_FILE
)

print(
    "総台データ:",
    len(rows)
)

print(
    "収録日数:",
    len(dates)
)

if dates:

    print(
        "収録期間:",
        dates[0],
        "～",
        dates[-1]
    )


# ==========================================
# 前日 → 翌日分析
# ==========================================

print()
print("========================================")
print(" 【前日 → 翌日分析】")
print("========================================")
print()


transitions = []


for machine_number, history in machine_history.items():

    for i in range(1, len(history)):

        previous = history[i - 1]
        current = history[i]

        # 本当に連続した営業日かどうかではなく、
        # 同じ台番号の「前回データ → 次回データ」として扱う
        transitions.append(
            {
                "台番号": machine_number,
                "機種名": current.get(
                    "機種名",
                    ""
                ),
                "前日": previous,
                "翌日": current
            }
        )


print(
    "比較できる台データ:",
    len(transitions)
)


# ==========================================
# 前日差枚区分
# ==========================================

transition_groups = {
    "前日3000枚以上": [],
    "前日1000～2999枚": [],
    "前日0～999枚": [],
    "前日-1～-999枚": [],
    "前日-1000～-2999枚": [],
    "前日-3000枚以下": []
}


for item in transitions:

    diff = item["前日"]["_差枚"]

    if diff >= 3000:

        transition_groups[
            "前日3000枚以上"
        ].append(item)

    elif diff >= 1000:

        transition_groups[
            "前日1000～2999枚"
        ].append(item)

    elif diff >= 0:

        transition_groups[
            "前日0～999枚"
        ].append(item)

    elif diff >= -1000:

        transition_groups[
            "前日-1～-999枚"
        ].append(item)

    elif diff >= -3000:

        transition_groups[
            "前日-1000～-2999枚"
        ].append(item)

    else:

        transition_groups[
            "前日-3000枚以下"
        ].append(item)


for name, data in transition_groups.items():

    count = len(data)

    if count == 0:

        print(
            f"{name}: データなし"
        )

        continue

    next_plus = sum(
        x["翌日"]["_差枚"] > 0
        for x in data
    )

    next_big_plus = sum(
        x["翌日"]["_差枚"] >= 3000
        for x in data
    )

    avg_next = sum(
        x["翌日"]["_差枚"]
        for x in data
    ) / count

    print(
        f"{name}"
    )

    print(
        f"  件数: {count} "
        f"翌日平均差枚: {avg_next:.1f} "
        f"翌日プラス率: "
        f"{next_plus / count * 100:.1f}% "
        f"翌日3000枚以上率: "
        f"{next_big_plus / count * 100:.1f}%"
    )


# ==========================================
# 上げ・据え置き・下げ分析
# ==========================================

print()
print("========================================")
print(" 【前日 → 翌日の上げ・据え置き・下げ】")
print("========================================")
print()

# ここでは差枚の変化そのものを確認する
# 設定変更を直接意味するものではない

up_data = []
same_data = []
down_data = []


for item in transitions:

    previous_diff = item["前日"]["_差枚"]
    current_diff = item["翌日"]["_差枚"]

    change = current_diff - previous_diff

    if change >= 1000:

        up_data.append(item)

    elif change <= -1000:

        down_data.append(item)

    else:

        same_data.append(item)


def print_transition_group(
    name,
    data
):

    if not data:

        print(
            f"{name}: データなし"
        )

        return

    avg_change = sum(
        x["翌日"]["_差枚"]
        - x["前日"]["_差枚"]
        for x in data
    ) / len(data)

    next_plus = sum(
        x["翌日"]["_差枚"] > 0
        for x in data
    )

    print(
        f"{name}: {len(data)}件 "
        f"平均差枚変化: {avg_change:.1f} "
        f"翌日プラス率: "
        f"{next_plus / len(data) * 100:.1f}%"
    )


print_transition_group(
    "上昇1000枚以上",
    up_data
)

print_transition_group(
    "ほぼ横ばい",
    same_data
)

print_transition_group(
    "下降1000枚以上",
    down_data
)


# ==========================================
# 台番号別履歴
# ==========================================

print()
print("========================================")
print(" 【台番号別 成績】")
print("========================================")
print()


number_results = []


for machine_number, history in machine_history.items():

    stats = calc_stats(history)

    machine_name = history[-1].get(
        "機種名",
        ""
    )

    number_results.append(
        (
            machine_number,
            machine_name,
            stats,
            history
        )
    )


# 平均差枚順
number_results.sort(
    key=lambda x: x[2]["平均差枚"],
    reverse=True
)


for rank, (
    machine_number,
    machine_name,
    stats,
    history
) in enumerate(
    number_results[:30],
    1
):

    print(
        f"{rank:2d}. "
        f"{machine_number} "
        f"{machine_name} "
        f"データ:{stats['台数']} "
        f"平均差枚:{stats['平均差枚']:.1f} "
        f"中央値:{stats['中央値']:.1f} "
        f"プラス率:{stats['プラス率']:.1f}%"
    )


# ==========================================
# 台番号別 連続プラス
# ==========================================

print()
print("========================================")
print(" 【台番号別 連続プラス】")
print("========================================")
print()


streak_results = []


for machine_number, history in machine_history.items():

    current_streak = 0
    max_streak = 0

    for row in history:

        if row["_差枚"] > 0:

            current_streak += 1

            if current_streak > max_streak:

                max_streak = current_streak

        else:

            current_streak = 0

    if max_streak >= 2:

        machine_name = history[-1].get(
            "機種名",
            ""
        )

        streak_results.append(
            (
                max_streak,
                machine_number,
                machine_name,
                len(history)
            )
        )


streak_results.sort(
    reverse=True
)


if not streak_results:

    print(
        "2日以上連続プラスの台はありません。"
    )

else:

    for rank, item in enumerate(
        streak_results[:30],
        1
    ):

        max_streak, machine_number, machine_name, count = item

        print(
            f"{rank:2d}. "
            f"{machine_number} "
            f"{machine_name} "
            f"最大連続プラス:{max_streak}回 "
            f"データ:{count}"
        )


# ==========================================
# 機種 × 曜日
# ==========================================

print()
print("========================================")
print(" 【機種 × 曜日】")
print("========================================")
print()


machine_weekday = defaultdict(list)


for row in rows:

    key = (
        row.get("機種名", ""),
        row.get("曜日", "")
    )

    machine_weekday[key].append(
        row
    )


machine_weekday_results = []


for key, data in machine_weekday.items():

    machine_name, weekday = key

    stats = calc_stats(data)

    machine_weekday_results.append(
        (
            machine_name,
            weekday,
            stats
        )
    )


machine_weekday_results.sort(
    key=lambda x: x[2]["平均差枚"],
    reverse=True
)


for rank, (
    machine_name,
    weekday,
    stats
) in enumerate(
    machine_weekday_results[:50],
    1
):

    print(
        f"{rank:2d}. "
        f"{machine_name} "
        f"{weekday}曜日 "
        f"台数:{stats['台数']} "
        f"平均差枚:{stats['平均差枚']:.1f} "
        f"プラス率:{stats['プラス率']:.1f}%"
    )


# ==========================================
# 前日大幅マイナス → 翌日
# ==========================================

print()
print("========================================")
print(" 【前日-3000枚以下 → 翌日】")
print("========================================")
print()


big_minus = transition_groups[
    "前日-3000枚以下"
]


if not big_minus:

    print(
        "該当データなし"
    )

else:

    next_plus = sum(
        x["翌日"]["_差枚"] > 0
        for x in big_minus
    )

    next_big_plus = sum(
        x["翌日"]["_差枚"] >= 3000
        for x in big_minus
    )

    avg_next = sum(
        x["翌日"]["_差枚"]
        for x in big_minus
    ) / len(big_minus)

    print(
        f"対象台数: {len(big_minus)}"
    )

    print(
        f"翌日平均差枚: {avg_next:.1f}"
    )

    print(
        f"翌日プラス率: "
        f"{next_plus / len(big_minus) * 100:.1f}%"
    )

    print(
        f"翌日3000枚以上率: "
        f"{next_big_plus / len(big_minus) * 100:.1f}%"
    )


# ==========================================
# 前日大幅プラス → 翌日
# ==========================================

print()
print("========================================")
print(" 【前日+3000枚以上 → 翌日】")
print("========================================")
print()


big_plus = transition_groups[
    "前日3000枚以上"
]


if not big_plus:

    print(
        "該当データなし"
    )

else:

    next_plus = sum(
        x["翌日"]["_差枚"] > 0
        for x in big_plus
    )

    next_big_plus = sum(
        x["翌日"]["_差枚"] >= 3000
        for x in big_plus
    )

    avg_next = sum(
        x["翌日"]["_差枚"]
        for x in big_plus
    ) / len(big_plus)

    print(
        f"対象台数: {len(big_plus)}"
    )

    print(
        f"翌日平均差枚: {avg_next:.1f}"
    )

    print(
        f"翌日プラス率: "
        f"{next_plus / len(big_plus) * 100:.1f}%"
    )

    print(
        f"翌日3000枚以上率: "
        f"{next_big_plus / len(big_plus) * 100:.1f}%"
    )


# ==========================================
# 次回分析用の注意
# ==========================================

print()
print("========================================")
print(" 【分析上の注意】")
print("========================================")
print()

print(
    "現在の収録日数:",
    len(dates)
)

if len(dates) < 7:

    print(
        "まだデータが少ないため、"
        "傾向判断は暫定です。"
    )

elif len(dates) < 30:

    print(
        "データが増えてきました。"
        "曜日・機種・台番号の傾向分析に"
        "利用できます。"
    )

else:

    print(
        "十分なデータ量になってきました。"
        "予測モデルへの統合を検討できます。"
    )


print()
print("========================================")
print(" V2 分析完了")
print("========================================")
print()