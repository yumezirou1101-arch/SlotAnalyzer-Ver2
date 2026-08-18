import csv
from pathlib import Path


# ============================================================
# 狙い台候補判定プログラム
# 第1段階
#
# 入力:
#   investment_score_v2_1.csv
#
# 出力:
#   target_selection.csv
#
# all_data.csv は変更しません。
# ============================================================


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    BASE_DIR
    / "data"
    / "maruhan_maebashi"
)

INPUT_FILE = (
    DATA_DIR
    / "machine_number"
    / "investment_score_v2_1.csv"
)

OUTPUT_FILE = (
    DATA_DIR
    / "machine_number"
    / "target_selection.csv"
)


# ============================================================
# 数値変換
# ============================================================

def to_float(value):

    if value is None:
        return 0.0

    value = str(value).strip()

    if value == "":
        return 0.0

    value = value.replace(",", "")
    value = value.replace("+", "")

    try:
        return float(value)

    except ValueError:
        return 0.0


def to_int(value):

    return int(
        to_float(value)
    )


# ============================================================
# CSV保存
# ============================================================

def save_csv(
    path,
    fieldnames,
    rows
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("★ CSV保存成功")
    print(path)


# ============================================================
# 開始
# ============================================================

print("=" * 70)
print("狙い台候補判定プログラム")
print("=" * 70)

print()
print("入力ファイル:")
print(INPUT_FILE)


if not INPUT_FILE.exists():

    print()
    print("[エラー]")
    print("investment_score_v2_1.csv が見つかりません。")

    input("Enterキーで終了...")
    raise SystemExit


# ============================================================
# CSV読み込み
# ============================================================

print()
print("investment_score_v2_1.csv を読み込みます...")

with open(
    INPUT_FILE,
    "r",
    newline="",
    encoding="utf-8-sig"
) as f:

    reader = csv.DictReader(f)

    rows = list(reader)

    fieldnames = reader.fieldnames


print(
    f"読み込み台数: {len(rows):,}台"
)


required_columns = [
    "台番号",
    "現在機種",
    "分析日数",
    "平均差枚",
    "プラス率",
    "直近3日平均差枚",
    "直近3日プラス率",
    "直近5日平均差枚",
    "信頼度",
    "総合スコア",
    "ランク"
]


missing = [
    column
    for column in required_columns
    if column not in fieldnames
]


if missing:

    print()
    print("[エラー]")
    print("必要な列がありません。")

    for column in missing:
        print(
            f"  {column}"
        )

    input("Enterキーで終了...")
    raise SystemExit


print("必要な列: OK")


# ============================================================
# 狙い台評価
# ============================================================

results = []


for row in rows:

    machine_number = to_int(
        row["台番号"]
    )

    analysis_days = to_int(
        row["分析日数"]
    )

    average_difference = to_float(
        row["平均差枚"]
    )

    positive_rate = to_float(
        row["プラス率"]
    )

    recent_3_difference = to_float(
        row["直近3日平均差枚"]
    )

    recent_3_positive_rate = to_float(
        row["直近3日プラス率"]
    )

    recent_5_difference = to_float(
        row["直近5日平均差枚"]
    )

    confidence = to_float(
        row["信頼度"]
    )

    total_score = to_float(
        row["総合スコア"]
    )


    # ========================================================
    # 狙い台スコア
    #
    # V2.1の総合スコアを中心に、
    # 「次回狙う」という観点を少し加える。
    #
    # 総合スコア        60%
    # 直近3日実績       15%
    # プラス率          10%
    # 信頼度             10%
    # 直近3日プラス率     5%
    # ========================================================

    recent_score = (
        50
        + 50
        * max(
            -1,
            min(
                1,
                recent_3_difference
                / 3000
            )
        )
    )


    recent_score = max(
        0,
        min(
            100,
            recent_score
        )
    )


    target_score = (

        total_score * 0.60

        + recent_score * 0.15

        + positive_rate * 0.10

        + confidence * 0.10

        + recent_3_positive_rate * 0.05
    )


    # ========================================================
    # 少数データ補正
    #
    # 2～3日しかない台は、
    # 高成績でも「狙い台」としては慎重にする。
    # ========================================================

    if analysis_days <= 2:

        target_score -= 8

    elif analysis_days == 3:

        target_score -= 5

    elif analysis_days == 4:

        target_score -= 3

    elif analysis_days == 5:

        target_score -= 1


    target_score = max(
        0,
        min(
            100,
            target_score
        )
    )


    # ========================================================
    # 判定
    # ========================================================

    if target_score >= 75:

        judgment = "◎ 最優先"

    elif target_score >= 68:

        judgment = "○ 有力"

    elif target_score >= 60:

        judgment = "△ 注目"

    elif target_score >= 50:

        judgment = "▲ 候補"

    else:

        judgment = "× 見送り"


    # ========================================================
    # 注意事項
    # ========================================================

    notes = []


    if analysis_days <= 2:

        notes.append(
            "データ不足"
        )


    if average_difference >= 2000:

        notes.append(
            "平均差枚強"
        )


    if positive_rate >= 70:

        notes.append(
            "プラス率高"
        )


    if recent_3_difference >= 2000:

        notes.append(
            "直近強"
        )


    if recent_3_difference <= -1000:

        notes.append(
            "直近弱"
        )


    if confidence < 60:

        notes.append(
            "信頼度注意"
        )


    note_text = (
        " / ".join(notes)
        if notes
        else ""
    )


    results.append({

        "順位": 0,

        "台番号":
            machine_number,

        "機種":
            row["現在機種"],

        "分析日数":
            analysis_days,

        "平均差枚":
            average_difference,

        "プラス率":
            positive_rate,

        "直近3日平均差枚":
            recent_3_difference,

        "直近3日プラス率":
            recent_3_positive_rate,

        "直近5日平均差枚":
            recent_5_difference,

        "信頼度":
            confidence,

        "V2.1総合スコア":
            total_score,

        "狙い台スコア":
            round(
                target_score,
                1
            ),

        "判定":
            judgment,

        "注意事項":
            note_text
    })


# ============================================================
# 狙い台スコア順
# ============================================================

results.sort(
    key=lambda x: (
        x["狙い台スコア"],
        x["信頼度"],
        x["平均差枚"]
    ),
    reverse=True
)


# ============================================================
# 順位付け
# ============================================================

for index, row in enumerate(
    results,
    1
):

    row["順位"] = index


# ============================================================
# TOP30表示
# ============================================================

print()
print("=" * 70)
print("【次回 狙い台候補 TOP30】")
print("=" * 70)


for row in results[:30]:

    print(
        f"{row['順位']:2d}. "
        f"台{row['台番号']} / "
        f"{row['機種']} / "
        f"平均差枚 "
        f"{row['平均差枚']:+.1f}枚 / "
        f"プラス率 "
        f"{row['プラス率']:.1f}% / "
        f"直近3日 "
        f"{row['直近3日平均差枚']:+.1f}枚 / "
        f"V2.1 "
        f"{row['V2.1総合スコア']:.1f} / "
        f"狙い台 "
        f"{row['狙い台スコア']:.1f} / "
        f"{row['判定']}"
    )


# ============================================================
# 判定別集計
# ============================================================

print()
print("=" * 70)
print("【判定別台数】")
print("=" * 70)


judgment_counts = {}


for row in results:

    judgment = row["判定"]

    judgment_counts[judgment] = (
        judgment_counts.get(
            judgment,
            0
        )
        + 1
    )


for judgment in [
    "◎ 最優先",
    "○ 有力",
    "△ 注目",
    "▲ 候補",
    "× 見送り"
]:

    print(
        f"{judgment}: "
        f"{judgment_counts.get(judgment, 0)}台"
    )


# ============================================================
# CSV保存
# ============================================================

output_fields = [

    "順位",
    "台番号",
    "機種",
    "分析日数",
    "平均差枚",
    "プラス率",
    "直近3日平均差枚",
    "直近3日プラス率",
    "直近5日平均差枚",
    "信頼度",
    "V2.1総合スコア",
    "狙い台スコア",
    "判定",
    "注意事項"
]


save_csv(
    OUTPUT_FILE,
    output_fields,
    results
)


# ============================================================
# 完了
# ============================================================

print()
print("=" * 70)
print("★★★★★ 狙い台候補判定 完了 ★★★★★")
print("=" * 70)

print()
print("保存ファイル:")
print(OUTPUT_FILE)

print()
print("all_data.csv は変更していません。")

input("Enterキーで終了...")