import csv
import math
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment


# ==========================================
# マイジャグラーV 設定スペック
# 北電子公式スペック
# ==========================================

SETTINGS = {
    1: {
        "bb": 273.1,
        "rb": 409.6,
        "combined": 163.8,
        "payout": 97.0,
    },
    2: {
        "bb": 270.8,
        "rb": 385.5,
        "combined": 159.1,
        "payout": 98.0,
    },
    3: {
        "bb": 266.4,
        "rb": 336.1,
        "combined": 148.6,
        "payout": 99.9,
    },
    4: {
        "bb": 254.0,
        "rb": 290.0,
        "combined": 135.4,
        "payout": 102.8,
    },
    5: {
        "bb": 240.1,
        "rb": 268.6,
        "combined": 126.8,
        "payout": 105.3,
    },
    6: {
        "bb": 229.1,
        "rb": 229.1,
        "combined": 114.6,
        "payout": 109.4,
    },
}


# ==========================================
# CSV読み込み
# ==========================================

with open(
    "slot_data.csv",
    "r",
    encoding="utf-8-sig"
) as f:

    rows = list(csv.DictReader(f))


# ==========================================
# 数値変換
# ==========================================

def to_int(value):
    try:
        return int(value)
    except:
        return 0


# ==========================================
# 対象機種
# ==========================================

target = [
    row
    for row in rows
    if row["機種名"] == "マイV"
]


print("マイV 台数:", len(target))


# ==========================================
# ポアソン分布の対数尤度
# ==========================================

def poisson_log_likelihood(k, lam):

    if lam <= 0:
        return -999999

    return (
        k * math.log(lam)
        - lam
        - math.lgamma(k + 1)
    )


# ==========================================
# 各台を設定1～6で評価
# ==========================================

results = []


for row in target:

    dai = row["台番号"]

    g = to_int(row["G数"])
    bb = to_int(row["BB"])
    rb = to_int(row["RB"])
    diff = to_int(row["差枚"])

    if g <= 0:
        continue

    # 実測確率
    bb_rate = g / bb if bb > 0 else None
    rb_rate = g / rb if rb > 0 else None
    combined_count = bb + rb
    combined_rate = (
        g / combined_count
        if combined_count > 0
        else None
    )

    likelihoods = {}

    # ======================================
    # 設定1～6
    # ======================================

    for setting, spec in SETTINGS.items():

        bb_lambda = g / spec["bb"]
        rb_lambda = g / spec["rb"]

        ll_bb = poisson_log_likelihood(
            bb,
            bb_lambda
        )

        ll_rb = poisson_log_likelihood(
            rb,
            rb_lambda
        )

        # BB + RBの尤度
        likelihoods[setting] = (
            ll_bb + ll_rb
        )

    # ======================================
    # 最大尤度
    # ======================================

    max_ll = max(
        likelihoods.values()
    )

    # ======================================
    # 相対尤度 → 確率化
    # ======================================

    weights = {}

    total_weight = 0

    for setting, ll in likelihoods.items():

        weight = math.exp(
            ll - max_ll
        )

        weights[setting] = weight
        total_weight += weight

    probabilities = {}

    for setting in SETTINGS:

        probabilities[setting] = (
            weights[setting]
            / total_weight
            * 100
        )

    # ======================================
    # 最有力設定
    # ======================================

    best_setting = max(
        probabilities,
        key=probabilities.get
    )

    best_probability = probabilities[
        best_setting
    ]

    # ======================================
    # 高設定確率
    # ======================================

    high_setting_probability = (
        probabilities[4]
        + probabilities[5]
        + probabilities[6]
    )

    # ======================================
    # 期待設定値
    # ======================================

    expected_setting = sum(
        setting * probabilities[setting]
        for setting in probabilities
    ) / 100

    # ======================================
    # 結果
    # ======================================

    results.append([
        dai,
        g,
        bb,
        rb,
        diff,
        bb_rate,
        rb_rate,
        combined_rate,

        round(probabilities[1], 2),
        round(probabilities[2], 2),
        round(probabilities[3], 2),
        round(probabilities[4], 2),
        round(probabilities[5], 2),
        round(probabilities[6], 2),

        round(high_setting_probability, 2),
        round(expected_setting, 2),

        best_setting,
        round(best_probability, 2),
    ])


# ==========================================
# 高設定確率順
# ==========================================

results.sort(
    key=lambda x: x[14],
    reverse=True
)


# ==========================================
# Excel
# ==========================================

wb = Workbook()

ws = wb.active
ws.title = "マイV設定推測"


headers = [
    "台番号",
    "G数",
    "BB",
    "RB",
    "差枚",
    "実測BB確率",
    "実測RB確率",
    "実測合成",

    "設定1確率%",
    "設定2確率%",
    "設定3確率%",
    "設定4確率%",
    "設定5確率%",
    "設定6確率%",

    "設定4-6確率%",
    "期待設定値",
    "最有力設定",
    "最有力確率%",
]

ws.append(headers)


for row in results:
    ws.append(row)


# ==========================================
# 書式
# ==========================================

for cell in ws[1]:

    cell.font = Font(
        bold=True
    )

    cell.alignment = Alignment(
        horizontal="center"
    )


ws.auto_filter.ref = ws.dimensions
ws.freeze_panes = "A2"


widths = {
    "A": 10,
    "B": 10,
    "C": 8,
    "D": 8,
    "E": 10,
    "F": 14,
    "G": 14,
    "H": 14,
    "I": 13,
    "J": 13,
    "K": 13,
    "L": 13,
    "M": 13,
    "N": 13,
    "O": 14,
    "P": 12,
    "Q": 12,
    "R": 14,
}


for col, width in widths.items():
    ws.column_dimensions[col].width = width


# ==========================================
# 保存
# ==========================================

filename = "setting_model_myv.xlsx"

wb.save(filename)


# ==========================================
# コンソール
# ==========================================

print()
print("保存完了")
print("対象台数:", len(results))
print("ファイル:", filename)

print()
print("===== マイV 設定推測ランキング =====")

for rank, row in enumerate(
    results,
    start=1
):

    print(
        rank,
        row[0],
        "G:", row[1],
        "BB:", row[2],
        "RB:", row[3],
        "差枚:", row[4],
        "設定4-6:", row[14], "%",
        "期待設定:", row[15],
        "最有力:", row[16],
        "確率:", row[17], "%"
    )