# -*- coding: utf-8 -*-
"""
アナスロ 狙い台スコア Ver.3
対象: マルハンメガシティ前橋インター
期間: 2026-07-11 ～ 2026-08-10

目的
- Ver.2の「少数サンプルの最強曜日」による過大評価を抑制
- 直近7日、31日平均、曜日、前日→翌日、機種傾向、隣接傾向を統合
- データ日数に応じた信頼度補正
- 将来日の実績を一切使わない予測形式

重要:
このスクリプトは探索・検証用。設定投入や勝利を保証するものではありません。
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import math
import pandas as pd
import numpy as np

BASE = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
DATA_DIR = BASE / "data" / "maruhan_maebashi" / "machine_number"
OUT_DIR = DATA_DIR / "analysis_31days_deep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_0711 = DATA_DIR / "ana_slo_20260711.csv"
CSV_0712_0810 = DATA_DIR / "ana_slo_20260712_20260810.csv"

OUTPUT = OUT_DIR / "06_狙い台スコア_Ver3.csv"
SUMMARY = OUT_DIR / "06_狙い台スコア_Ver3_上位30台.txt"

START = pd.Timestamp("2026-07-11")
END = pd.Timestamp("2026-08-10")
EXPECTED_MACHINES = 514

# 最低履歴日数。
# これ未満でも参考値として計算するが、信頼度を強く下げる。
MIN_HISTORY = 15

# Empirical Bayes の縮約強度。
# 勝率・強出率を「全体平均」に寄せる。
PRIOR_N = 15.0


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"CSVを読み込めません: {path}")


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_data():
    frames = []
    if CSV_0711.exists():
        frames.append(read_csv(CSV_0711))
    if CSV_0712_0810.exists():
        frames.append(read_csv(CSV_0712_0810))

    if not frames:
        raise FileNotFoundError("対象CSVが見つかりません。")

    df = pd.concat(frames, ignore_index=True)

    date_col = find_col(df, ["date", "日付", "譌･莉・"])
    no_col = find_col(df, ["machine_no", "台番号", "台番", "蜿ｰ逡ｪ蜿ｷ"])
    name_col = find_col(df, ["machine_name", "機種名", "讖溽ｨｮ蜷・", "讖溽ｨｮ"])
    diff_col = find_col(df, ["diff", "差枚", "蟾ｮ譫・"])

    if not all([date_col, no_col, name_col, diff_col]):
        raise ValueError(
            f"必要列が見つかりません: date={date_col}, no={no_col}, "
            f"name={name_col}, diff={diff_col}"
        )

    df = df.rename(columns={
        date_col: "date",
        no_col: "machine_no",
        name_col: "machine_name",
        diff_col: "diff",
    })

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["machine_no"] = pd.to_numeric(df["machine_no"], errors="coerce")
    df["diff"] = (
    df["diff"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("+", "", regex=False)
    .str.strip()
)

    df["diff"] = pd.to_numeric(
    df["diff"],
    errors="coerce"
)

    df = df.dropna(subset=["date", "machine_no", "diff"]).copy()
    df["machine_no"] = df["machine_no"].astype(int)
    df["machine_name"] = df["machine_name"].astype(str).str.strip()
    df = df[(df["date"] >= START) & (df["date"] <= END)].copy()

    # 1日1台にする。重複があれば最後を採用。
    df = df.sort_values(["date", "machine_no"])
    df = df.drop_duplicates(["date", "machine_no"], keep="last")

    df["win"] = (df["diff"] > 0).astype(int)
    df["plus500"] = (df["diff"] >= 500).astype(int)
    df["plus1000"] = (df["diff"] >= 1000).astype(int)
    df["plus2000"] = (df["diff"] >= 2000).astype(int)
    df["plus3000"] = (df["diff"] >= 3000).astype(int)

    return df


def eb_rate(successes, n, prior_rate, prior_n=PRIOR_N):
    """二項比率を全体平均へ縮約。少数サンプルの100%/0%を抑える。"""
    if n <= 0:
        return prior_rate
    return (successes + prior_rate * prior_n) / (n + prior_n)


def minmax(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    if len(s) == 0 or s.max() == s.min():
        return pd.Series(50.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min()) * 100.0


def z_to_50(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    std = float(s.std(ddof=0))
    if std == 0:
        return pd.Series(50.0, index=s.index)
    z = (s - float(s.mean())) / std
    return (50.0 + z * 12.5).clip(0, 100)


def reliability(days):
    """履歴日数に対する信頼度。"""
    if days <= 0:
        return 0.0
    # 31日でほぼ1、15日で約0.83、8日なら約0.65。
    return math.sqrt(min(days, 31) / 31.0)


def build_features(df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    # 予測対象日より前だけを使用
    hist = df[df["date"] < target_date].copy()
    if hist.empty:
        raise ValueError("予測対象日前の履歴がありません。")

    overall_win = float(hist["win"].mean())
    overall_p500 = float(hist["plus500"].mean())
    overall_p1000 = float(hist["plus1000"].mean())
    overall_p2000 = float(hist["plus2000"].mean())
    overall_p3000 = float(hist["plus3000"].mean())

    target_weekday = target_date.dayofweek

    # 機種別母集団
    type_stats = {}
    for name, g in hist.groupby("machine_name"):
        n = len(g)
        type_stats[name] = {
            "avg_diff": float(g["diff"].mean()),
            "win": eb_rate(int(g["win"].sum()), n, overall_win),
            "p1000": eb_rate(int(g["plus1000"].sum()), n, overall_p1000),
            "p2000": eb_rate(int(g["plus2000"].sum()), n, overall_p2000),
        }

    # 全台を対象に、最新日が存在する台を候補とする
    latest = hist.sort_values("date").groupby("machine_no").tail(1)
    candidates = sorted(latest["machine_no"].unique())

    rows = []

    for no in candidates:
        m = hist[hist["machine_no"] == no].sort_values("date").copy()
        if m.empty:
            continue

        name = str(m.iloc[-1]["machine_name"])
        days = len(m)

        diffs = m["diff"].astype(float).tolist()
        last = float(diffs[-1])
        prev = float(diffs[-2]) if len(diffs) >= 2 else last
        prev_change = last - prev

        recent7 = m.tail(7)
        recent3 = m.tail(3)

        avg31 = float(m["diff"].mean())
        win31_raw = float(m["win"].mean())
        p1000_raw = float(m["plus1000"].mean())
        p2000_raw = float(m["plus2000"].mean())
        p3000_raw = float(m["plus3000"].mean())

        # 少数サンプル補正
        win31 = eb_rate(int(m["win"].sum()), days, overall_win)
        p1000 = eb_rate(int(m["plus1000"].sum()), days, overall_p1000)
        p2000 = eb_rate(int(m["plus2000"].sum()), days, overall_p2000)
        p3000 = eb_rate(int(m["plus3000"].sum()), days, overall_p3000)

        recent7_avg = float(recent7["diff"].mean())
        recent7_win = float(recent7["win"].mean())

        # 曜日傾向も縮約。対象曜日の実績が少ないほど全体へ寄せる。
        wd = m[m["date"].dt.dayofweek == target_weekday]
        wd_n = len(wd)
        if wd_n:
            wd_avg_raw = float(wd["diff"].mean())
            wd_win = eb_rate(int(wd["win"].sum()), wd_n, overall_win)
            # 平均差枚も日数に応じて縮約
            wd_weight = wd_n / (wd_n + PRIOR_N)
            wd_avg = wd_avg_raw * wd_weight + avg31 * (1 - wd_weight)
        else:
            wd_avg = avg31
            wd_win = win31

        # 前日→翌日: 「前日が凹み/好調だった時」の翌日差枚
        transitions = []
        prev_map = m.set_index("date")["diff"].to_dict()
        for d, val in prev_map.items():
            nxt = d + pd.Timedelta(days=1)
            if nxt in prev_map:
                transitions.append((float(val), float(prev_map[nxt])))

        if transitions:
            tdf = pd.DataFrame(transitions, columns=["prev_diff", "next_diff"])
            # 前日-500枚以下、-1000枚以下からの翌日平均
            t_neg500 = tdf[tdf["prev_diff"] <= -500]["next_diff"]
            t_neg1000 = tdf[tdf["prev_diff"] <= -1000]["next_diff"]
            next_after_neg500 = float(t_neg500.mean()) if len(t_neg500) else 0.0
            next_after_neg1000 = float(t_neg1000.mean()) if len(t_neg1000) else 0.0
            transition_n = len(transitions)
        else:
            next_after_neg500 = 0.0
            next_after_neg1000 = 0.0
            transition_n = 0

        # 今回の予測で「前日が凹んだ」という事実は使える。
        # ただし大きな単発補正にはしない。
        bounce_signal = 0.0
        if last <= -1000:
            bounce_signal = 1.0
        elif last <= -500:
            bounce_signal = 0.5
        elif last >= 1000:
            bounce_signal = -0.25

        # 隣接台: 直近日の左右台の平均差枚。
        latest_date = hist["date"].max()
        latest_day = hist[hist["date"] == latest_date].set_index("machine_no")
        neighbor_values = []
        for n2 in (no - 1, no + 1):
            if n2 in latest_day.index:
                neighbor_values.append(float(latest_day.loc[n2, "diff"]))
        neighbor_avg = float(np.mean(neighbor_values)) if neighbor_values else 0.0

        # 機種傾向
        ts = type_stats.get(name, {})
        type_avg = float(ts.get("avg_diff", 0.0))
        type_win = float(ts.get("win", overall_win))

        rows.append({
            "machine_no": no,
            "machine_name": name,
            "history_days": days,
            "avg_diff_31": avg31,
            "win_rate_31": win31,
            "win_rate_31_raw": win31_raw,
            "plus1000_rate": p1000,
            "plus2000_rate": p2000,
            "plus3000_rate": p3000,
            "recent7_avg": recent7_avg,
            "recent7_win": recent7_win,
            "recent3_avg": float(recent3["diff"].mean()),
            "last_diff": last,
            "prev_change": prev_change,
            "weekday_avg": wd_avg,
            "weekday_win": wd_win,
            "weekday_n": wd_n,
            "type_avg": type_avg,
            "type_win": type_win,
            "next_after_neg500": next_after_neg500,
            "next_after_neg1000": next_after_neg1000,
            "transition_n": transition_n,
            "bounce_signal": bounce_signal,
            "neighbor_avg": neighbor_avg,
            "reliability": reliability(days),
        })

    out = pd.DataFrame(rows)

    # 正規化
    out["s_avg31"] = z_to_50(out["avg_diff_31"])
    out["s_recent7"] = z_to_50(out["recent7_avg"])
    out["s_recent7_win"] = z_to_50(out["recent7_win"])
    out["s_last"] = z_to_50(out["last_diff"])
    out["s_prev_change"] = z_to_50(out["prev_change"])
    out["s_weekday"] = z_to_50(out["weekday_avg"])
    out["s_type"] = z_to_50(out["type_avg"])
    out["s_p1000"] = z_to_50(out["plus1000_rate"])
    out["s_p2000"] = z_to_50(out["plus2000_rate"])
    out["s_neighbor"] = z_to_50(out["neighbor_avg"])

    # ------------------------------------------------------------
    # Ver.3 重み
    #
    # 長期安定性       18%
    # 直近7日          18%
    # 直近7日勝率       8%
    # 前日差枚          8%
    # 前回からの変化    7%
    # 曜日              8%
    # 機種傾向          8%
    # +1000/+2000出率  8%
    # 隣接傾向          4%
    # 前日凹み補正      3%
    # ------------------------------------------------------------
    raw = (
        out["s_avg31"] * 0.18
        + out["s_recent7"] * 0.18
        + out["s_recent7_win"] * 0.08
        + out["s_last"] * 0.08
        + out["s_prev_change"] * 0.07
        + out["s_weekday"] * 0.08
        + out["s_type"] * 0.08
        + ((out["s_p1000"] * 0.5) + (out["s_p2000"] * 0.5)) * 0.08
        + out["s_neighbor"] * 0.04
        + out["bounce_signal"] * 3.0
    )

    # 履歴の少ない台を強く抑制。
    # 31日=1、15日=約0.83、8日=約0.51。
    confidence = out["reliability"]

    # rawは50を中心とするスコアなので、信頼度補正も50へ縮約する。
    out["狙い台スコアVer3"] = (
        50.0 + (raw - 50.0) * confidence
    ).clip(0, 100)

    # 信頼度ランク
    out["信頼度"] = np.select(
        [
            out["history_days"] >= 25,
            out["history_days"] >= 20,
            out["history_days"] >= 15,
        ],
        ["高", "中", "低"],
        default="参考"
    )

    # 理由を簡潔に生成
    def reason(r):
        reasons = []
        if r["recent7_avg"] >= 300:
            reasons.append("直近7日プラス")
        if r["avg_diff_31"] >= 300:
            reasons.append("31日平均プラス")
        if r["weekday_avg"] >= 300:
            reasons.append("対象曜日強め")
        if r["last_diff"] <= -500:
            reasons.append("前日凹み")
        if r["plus2000_rate"] >= 0.15:
            reasons.append("+2000出率高")
        if r["type_avg"] >= 200:
            reasons.append("機種傾向良")
        if r["neighbor_avg"] >= 300:
            reasons.append("隣接台強め")
        if not reasons:
            reasons.append("複数要素の総合評価")
        return " / ".join(reasons[:4])

    out["主な根拠"] = out.apply(reason, axis=1)

    out = out.sort_values(
        ["狙い台スコアVer3", "信頼度", "history_days"],
        ascending=[False, True, False]
    ).reset_index(drop=True)

    out.insert(0, "順位", np.arange(1, len(out) + 1))

    return out


def main():
    print("=" * 70)
    print("アナスロ 狙い台スコア Ver.3")
    print("=" * 70)

    df = load_data()

    # 次回予測日 = 履歴最終日の翌日
    latest_date = df["date"].max()
    target_date = latest_date + pd.Timedelta(days=1)

    print(f"履歴最終日: {latest_date.date()}")
    print(f"予測対象日: {target_date.date()}")
    print(f"レコード: {len(df):,}")

    result = build_features(df, target_date)

    cols = [
        "順位", "machine_no", "machine_name", "history_days",
        "信頼度", "狙い台スコアVer3",
        "avg_diff_31", "win_rate_31",
        "recent7_avg", "recent7_win",
        "last_diff", "prev_change",
        "weekday_avg", "weekday_win", "weekday_n",
        "type_avg", "plus1000_rate", "plus2000_rate", "plus3000_rate",
        "next_after_neg500", "next_after_neg1000",
        "neighbor_avg", "transition_n",
        "主な根拠",
    ]

    result[cols].to_csv(
        OUTPUT, index=False, encoding="utf-8-sig"
    )

    top = result.head(30)

    lines = [
        "アナスロ 狙い台スコア Ver.3 上位30台",
        f"履歴期間: {START.date()} ～ {latest_date.date()}",
        f"予測対象日: {target_date.date()}",
        f"対象台数: {len(result)}",
        "",
    ]

    for _, r in top.iterrows():
        lines.append(
            f'{int(r["順位"]):2d}. 台{int(r["machine_no"])} '
            f'{r["machine_name"]} '
            f'スコア={r["狙い台スコアVer3"]:.2f} '
            f'信頼度={r["信頼度"]} '
            f'31日平均={r["avg_diff_31"]:+.0f} '
            f'直近7日={r["recent7_avg"]:+.0f} '
            f'前日={r["last_diff"]:+.0f} '
            f'曜日={r["weekday_avg"]:+.0f} '
            f'根拠={r["主な根拠"]}'
        )

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("保存:")
    print(OUTPUT)
    print(SUMMARY)
    print()
    print("===== 上位30台 =====")
    print("\n".join(lines[4:]))
    print()
    print("完了")


if __name__ == "__main__":
    main()
