from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
ANA_DIR = ROOT / "data" / "maruhan_maebashi" / "machine_number"
OUT_DIR = ROOT / "data" / "maruhan_maebashi" / "external_validation" / "minrepo_multi"


def num(v):
    if pd.isna(v):
        return np.nan
    m = re.search(r"-?\d+(?:\.\d+)?", str(v).replace(",", "").replace("+", ""))
    return float(m.group()) if m else np.nan


def norm_name(v):
    if pd.isna(v):
        return ""
    s = unicodedata.normalize("NFKC", str(v))
    s = re.sub(r"\s+", "", s)
    # Remove common source-specific machine-type prefixes for comparison only.
    s = re.sub(r"^(スマスロ|Lパチスロ|Lスロット|L|スロット)", "", s)
    return s


def load_ana(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    x = df[["台番号", "機種名", "差枚", "G数"]].copy()
    x.columns = ["machine_no", "ana_name", "ana_diff", "ana_g"]
    x["machine_no"] = pd.to_numeric(x["machine_no"], errors="coerce")
    x["ana_diff"] = x["ana_diff"].map(num)
    x["ana_g"] = x["ana_g"].map(num)
    x = x.dropna(subset=["machine_no"]).copy()
    x["machine_no"] = x["machine_no"].astype(int)
    x["ana_name_norm"] = x["ana_name"].map(norm_name)
    return x


def load_minrepo(path):
    tables = pd.read_html(path)
    candidates = []
    for t in tables:
        cols = [str(c) for c in t.columns]
        if all(c in cols for c in ["機種", "台番", "差枚", "G数"]):
            candidates.append(t)
    if not candidates:
        raise RuntimeError(f"全台表が見つかりません: {path.name}")
    raw = max(candidates, key=len).copy()
    x = raw[["機種", "台番", "差枚", "G数"]].copy()
    x.columns = ["minrepo_name", "machine_no", "minrepo_diff", "minrepo_g"]
    x["machine_no"] = x["machine_no"].map(num)
    x["minrepo_diff"] = x["minrepo_diff"].map(num)
    x["minrepo_g"] = x["minrepo_g"].map(num)
    x = x.dropna(subset=["machine_no"]).copy()
    x["machine_no"] = x["machine_no"].astype(int)
    x["minrepo_name_norm"] = x["minrepo_name"].map(norm_name)
    return raw, x


def pct(s):
    return float(s.mean() * 100) if len(s) else np.nan


def main():
    htmls = sorted(ROOT.glob("minrepo_????????_allmachines.html"))
    if not htmls:
        raise SystemExit("minrepo_YYYYMMDD_allmachines.html が見つかりません。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_matches = []

    print("=" * 105)
    print("Ana-Slo vs Min-Repo MULTI-DAY CROSS CHECK")
    print("=" * 105)

    for html in htmls:
        m = re.fullmatch(r"minrepo_(\d{8})_allmachines\.html", html.name)
        if not m:
            continue
        ymd = m.group(1)
        ana_path = ANA_DIR / f"ana_slo_{ymd}.csv"

        print()
        print("-" * 105)
        print(f"DATE: {ymd}")
        print("-" * 105)

        if not ana_path.exists():
            print(f"[SKIP] Ana-Slo CSVなし: {ana_path}")
            continue

        ana = load_ana(ana_path)
        raw, mr = load_minrepo(html)

        ana_dup = int(ana.duplicated("machine_no").sum())
        mr_dup = int(mr.duplicated("machine_no").sum())

        a = ana.drop_duplicates("machine_no", keep="first")
        b = mr.drop_duplicates("machine_no", keep="first")

        merged = a.merge(b, on="machine_no", how="outer", indicator=True)
        both = merged[merged["_merge"] == "both"].copy()
        ana_only = merged[merged["_merge"] == "left_only"].copy()
        mr_only = merged[merged["_merge"] == "right_only"].copy()

        both["name_match"] = both["ana_name_norm"] == both["minrepo_name_norm"]
        both["diff_delta"] = both["minrepo_diff"] - both["ana_diff"]
        both["g_delta"] = both["minrepo_g"] - both["ana_g"]
        both["diff_exact"] = both["diff_delta"] == 0
        both["g_exact"] = both["g_delta"] == 0
        both["date"] = ymd

        coverage = len(both) / len(a) * 100 if len(a) else np.nan
        diff_mae = float(both["diff_delta"].abs().mean()) if len(both) else np.nan
        g_mae = float(both["g_delta"].abs().mean()) if len(both) else np.nan

        print(f"Ana rows / unique       : {len(ana)} / {ana.machine_no.nunique()}")
        print(f"MinRepo raw/numeric/uniq: {len(raw)} / {len(mr)} / {mr.machine_no.nunique()}")
        print(f"duplicates Ana/MinRepo  : {ana_dup} / {mr_dup}")
        print(f"matched                 : {len(both)}")
        print(f"Ana only / MinRepo only : {len(ana_only)} / {len(mr_only)}")
        print(f"coverage                 : {coverage:.2f}%")
        print(f"name normalized match    : {pct(both['name_match']):.2f}%")
        print(f"diff exact               : {pct(both['diff_exact']):.2f}%")
        print(f"G exact                  : {pct(both['g_exact']):.2f}%")
        print(f"diff MAE                 : {diff_mae:.2f}")
        print(f"G MAE                    : {g_mae:.2f}")

        day_dir = OUT_DIR / ymd
        day_dir.mkdir(parents=True, exist_ok=True)
        both.to_csv(day_dir / "matched.csv", index=False, encoding="utf-8-sig")
        ana_only.to_csv(day_dir / "ana_only.csv", index=False, encoding="utf-8-sig")
        mr_only.to_csv(day_dir / "minrepo_only.csv", index=False, encoding="utf-8-sig")

        summaries.append({
            "date": ymd,
            "ana_rows": len(ana),
            "ana_unique": ana.machine_no.nunique(),
            "minrepo_raw_rows": len(raw),
            "minrepo_numeric_rows": len(mr),
            "minrepo_unique": mr.machine_no.nunique(),
            "matched": len(both),
            "ana_only": len(ana_only),
            "minrepo_only": len(mr_only),
            "coverage_percent": coverage,
            "name_match_percent": pct(both["name_match"]),
            "diff_exact_percent": pct(both["diff_exact"]),
            "g_exact_percent": pct(both["g_exact"]),
            "diff_mae": diff_mae,
            "g_mae": g_mae,
        })
        all_matches.append(both)

    if not summaries:
        raise SystemExit("比較できる日付がありませんでした。")

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT_DIR / "summary_by_date.csv", index=False, encoding="utf-8-sig")

    all_df = pd.concat(all_matches, ignore_index=True)
    overall = pd.DataFrame([{
        "days": len(summary),
        "matched_rows": len(all_df),
        "name_match_percent": pct(all_df["name_match"]),
        "diff_exact_percent": pct(all_df["diff_exact"]),
        "g_exact_percent": pct(all_df["g_exact"]),
        "diff_mae": float(all_df["diff_delta"].abs().mean()),
        "g_mae": float(all_df["g_delta"].abs().mean()),
    }])
    overall.to_csv(OUT_DIR / "summary_overall.csv", index=False, encoding="utf-8-sig")

    print()
    print("=" * 105)
    print("OVERALL")
    print("=" * 105)
    print(summary[["date", "matched", "coverage_percent", "name_match_percent",
                   "diff_exact_percent", "g_exact_percent", "diff_mae", "g_mae"]].to_string(index=False))
    print()
    print(f"days                     : {len(summary)}")
    print(f"matched rows             : {len(all_df)}")
    print(f"name normalized match    : {pct(all_df['name_match']):.2f}%")
    print(f"diff exact               : {pct(all_df['diff_exact']):.2f}%")
    print(f"G exact                  : {pct(all_df['g_exact']):.2f}%")
    print(f"diff MAE                 : {all_df['diff_delta'].abs().mean():.2f}")
    print(f"G MAE                    : {all_df['g_delta'].abs().mean():.2f}")
    print()
    print(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
