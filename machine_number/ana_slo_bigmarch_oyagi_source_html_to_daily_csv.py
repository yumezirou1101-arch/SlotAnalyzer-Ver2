from __future__ import annotations

from pathlib import Path
import argparse
import re
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
OUTPUT_DIR = PROJECT_ROOT / "data" / "bigmarch_takasaki_oyagi" / "machine_number"
DEFAULT_EXPECTED_MACHINES = 276
STORE_NAMES = ("ビックマーチ高崎おおやぎ店", "ビッグマーチ高崎おおやぎ店")

def header(title: str) -> None:
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("html", nargs="?", default=None)
    p.add_argument("--expected-machines", type=int, default=DEFAULT_EXPECTED_MACHINES)
    return p.parse_args()

def resolve_html(text):
    if text:
        p = Path(text)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    files = sorted(
        PROJECT_ROOT.glob("ana_slo_bigmarch_oyagi_????????_source.html"),
        key=lambda x: x.stat().st_mtime,
    )
    if not files:
        raise FileNotFoundError("No Big March source HTML found.")
    return files[-1]

def read_text(path):
    err = None
    for enc in ("utf-8", "utf-8-sig", "cp932"):
        try:
            return path.read_text(encoding=enc)
        except Exception as e:
            err = e
    raise RuntimeError(err)

def detect_date(text):
    m = re.search(r"(20\d{2})/(\d{2})/(\d{2})", text)
    if not m:
        raise RuntimeError("Could not detect page date.")
    return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3)))

def detect_store(text):
    for name in STORE_NAMES:
        if name in text:
            return name
    raise RuntimeError("Store name validation failed.")

def load_main_table(path):
    tables = pd.read_html(path)
    candidates = []
    for i, df in enumerate(tables):
        cols = {str(c).strip() for c in df.columns}
        if {"機種名","台番号","G数","差枚"}.issubset(cols):
            candidates.append((len(df), i, df))
    if not candidates:
        raise RuntimeError("Main machine table not found.")
    candidates.sort(reverse=True, key=lambda x: x[0])
    rows, idx, df = candidates[0]
    print(f"selected table index  : {idx}")
    print(f"raw table rows        : {rows}")
    return df.copy()

def clean(df, page_date):
    x = df.copy()
    x.columns = [str(c).strip() for c in x.columns]
    rename = {
        "機種名":"machine_name","台番号":"machine_no","G数":"G","差枚":"diff",
        "BB":"BB","RB":"RB","ART":"ART","合成確率":"combined_prob",
        "BB確率":"BB_prob","RB確率":"RB_prob","ART確率":"ART_prob"
    }
    x = x.rename(columns={k:v for k,v in rename.items() if k in x.columns})
    x["machine_name"] = x["machine_name"].astype(str).str.strip()
    x["machine_no"] = pd.to_numeric(x["machine_no"], errors="coerce")
    for c in ("G","diff","BB","RB","ART"):
        if c in x.columns:
            x[c] = x[c].astype(str).str.replace(",","",regex=False).str.replace("+","",regex=False).str.strip()
            x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["machine_no","machine_name","G","diff"]).copy()
    x["machine_no"] = x["machine_no"].astype(int)
    x["date"] = page_date.date()
    first = ["date","machine_name","machine_no","G","diff"]
    rest = [c for c in x.columns if c not in first]
    return x[first + rest]

def quality(df, expected):
    vals = {
        "records": len(df),
        "unique": df["machine_no"].nunique(),
        "duplicates": int(df["machine_no"].duplicated(keep=False).sum()),
        "missing_name": int((df["machine_name"].astype(str).str.strip()=="").sum()),
        "invalid_diff": int(pd.to_numeric(df["diff"], errors="coerce").isna().sum()),
        "invalid_G": int(pd.to_numeric(df["G"], errors="coerce").isna().sum()),
        "negative_G": int((pd.to_numeric(df["G"], errors="coerce") < 0).sum()),
    }
    header("QUALITY CHECK")
    print(f"records              : {vals['records']}")
    print(f"unique machines      : {vals['unique']}")
    print(f"duplicates           : {vals['duplicates']}")
    print(f"missing name         : {vals['missing_name']}")
    print(f"invalid diff         : {vals['invalid_diff']}")
    print(f"invalid G            : {vals['invalid_G']}")
    print(f"negative G           : {vals['negative_G']}")
    print(f"diff min/max         : {df['diff'].min()} / {df['diff'].max()}")
    print(f"G min/max            : {df['G'].min()} / {df['G'].max()}")
    problems = []
    if vals["records"] != expected: problems.append(f"records={vals['records']}, expected={expected}")
    if vals["unique"] != expected: problems.append(f"unique={vals['unique']}, expected={expected}")
    if vals["duplicates"] != 0: problems.append(f"duplicates={vals['duplicates']}")
    if vals["missing_name"] != 0: problems.append(f"missing_name={vals['missing_name']}")
    if vals["invalid_diff"] != 0: problems.append(f"invalid_diff={vals['invalid_diff']}")
    if vals["invalid_G"] != 0: problems.append(f"invalid_G={vals['invalid_G']}")
    if vals["negative_G"] != 0: problems.append(f"negative_G={vals['negative_G']}")
    if problems:
        print("\nRESULT: DAILY DATA QUALITY FAILED")
        for p in problems: print("-", p)
        raise RuntimeError("Daily data quality check failed.")
    print("\nRESULT: DAILY DATA QUALITY OK")

def main():
    args = parse_args()
    header("Big March Takasaki Oyagi - Ana-Slo HTML -> Daily CSV")
    html = resolve_html(args.html)
    text = read_text(html)
    page_date = detect_date(text)
    store_name = detect_store(text)
    header("SOURCE")
    print(f"file                 : {html}")
    print(f"html_chars           : {len(text):,}")
    print(f"page_date            : {page_date.date()}")
    print(f"store_name           : {store_name}")
    table = load_main_table(html)
    print(f"detected_headers     : {list(table.columns)}")
    daily = clean(table, page_date)
    quality(daily, args.expected_machines)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"ana_slo_bigmarch_oyagi_{page_date.strftime('%Y%m%d')}.csv"
    daily.to_csv(out, index=False, encoding="utf-8-sig")
    header("SAVED")
    print(out)
    print(f"saved records        : {len(daily)}")
    print("\nBig March daily CSV conversion complete.")
    print("Maruhan Maebashi data was not modified.")

if __name__ == "__main__":
    main()
