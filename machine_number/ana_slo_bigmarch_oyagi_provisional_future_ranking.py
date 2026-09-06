from __future__ import annotations

import argparse, importlib.util, json, os, re, shutil, tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
DATA_REL = Path("data/bigmarch_takasaki_oyagi/machine_number")
OUTPUT_REL = DATA_REL / "analysis_31days_deep/90_provisional_future_ranking"
DAILY_RE = re.compile(r"^ana_slo_bigmarch_oyagi_(\d{8})\.csv$", re.I)
REQUIRED_DAILY_COLUMNS = {"date", "machine_name", "machine_no", "G", "diff"}
MIN_MACHINES = 200
JST = ZoneInfo("Asia/Tokyo")

class ProvisionalBlockedError(RuntimeError): pass
class ProvisionalManualReviewError(RuntimeError): pass

@dataclass(frozen=True)
class Eligibility:
    operation_date: date
    expected_data_date: date
    latest_data_date: date
    latest_daily_path: Path
    expected_source_path: Path
    expected_daily_path: Path

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def load_ranking_modules(project_root: Path):
    machine_dir = project_root / "machine_number"
    return (
        _load_module("bigmarch_provisional_juggler", machine_dir / "ana_slo_bigmarch_oyagi_juggler_recent7_future_ranking.py"),
        _load_module("bigmarch_provisional_nonjuggler", machine_dir / "ana_slo_bigmarch_oyagi_nonjuggler_weekday_future_ranking.py"),
    )

def discover_daily_files(data_dir: Path) -> list[tuple[date, Path]]:
    found=[]
    for path in data_dir.glob("ana_slo_bigmarch_oyagi_*.csv"):
        match=DAILY_RE.fullmatch(path.name)
        if match: found.append((datetime.strptime(match.group(1), "%Y%m%d").date(), path))
    return sorted(found)

def validate_latest_daily(path: Path, expected_date: date) -> pd.DataFrame:
    frame=pd.read_csv(path, encoding="utf-8-sig")
    missing=sorted(REQUIRED_DAILY_COLUMNS-set(frame.columns))
    if missing: raise ProvisionalBlockedError(f"LATEST_DAILY_SCHEMA_INVALID: missing={missing}")
    if len(frame)<MIN_MACHINES: raise ProvisionalBlockedError(f"LATEST_DAILY_TOO_SMALL: rows={len(frame)}")
    dates=pd.to_datetime(frame["date"], errors="raise").dt.date.unique().tolist()
    machine_no=pd.to_numeric(frame["machine_no"], errors="coerce")
    machine_name=frame["machine_name"].astype("string").str.strip()
    games=pd.to_numeric(frame["G"], errors="coerce"); differences=pd.to_numeric(frame["diff"], errors="coerce")
    if dates != [expected_date]: raise ProvisionalBlockedError("LATEST_DAILY_INTERNAL_DATE_MISMATCH")
    if machine_no.isna().any() or machine_no.duplicated().any() or machine_no.nunique()!=len(frame): raise ProvisionalBlockedError("LATEST_DAILY_MACHINE_NO_INVALID")
    if machine_name.isna().any() or machine_name.isin(["", "nan", "None"]).any(): raise ProvisionalBlockedError("LATEST_DAILY_MACHINE_NAME_INVALID")
    if games.isna().any() or differences.isna().any() or (games<0).any(): raise ProvisionalBlockedError("LATEST_DAILY_VALUES_INVALID")
    return frame

def assess_eligibility(project_root: Path, operation_date: date) -> Eligibility:
    operation_date=date.fromisoformat(str(operation_date)); expected=operation_date-timedelta(days=1); required_latest=expected-timedelta(days=1)
    data_dir=project_root/DATA_REL
    expected_source=project_root/f"ana_slo_bigmarch_oyagi_{expected:%Y%m%d}_source.html"
    expected_daily=data_dir/f"ana_slo_bigmarch_oyagi_{expected:%Y%m%d}.csv"
    if expected_source.exists(): raise ProvisionalBlockedError("EXPECTED_SOURCE_PRESENT")
    if expected_daily.exists(): raise ProvisionalBlockedError("EXPECTED_DAILY_PRESENT")
    daily_files=discover_daily_files(data_dir)
    if not daily_files: raise ProvisionalBlockedError("NO_DAILY_FILES")
    latest_date,latest_path=daily_files[-1]
    if latest_date>=expected: raise ProvisionalBlockedError("LATEST_DATE_NOT_BEFORE_EXPECTED")
    if latest_date!=required_latest: raise ProvisionalBlockedError(f"LATEST_DATE_GAP_NOT_ONE_DAY: latest={latest_date} expected={expected}")
    validate_latest_daily(latest_path, latest_date)
    return Eligibility(operation_date,expected,latest_date,latest_path,expected_source,expected_daily)

def output_paths(project_root: Path, operation_date: date) -> dict[str,Path]:
    compact=operation_date.strftime("%Y%m%d"); directory=project_root/OUTPUT_REL/compact; prefix=f"90_provisional_{compact}"
    return {"directory":directory,"juggler_all":directory/f"{prefix}_juggler_all.csv","juggler_top10":directory/f"{prefix}_juggler_top10.csv","nonjuggler_all":directory/f"{prefix}_nonjuggler_all.csv","nonjuggler_top10":directory/f"{prefix}_nonjuggler_top10.csv","metadata":directory/f"{prefix}_metadata.csv","status":directory/f"{prefix}_status.csv"}

def _expected_metadata(e: Eligibility, models: str) -> dict:
    return {"ranking_class":"PROVISIONAL","provisional":True,"formal":False,"forward_valid":False,"operation_date":e.operation_date.isoformat(),"target_date":e.operation_date.isoformat(),"expected_data_date":e.expected_data_date.isoformat(),"latest_data_date":e.latest_data_date.isoformat(),"expected_gap_days":1,"target_to_latest_gap_days":2,"source_status":"EXPECTED_DATE_MISSING","model":models,"automatic_promotion":False,"eligible_for_formal_evaluation":False}

def _bool_value(value) -> bool: return str(value).strip().lower() in {"1","true","yes"}

def validate_existing(paths: dict[str,Path], expected: dict) -> bool:
    artifacts=[v for k,v in paths.items() if k!="directory"]
    if not any(p.exists() for p in artifacts):
        if paths["directory"].exists(): raise ProvisionalManualReviewError("PROVISIONAL_DIRECTORY_EXISTS_WITHOUT_ARTIFACTS")
        return False
    if not all(p.is_file() and p.stat().st_size>0 for p in artifacts): raise ProvisionalManualReviewError("PROVISIONAL_ARTIFACTS_PARTIAL")
    metadata=pd.read_csv(paths["metadata"],encoding="utf-8-sig")
    if len(metadata)!=1: raise ProvisionalManualReviewError("PROVISIONAL_METADATA_INVALID")
    row=metadata.iloc[0]
    for key,value in expected.items():
        actual=row.get(key); matches=(_bool_value(actual)==value) if isinstance(value,bool) else (str(actual)==str(value))
        if not matches: raise ProvisionalManualReviewError(f"PROVISIONAL_METADATA_MISMATCH: {key}")
    required={"target_date","expected_data_date","latest_data_date","ranking_class","provisional","forward_valid"}
    for key in ("juggler_all","juggler_top10","nonjuggler_all","nonjuggler_top10"):
        frame=pd.read_csv(paths[key],encoding="utf-8-sig")
        if frame.empty or not required.issubset(frame.columns): raise ProvisionalManualReviewError(f"PROVISIONAL_RANKING_INVALID: {key}")
        if not (frame["target_date"].astype(str)==expected["target_date"]).all(): raise ProvisionalManualReviewError(f"PROVISIONAL_TARGET_MISMATCH: {key}")
        if not (frame["expected_data_date"].astype(str)==expected["expected_data_date"]).all(): raise ProvisionalManualReviewError(f"PROVISIONAL_EXPECTED_DATE_MISMATCH: {key}")
        if not (frame["latest_data_date"].astype(str)==expected["latest_data_date"]).all(): raise ProvisionalManualReviewError(f"PROVISIONAL_LATEST_DATE_MISMATCH: {key}")
        if not (frame["ranking_class"].astype(str)=="PROVISIONAL").all(): raise ProvisionalManualReviewError(f"PROVISIONAL_CLASS_MISMATCH: {key}")
        if not frame["provisional"].map(_bool_value).all(): raise ProvisionalManualReviewError(f"PROVISIONAL_FLAG_MISMATCH: {key}")
        if frame["forward_valid"].map(_bool_value).any(): raise ProvisionalManualReviewError(f"PROVISIONAL_FORWARD_FLAG_MISMATCH: {key}")
    status=pd.read_csv(paths["status"],encoding="utf-8-sig")
    if len(status)!=1 or status.iloc[0].get("status")!="PROVISIONAL": raise ProvisionalManualReviewError("PROVISIONAL_STATUS_INVALID")
    return True

def _annotate(ranking: pd.DataFrame, e: Eligibility) -> pd.DataFrame:
    result=ranking.copy(); result["target_date"]=e.operation_date.isoformat(); result["expected_data_date"]=e.expected_data_date.isoformat(); result["latest_data_date"]=e.latest_data_date.isoformat(); result["ranking_class"]="PROVISIONAL"; result["provisional"]=True; result["forward_valid"]=False; return result

def generate(project_root: Path, operation_date: date) -> dict:
    project_root=Path(project_root); e=assess_eligibility(project_root,operation_date); j,n=load_ranking_modules(project_root); models=f"{j.MODEL_NAME}|{n.MODEL_NAME}"; metadata_base=_expected_metadata(e,models); paths=output_paths(project_root,e.operation_date)
    if validate_existing(paths,metadata_base): return {"status":"ALREADY_PROVISIONAL","paths":paths,"metadata":metadata_base}
    jh,_,_=j.load_frozen_history(); nh,_,_=n.load_frozen_history()
    for label,history in (("JUGGLER",jh),("NON_JUGGLER",nh)):
        latest=pd.Timestamp(history["date"].max()).date()
        if latest!=e.latest_data_date: raise ProvisionalBlockedError(f"{label}_HISTORY_LATEST_MISMATCH: {latest}")
    target=pd.Timestamp(e.operation_date)
    jr=_annotate(j.build_future_ranking(jh,pd.Timestamp(e.latest_data_date),target),e); nr=_annotate(n.build_future_ranking(nh,pd.Timestamp(e.latest_data_date),target),e)
    frames={"juggler_all":jr,"juggler_top10":jr.head(10).copy(),"nonjuggler_all":nr,"nonjuggler_top10":nr.head(10).copy()}
    parent=paths["directory"].parent; parent.mkdir(parents=True,exist_ok=True); temporary=Path(tempfile.mkdtemp(prefix=f".{e.operation_date:%Y%m%d}_",dir=parent))
    try:
        for key,frame in frames.items(): frame.to_csv(temporary/paths[key].name,index=False,encoding="utf-8-sig")
        metadata={**metadata_base,"generated_at_jst":datetime.now(JST).isoformat()}
        pd.DataFrame([metadata]).to_csv(temporary/paths["metadata"].name,index=False,encoding="utf-8-sig")
        pd.DataFrame([{**metadata,"status":"PROVISIONAL"}]).to_csv(temporary/paths["status"].name,index=False,encoding="utf-8-sig")
        os.replace(temporary,paths["directory"])
    except Exception:
        if temporary.exists(): shutil.rmtree(temporary)
        raise
    return {"status":"PROVISIONAL","paths":paths,"metadata":metadata}

def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description="Generate isolated Big March provisional rankings."); parser.add_argument("--operation-date",required=True); parser.add_argument("--project-root",type=Path,default=PROJECT_ROOT); return parser.parse_args()

def main() -> int:
    args=parse_args()
    try: result=generate(args.project_root,date.fromisoformat(args.operation_date))
    except ProvisionalManualReviewError as exc: print(json.dumps({"status":"MANUAL_REVIEW","error":str(exc)},ensure_ascii=False)); return 2
    except Exception as exc: print(json.dumps({"status":"BLOCKED","error":f"{type(exc).__name__}: {exc}"},ensure_ascii=False)); return 1
    print(json.dumps({"status":result["status"],"target_date":args.operation_date},ensure_ascii=False)); return 0

if __name__=="__main__": raise SystemExit(main())
