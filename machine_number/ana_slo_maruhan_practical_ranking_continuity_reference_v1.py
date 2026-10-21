from __future__ import annotations
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, re
import numpy as np
import pandas as pd
from ana_slo_prediction_v4_2_forward_guard import validate_consecutive_latest_date, validate_forward_time, validate_target_actual_absent

PROJECT_ROOT=Path(os.environ.get("SLOTANALYZER_PROJECT_ROOT", r"C:\Users\user\Desktop\Documents\SlotAnalyzer"))
DATA_DIR=PROJECT_ROOT/"data"/"maruhan_maebashi"/"machine_number"
OUTPUT_DIR=DATA_DIR/"analysis_31days_deep"/"80_inventory_practical_ranking_reference"
SOURCE_56=PROJECT_ROOT/"machine_number"/"ana_slo_prediction_v4_2_machine_number_position_ablation_oos.py"
SOURCE_63=PROJECT_ROOT/"machine_number"/"ana_slo_prediction_v4_2_forward_champion_challenger.py"
EXPECTED_MACHINES=514; STABILITY_WINDOW=7; TOP_N=15
A_TYPE_PATTERNS=[r"ジャグラー",r"ハナハナ",r"(^|[^ァ-ヶ])ハナビ|新ハナビ",r"バーサス",r"ディスクアップ",r"クランキー",r"アレックス",r"サンダーV|サンダーＶ",r"ゲッターマウス",r"ニューパルサー",r"パルサー",r"ケロット"]

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f"Could not import: {path}")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def fingerprint(weights):
    text="|".join(f"{k}:{weights[k]:.15f}" for k in sorted(weights)); return hashlib.sha256(text.encode()).hexdigest()[:16]

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def inventory_flags(df, latest_date, window=STABILITY_WINDOW):
    hist=df[df["date"]<=latest_date].copy(); dates=sorted(pd.Timestamp(x).normalize() for x in hist["date"].dropna().unique())
    if not dates or dates[-1]!=pd.Timestamp(latest_date).normalize(): raise RuntimeError("Latest inventory date missing.")
    latest=hist[hist["date"]==latest_date][["machine_no","machine_name"]].copy()
    if len(latest)!=EXPECTED_MACHINES or latest["machine_no"].nunique()!=EXPECTED_MACHINES: raise RuntimeError("Latest inventory snapshot invalid.")
    maps={d:hist[hist["date"]==d].drop_duplicates("machine_no").set_index("machine_no")["machine_name"].astype(str).to_dict() for d in dates[-max(window,2):]}
    prior=dates[-2] if len(dates)>=2 else None; rows=[]
    for r in latest.itertuples(index=False):
        no=int(r.machine_no); name=str(r.machine_name); changed=prior is not None and maps[prior].get(no)!=name
        stable_dates=dates[-window:]; stable=len(stable_dates)>=window and all(maps.get(d,{}).get(no)==name for d in stable_dates)
        previous_names=set(hist[(hist["date"]<latest_date)&(hist["machine_no"]==no)]["machine_name"].astype(str)); risk=bool(previous_names-{name})
        new_to_window=hist[(hist["date"]<latest_date)&(hist["machine_name"].astype(str)==name)].empty
        rows.append({"machine_no":no,"inventory_changed_latest":changed,"stable_7":stable,"new_to_history_window":new_to_window,"old_model_history_risk":risk})
    return pd.DataFrame(rows)

def category(name):
    s=str(name)
    if re.search(r"ジャグラー",s,re.I): return "JUGGLER"
    if any(re.search(p,s,re.I) for p in A_TYPE_PATTERNS): return "A_TYPE"
    return "NORMAL"

def write_new(df,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise RuntimeError(f"Practical artifact already exists; no overwrite: {path}")
    df.to_csv(path,index=False,encoding="utf-8-sig",mode="x")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--target-date",required=True); a=ap.parse_args()
    target=pd.Timestamp(pd.to_datetime(a.target_date,format="%Y-%m-%d",errors="raise")).normalize(); latest=target-pd.Timedelta(days=1)
    validate_consecutive_latest_date(target,latest); generated=validate_forward_time(target); validate_target_actual_absent(PROJECT_ROOT,DATA_DIR,target)
    m56=load_module(SOURCE_56,"practical_source56"); m63=load_module(SOURCE_63,"practical_source63"); weights=m56.V42_C_WEIGHTS.copy(); fp=fingerprint(weights)
    if fp!="a1eaf45d71ded209": raise RuntimeError(f"Champion fingerprint mismatch: {fp}")
    df,_=m63.assemble_dataset(); df=df[df["date"]<target].copy()
    if df.empty or pd.Timestamp(df["date"].max()).normalize()!=latest: raise RuntimeError("Latest data date mismatch.")
    snap=df[df["date"]==latest]
    if len(snap)!=EXPECTED_MACHINES or snap["machine_no"].nunique()!=EXPECTED_MACHINES or snap["machine_no"].duplicated().any(): raise RuntimeError("Latest snapshot invalid.")
    fixed=load_module(PROJECT_ROOT/"machine_number"/"ana_slo_prediction_v4_2_future_top10_fixed.py","practical_fixed_helpers"); fixed.EXPECTED_MACHINES=EXPECTED_MACHINES
    panel,panel_latest=fixed.build_future_panel(m56,df,target); ranked=m56.rank_score(panel.copy(),weights).reset_index(drop=True)
    ranked["prediction_rank"]=np.arange(1,len(ranked)+1); ranked["target_date"]=target.date(); ranked["latest_data_date"]=panel_latest.date()
    ranked=ranked.merge(inventory_flags(df,latest),on="machine_no",how="left",validate="one_to_one"); ranked["inventory_category"]=ranked["machine_name"].map(category)
    ranked["inventory_warning"]=np.where(ranked["old_model_history_risk"],"OLD_MODEL_HISTORY_RISK","")
    stable=ranked[ranked["stable_7"] & ~ranked["old_model_history_risk"]].copy().sort_values(["score","prediction_rank","machine_no"],ascending=[False,True,True]).reset_index(drop=True); stable["stable_rank"]=np.arange(1,len(stable)+1)
    ymd=target.strftime("%Y%m%d"); paths={"all":OUTPUT_DIR/f"80_practical_{ymd}_all514.csv","top15":OUTPUT_DIR/f"80_practical_{ymd}_top15.csv","stable_all":OUTPUT_DIR/f"80_stable7_{ymd}_all.csv","stable_top15":OUTPUT_DIR/f"80_stable7_{ymd}_top15.csv","atype_top15":OUTPUT_DIR/f"80_practical_atype_{ymd}_top15.csv","juggler_top15":OUTPUT_DIR/f"80_practical_juggler_{ymd}_top15.csv"}
    subsets={"all":ranked,"top15":ranked.head(TOP_N),"stable_all":stable,"stable_top15":stable.head(TOP_N),"atype_top15":ranked[ranked["inventory_category"].isin(["A_TYPE","JUGGLER"])].head(TOP_N).copy(),"juggler_top15":ranked[ranked["inventory_category"]=="JUGGLER"].head(TOP_N).copy()}
    for k,p in paths.items(): write_new(subsets[k],p)
    state={}; sp=DATA_DIR/"inventory_guard_state.json"
    if sp.is_file():
        try: state=json.loads(sp.read_text(encoding="utf-8-sig"))
        except Exception as e: state={"read_error":f"{type(e).__name__}: {e}"}
    meta=pd.DataFrame([{"generated_at_jst":generated.isoformat(),"target_date":target.date(),"latest_data_date":latest.date(),"model":"CHAMPION_V4.2_C","weight_fingerprint":fp,"practical_reference":True,"formal_forward_valid":False,"target_actual_used":False,"future_leakage_guard":"PASS","machines_ranked":len(ranked),"stable_window":STABILITY_WINDOW,"stable_machines":len(stable),"changed_latest_count":int(ranked["inventory_changed_latest"].sum()),"old_model_history_risk_count":int(ranked["old_model_history_risk"].sum()),"inventory_guard_status":state.get("status","UNKNOWN"),"inventory_incident_kind":state.get("incident_kind",""),"policy":"PRACTICAL_RANKING_CONTINUITY_REFERENCE; FORMAL_OUTPUTS_UNCHANGED",**{f"{k}_sha256":sha256_file(v) for k,v in paths.items()}}])
    mp=OUTPUT_DIR/f"80_practical_{ymd}_metadata.csv"; write_new(meta,mp)
    print("PRACTICAL_RANKING_RESULT=OK"); print(f"machines_ranked={len(ranked)} stable7={len(stable)} changed_latest={int(ranked.inventory_changed_latest.sum())} old_model_history_risk={int(ranked.old_model_history_risk.sum())}"); print("FORMAL_FORWARD_MODIFIED=False"); print("CHAMPION_WEIGHTS_MODIFIED=False"); print("TARGET_ACTUAL_USED=False")
if __name__=="__main__": main()
