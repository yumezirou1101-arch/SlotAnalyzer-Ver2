from __future__ import annotations

import argparse, hashlib, importlib.util, json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
DATA_DIR = ROOT / "data/maruhan_maebashi/machine_number"
SOURCE_56 = ROOT / "machine_number/ana_slo_prediction_v4_2_machine_number_position_ablation_oos.py"
OUTPUT_DIR = DATA_DIR / "analysis_31days_deep/research_inventory_epoch_walk_forward"
FINGERPRINT = "a1eaf45d71ded209"
FEATURES = ("avg31","recent7_avg","last_diff","prev_change","weekday_avg","type_avg","plus1000_rate","plus2000_rate","neighbor_avg")
EPOCH_FEATURES = ("avg31","recent7_avg","last_diff","prev_change","weekday_avg","plus1000_rate","plus2000_rate")
POLICIES = ("BASELINE","EPOCH_PRIOR","EPOCH_EXCLUDE_LT7","EPOCH_EXCLUDE_LT14","EPOCH_EXCLUDE_LT21","EPOCH_EXCLUDE_LT31","EPOCH_EXCLUDE_CHANGED","EPOCH_EXCLUDE_LT14_ZPOP_EXCLUDE")
TOP_NS = (1,3,5,10)

@dataclass(frozen=True)
class Event:
    change_date: pd.Timestamp
    previous_date: pd.Timestamp
    machine_no: int
    old_machine_name: str
    new_machine_name: str

def load_module(path):
    spec=importlib.util.spec_from_file_location("epoch_source56",path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def fingerprint(weights):
    text="|".join(f"{k}:{weights[k]:.15f}" for k in sorted(weights))
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def read_daily(path, day):
    raw=None
    for enc in ("utf-8-sig","utf-8","cp932"):
        try: raw=pd.read_csv(path,encoding=enc); break
        except UnicodeDecodeError: pass
    if raw is None: raise RuntimeError(f"Unreadable: {path}")
    aliases={"date":("date","日付"),"machine_no":("machine_no","台番号"),"machine_name":("machine_name","機種名"),"diff":("diff","差枚")}
    cols={k:next((x for x in v if x in raw.columns),None) for k,v in aliases.items()}
    if any(v is None for v in cols.values()): raise RuntimeError(f"Columns: {path}")
    x=raw[[cols[k] for k in aliases]].rename(columns={v:k for k,v in cols.items()})
    x["date"]=pd.to_datetime(x["date"]).dt.normalize()
    if not (x["date"]==day).all(): raise RuntimeError(f"Date mismatch: {path}")
    x["machine_no"]=pd.to_numeric(x["machine_no"]).astype(int)
    x["machine_name"]=x["machine_name"].astype(str).str.strip()
    x["diff"]=pd.to_numeric(x["diff"].astype(str).str.replace(",","",regex=False)).astype(float)
    if x.machine_no.duplicated().any(): raise RuntimeError(f"Duplicate: {path}")
    x["win"]=(x["diff"]>0).astype(int); x["plus1000"]=(x["diff"]>=1000).astype(int); x["plus2000"]=(x["diff"]>=2000).astype(int)
    return x.sort_values("machine_no").reset_index(drop=True)

def load_data(data_dir=DATA_DIR):
    snapshots={}
    for path in Path(data_dir).glob("ana_slo_????????.csv"):
        token=path.stem.replace("ana_slo_","")
        if token.isdigit():
            day=pd.to_datetime(token,format="%Y%m%d"); snapshots[day]=read_daily(path,day)
    snapshots=dict(sorted(snapshots.items()))
    return pd.concat(snapshots.values(),ignore_index=True),snapshots

def detect_events(snapshots):
    out=[]; dates=list(snapshots)
    for prev,cur in zip(dates,dates[1:]):
        if (cur-prev).days != 1: continue
        a=snapshots[prev].set_index("machine_no").machine_name; b=snapshots[cur].set_index("machine_no").machine_name
        for no in sorted(set(a.index)&set(b.index)):
            if a[no]!=b[no]: out.append(Event(cur,prev,int(no),str(a[no]),str(b[no])))
    return out

def latest_event(events,no,name,asof):
    found=[e for e in events if e.machine_no==no and e.change_date<=asof]
    if not found:return None
    event=max(found,key=lambda e:e.change_date)
    return event if event.new_machine_name==name else None

def priors(hist,target,name):
    latest=hist.date.max(); recent=hist[hist.date>=latest-pd.Timedelta(days=6)]; wd=hist[hist.date.dt.dayofweek==target.dayofweek]
    avg=float(hist["diff"].mean())
    values={"avg31":avg,"recent7_avg":float(recent["diff"].mean()),"last_diff":float(recent["diff"].mean()),"prev_change":0.0,"weekday_avg":float(wd["diff"].mean()) if len(wd) else avg,"plus1000_rate":float(hist.plus1000.mean()),"plus2000_rate":float(hist.plus2000.mean())}
    typed=hist[hist.machine_name==name]
    if len(typed):
        twd=typed[typed.date.dt.dayofweek==target.dayofweek]
        values.update(avg31=float(typed["diff"].mean()),recent7_avg=float(typed.tail(7)["diff"].mean()),last_diff=float(typed.tail(7)["diff"].mean()),weekday_avg=float(twd["diff"].mean()) if len(twd) else float(typed["diff"].mean()),plus1000_rate=float(typed.plus1000.mean()),plus2000_rate=float(typed.plus2000.mean()))
        return values,"TYPE_PRIOR"
    return values,"STORE_PRIOR"

def stats(epoch,hist,target,name):
    if epoch.empty:return priors(hist,target,name)
    m=epoch.sort_values("date"); avg=float(m["diff"].mean()); wd=m[m.date.dt.dayofweek==target.dayofweek]
    weekday=avg
    if len(wd):
        w=len(wd)/(len(wd)+15.0); weekday=float(wd["diff"].mean())*w+avg*(1-w)
    return {"avg31":avg,"recent7_avg":float(m.tail(7)["diff"].mean()),"last_diff":float(m.iloc[-1]["diff"]),"prev_change":float(m.iloc[-1]["diff"]-m.iloc[-2]["diff"]) if len(m)>=2 else 0.0,"weekday_avg":weekday,"plus1000_rate":float(m.plus1000.mean()),"plus2000_rate":float(m.plus2000.mean())},"EPOCH_HISTORY"

def baseline_panel(m56,hist,inventory,target):
    synthetic=inventory[["machine_no","machine_name"]].copy(); synthetic["date"]=target; synthetic["diff"]=0.; synthetic["win"]=0; synthetic["plus1000"]=0; synthetic["plus2000"]=0
    work=pd.concat([hist,synthetic.reindex(columns=hist.columns)],ignore_index=True)
    panel=m56.build_features(work,target,m56.build_number_edge_distance(hist.machine_no.tolist()))
    return panel.drop(columns="diff").sort_values("machine_no").reset_index(drop=True)

def epoch_panel(base,hist,inventory,events,target):
    out=base.copy(); out["inventory_changed"]=False; out["change_date"]=""; out["epoch_history_n"]=0; out["fallback_policy"]="UNCHANGED"; out["history_start_date"]=""; out["max_history_date"]=""
    names=inventory.set_index("machine_no").machine_name.to_dict(); asof=target-pd.Timedelta(days=1)
    for i,row in out.iterrows():
        no=int(row.machine_no); name=str(names[no]); event=latest_event(events,no,name,asof)
        if event is None:continue
        eh=hist[(hist.machine_no==no)&(hist.date>=event.change_date)&(hist.date<=asof)&(hist.machine_name==name)].copy()
        values,policy=stats(eh,hist,target,name)
        for key in EPOCH_FEATURES:out.at[i,key]=values[key]
        out.at[i,"inventory_changed"]=True; out.at[i,"change_date"]=event.change_date.date().isoformat(); out.at[i,"epoch_history_n"]=len(eh); out.at[i,"fallback_policy"]=policy
        if len(eh):out.at[i,"history_start_date"]=eh.date.min().date().isoformat(); out.at[i,"max_history_date"]=eh.date.max().date().isoformat()
        if len(eh) and eh.date.min()<event.change_date:raise AssertionError("pre-change history")
        if len(eh)==1 and values["prev_change"]!=0:raise AssertionError("boundary prev_change")
    return out

def rank(m56,panel,weights,policy):
    thresholds={"EPOCH_EXCLUDE_LT7":7,"EPOCH_EXCLUDE_LT14":14,"EPOCH_EXCLUDE_LT21":21,"EPOCH_EXCLUDE_LT31":31,"EPOCH_EXCLUDE_LT14_ZPOP_EXCLUDE":14}
    eligible=pd.Series(True,index=panel.index)
    if policy=="EPOCH_EXCLUDE_CHANGED":eligible=~panel.inventory_changed
    elif policy in thresholds:eligible=(~panel.inventory_changed)|(panel.epoch_history_n>=thresholds[policy])
    # Normal exclusion intentionally preserves the full z-score population; the ZPOP policy does not.
    source=panel.loc[eligible].copy() if policy.endswith("ZPOP_EXCLUDE") else panel.copy()
    scored=m56.rank_score(source,weights)
    if policy not in ("BASELINE","EPOCH_PRIOR") and not policy.endswith("ZPOP_EXCLUDE"):
        scored=scored[scored.machine_no.isin(set(panel.loc[eligible,"machine_no"]))]
    scored=scored.reset_index(drop=True); scored["prediction_rank"]=np.arange(1,len(scored)+1); return scored

def metrics(series):
    x=pd.to_numeric(series); return {"selected_n":len(x),"avg_diff":x.mean(),"total_diff":x.sum(),"win_rate":100*(x>0).mean(),"plus1000_rate":100*(x>=1000).mean(),"plus2000_rate":100*(x>=2000).mean()}

def bucket(lag):return f"DAY{lag}" if lag<=3 else "DAY4-7" if lag<=7 else "DAY8-14" if lag<=14 else "DAY15-21" if lag<=21 else "DAY22-31" if lag<=31 else "DAY32+"

def run(data_dir=DATA_DIR,output_dir=OUTPUT_DIR):
    m56=load_module(SOURCE_56); weights=m56.V42_C_WEIGHTS.copy(); fp=fingerprint(weights)
    if fp!=FINGERPRINT:raise AssertionError(fp)
    data,snaps=load_data(data_dir); events=detect_events(snaps); change_dates={e.change_date for e in events}
    if len(events)!=102 or len(change_dates)!=4:raise AssertionError((len(events),len(change_dates)))
    primary_targets=sorted({e.change_date+pd.Timedelta(days=lag) for e in events for lag in range(1,15) if e.change_date+pd.Timedelta(days=lag) in snaps and e.change_date+pd.Timedelta(days=lag-1) in snaps and e.change_date+pd.Timedelta(days=lag) not in change_dates})
    targets=sorted({d for e in events for d in snaps if d>e.change_date and d-pd.Timedelta(days=1) in snaps and d not in change_dates})
    details=[];daily=[];ripple=[];checks=[];ranked={};panels={};decomp=[]
    for target in targets:
        asof=target-pd.Timedelta(days=1); hist=data[data.date<=asof].copy(); inventory=snaps[asof]; actual=snaps[target][["machine_no","machine_name","diff"]]
        base=baseline_panel(m56,hist,inventory,target); ep=epoch_panel(base,hist,inventory,events,target)
        panels[target,"BASELINE"]=base;panels[target,"EPOCH_PRIOR"]=ep
        changed=set(ep.loc[ep.inventory_changed,"machine_no"]); unchanged=sorted(set(base.machine_no)-changed)
        equal=base.set_index("machine_no").loc[unchanged,list(FEATURES)].equals(ep.set_index("machine_no").loc[unchanged,list(FEATURES)])
        if not equal:raise AssertionError(f"unchanged features {target}")
        for feature in FEATURES:
            bv=pd.to_numeric(base[feature]);ev=pd.to_numeric(ep[feature]);delta=ev-bv;cm=ep.inventory_changed
            decomp.append({"target_date":target.date().isoformat(),"feature":feature,"baseline_mean":bv.mean(),"epoch_mean":ev.mean(),"mean_diff":ev.mean()-bv.mean(),"baseline_std":bv.std(ddof=0),"epoch_std":ev.std(ddof=0),"std_diff":ev.std(ddof=0)-bv.std(ddof=0),"max_abs_raw_change":delta.abs().max(),"changed_mean_contribution":delta[cm].sum()/len(delta),"unchanged_max_raw_change":delta[~cm].abs().max(),"weight":weights[feature]})
        for policy in POLICIES:
            panel=base if policy=="BASELINE" else ep
            rr=rank(m56,panel,weights,policy); ranked[target,policy]=rr; ev=rr.merge(actual,on="machine_no",suffixes=("","_actual"))
            if target in primary_targets:
                for n in TOP_NS:daily.append({"target_date":target.date().isoformat(),"policy":policy,"band":f"TOP{n}",**metrics(ev.head(n)["diff"]),"positive_day":int(ev.head(n)["diff"].sum()>0)})
                for x in ev.head(10).to_dict("records"):details.append({"target_date":target.date().isoformat(),"policy":policy,"prediction_rank":int(x["prediction_rank"]),"machine_no":int(x["machine_no"]),"machine_name":x["machine_name"],"score":x["score"],"actual_diff":x["diff"],"inventory_changed":x.get("inventory_changed",False),"epoch_history_n":x.get("epoch_history_n",0),"fallback_policy":x.get("fallback_policy","UNCHANGED")})
        br=ranked[target,"BASELINE"].set_index("machine_no");er=ranked[target,"EPOCH_PRIOR"].set_index("machine_no"); common=sorted(set(unchanged)&set(br.index)&set(er.index));sd=(er.loc[common,"score"]-br.loc[common,"score"]).abs();rd=(er.loc[common,"prediction_rank"]-br.loc[common,"prediction_rank"]).abs();bt=set(br.nsmallest(10,"prediction_rank").index);et=set(er.nsmallest(10,"prediction_rank").index)
        ripple.append({"target_date":target.date().isoformat(),"unchanged_n":len(common),"max_abs_score_diff":sd.max(),"max_score_machine_no":int(sd.idxmax()),"mean_abs_score_diff":sd.mean(),"rank_changed_machines":int((rd>0).sum()),"max_abs_rank_diff":int(rd.max()),"max_rank_machine_no":int(rd.idxmax()),"top10_replacements":len(bt-et),"baseline_top10":"|".join(map(str,sorted(bt))),"epoch_top10":"|".join(map(str,sorted(et))),"baseline_only_top10":"|".join(map(str,sorted(bt-et))),"epoch_only_top10":"|".join(map(str,sorted(et-bt))),"changed_machines":len(changed)})
        checks.append({"target_date":target.date().isoformat(),"max_history_before_target":hist.date.max()<target,"unchanged_features_equal":equal,"fingerprint_ok":fp==FINGERPRINT,"changed_machines":len(changed)})
    daily=pd.DataFrame(daily); summary=daily.groupby(["policy","band"],as_index=False).agg(evaluation_days=("target_date","nunique"),avg_diff=("avg_diff","mean"),total_diff=("total_diff","sum"),win_rate=("win_rate","mean"),plus1000_rate=("plus1000_rate","mean"),plus2000_rate=("plus2000_rate","mean"),positive_days=("positive_day","sum"))
    expected={"TOP1":(307.5,12300,45.0),"TOP3":(-20.8333333333333,-2500,37.5),"TOP5":(150.5,30100,41.0),"TOP10":(106.25,42500,40.25)}
    bs=summary[summary.policy=="BASELINE"].set_index("band")
    for band,(avg,total,win) in expected.items():
        if not (np.isclose(bs.at[band,"avg_diff"],avg) and np.isclose(bs.at[band,"total_diff"],total) and np.isclose(bs.at[band,"win_rate"],win)):raise AssertionError((band,bs.loc[band].to_dict()))
    compositions=[]
    for target in primary_targets:
        bt=set(ranked[target,"BASELINE"].head(10).machine_no)
        for policy in POLICIES:compositions.append({"target_date":target.date().isoformat(),"policy":policy,"top10_replacements":len(bt-set(ranked[target,policy].head(10).machine_no))})
    comp=pd.DataFrame(compositions).groupby("policy",as_index=False).agg(top10_composition_diff_avg=("top10_replacements","mean"),top10_composition_diff_max=("top10_replacements","max"))
    baseline=summary[summary.policy=="BASELINE"][["band","avg_diff","total_diff","win_rate"]].rename(columns={x:f"baseline_{x}" for x in ("avg_diff","total_diff","win_rate")})
    comparison=summary.merge(baseline,on="band").merge(comp,on="policy")
    for x in ("avg_diff","total_diff","win_rate"):comparison[f"{x}_delta_vs_baseline"]=comparison[x]-comparison[f"baseline_{x}"]
    selections=[]
    for e in events:
        for lag in range(1,1000):
            target=e.change_date+pd.Timedelta(days=lag)
            if target>max(snaps):break
            if (target,"BASELINE") not in ranked:continue
            actual=snaps[target];actual=actual[(actual.machine_no==e.machine_no)&(actual.machine_name==e.new_machine_name)]
            if actual.empty:continue
            diff=float(actual.iloc[0]["diff"])
            for policy in POLICIES:
                row=ranked[target,policy];row=row[row.machine_no==e.machine_no];r=int(row.iloc[0].prediction_rank) if len(row) else 0
                selections.append({"change_date":e.change_date.date().isoformat(),"target_date":target.date().isoformat(),"lag":lag,"lag_bucket":bucket(lag),"machine_no":e.machine_no,"policy":policy,"rank":r,"selected_top10":int(0<r<=10),"actual_diff":diff,"win":int(diff>0),"plus1000":int(diff>=1000),"plus2000":int(diff>=2000)})
    sel=pd.DataFrame(selections); picked=sel[sel.selected_top10==1]
    buckets=("DAY1","DAY2","DAY3","DAY4-7","DAY8-14","DAY15-21","DAY22-31","DAY32+")
    lag=picked.groupby(["policy","lag_bucket"],as_index=False).agg(changed_selected=("machine_no","size"),selected_event_dates=("change_date","nunique"),avg_diff=("actual_diff","mean"),win_rate=("win","mean"),plus1000_rate=("plus1000","mean"),plus2000_rate=("plus2000","mean")) if len(picked) else pd.DataFrame()
    grid=pd.MultiIndex.from_product([POLICIES,buckets],names=["policy","lag_bucket"]).to_frame(index=False);lag=grid.merge(lag,on=["policy","lag_bucket"],how="left");lag[["changed_selected","selected_event_dates"]]=lag[["changed_selected","selected_event_dates"]].fillna(0).astype(int);lag[["win_rate","plus1000_rate","plus2000_rate"]]*=100
    by_event=picked.groupby(["policy","change_date"],as_index=False).agg(changed_selected=("machine_no","size"),target_dates=("target_date","nunique"),avg_diff=("actual_diff","mean"),total_diff=("actual_diff","sum"),win_rate=("win","mean")) if len(picked) else pd.DataFrame()
    if len(by_event):by_event["win_rate"]*=100
    event_rows=[]
    for e in events:
        before=data[(data.machine_no==e.machine_no)&(data.date<e.change_date)].sort_values("date");after=data[(data.machine_no==e.machine_no)&(data.date>=e.change_date)].sort_values("date");oldn=0;newn=0
        for name in reversed(before.machine_name.tolist()):
            if name!=e.old_machine_name:break
            oldn+=1
        for name in after.machine_name.tolist():
            if name!=e.new_machine_name:break
            newn+=1
        event_rows.append({**asdict(e),"change_date":e.change_date.date().isoformat(),"previous_date":e.previous_date.date().isoformat(),"prior_observed_days":len(before),"old_epoch_days":oldn,"post_actual_days":len(after),"new_epoch_actual_days":newn})
    cohorts=[]
    for label,mask in (("ALL",pd.Series(True,index=picked.index)),("MASS_2026-08-03",picked.change_date=="2026-08-03"),("SMALL_EXCLUDING_2026-08-03",picked.change_date!="2026-08-03")):
        z=picked[mask]
        g=z.groupby("policy",as_index=False).agg(changed_selected=("machine_no","size"),independent_change_dates=("change_date","nunique"),avg_diff=("actual_diff","mean"),total_diff=("actual_diff","sum"),win_rate=("win","mean"),plus1000_rate=("plus1000","mean"),plus2000_rate=("plus2000","mean"));g=pd.DataFrame({"policy":POLICIES}).merge(g,on="policy",how="left");g[["changed_selected","independent_change_dates","total_diff"]]=g[["changed_selected","independent_change_dates","total_diff"]].fillna(0);g.insert(0,"cohort",label);cohorts.append(g)
    mass=pd.concat(cohorts,ignore_index=True);mass[["win_rate","plus1000_rate","plus2000_rate"]]*=100
    ripple_df=pd.DataFrame(ripple);extreme_dates=set(ripple_df.nlargest(1,"max_abs_score_diff").target_date)|set(ripple_df.nlargest(1,"max_abs_rank_diff").target_date)|set(ripple_df[ripple_df.top10_replacements==ripple_df.top10_replacements.max()].target_date)
    extreme_rows=[]
    for day_text in sorted(extreme_dates):
        target=pd.Timestamp(day_text);summary_row=ripple_df[ripple_df.target_date==day_text].iloc[0].to_dict();bp=panels[target,"BASELINE"].set_index("machine_no");ep=panels[target,"EPOCH_PRIOR"].set_index("machine_no");br=ranked[target,"BASELINE"].set_index("machine_no");er=ranked[target,"EPOCH_PRIOR"].set_index("machine_no")
        for no in ep[ep.inventory_changed].index:
            row={**summary_row,"machine_no":int(no),"baseline_score":br.at[no,"score"],"epoch_score":er.at[no,"score"],"baseline_rank":int(br.at[no,"prediction_rank"]),"epoch_rank":int(er.at[no,"prediction_rank"])}
            for feature in FEATURES:row[f"baseline_{feature}"]=bp.at[no,feature];row[f"epoch_{feature}"]=ep.at[no,feature]
            extreme_rows.append(row)
    extremes=pd.DataFrame(extreme_rows);decomp_df=pd.DataFrame(decomp);decomp_df["extreme_day"]=decomp_df.target_date.isin(extreme_dates);decomp_df["weighted_population_shift_abs"]=decomp_df.mean_diff.abs()*decomp_df.weight.abs()
    zpol=("EPOCH_EXCLUDE_LT14","EPOCH_EXCLUDE_LT14_ZPOP_EXCLUDE")
    zpop=comparison[comparison.policy.isin(zpol)].copy()
    zr=[]
    for target in primary_targets:
        a=ranked[target,zpol[0]].set_index("machine_no");b=ranked[target,zpol[1]].set_index("machine_no");unchanged=set(panels[target,"EPOCH_PRIOR"].loc[~panels[target,"EPOCH_PRIOR"].inventory_changed,"machine_no"]);common=sorted(unchanged&set(a.index)&set(b.index));sd=(b.loc[common,"score"]-a.loc[common,"score"]).abs();rd=(b.loc[common,"prediction_rank"]-a.loc[common,"prediction_rank"]).abs()
        zr.append({"nonchanged_mean_abs_score_diff":sd.mean(),"nonchanged_max_abs_score_diff":sd.max(),"nonchanged_mean_abs_rank_diff":rd.mean(),"nonchanged_max_abs_rank_diff":rd.max(),"top10_replacements_ranking_vs_zpop":len(set(a.head(10).index)-set(b.head(10).index))})
    za=pd.DataFrame(zr).agg(["mean","max"]);zpop["nonchanged_mean_abs_score_diff_mean"]=za.at["mean","nonchanged_mean_abs_score_diff"];zpop["nonchanged_max_abs_score_diff"]=za.at["max","nonchanged_max_abs_score_diff"];zpop["nonchanged_mean_abs_rank_diff_mean"]=za.at["mean","nonchanged_mean_abs_rank_diff"];zpop["nonchanged_max_abs_rank_diff"]=za.at["max","nonchanged_max_abs_rank_diff"];zpop["top10_replacements_mean_ranking_vs_zpop"]=za.at["mean","top10_replacements_ranking_vs_zpop"];zpop["top10_replacements_max_ranking_vs_zpop"]=za.at["max","top10_replacements_ranking_vs_zpop"]
    frames={"events":pd.DataFrame(event_rows),"prediction_detail":pd.DataFrame(details),"daily_topn":daily,"topn_summary":summary,"changed_lag":lag,"changed_event_date":by_event,"baseline_vs_epoch":comparison,"score_ripple":ripple_df,"safety_asserts":pd.DataFrame(checks),"changed_selection_detail":sel,"threshold_policy_summary":comparison,"threshold_policy_topn":daily,"threshold_policy_changed_lag":lag,"threshold_policy_event_date":by_event,"mass_change_vs_small_change":mass,"score_ripple_extreme_days":extremes,"score_ripple_feature_decomposition":decomp_df,"zpop_policy_comparison":zpop}
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    for name,frame in frames.items():
        filename=f"{name}.csv" if name.startswith(("threshold_","mass_","score_ripple_extreme","score_ripple_feature","zpop_")) else f"inventory_epoch_{name}.csv"
        frame.to_csv(output_dir/filename,index=False,encoding="utf-8-sig")
    meta={"generated_at":datetime.now().astimezone().isoformat(),"data_start":min(snaps).date().isoformat(),"data_end":max(snaps).date().isoformat(),"daily_files":len(snaps),"change_events":len(events),"change_dates":len(change_dates),"change_date_values":[x.date().isoformat() for x in sorted(change_dates)],"primary_evaluation_days":len(primary_targets),"extended_lag_days":len(targets),"policies":list(POLICIES),"model":"CHAMPION_V4.2_C","weight_fingerprint":fp,"weights":weights,"history_policy":"latest detected contiguous inventory epoch for changed machines","fallback_policy":"past-only current type prior, else store prior","exclusion_policy":"ranking-only LT7/LT14/LT21/LT31/changed and research LT14 z-score-population exclusion","leakage_guard":"inventory through D-1; features through D-1; D actual evaluation-only","baseline_asserted":expected,"production_outputs_written":False}
    (output_dir/"inventory_epoch_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return {"metadata":meta,**frames}

def main():
    p=argparse.ArgumentParser();p.add_argument("--data-dir",type=Path,default=DATA_DIR);p.add_argument("--output-dir",type=Path,default=OUTPUT_DIR);a=p.parse_args();r=run(a.data_dir,a.output_dir);print(json.dumps(r["metadata"],ensure_ascii=False,indent=2))
if __name__=="__main__":main()
