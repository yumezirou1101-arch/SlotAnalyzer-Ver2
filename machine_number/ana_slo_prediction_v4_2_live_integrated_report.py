from __future__ import annotations
from pathlib import Path
import argparse, re
import pandas as pd

ROOT=Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
BASE=ROOT/"data"/"maruhan_maebashi"/"machine_number"/"analysis_31days_deep"
D64=BASE/"64_Ver4_2_future_top10"
D74=BASE/"74_Ver4_2_A_type_prediction"
D75=BASE/"75_Ver4_2_Juggler_prediction"
OUT=BASE/"77_live_integrated_prediction_report"

def read_csv(p):
    err=None
    for e in ("utf-8-sig","utf-8","cp932"):
        try: return pd.read_csv(p,encoding=e)
        except Exception as x: err=x
    raise RuntimeError(f"CSV read failed: {p}\n{err}")

def files(d,rx):
    out={}
    if not d.exists(): return out
    for p in d.glob("*.csv"):
        m=rx.fullmatch(p.name)
        if m:
            dt=pd.to_datetime(m.group(1),format="%Y%m%d",errors="coerce")
            if not pd.isna(dt): out[pd.Timestamp(dt).normalize()]=p
    return out

def norm(df,target,kind):
    rank={"NORMAL":"prediction_rank","A_TYPE":"a_type_rank","JUGGLER":"juggler_rank"}[kind]
    tier={"NORMAL":"tier","A_TYPE":"a_type_tier","JUGGLER":"juggler_tier"}[kind]
    need=["machine_no","machine_name","score",rank,tier,"target_date","latest_data_date"]
    miss=[c for c in need if c not in df.columns]
    if miss: raise ValueError(f"{kind}: missing {miss}")
    dates=pd.to_datetime(df["target_date"],errors="coerce").dropna()
    if len(dates)==0 or dates.nunique()!=1 or pd.Timestamp(dates.iloc[0]).normalize()!=target:
        raise RuntimeError(f"{kind}: target_date mismatch")
    x=df.copy()
    x["machine_no"]=pd.to_numeric(x["machine_no"],errors="coerce")
    x["score"]=pd.to_numeric(x["score"],errors="coerce")
    x[rank]=pd.to_numeric(x[rank],errors="coerce")
    x=x.dropna(subset=["machine_no","score",rank]).copy()
    x["machine_no"]=x["machine_no"].astype(int); x[rank]=x[rank].astype(int)
    x=x.sort_values(rank).head(10).copy()
    x=x.rename(columns={rank:f"{kind.lower()}_rank",tier:f"{kind.lower()}_tier"})
    return x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--target-date",default=None)
    a=ap.parse_args()
    f64=files(D64,re.compile(r"64_prediction_(\d{8})_top10\.csv",re.I))
    f74=files(D74,re.compile(r"74_A_type_prediction_(\d{8})_top10\.csv",re.I))
    f75=files(D75,re.compile(r"75_Juggler_prediction_(\d{8})_top10\.csv",re.I))
    common=sorted(set(f64)&set(f74)&set(f75))
    if a.target_date:
        target=pd.Timestamp(pd.to_datetime(a.target_date,format="%Y-%m-%d")).normalize()
    else:
        if not common: raise RuntimeError("No common target date for 64/74/75.")
        target=common[-1]
    if target not in common:
        raise FileNotFoundError(f"{target.date()}: 64/74/75 are not all present.")

    n=norm(read_csv(f64[target]),target,"NORMAL")
    at=norm(read_csv(f74[target]),target,"A_TYPE")
    j=norm(read_csv(f75[target]),target,"JUGGLER")

    rows={}
    for kind,df,rankcol in [
        ("NORMAL",n,"normal_rank"),
        ("A_TYPE",at,"a_type_rank"),
        ("JUGGLER",j,"juggler_rank"),
    ]:
        for _,r in df.iterrows():
            no=int(r["machine_no"])
            if no not in rows:
                rows[no]={"machine_no":no,"machine_name":r["machine_name"],
                          "score":r["score"],"target_date":r["target_date"],
                          "latest_data_date":r["latest_data_date"]}
            rows[no][rankcol]=int(r[rankcol])

    z=pd.DataFrame(rows.values())
    for c in ("normal_rank","a_type_rank","juggler_rank"):
        if c not in z.columns: z[c]=pd.NA
    z["in_normal"]=z["normal_rank"].notna()
    z["in_a_type"]=z["a_type_rank"].notna()
    z["in_juggler"]=z["juggler_rank"].notna()
    z["overlap_count"]=z[["in_normal","in_a_type","in_juggler"]].sum(axis=1)

    def sel(r):
        s=[]
        if r["in_normal"]: s.append("NORMAL")
        if r["in_a_type"]: s.append("A_TYPE")
        if r["in_juggler"]: s.append("JUGGLER")
        return "+".join(s)
    z["selected_by"]=z.apply(sel,axis=1)

    big=9999
    z["_n"]=z["normal_rank"].fillna(big)
    z["_a"]=z["a_type_rank"].fillna(big)
    z["_j"]=z["juggler_rank"].fillna(big)
    z=z.sort_values(["overlap_count","_n","_a","_j","score"],
                    ascending=[False,True,True,True,False]).reset_index(drop=True)
    z["report_order"]=range(1,len(z)+1)
    z=z.drop(columns=["_n","_a","_j"])

    summary=pd.DataFrame([{
        "target_date":target.date(),
        "normal_top10":len(n),"a_type_top10":len(at),"juggler_top10":len(j),
        "unique_candidates":z["machine_no"].nunique(),
        "selected_by_3":int((z["overlap_count"]==3).sum()),
        "selected_by_2":int((z["overlap_count"]==2).sum()),
        "selected_by_1":int((z["overlap_count"]==1).sum()),
        "new_model_score_created":False,
        "model_changed":False
    }])

    print("="*110)
    print("77 - Live Integrated Prediction Report")
    print("="*110)
    print(f"target date : {target.date()}")
    print(z[["report_order","machine_no","machine_name","selected_by","overlap_count",
             "normal_rank","a_type_rank","juggler_rank","score"]].to_string(index=False))
    print("\nNOTE: report_order is NOT a new prediction score/rank.")
    print("It is only a practical display order: overlap -> source ranks -> frozen score.")

    OUT.mkdir(parents=True,exist_ok=True)
    y=target.strftime("%Y%m%d")
    p1=OUT/f"77_integrated_prediction_{y}.csv"
    p2=OUT/f"77_integrated_prediction_{y}_summary.csv"
    p3=OUT/f"77_source_NORMAL_{y}.csv"
    p4=OUT/f"77_source_A_TYPE_{y}.csv"
    p5=OUT/f"77_source_JUGGLER_{y}.csv"
    z.to_csv(p1,index=False,encoding="utf-8-sig")
    summary.to_csv(p2,index=False,encoding="utf-8-sig")
    n.to_csv(p3,index=False,encoding="utf-8-sig")
    at.to_csv(p4,index=False,encoding="utf-8-sig")
    j.to_csv(p5,index=False,encoding="utf-8-sig")
    print("\nFILES SAVED")
    for p in (p1,p2,p3,p4,p5): print(p)
    print("\n77 integrated report complete.")
    print("64 / 74 / 75 / 76 were not modified.")

if __name__=="__main__":
    main()
