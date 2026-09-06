from __future__ import annotations
import importlib.util, sys, unittest
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"machine_number/ana_slo_prediction_v4_2_inventory_epoch_walk_forward.py"
SPEC=importlib.util.spec_from_file_location("epoch_bt",SCRIPT); epoch=importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name]=epoch; SPEC.loader.exec_module(epoch)

def frame(values):
    x=pd.DataFrame(values,columns=["date","machine_no","machine_name","diff"]);x["date"]=pd.to_datetime(x.date);x["win"]=(x["diff"]>0).astype(int);x["plus1000"]=(x["diff"]>=1000).astype(int);x["plus2000"]=(x["diff"]>=2000).astype(int);return x

class EpochBacktestTests(unittest.TestCase):
    def test_consecutive_change_only(self):
        snaps={pd.Timestamp("2026-01-01"):frame([("2026-01-01",1,"A",0)]),pd.Timestamp("2026-01-02"):frame([("2026-01-02",1,"B",0)]),pd.Timestamp("2026-01-04"):frame([("2026-01-04",1,"C",0)])}
        events=epoch.detect_events(snaps);self.assertEqual(len(events),1);self.assertEqual(events[0].new_machine_name,"B")
    def test_added_removed_and_renamed_are_not_confused(self):
        snaps={pd.Timestamp("2026-01-01"):frame([("2026-01-01",1,"OLD",0),("2026-01-01",2,"REMOVE",0)]),pd.Timestamp("2026-01-02"):frame([("2026-01-02",1,"NEW",0),("2026-01-02",3,"ADD",0)])}
        events=epoch.detect_events(snaps);self.assertEqual([(x.machine_no,x.old_machine_name,x.new_machine_name) for x in events],[(1,"OLD","NEW")])
    def test_same_name_return_uses_latest_epoch(self):
        events=[epoch.Event(pd.Timestamp("2026-01-02"),pd.Timestamp("2026-01-01"),1,"A","B"),epoch.Event(pd.Timestamp("2026-01-05"),pd.Timestamp("2026-01-04"),1,"B","A")]
        self.assertEqual(epoch.latest_event(events,1,"A",pd.Timestamp("2026-01-06")).change_date,pd.Timestamp("2026-01-05"))
    def test_zero_day_type_and_store_prior(self):
        hist=frame([("2026-01-01",1,"KNOWN",1000),("2026-01-02",2,"OTHER",-1000)])
        values,policy=epoch.stats(hist.iloc[:0],hist,pd.Timestamp("2026-01-03"),"KNOWN");self.assertEqual((policy,values["avg31"]),("TYPE_PRIOR",1000))
        values,policy=epoch.stats(hist.iloc[:0],hist,pd.Timestamp("2026-01-03"),"NEW");self.assertEqual((policy,values["avg31"]),("STORE_PRIOR",0))
    def test_one_two_three_six_seven_days(self):
        values=[100,-200,300,400,-500,600,700]; current=frame([(f"2026-01-{d:02d}",1,"NEW",v) for d,v in enumerate(values,2)])
        for n in (1,2,3,6,7):
            result,policy=epoch.stats(current.head(n),current.head(n),pd.Timestamp("2026-01-10"),"NEW");self.assertEqual(policy,"EPOCH_HISTORY");self.assertEqual(result["recent7_avg"],sum(values[:n])/n);self.assertEqual(result["prev_change"],0 if n==1 else values[n-1]-values[n-2])
    def test_changed_only_keeps_unchanged_raw_features(self):
        base=pd.DataFrame([{"machine_no":1,"machine_name":"NEW",**{k:10. for k in epoch.FEATURES}},{"machine_no":2,"machine_name":"KEEP",**{k:20. for k in epoch.FEATURES}}])
        hist=frame([("2026-01-01",1,"OLD",1000),("2026-01-02",1,"NEW",-100),("2026-01-01",2,"KEEP",500),("2026-01-02",2,"KEEP",600)])
        inv=frame([("2026-01-02",1,"NEW",-100),("2026-01-02",2,"KEEP",600)]); events=[epoch.Event(pd.Timestamp("2026-01-02"),pd.Timestamp("2026-01-01"),1,"OLD","NEW")]
        out=epoch.epoch_panel(base,hist,inv,events,pd.Timestamp("2026-01-03"));pd.testing.assert_series_equal(base.set_index("machine_no").loc[2,list(epoch.FEATURES)],out.set_index("machine_no").loc[2,list(epoch.FEATURES)]);changed=out.set_index("machine_no").loc[1];self.assertEqual(changed.epoch_history_n,1);self.assertEqual(changed.prev_change,0);self.assertEqual(changed.history_start_date,"2026-01-02")
    def test_lt7_exclusion_boundary(self):
        x=pd.DataFrame({"machine_no":[1,2,3],"inventory_changed":[True,True,False],"epoch_history_n":[6,7,0]});eligible=x[(~x.inventory_changed)|(x.epoch_history_n>=7)];self.assertEqual(eligible.machine_no.tolist(),[2,3])
    def test_all_exclusion_boundaries(self):
        x=pd.DataFrame({"machine_no":range(1,11),"inventory_changed":[True]*8+[False]*2,"epoch_history_n":[6,7,13,14,20,21,30,31,0,99]})
        for threshold,expected in ((7,[2,3,4,5,6,7,8,9,10]),(14,[4,5,6,7,8,9,10]),(21,[6,7,8,9,10]),(31,[8,9,10])):
            eligible=x[(~x.inventory_changed)|(x.epoch_history_n>=threshold)];self.assertEqual(eligible.machine_no.tolist(),expected)
        self.assertEqual(x[~x.inventory_changed].machine_no.tolist(),[9,10])
    def test_ranking_only_and_zpop_exclusion_are_distinct(self):
        class Ranker:
            def __init__(self):self.sizes=[]
            def rank_score(self,x,weights):self.sizes.append(len(x));y=x.copy();y["score"]=y["raw"];return y.sort_values("score",ascending=False)
        panel=pd.DataFrame({"machine_no":[1,2,3],"raw":[100.,2.,1.],"inventory_changed":[True,False,False],"epoch_history_n":[13,0,0]})
        a=Ranker();self.assertEqual(epoch.rank(a,panel,{},"EPOCH_EXCLUDE_LT14").machine_no.tolist(),[2,3]);self.assertEqual(a.sizes,[3])
        b=Ranker();self.assertEqual(epoch.rank(b,panel,{},"EPOCH_EXCLUDE_LT14_ZPOP_EXCLUDE").machine_no.tolist(),[2,3]);self.assertEqual(b.sizes,[2])
    def test_lag_buckets_extended(self):
        expected={1:"DAY1",2:"DAY2",3:"DAY3",4:"DAY4-7",7:"DAY4-7",8:"DAY8-14",14:"DAY8-14",15:"DAY15-21",21:"DAY15-21",22:"DAY22-31",31:"DAY22-31",32:"DAY32+"}
        self.assertEqual({x:epoch.bucket(x) for x in expected},expected)
    def test_output_is_research_only(self):
        self.assertIn("research_inventory_epoch",str(epoch.OUTPUT_DIR));self.assertNotIn("64_Ver4_2_future_top10",str(epoch.OUTPUT_DIR))
if __name__=="__main__":unittest.main()
