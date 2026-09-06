from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"machine_number/ana_slo_bigmarch_oyagi_provisional_future_ranking.py"
JUGGLER=ROOT/"machine_number/ana_slo_bigmarch_oyagi_juggler_recent7_future_ranking.py"

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module); return module

provisional=load("bigmarch_provisional_test",SCRIPT)
juggler=load("bigmarch_juggler_test",JUGGLER)

def write_daily(root:Path,day:date,rows:int=200,duplicate:bool=False):
    data=root/provisional.DATA_REL;data.mkdir(parents=True,exist_ok=True)
    numbers=list(range(1,rows+1));
    if duplicate and len(numbers)>1:numbers[-1]=numbers[0]
    frame=pd.DataFrame({"date":[day.isoformat()]*rows,"machine_name":[f"台{x}" for x in range(rows)],"machine_no":numbers,"G":[1000]*rows,"diff":[0]*rows})
    path=data/f"ana_slo_bigmarch_oyagi_{day:%Y%m%d}.csv";frame.to_csv(path,index=False,encoding="utf-8-sig");return path

def fake_modules(latest=date(2026,9,4),calls=None):
    history=pd.DataFrame({"date":[pd.Timestamp(latest)],"machine_no":[1],"machine_name":["X"]})
    calls=calls if calls is not None else []
    def build_j(h,l,t=None):
        calls.append(("J",pd.Timestamp(l).date(),pd.Timestamp(t).date()));return pd.DataFrame({"machine_no":[1],"machine_name":["J"],"prediction_rank":[1],"target_date":[pd.Timestamp(t).date()],"latest_data_date":[pd.Timestamp(l).date()]})
    def build_n(h,l,t):
        calls.append(("N",pd.Timestamp(l).date(),pd.Timestamp(t).date(),pd.Timestamp(t).weekday()));return pd.DataFrame({"machine_no":[2],"machine_name":["N"],"prediction_rank":[1],"target_date":[pd.Timestamp(t).date()],"latest_data_date":[pd.Timestamp(l).date()]})
    common=lambda:(history.copy(),Path("locked.csv"),[])
    return SimpleNamespace(MODEL_NAME="J_MODEL",load_frozen_history=common,build_future_ranking=build_j),SimpleNamespace(MODEL_NAME="N_MODEL",load_frozen_history=common,build_future_ranking=build_n)

class BigMarchProvisionalTests(unittest.TestCase):
    def test_eligible_exactly_one_missing_day(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);write_daily(root,date(2026,9,4));e=provisional.assess_eligibility(root,date(2026,9,6));self.assertEqual((e.expected_data_date,e.latest_data_date),(date(2026,9,5),date(2026,9,4)))

    def test_two_days_old_is_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);write_daily(root,date(2026,9,3))
            with self.assertRaisesRegex(provisional.ProvisionalBlockedError,"GAP_NOT_ONE_DAY"):provisional.assess_eligibility(root,date(2026,9,6))

    def test_expected_source_or_daily_blocks(self):
        for kind in ("source","daily"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                root=Path(td);write_daily(root,date(2026,9,4))
                if kind=="source":(root/"ana_slo_bigmarch_oyagi_20260905_source.html").write_text("broken",encoding="utf-8")
                else:write_daily(root,date(2026,9,5))
                with self.assertRaises(provisional.ProvisionalBlockedError):provisional.assess_eligibility(root,date(2026,9,6))

    def test_latest_daily_quality_fail_closed(self):
        for rows,duplicate in ((199,False),(200,True)):
            with self.subTest(rows=rows,duplicate=duplicate),tempfile.TemporaryDirectory() as td:
                root=Path(td);write_daily(root,date(2026,9,4),rows,duplicate)
                with self.assertRaises(provisional.ProvisionalBlockedError):provisional.assess_eligibility(root,date(2026,9,6))

    def test_generate_contract_dates_weekday_and_isolated_output(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);write_daily(root,date(2026,9,4));calls=[];modules=fake_modules(calls=calls)
            formal=[]
            for rel in ("08_juggler_recent7_top3_forward","09_juggler_recent7_future_ranking","11_nonjuggler_weekday_top1_forward","12_nonjuggler_weekday_future_ranking"):
                marker=root/provisional.DATA_REL/"analysis_31days_deep"/rel/"marker";marker.parent.mkdir(parents=True,exist_ok=True);marker.write_text("unchanged",encoding="utf-8");formal.append(marker)
            with mock.patch.object(provisional,"load_ranking_modules",return_value=modules):result=provisional.generate(root,date(2026,9,6))
            self.assertEqual(result["status"],"PROVISIONAL");self.assertEqual(calls,[('J',date(2026,9,4),date(2026,9,6)),('N',date(2026,9,4),date(2026,9,6),6)])
            paths=result["paths"];self.assertIn("90_provisional_future_ranking",str(paths["directory"]));self.assertTrue(all(p.read_text(encoding="utf-8")=="unchanged" for p in formal))
            metadata=pd.read_csv(paths["metadata"],encoding="utf-8-sig").iloc[0]
            self.assertEqual((metadata.target_date,metadata.expected_data_date,metadata.latest_data_date),("2026-09-06","2026-09-05","2026-09-04"));self.assertEqual((int(metadata.expected_gap_days),int(metadata.target_to_latest_gap_days)),(1,2));self.assertEqual(metadata.ranking_class,"PROVISIONAL");self.assertTrue(bool(metadata.provisional));self.assertFalse(bool(metadata.formal));self.assertFalse(bool(metadata.forward_valid));self.assertFalse(bool(metadata.eligible_for_formal_evaluation))
            for key in ("juggler_all","juggler_top10","nonjuggler_all","nonjuggler_top10"):
                frame=pd.read_csv(paths[key],encoding="utf-8-sig");self.assertEqual(frame.ranking_class.unique().tolist(),["PROVISIONAL"]);self.assertEqual(frame.target_date.astype(str).unique().tolist(),["2026-09-06"]);self.assertFalse(frame.forward_valid.astype(bool).any())

    def test_safe_rerun_and_conflict(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);write_daily(root,date(2026,9,4));modules=fake_modules()
            with mock.patch.object(provisional,"load_ranking_modules",return_value=modules):
                first=provisional.generate(root,date(2026,9,6));before={k:p.stat().st_mtime_ns for k,p in first["paths"].items() if k!="directory"};second=provisional.generate(root,date(2026,9,6));self.assertEqual(second["status"],"ALREADY_PROVISIONAL");self.assertEqual(before,{k:p.stat().st_mtime_ns for k,p in first["paths"].items() if k!="directory"})
                metadata=pd.read_csv(first["paths"]["metadata"],encoding="utf-8-sig");metadata.loc[0,"latest_data_date"]="2026-09-03";metadata.to_csv(first["paths"]["metadata"],index=False,encoding="utf-8-sig")
                with self.assertRaises(provisional.ProvisionalManualReviewError):provisional.generate(root,date(2026,9,6))

    def test_juggler_default_backward_compatible_and_explicit_target(self):
        history=pd.DataFrame({"date":pd.to_datetime(["2026-09-04","2026-09-03"]),"machine_no":[1,1],"machine_name":["ジャグラー","ジャグラー"],"is_juggler":[True,True],"win":[1,0],"diff":[100,-100]})
        default=juggler.build_future_ranking(history,pd.Timestamp("2026-09-04"));explicit=juggler.build_future_ranking(history,pd.Timestamp("2026-09-04"),pd.Timestamp("2026-09-06"));self.assertEqual(str(default.iloc[0].target_date),"2026-09-05");self.assertEqual(str(explicit.iloc[0].target_date),"2026-09-06")

    def test_no_forward_allow_gap_or_formal_output_references(self):
        source=SCRIPT.read_text(encoding="utf-8");self.assertNotIn("--allow-gap",source);self.assertNotIn("top3_forward",source);self.assertNotIn("top1_forward",source);self.assertNotIn("09_juggler_recent7_future_ranking",str(provisional.OUTPUT_REL));self.assertNotIn("12_nonjuggler_weekday_future_ranking",str(provisional.OUTPUT_REL))

if __name__=="__main__":unittest.main()
