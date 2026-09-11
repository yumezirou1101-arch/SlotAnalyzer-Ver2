from __future__ import annotations

import csv
import io
import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'machine_number'))
import maruhan_maebashi_floor_map_generator as floor


def csv_bytes(rows):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode('utf-8')


def fixture_rows():
    return [{'日付':'2026-09-09','台番号':str(seat.machine_no),'機種名':' 同一機種A+ 半角　全角 '} for seat in floor.parse_layout(floor.DEFAULT_LAYOUT.read_bytes())]


class FloorMapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.daily = self.root/'ana_slo_20260909.csv'
        self.output = self.root/'output'
        self.daily.write_bytes(csv_bytes(fixture_rows()))

    def tearDown(self):
        self.temp.cleanup()

    def test_layout_has_exact_514_unique_20slot_numbers(self):
        self.assertNotIn(b'\r', floor.DEFAULT_LAYOUT.read_bytes(), 'Layout must remain LF-only for Git and SHA-256')
        seats = floor.parse_layout(floor.DEFAULT_LAYOUT.read_bytes())
        numbers = {seat.machine_no for seat in seats}
        self.assertEqual(len(seats),514)
        self.assertEqual(len(numbers),514)
        self.assertEqual(numbers,set(range(561,1115))-set(range(581,621)))
        self.assertEqual(len({seat.island_id for seat in seats}),20)
        self.assertFalse(numbers & set(range(581,621)))
        provenance = json.loads(floor.DEFAULT_LAYOUT.with_suffix('.json').read_text(encoding='utf-8'))
        self.assertEqual(provenance['layout_sha256'],floor.sha256_bytes(floor.DEFAULT_LAYOUT.read_bytes()))

    def test_physical_landmarks_directions_and_empty_slots(self):
        by_no = {seat.machine_no:seat for seat in floor.parse_layout(floor.DEFAULT_LAYOUT.read_bytes())}
        self.assertEqual(by_no[920].x,by_no[1114].x)
        self.assertLess(by_no[901].y,by_no[1114].y)
        self.assertLess(by_no[900].y,by_no[1113].y)
        self.assertLess(by_no[885].y,by_no[1112].y)
        self.assertLess(by_no[721].x,by_no[736].x)
        self.assertLess(by_no[884].x,by_no[872].x)
        self.assertLess(by_no[777].x,by_no[776].x)
        self.assertLess(by_no[777].y,by_no[796].y)
        self.assertLess(by_no[776].y,by_no[757].y)
        self.assertEqual(by_no[843].slot_index-by_no[842].slot_index,2)
        self.assertEqual(by_no[1014].slot_index-by_no[1015].slot_index,2)
        self.assertAlmostEqual(by_no[1006].x,(by_no[1015].x+by_no[1014].x)/2)
        ring = [by_no[n] for n in range(797,814)]
        self.assertEqual({seat.shape for seat in ring},{'arc'})
        self.assertEqual({seat.island_id for seat in ring},{'C05'})
        self.assertEqual([s.position_order for s in ring],list(range(1,18)))
        self.assertLess(by_no[797].x,by_no[813].x)
        self.assertGreater(by_no[812].x,by_no[813].x)
        self.assertTrue(all(54 < math.hypot(s.x-744,s.y-416) < 61 for s in ring))

    def test_daily_validation_and_exact_name_join(self):
        seats = floor.parse_layout(floor.DEFAULT_LAYOUT.read_bytes())
        data_date,names = floor.parse_daily(self.daily.read_bytes(),self.daily.name)
        joined = floor.join_daily(seats,names)
        self.assertEqual(data_date,date(2026,9,9))
        self.assertEqual(len(joined),514)
        self.assertEqual(joined[561],' 同一機種A+ 半角　全角 ')
        self.assertEqual(joined,names)

    def test_invalid_daily_never_renders_or_creates_output(self):
        cases = ['remove','add','duplicate','five_slot','blank_number','blank_name','unknown','wrong_date','mixed_date']
        for case in cases:
            with self.subTest(case=case):
                rows = fixture_rows()
                if case=='remove': rows.pop()
                elif case=='add': rows.append(dict(rows[0],台番号='9999'))
                elif case=='duplicate': rows[-1]['台番号']=rows[0]['台番号']
                elif case=='five_slot': rows[0]['台番号']='581'
                elif case=='blank_number': rows[0]['台番号']=''
                elif case=='blank_name': rows[0]['機種名']='　 '
                elif case=='unknown': rows[0]['台番号']='9999'
                elif case=='wrong_date':
                    for row in rows: row['日付']='2026-09-08'
                elif case=='mixed_date': rows[0]['日付']='2026-09-08'
                self.daily.write_bytes(csv_bytes(rows))
                with mock.patch.object(floor,'render_png') as render:
                    with self.assertRaises(floor.FloorMapValidationError):
                        floor.generate(self.daily,self.output)
                    render.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_invalid_input_keeps_previous_png_and_latest_untouched(self):
        self.output.mkdir()
        paths = [self.output/f'{floor.PREFIX}_latest.png',self.output/f'{floor.PREFIX}_2026-09-09.png']
        for path in paths: path.write_bytes(b'previous-output')
        rows = fixture_rows()
        rows[0]['機種名']=''
        self.daily.write_bytes(csv_bytes(rows))
        with self.assertRaises(floor.FloorMapValidationError):
            floor.generate(self.daily,self.output)
        self.assertTrue(all(p.read_bytes()==b'previous-output' for p in paths))

    def test_layout_invalid_counts_duplicates_coordinates_and_nan(self):
        original = floor._csv_rows(floor.DEFAULT_LAYOUT.read_bytes())
        for case in ['remove','duplicate','five_slot','blank','order','coordinate','nan']:
            with self.subTest(case=case):
                rows=[dict(row) for row in original]
                if case=='remove':rows.pop()
                elif case=='duplicate':rows[-1]['machine_no']=rows[0]['machine_no']
                elif case=='five_slot':rows[0]['machine_no']='620'
                elif case=='blank':rows[0]['machine_no']=''
                elif case=='order':rows[1]['position_order']=rows[0]['position_order']
                elif case=='coordinate':rows[1]['x'],rows[1]['y']=rows[0]['x'],rows[0]['y']
                elif case=='nan':rows[0]['x']='nan'
                with self.assertRaises(floor.FloorMapValidationError):
                    floor.parse_layout(csv_bytes(rows))

    def test_layout_sha_mismatch_blocks_generation(self):
        path=self.root/'layout.csv'
        path.write_bytes(floor.DEFAULT_LAYOUT.read_bytes()+b'\n')
        path.with_suffix('.json').write_bytes(floor.DEFAULT_LAYOUT.with_suffix('.json').read_bytes())
        with self.assertRaisesRegex(floor.FloorMapValidationError,'SHA-256'):
            floor.generate(self.daily,self.output,path)
        self.assertFalse(self.output.exists())

    def test_groups_do_not_cross_gap_or_opposite_side(self):
        seats=floor.parse_layout(floor.DEFAULT_LAYOUT.read_bytes())
        names={s.machine_no:'same' for s in seats}
        groups=floor.make_groups(seats,names)
        lower=[g for g in groups if g.island_id=='C03' and g.side=='S']
        upper=[g for g in groups if g.island_id=='C08' and g.side=='N']
        self.assertEqual(len(lower),2)
        self.assertEqual(lower[0].machine_numbers[-1],842)
        self.assertEqual(lower[1].machine_numbers[0],843)
        self.assertEqual(len(upper),2)
        self.assertFalse(any(777 in g.machine_numbers and 776 in g.machine_numbers for g in groups))

    def test_ring_wrap_groups_and_single_machine(self):
        seats=floor.parse_layout(floor.DEFAULT_LAYOUT.read_bytes())
        names={s.machine_no:'same' for s in seats}
        names[803]='single'
        ring=[g for g in floor.make_groups(seats,names) if g.island_id=='C05']
        self.assertEqual(len(ring),2)
        self.assertEqual(next(g.machine_numbers for g in ring if g.machine_name=='single'),(803,))
        contiguous=next(g.machine_numbers for g in ring if g.machine_name=='same')
        self.assertEqual(len(contiguous),16)
        self.assertEqual(set(contiguous),set(range(797,814))-{803})
        self.assertIn((813,797),list(zip(contiguous,contiguous[1:])))

    def test_png_date_join_hash_latest_and_byte_reproducibility(self):
        before=self.daily.read_bytes()
        metadata=floor.generate(self.daily,self.output,expected_date=date(2026,9,9))
        path=self.output/f'{floor.PREFIX}_2026-09-09.png'
        first=path.read_bytes()
        self.assertGreater(len(first),0)
        self.assertEqual(first,(self.output/f'{floor.PREFIX}_latest.png').read_bytes())
        with Image.open(path) as image:
            self.assertEqual(image.format,'PNG')
            self.assertEqual(image.info['data_date'],'2026-09-09')
            self.assertEqual(image.info['machine_count'],'514')
            self.assertEqual(image.width,4920)
            self.assertGreater(image.height,3550)
            image.verify()
        self.assertEqual(metadata['matched'],514)
        self.assertEqual(metadata['missing'],0)
        self.assertEqual(metadata['unknown'],0)
        self.assertEqual(metadata['png_sha256'],floor.sha256_bytes(first))
        self.assertEqual(metadata['rendering']['text_overlap_count'],0)
        self.assertEqual(metadata['rendering']['text_clipping_count'],0)
        self.assertEqual(sum(len(g['machine_numbers']) for g in metadata['rendering']['groups']),514)
        self.assertTrue(all(g['machine_name']==' 同一機種A+ 半角　全角 ' for g in metadata['rendering']['groups']))
        floor.generate(self.daily,self.output)
        self.assertEqual(first,path.read_bytes())
        self.assertEqual(before,self.daily.read_bytes())

    def test_newest_invalid_file_never_falls_back(self):
        new=self.root/'ana_slo_20260910.csv'
        new.write_bytes(b'bad')
        self.assertEqual(floor.latest_daily(self.root),new)
        with self.assertRaises(floor.FloorMapValidationError):
            floor.generate(floor.latest_daily(self.root),self.output)
        self.assertFalse(self.output.exists())

    def test_latest_does_not_move_backwards(self):
        self.output.mkdir()
        path=self.output/f'{floor.PREFIX}_latest.json'
        path.write_text('{"data_date":"2026-09-10"}',encoding='utf-8')
        before=path.read_bytes()
        with self.assertRaisesRegex(floor.FloorMapValidationError,'backwards'):
            floor.generate(self.daily,self.output)
        self.assertEqual(path.read_bytes(),before)

    def test_text_wrapping_preserves_all_characters(self):
        font=floor.ImageFont.truetype(str(floor.choose_font()),28)
        name=' 交響詩篇エウレカセブン HI-EVOLUTION ZERO TYPE‐ART  '
        lines=floor._wrap(name,font,180)
        self.assertEqual(''.join(lines),name)
        self.assertTrue(all(font.getlength(line)<=180 for line in lines))

    def test_cli_bad_input_returns_nonzero_without_output(self):
        self.daily.write_bytes(b'bad')
        self.assertEqual(floor.main(['--daily',str(self.daily),'--output-dir',str(self.output)]),1)
        self.assertFalse(self.output.exists())


if __name__=='__main__':
    unittest.main()
