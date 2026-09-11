"""Read-only daily CSV -> validated physical layout -> deterministic PNG.

Standalone visualization: does not import/run prediction, guards or automation.
Coordinates are original plan pixels, not surveyed distances or model features.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin, __version__ as PILLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAYOUT = Path(__file__).with_name('maruhan_maebashi_floor_map_layout.csv')
DEFAULT_DATA_DIR = ROOT / 'data/maruhan_maebashi/machine_number'
DEFAULT_OUTPUT_DIR = ROOT / 'outputs/floor_map'
EXPECTED_COUNT = 514
PREFIX = 'maruhan_maebashi_floor_map'
VERSION = '1.0'
SIDE_ORDER = ('N', 'S', 'W', 'E', 'RING')
SCALE = 4
WIDTH = 4920
MAP_BOTTOM = 3550
PALETTE = ('#dcebf9', '#e2f1e8', '#f8ead5', '#ece4f6', '#f7e1e5', '#dff0f2')


class FloorMapValidationError(ValueError):
    """Invalid evidence: no official image may be published."""


@dataclass(frozen=True)
class Seat:
    machine_no: int
    island_id: str
    side: str
    position_order: int
    x: float
    y: float
    rotation: float
    slot_index: int
    shape: str


@dataclass(frozen=True)
class LabelGroup:
    label_id: str
    island_id: str
    side: str
    machine_name: str
    machine_numbers: tuple[int, ...]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _csv_rows(payload: bytes) -> list[dict[str, str]]:
    try:
        reader = csv.DictReader(io.StringIO(payload.decode('utf-8-sig'), newline=''))
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise FloorMapValidationError('CSV header missing or duplicated')
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise FloorMapValidationError('CSV column count mismatch')
        return rows
    except (UnicodeError, csv.Error) as exc:
        raise FloorMapValidationError(f'Unreadable CSV: {exc}') from exc


def _integer(value: str, label: str) -> int:
    if not re.fullmatch(r'[0-9]+', value or ''):
        raise FloorMapValidationError(f'{label}: missing or invalid integer {value!r}')
    return int(value)


def _validate_numbers(numbers: list[int], label: str) -> None:
    if len(numbers) != EXPECTED_COUNT:
        raise FloorMapValidationError(f'{label}: rows={len(numbers)}, expected 514')
    duplicates = sorted(n for n, count in Counter(numbers).items() if count > 1)
    if duplicates:
        raise FloorMapValidationError(f'{label}: duplicate machine_no {duplicates}')
    five_slot = sorted(n for n in numbers if 581 <= n <= 620)
    if five_slot:
        raise FloorMapValidationError(f'{label}: 5-slot contamination {five_slot}')


def parse_layout(payload: bytes) -> list[Seat]:
    rows = _csv_rows(payload)
    fields = set(Seat.__dataclass_fields__)
    if not rows or not fields.issubset(rows[0]):
        raise FloorMapValidationError('Layout columns missing')
    seats = []
    for row in rows:
        try:
            seat = Seat(_integer(row['machine_no'], 'layout machine_no'), row['island_id'], row['side'],
                        _integer(row['position_order'], 'position_order'), float(row['x']), float(row['y']),
                        float(row['rotation']), _integer(row['slot_index'], 'slot_index'), row['shape'])
        except ValueError as exc:
            raise FloorMapValidationError(f'Invalid layout value: {exc}') from exc
        if (not seat.island_id or seat.side not in SIDE_ORDER or seat.position_order < 1
                or seat.shape not in {'rect', 'arc'} or (seat.side == 'RING') != (seat.shape == 'arc')
                or not all(math.isfinite(v) for v in (seat.x, seat.y, seat.rotation))
                or not (350 <= seat.x <= 1500 and 45 <= seat.y <= 830)):
            raise FloorMapValidationError(f'Invalid layout geometry: {seat.machine_no}')
        seats.append(seat)
    _validate_numbers([s.machine_no for s in seats], 'layout')
    if len({(s.x, s.y) for s in seats}) != EXPECTED_COUNT:
        raise FloorMapValidationError('Duplicate physical coordinates')
    sides = defaultdict(list)
    for seat in seats:
        sides[seat.island_id, seat.side].append(seat)
    for key, side in sides.items():
        side.sort(key=lambda s: s.position_order)
        if [s.position_order for s in side] != list(range(1, len(side) + 1)):
            raise FloorMapValidationError(f'Non-unique/non-contiguous position order: {key}')
        if any(b.slot_index <= a.slot_index for a, b in zip(side, side[1:])):
            raise FloorMapValidationError(f'Invalid physical slot sequence: {key}')
    return sorted(seats, key=lambda s: (s.island_id, SIDE_ORDER.index(s.side), s.position_order))


def parse_daily(payload: bytes, source_name: str) -> tuple[date, dict[int, str]]:
    rows = _csv_rows(payload)
    if not rows or not {'日付', '台番号', '機種名'}.issubset(rows[0]):
        raise FloorMapValidationError('Daily columns 日付/台番号/機種名 are required')
    numbers = [_integer(row['台番号'], 'daily machine_no') for row in rows]
    _validate_numbers(numbers, 'daily')
    if any(not row['機種名'].strip() for row in rows):
        raise FloorMapValidationError('Daily machine_name is missing')
    dates = {row['日付'] for row in rows}
    if len(dates) != 1:
        raise FloorMapValidationError('Daily internal dates differ')
    try:
        target = date.fromisoformat(next(iter(dates)))
    except ValueError as exc:
        raise FloorMapValidationError('Invalid daily date') from exc
    if dates != {target.isoformat()} or source_name != f'ana_slo_{target:%Y%m%d}.csv':
        raise FloorMapValidationError('Daily filename/internal date mismatch')
    # Preserve the exact name, including whitespace and punctuation. No normalization.
    return target, {number: row['機種名'] for number, row in zip(numbers, rows)}


def join_daily(seats: list[Seat], names: dict[int, str]) -> dict[int, str]:
    layout_numbers = {seat.machine_no for seat in seats}
    missing = sorted(layout_numbers - names.keys())
    unknown = sorted(names.keys() - layout_numbers)
    if missing or unknown:
        raise FloorMapValidationError(f'Layout/daily mismatch: missing={missing}; unknown={unknown}')
    return {seat.machine_no: names[seat.machine_no] for seat in seats}


def make_groups(seats: list[Seat], names: dict[int, str]) -> list[LabelGroup]:
    sides = defaultdict(list)
    for seat in seats:
        sides[seat.island_id, seat.side].append(seat)
    groups = []
    for (island, side), members in sorted(sides.items(), key=lambda kv: (kv[0][0], SIDE_ORDER.index(kv[0][1]))):
        members.sort(key=lambda seat: seat.position_order)
        runs = []
        for seat in members:
            if runs and names[runs[-1][-1].machine_no] == names[seat.machine_no] and seat.slot_index == runs[-1][-1].slot_index + 1:
                runs[-1].append(seat)
            else:
                runs.append([seat])
        if side == 'RING' and len(runs) > 1 and names[runs[0][0].machine_no] == names[runs[-1][-1].machine_no]:
            runs[0] = runs[-1] + runs[0]
            runs.pop()
        for run in runs:
            groups.append(LabelGroup(f'G{len(groups)+1:03d}', island, side, names[run[0].machine_no], tuple(s.machine_no for s in run)))
    return groups


def choose_font(explicit: Path | None = None) -> Path:
    candidates = [explicit] if explicit else [
        Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/meiryo.ttc',
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'),
    ]
    for path in candidates:
        if path and path.is_file():
            return path
    raise FloorMapValidationError('Japanese font unavailable; specify --font (TTF/TTC/OTF)')


def _wrap(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    """Character wrap without dropping/normalizing any source characters."""
    lines, current = [], ''
    for character in text:
        if character in '\r\n':
            raise FloorMapValidationError('Embedded line break in display text')
        if font.getlength(current + character) > width and current:
            lines.append(current)
            current = ''
        if font.getlength(character) > width:
            raise FloorMapValidationError('A glyph exceeds label width')
        current += character
    return lines + [current]


def _number_ranges(numbers: tuple[int, ...]) -> str:
    parts, start, end, direction = [], numbers[0], numbers[0], 0
    for number in numbers[1:]:
        delta = number - end
        if abs(delta) == 1 and direction in (0, delta):
            end, direction = number, delta
        else:
            parts.append(str(start) if start == end else f'{start}–{end}')
            start = end = number
            direction = 0
    parts.append(str(start) if start == end else f'{start}–{end}')
    return ', '.join(parts)


def render_png(seats: list[Seat], names: dict[int, str], provenance: dict,
               target: date, source_sha: str, layout_sha: str, font_path: Path) -> tuple[bytes, dict]:
    groups = make_groups(seats, names)
    group_by_no = {n: g for g in groups for n in g.machine_numbers}
    if len(group_by_no) != EXPECTED_COUNT:
        raise FloorMapValidationError('Label coverage must be 514/514')
    fonts = {size: ImageFont.truetype(str(font_path), size) for size in (16, 24, 26, 28, 32, 42, 54)}
    line_height = 46
    card_width, margin, gap = 1170, 60, 40
    cards = []
    column_bottoms = [MAP_BOTTOM + 190] * 4
    for island in sorted(provenance['islands']):
        island_groups = [g for g in groups if g.island_id == island]
        content = []
        for group in island_groups:
            name_lines = _wrap(f'{group.label_id}  {group.machine_name}', fonts[28], card_width - 64)
            number_lines = _wrap(f'{group.side}  {_number_ranges(group.machine_numbers)}  ({len(group.machine_numbers)}台)', fonts[24], card_width - 64)
            content.append((group, name_lines, number_lines))
        height = 88 + sum((len(n) + len(m)) * line_height + 18 for _, n, m in content) + 24
        column = min(range(4), key=lambda c: (column_bottoms[c], c))
        x, y = margin + column * (card_width + gap), column_bottoms[column]
        cards.append((island, x, y, height, content))
        column_bottoms[column] += height + gap
    image = Image.new('RGB', (WIDTH, max(column_bottoms) + 120), '#f4f6f8')
    draw = ImageDraw.Draw(image)
    text_boxes = []
    def text(x, y, value, size=28, color='#182e42', centered=False):
        font = fonts[size]
        if centered:
            x -= font.getlength(value) / 2
        box = draw.textbbox((x, y), value, font=font, anchor='lt')
        if min(box[0], box[1]) < 0 or box[2] > image.width or box[3] > image.height:
            raise FloorMapValidationError(f'Text outside image: {value}')
        draw.text((x, y), value, font=font, fill=color, anchor='lt')
        if value.strip():
            text_boxes.append(box)
    draw.rectangle((0, 0, WIDTH, 175), fill='#173c61')
    text(60, 32, 'マルハン メガシティ前橋インター  |  20スロ フロアマップ V1', 54, '#ffffff')
    text(60, 112, f'データ日付 {target.isoformat()}  ・  514 / 514台照合済み  ・  5スロ581–620は正式対象外', 32, '#ffffff')
    text(60, 202, '台番号の下のG番号 → 下段「島別 機種ラベル」の同じID。機種名はdaily CSV原文。配置は模式図です。', 28)
    draw.rounded_rectangle((35, 262, WIDTH-35, MAP_BOTTOM), radius=16, fill='#ffffff', outline='#c8d1da', width=2)
    def xy(x, y):
        return (x - 350) * SCALE + 140, (y - 45) * SCALE + 330
    seat_rects = []
    for seat in seats:
        group = group_by_no[seat.machine_no]
        fill = PALETTE[(int(group.label_id[1:])-1) % len(PALETTE)]
        cx, cy = xy(seat.x, seat.y)
        if seat.shape == 'rect':
            # Dimensions fit both straight vertical and horizontal banks.
            box = (cx-46, cy-29, cx+46, cy+29)
            seat_rects.append((seat.machine_no, box))
            draw.rectangle(box, fill=fill, outline='#42617a', width=2)
        else:
            ring = provenance['circular_island']
            ring_seats = sorted((s for s in seats if s.shape == 'arc'), key=lambda s: s.rotation)
            index = ring_seats.index(seat)
            angle = seat.rotation
            previous = ring_seats[index-1].rotation if index else ring_seats[-1].rotation - 360
            following = ring_seats[index+1].rotation if index+1 < len(ring_seats) else ring_seats[0].rotation + 360
            low, high = (previous+angle)/2, (angle+following)/2
            polygon = []
            # Widen the band inward for the second (G-ID) line, keeping seat centers
            # and the outer footprint fixed to the original plan.
            for radius, values in [(ring['outer_radius'], range(11)), (ring['inner_radius']-4, range(10,-1,-1))]:
                for step in values:
                    radians = math.radians(low + (high-low)*step/10)
                    polygon.append(xy(ring['center_x'] + radius*math.cos(radians), ring['center_y'] + radius*math.sin(radians)))
            draw.polygon(polygon, fill=fill, outline='#42617a', width=2)
        text(cx, cy-24, str(seat.machine_no), 26, centered=True)
        text(cx, cy+7, group.label_id, 16, centered=True)
    for island in sorted(provenance['islands']):
        members = [s for s in seats if s.island_id == island]
        if island == 'C05':
            x, y = xy(744, 411)
            text(x, y-10, island, 32, centered=True)
            text(x, y+37, '円形17台', 24, centered=True)
        else:
            x, y = xy(min(s.x for s in members)-13, min(s.y for s in members)-21)
            text(x, y-18, island, 24)
    for x, label in [(1366, '601–620'), (1478, '581–600')]:
        x1, y1 = xy(x-13, 495)
        x2, y2 = xy(x+14, 830)
        draw.rectangle((x1,y1,x2,y2), fill='#f1f1f1', outline='#aaaaaa', width=2)
        for i, line in enumerate(['5スロ', '対象外']):
            text((x1+x2)/2, y1+30+i*45, line, 24, '#666666', centered=True)
        # Machine numbers of excluded slots are not added to the layout/join.
        for i, character in enumerate(label):
            text((x1+x2)/2, y1+155+i*35, character, 24, '#666666', centered=True)
    text(60, MAP_BOTTOM+40, '島別 機種ラベル  |  G番号は配置図と一致・機種名を省略せず掲載', 42)
    text(60, MAP_BOTTOM+110, 'N/S＝横島の上/下、W/E＝縦島の左/右、RING＝円形。空き位置・通路をまたぐ同一機種は別ラベル。', 26)
    for island, x, y, height, content in cards:
        draw.rounded_rectangle((x,y,x+card_width,y+height), radius=12, fill='#ffffff', outline='#c8d1da', width=2)
        text(x+28, y+24, f'{island}  {provenance["islands"][island]}', 28)
        cursor = y+88
        for group, name_lines, number_lines in content:
            fill = PALETTE[(int(group.label_id[1:])-1) % len(PALETTE)]
            block_height = (len(name_lines)+len(number_lines))*line_height
            draw.rectangle((x+14,cursor-6,x+card_width-14,cursor+block_height), fill=fill)
            for line in name_lines:
                text(x+30,cursor,line,28)
                cursor += line_height
            for line in number_lines:
                text(x+30,cursor,line,24,'#425466')
                cursor += line_height
            cursor += 18
    text(60, image.height-78, '位置マスターV1・現行機種はdaily CSVのみからJOIN / 予測・正式成績・Guard・朝自動化には接続していません。', 26)
    # Check actual font ink bounds, not just string lengths. Fail rather than clip.
    overlap_count = sum(1 for i,a in enumerate(text_boxes) for b in text_boxes[i+1:]
                        if min(a[2],b[2]) > max(a[0],b[0]) and min(a[3],b[3]) > max(a[1],b[1]))
    seat_overlaps = [(n,m) for i,(n,a) in enumerate(seat_rects) for m,b in seat_rects[i+1:]
                     if min(a[2],b[2]) > max(a[0],b[0]) and min(a[3],b[3]) > max(a[1],b[1])]
    if overlap_count or seat_overlaps:
        raise FloorMapValidationError(f'Render overlap: text={overlap_count}, seats={seat_overlaps[:5]}')
    info = PngImagePlugin.PngInfo()
    for key, value in {'data_date': target.isoformat(), 'machine_count': '514',
                       'source_sha256': source_sha, 'layout_sha256': layout_sha}.items():
        info.add_text(key, value)
    stream = io.BytesIO()
    image.save(stream, format='PNG', pnginfo=info, optimize=False, compress_level=9)
    return stream.getvalue(), dict(width=image.width,height=image.height,label_count=len(groups),
        labels_by_island=dict(sorted(Counter(g.island_id for g in groups).items())),
        text_overlap_count=overlap_count,text_clipping_count=0,seat_overlap_count=0,
        groups=[dict(label_id=g.label_id,island_id=g.island_id,side=g.side,machine_name=g.machine_name,
                     machine_numbers=list(g.machine_numbers)) for g in groups])


def generate(daily_path: Path, output_dir: Path = DEFAULT_OUTPUT_DIR, layout_path: Path = DEFAULT_LAYOUT,
             font_path: Path | None = None, expected_date: date | None = None) -> dict:
    daily_path, layout_path, output_dir = Path(daily_path), Path(layout_path), Path(output_dir)
    # Read input bytes once; validation, join, rendering and SHA refer to that snapshot.
    daily_bytes, layout_bytes = daily_path.read_bytes(), layout_path.read_bytes()
    provenance_path = layout_path.with_suffix('.json')
    provenance_bytes = provenance_path.read_bytes()
    provenance = json.loads(provenance_bytes)
    if provenance.get('schema_version') != 1 or provenance.get('layout_sha256') != sha256_bytes(layout_bytes):
        raise FloorMapValidationError('Layout provenance version/SHA-256 mismatch')
    seats = parse_layout(layout_bytes)
    if set(provenance['islands']) != {s.island_id for s in seats}:
        raise FloorMapValidationError('Provenance island set mismatch')
    target, names = parse_daily(daily_bytes, daily_path.name)
    if expected_date is not None and target != expected_date:
        raise FloorMapValidationError('Requested date differs from daily date')
    names = join_daily(seats, names)
    font_path = choose_font(font_path)
    latest_manifest = output_dir / f'{PREFIX}_latest.json'
    if latest_manifest.exists():
        old = json.loads(latest_manifest.read_text(encoding='utf-8'))
        if date.fromisoformat(old['data_date']) > target:
            raise FloorMapValidationError('Refusing to move latest backwards to an older data date')
    png, rendering = render_png(seats,names,provenance,target,sha256_bytes(daily_bytes),sha256_bytes(layout_bytes),font_path)
    metadata = dict(schema_version=1,generator_version=VERSION,data_date=target.isoformat(),
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_csv=str(daily_path.resolve()),source_sha256=sha256_bytes(daily_bytes),
        layout_master=str(layout_path.resolve()),layout_sha256=sha256_bytes(layout_bytes),
        layout_provenance_sha256=sha256_bytes(provenance_bytes),
        font_path=str(font_path.resolve()),font_sha256=sha256_bytes(font_path.read_bytes()),pillow_version=PILLOW_VERSION,
        machine_count=EXPECTED_COUNT,daily_rows=514,layout_rows=514,matched=514,duplicates=0,missing=0,unknown=0,
        daily_unique_machine_no=514,layout_unique_machine_no=514,blank_machine_no=0,blank_machine_name=0,
        five_slot_contamination=0,png_sha256=sha256_bytes(png),rendering=rendering)
    encoded = (json.dumps(metadata,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    # Nothing has been created/overwritten before all validation and rendering succeed.
    output_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.floor-map-',dir=output_dir) as staging:
        artifacts = {f'{PREFIX}_{target.isoformat()}.png':png, f'{PREFIX}_{target.isoformat()}.json':encoded,
                     f'{PREFIX}_latest.png':png, f'{PREFIX}_latest.json':encoded}
        for name, payload in artifacts.items():
            (Path(staging)/name).write_bytes(payload)
        # Each replace is atomic; latest metadata is the final publication marker.
        for name in artifacts:
            os.replace(Path(staging)/name,output_dir/name)
    return metadata


def latest_daily(directory: Path = DEFAULT_DATA_DIR) -> Path:
    candidates = sorted(path for path in directory.glob('ana_slo_????????.csv')
                        if re.fullmatch(r'ana_slo_[0-9]{8}\.csv',path.name))
    if not candidates:
        raise FloorMapValidationError('No daily CSV found')
    return candidates[-1]  # An invalid newest file fails; never fall back to an older date.


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--daily',type=Path)
    parser.add_argument('--layout',type=Path,default=DEFAULT_LAYOUT)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--font',type=Path)
    parser.add_argument('--data-date',type=date.fromisoformat)
    args = parser.parse_args(argv)
    try:
        result = generate(args.daily or latest_daily(),args.output_dir,args.layout,args.font,args.data_date)
    except (FloorMapValidationError,OSError,ValueError,KeyError) as exc:
        print(f'FLOOR_MAP_BLOCKED: {exc}')
        return 1
    print(f'FLOOR_MAP_OK: {result["data_date"]} matched=514/514 labels={result["rendering"]["label_count"]}')
    print(args.output_dir / f'{PREFIX}_{result["data_date"]}.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
