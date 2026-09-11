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
VERSION = '2.0'
SIDE_ORDER = ('N', 'S', 'W', 'E', 'RING')
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


def display_positions(seats: list[Seat]) -> dict[int, tuple[float, float]]:
    """Schematic island translations; physical master and within-side order stay intact."""
    east = {'E01': 2714, 'E02': 3094, 'E03': 3474, 'E04': 3840,
            'E05': 2714, 'E06': 3094, 'E07': 3420}
    horizontal_y = {'C01': 240, 'C02': 560, 'C03': 880, 'C04': 1200,
                    'C08': 2970, 'C09': 3320}
    result = {}
    for seat in seats:
        island = seat.island_id
        if island in horizontal_y:
            members = [s for s in seats if s.island_id == island]
            x = 660 + (seat.x-550)*4
            y = horizontal_y[island] + (seat.y-min(s.y for s in members))*4
        elif island == 'C05':
            x, y = 660+(seat.x-550)*4, 1680+(seat.y-416)*4
        elif island in {'C06', 'C07'}:
            x, y = 660+(seat.x-550)*4, 2110+(seat.y-520)*4.5
        else:
            members = [s for s in seats if s.island_id == island]
            center = (min(s.x for s in members)+max(s.x for s in members))/2
            anchor = east.get(island, 100 if island in {'W01','W03'} else 410)
            x = anchor+(seat.x-center)*4
            lower = island in {'W03','W04','E05','E06','E07'}
            y = (2050 if lower else 288)+(seat.y-(507 if lower else 77))*4.5
        result[seat.machine_no] = (x, y)
    # The two protected holes must remain exactly two ordinary seat pitches apart.
    for left, right, neighbor in [(842,843,841),(1015,1014,1016)]:
        a,b,c = result[left],result[right],result[neighbor]
        if abs(abs(b[0]-a[0])-2*abs(a[0]-c[0])) > .01 or a[1] != b[1]:
            raise FloorMapValidationError('Protected one-seat gap changed')
    return result


def fit_label(value: str, font_path: Path, width: int, height: int):
    """Prefer two lines; use three for narrow single seats, never truncate."""
    for max_lines in (2, 3):
        for size in range(28, 11, -1):
            font = ImageFont.truetype(str(font_path), size)
            lines = _wrap(value, font, width-8)
            if len(lines) <= max_lines and len(lines)*(size+6) <= height-8:
                return font, lines, size
    raise FloorMapValidationError(f'Machine name cannot fit without clipping: {value!r}')


def render_png(seats: list[Seat], names: dict[int, str], provenance: dict,
               target: date, source_sha: str, layout_sha: str, font_path: Path) -> tuple[bytes, dict]:
    groups = make_groups(seats, names)
    group_by_no = {n:g for g in groups for n in g.machine_numbers}
    if len(group_by_no) != EXPECTED_COUNT:
        raise FloorMapValidationError('Label coverage must be 514/514')
    positions = display_positions(seats)
    image = Image.new('RGB', (4030, 3610), '#ffffff')
    draw = ImageDraw.Draw(image)
    text_boxes, seat_rects, label_boxes, rendered_labels = [], [], [], []
    def intersects(a,b):
        return min(a[2],b[2]) > max(a[0],b[0]) and min(a[3],b[3]) > max(a[1],b[1])
    def ink(x,y,value,size,color='#162f44',center=False):
        font = ImageFont.truetype(str(font_path),size)
        if center: x -= font.getlength(value)/2
        box = draw.textbbox((x,y),value,font=font,anchor='lt')
        if box[0]<0 or box[1]<0 or box[2]>image.width or box[3]>image.height:
            raise FloorMapValidationError('Text outside image')
        draw.text((x,y),value,font=font,anchor='lt',fill=color)
        text_boxes.append(box)
    draw.rectangle((0,0,image.width,110),fill='#173c61')
    ink(40,14,'マルハン メガシティ前橋インター  20スロ フロアマップ V2',42,'white')
    ink(40,70,f'{target.isoformat()}  |  514台照合済み  |  機種名：daily CSV原文  |  配置は模式図',25,'white')
    for seat in seats:
        cx,cy = positions[seat.machine_no]
        group = group_by_no[seat.machine_no]
        fill = PALETTE[(int(group.label_id[1:])-1)%len(PALETTE)]
        if seat.shape == 'rect':
            box = (cx-46,cy-29,cx+46,cy+29)
            seat_rects.append((seat.machine_no,box))
            draw.rectangle(box,fill=fill,outline='#42617a',width=2)
        else:
            ring = provenance['circular_island']
            members = sorted((s for s in seats if s.shape=='arc'),key=lambda s:s.rotation)
            i = members.index(seat)
            before = members[i-1].rotation if i else members[-1].rotation-360
            after = members[i+1].rotation if i+1<len(members) else members[0].rotation+360
            lo,hi = (before+seat.rotation)/2,(seat.rotation+after)/2
            polygon=[]
            for radius,steps in [(ring['outer_radius'],range(11)),(ring['inner_radius']-4,range(10,-1,-1))]:
                for step in steps:
                    angle=math.radians(lo+(hi-lo)*step/10)
                    polygon.append((660+(ring['center_x']-550+radius*math.cos(angle))*4,
                                    1680+(ring['center_y']-416+radius*math.sin(angle))*4))
            draw.polygon(polygon,fill=fill,outline='#42617a',width=2)
        ink(cx,cy-20,str(seat.machine_no),34,center=True)
    def label(group,box,rotate=False):
        x0,y0,x1,y1 = [int(round(v)) for v in box]
        if x0<0 or y0<0 or x1>image.width or y1>image.height or x1<=x0 or y1<=y0:
            raise FloorMapValidationError('Machine label outside image')
        width,height=x1-x0,y1-y0
        tw,th = (height,width) if rotate else (width,height)
        font,lines,size=fit_label(group.machine_name,font_path,tw,th)
        tile=Image.new('RGB',(tw,th),PALETTE[(int(group.label_id[1:])-1)%len(PALETTE)])
        td=ImageDraw.Draw(tile)
        for i,line in enumerate(lines):
            x=(tw-font.getlength(line))/2
            y=(th-len(lines)*(size+6))/2+i*(size+6)
            bounds=td.textbbox((x,y),line,font=font,anchor='lt')
            if bounds[0]<0 or bounds[1]<0 or bounds[2]>tw or bounds[3]>th:
                raise FloorMapValidationError('Machine name ink clipped')
            td.text((x,y),line,font=font,anchor='lt',fill='#162f44')
        if rotate: tile=tile.transpose(Image.Transpose.ROTATE_90)
        image.paste(tile,(x0,y0))
        label_boxes.append((x0,y0,x1,y1))
        rendered_labels.append(dict(label_id=group.label_id,island_id=group.island_id,side=group.side,
            machine_name=group.machine_name,machine_numbers=list(group.machine_numbers),
            lines=lines,font_size=size,box=[x0,y0,x1,y1],rotated=rotate))
    for group in groups:
        if group.side=='RING': continue
        points=[positions[n] for n in group.machine_numbers]
        if group.side in {'N','S'}:
            x0,x1=min(p[0] for p in points)-50,max(p[0] for p in points)+50
            cy=points[0][1]
            y0=cy-115 if group.side=='N' else cy+34
            label(group,(x0,y0,x1,y0+80))
        else:
            y0,y1=min(p[1] for p in points)-34,max(p[1] for p in points)+34
            cx=points[0][0]
            # Single west wall faces the floor, so its label goes on the right.
            right=group.side=='E' or group.island_id in {'W01','W03'}
            x0=cx+51 if right else cx-125
            label(group,(x0,y0,x0+74,y1),rotate=True)
    ring_groups=[g for g in groups if g.side=='RING']
    if len(ring_groups)==1:
        label(ring_groups[0],(1280,1630,1592,1730))
    else:
        # Local callouts beside the ring, ordered by the actual group centers.
        ordered=sorted(ring_groups,key=lambda g:sum(positions[n][1] for n in g.machine_numbers)/len(g.machine_numbers))
        for i,g in enumerate(ordered):
            left=i%2==0; row=i//2
            x0=650 if left else 1770; y0=1450+row*58
            points=[positions[n] for n in g.machine_numbers]
            point=min(points,key=lambda p:p[0]) if left else max(points,key=lambda p:p[0])
            draw.line((point,(x0+470 if left else x0,y0+26)),fill='#60758a',width=2)
            label(g,(x0,y0,x0+470,y0+52))
    ink(40,3560,'5スロ581–620は対象外。空白842–843 / 1015–1014は1台分を保持。',25)
    overlaps=sum(intersects(a,b) for i,a in enumerate(text_boxes) for b in text_boxes[i+1:])
    seat_overlaps=sum(intersects(a,b) for i,(_,a) in enumerate(seat_rects) for _,b in seat_rects[i+1:])
    label_overlaps=sum(intersects(a,b) for i,a in enumerate(label_boxes) for b in label_boxes[i+1:])
    label_obstructions=sum(intersects(a,b) for a in label_boxes for b in text_boxes)
    label_obstructions+=sum(intersects(a,b) for a in label_boxes for _,b in seat_rects)
    if overlaps or seat_overlaps or label_overlaps or label_obstructions:
        raise FloorMapValidationError(f'Render overlap: text={overlaps}, seats={seat_overlaps}, labels={label_overlaps}, obstruction={label_obstructions}')
    info=PngImagePlugin.PngInfo()
    for key,value in {'data_date':target.isoformat(),'machine_count':'514','source_sha256':source_sha,
                      'layout_sha256':layout_sha,'generator_version':VERSION}.items():info.add_text(key,value)
    stream=io.BytesIO();image.save(stream,format='PNG',pnginfo=info,optimize=False,compress_level=9)
    return stream.getvalue(),dict(width=image.width,height=image.height,label_count=len(groups),
        labels_by_island=dict(sorted(Counter(g.island_id for g in groups).items())),
        text_overlap_count=0,text_clipping_count=0,seat_overlap_count=0,label_overlap_count=0,
        internal_island_labels_visible=False,legend_visible=False,number_font_size=34,
        groups=rendered_labels)


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
