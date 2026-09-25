"""Display-only Maruhan morning floor maps for NORMAL/A-TYPE/JUGGLER Top15.

This module reuses the validated 514-seat physical layout and floor-map renderer.
It must never participate in prediction scoring, Formal Forward, inventory approval,
or Morning Automation success/sleep decisions.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from maruhan_maebashi_floor_map_generator import (
    DEFAULT_LAYOUT,
    FloorMapValidationError,
    choose_font,
    display_positions,
    join_daily,
    parse_daily,
    parse_layout,
    render_png,
    sha256_bytes,
)

MAIL_WIDTH = 1600
OUTPUT_DIR_NAME = "outputs/morning_floor_map"

CATEGORY_CONFIG = {
    "NORMAL": {
        "label": "NORMAL Top15",
        "slug": "normal",
        "cid_prefix": "maruhan-normal-top15-floor",
    },
    "A-TYPE": {
        "label": "A-TYPE Top15",
        "slug": "atype",
        "cid_prefix": "maruhan-atype-top15-floor",
    },
    "JUGGLER": {
        "label": "JUGGLER Top15",
        "slug": "juggler",
        "cid_prefix": "maruhan-juggler-top15-floor",
    },
}


@dataclass(frozen=True)
class FloorMapArtifact:
    category: str
    label: str
    cid: str
    filename: str
    email_bytes: bytes
    full_path: Path
    email_path: Path
    full_sha256: str
    email_sha256: str
    full_width: int
    full_height: int
    email_width: int
    email_height: int
    full_size_bytes: int
    email_size_bytes: int
    rank_machine_pairs: tuple[tuple[int, int], ...]
    marker_overlap_count: int
    marker_clipping_count: int


@dataclass(frozen=True)
class FloorMapCategoryResult:
    category: str
    label: str
    status: str
    artifact: FloorMapArtifact | None = None
    error: str = ""


def _safe_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


def _normalize_machine_no(value: object) -> int:
    text = str(value).strip()
    if not text:
        raise FloorMapValidationError("ranking machine_no is blank")
    try:
        number = int(float(text))
    except (TypeError, ValueError) as exc:
        raise FloorMapValidationError(f"invalid ranking machine_no: {value!r}") from exc
    if str(number) != text and text not in {f"{number}.0"}:
        raise FloorMapValidationError(f"non-integral ranking machine_no: {value!r}")
    return number


def _ranking_pairs(
    rows: list[dict[str, str]],
    rank_keys: tuple[str, ...],
) -> tuple[tuple[int, int], ...]:
    if len(rows) != 15:
        raise FloorMapValidationError(f"Top15 rows={len(rows)}, expected 15")

    pairs: list[tuple[int, int]] = []
    for index, row in enumerate(rows, 1):
        raw_rank = next((row.get(key) for key in rank_keys if row.get(key)), str(index))
        try:
            rank = int(str(raw_rank))
        except (TypeError, ValueError) as exc:
            raise FloorMapValidationError(f"invalid ranking rank: {raw_rank!r}") from exc
        machine_no = _normalize_machine_no(row.get("machine_no") or row.get("台番号") or "")
        pairs.append((rank, machine_no))

    ranks = [rank for rank, _ in pairs]
    machines = [machine for _, machine in pairs]
    if sorted(ranks) != list(range(1, 16)) or len(set(ranks)) != 15:
        raise FloorMapValidationError(f"Top15 ranks must be exactly 1..15: {ranks}")
    if len(set(machines)) != 15:
        raise FloorMapValidationError("Top15 machine_no contains duplicates")
    pairs.sort(key=lambda item: item[0])
    return tuple(pairs)


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _place_badge(
    cx: float,
    cy: float,
    radius: int,
    image_size: tuple[int, int],
    existing: list[tuple[int, int, int, int]],
) -> tuple[int, int, tuple[int, int, int, int]]:
    width, height = image_size
    offsets = (
        (42, -42),
        (-42, -42),
        (42, 42),
        (-42, 42),
        (0, -52),
        (0, 52),
    )
    for dx, dy in offsets:
        bx = int(round(cx + dx))
        by = int(round(cy + dy))
        box = (bx - radius, by - radius, bx + radius, by + radius)
        if box[0] < 2 or box[1] < 112 or box[2] > width - 2 or box[3] > height - 2:
            continue
        if any(_intersects(box, other) for other in existing):
            continue
        return bx, by, box
    raise FloorMapValidationError(f"marker placement overlap/clipping near ({cx:.1f}, {cy:.1f})")


def _draw_category_overlay(
    base_png: bytes,
    seats,
    pairs: tuple[tuple[int, int], ...],
    category: str,
    target_date: date,
    font_path: Path,
) -> tuple[bytes, dict]:
    image = Image.open(io.BytesIO(base_png)).convert("RGB")
    draw = ImageDraw.Draw(image)
    positions = display_positions(seats)
    layout_numbers = set(positions)
    missing = [machine for _, machine in pairs if machine not in layout_numbers]
    if missing:
        raise FloorMapValidationError(f"Top15 machine_no missing from layout: {missing}")

    badge_boxes: list[tuple[int, int, int, int]] = []
    marker_rows = []
    rank_font = ImageFont.truetype(str(font_path), 40)
    rank_font_top3 = ImageFont.truetype(str(font_path), 46)
    header_font = ImageFont.truetype(str(font_path), 27)

    # Self-identifying banner in the unused upper-right portion of the existing header.
    banner = (2730, 12, 4000, 100)
    draw.rounded_rectangle(banner, radius=14, fill="#ffffff", outline="#173c61", width=3)
    draw.text(
        (banner[0] + 18, banner[1] + 11),
        f"{category} Top15  Target {target_date.isoformat()}",
        font=header_font,
        fill="#173c61",
    )

    for rank, machine_no in pairs:
        cx, cy = positions[machine_no]
        top3 = rank <= 3
        radius = 31 if top3 else 26
        badge_x, badge_y, badge_box = _place_badge(
            cx, cy, radius, image.size, badge_boxes
        )
        badge_boxes.append(badge_box)

        outline = "#c62828" if top3 else "#0d47a1"
        line_width = 9 if top3 else 6
        seat_radius_x = 53 if top3 else 50
        seat_radius_y = 36 if top3 else 33
        draw.ellipse(
            (
                int(round(cx - seat_radius_x)),
                int(round(cy - seat_radius_y)),
                int(round(cx + seat_radius_x)),
                int(round(cy + seat_radius_y)),
            ),
            outline=outline,
            width=line_width,
        )
        draw.ellipse(badge_box, fill=outline, outline="#ffffff", width=3)
        font = rank_font_top3 if top3 else rank_font
        text = str(rank)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            (badge_x - (bbox[0] + bbox[2]) / 2, badge_y - (bbox[1] + bbox[3]) / 2),
            text,
            font=font,
            fill="#ffffff",
        )
        marker_rows.append(
            {
                "rank": rank,
                "machine_no": machine_no,
                "seat_center": [round(cx, 2), round(cy, 2)],
                "badge_box": list(badge_box),
                "top3": top3,
            }
        )

    overlap_count = sum(
        _intersects(a, b)
        for index, a in enumerate(badge_boxes)
        for b in badge_boxes[index + 1 :]
    )
    clipping_count = sum(
        box[0] < 0 or box[1] < 0 or box[2] > image.width or box[3] > image.height
        for box in badge_boxes
    )
    if overlap_count or clipping_count:
        raise FloorMapValidationError(
            f"marker QA failed: overlap={overlap_count}, clipping={clipping_count}"
        )

    info = PngImagePlugin.PngInfo()
    info.add_text("display_only", "true")
    info.add_text("category", category)
    info.add_text("target_date", target_date.isoformat())
    info.add_text("marker_count", "15")
    stream = io.BytesIO()
    image.save(stream, format="PNG", pnginfo=info, optimize=False, compress_level=9)
    return stream.getvalue(), {
        "width": image.width,
        "height": image.height,
        "marker_count": 15,
        "marker_overlap_count": overlap_count,
        "marker_clipping_count": clipping_count,
        "markers": marker_rows,
    }


def _mail_variant(full_png: bytes, width: int = MAIL_WIDTH) -> tuple[bytes, int, int]:
    image = Image.open(io.BytesIO(full_png)).convert("RGB")
    if width <= 0:
        raise FloorMapValidationError("mail width must be positive")
    height = max(1, round(image.height * width / image.width))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    resized.save(stream, format="PNG", optimize=True, compress_level=9)
    return stream.getvalue(), width, height


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _prepare_base(project_root: Path, operation_date: date, layout_path: Path | None = None):
    project_root = Path(project_root)
    layout_path = Path(layout_path) if layout_path is not None else (
        project_root / "machine_number/maruhan_maebashi_floor_map_layout.csv"
    )
    expected_daily_date = operation_date - timedelta(days=1)
    daily_path = (
        project_root
        / "data/maruhan_maebashi/machine_number"
        / f"ana_slo_{expected_daily_date:%Y%m%d}.csv"
    )

    layout_bytes = layout_path.read_bytes()
    provenance_path = layout_path.with_suffix(".json")
    provenance_bytes = provenance_path.read_bytes()
    provenance = json.loads(provenance_bytes)
    if provenance.get("schema_version") != 1:
        raise FloorMapValidationError("Layout provenance schema mismatch")
    if provenance.get("layout_sha256") != sha256_bytes(layout_bytes):
        raise FloorMapValidationError("Layout provenance SHA-256 mismatch")

    seats = parse_layout(layout_bytes)
    if set(provenance.get("islands", [])) != {seat.island_id for seat in seats}:
        raise FloorMapValidationError("Layout provenance island set mismatch")

    daily_bytes = daily_path.read_bytes()
    data_date, names = parse_daily(daily_bytes, daily_path.name)
    if data_date != expected_daily_date:
        raise FloorMapValidationError("Floor-map daily date is not operation_date - 1")
    names = join_daily(seats, names)
    font_path = choose_font()
    base_png, base_rendering = render_png(
        seats,
        names,
        provenance,
        data_date,
        sha256_bytes(daily_bytes),
        sha256_bytes(layout_bytes),
        font_path,
    )
    return {
        "seats": seats,
        "font_path": font_path,
        "base_png": base_png,
        "base_rendering": base_rendering,
        "data_date": data_date,
        "daily_path": daily_path,
        "layout_path": layout_path,
        "source_sha256": sha256_bytes(daily_bytes),
        "layout_sha256": sha256_bytes(layout_bytes),
    }


def generate_floor_map_bundle(
    project_root: Path,
    operation_date: date,
    rankings: dict[str, tuple[list[dict[str, str]], tuple[str, ...]]],
    *,
    output_dir: Path | None = None,
    mail_width: int = MAIL_WIDTH,
) -> dict[str, FloorMapCategoryResult]:
    """Generate independent category maps. Every failure is returned, never raised outward."""
    project_root = Path(project_root)
    output_dir = Path(output_dir) if output_dir is not None else project_root / OUTPUT_DIR_NAME
    results: dict[str, FloorMapCategoryResult] = {}

    requested = [category for category in CATEGORY_CONFIG if category in rankings]
    if not requested:
        return results

    try:
        base = _prepare_base(project_root, operation_date)
    except Exception as exc:
        error = _safe_error(exc)
        for category in requested:
            results[category] = FloorMapCategoryResult(
                category=category,
                label=CATEGORY_CONFIG[category]["label"],
                status="UNAVAILABLE",
                error=error,
            )
        return results

    manifest = {
        "schema_version": 1,
        "display_only": True,
        "operation_date": operation_date.isoformat(),
        "data_date": base["data_date"].isoformat(),
        "source_csv": str(base["daily_path"]),
        "source_sha256": base["source_sha256"],
        "layout_master": str(base["layout_path"]),
        "layout_sha256": base["layout_sha256"],
        "mail_width": mail_width,
        "categories": {},
    }

    for category in requested:
        config = CATEGORY_CONFIG[category]
        rows, rank_keys = rankings[category]
        try:
            pairs = _ranking_pairs(rows, rank_keys)
            full_png, marker_qa = _draw_category_overlay(
                base["base_png"],
                base["seats"],
                pairs,
                category,
                operation_date,
                base["font_path"],
            )
            mail_png, email_width, email_height = _mail_variant(full_png, mail_width)
            slug = config["slug"]
            ymd = operation_date.strftime("%Y%m%d")
            full_name = f"maruhan_{slug}_top15_floor_{ymd}_full.png"
            email_name = f"maruhan_{slug}_top15_floor_{ymd}_mail.png"
            full_path = output_dir / full_name
            email_path = output_dir / email_name
            _atomic_write(full_path, full_png)
            _atomic_write(email_path, mail_png)
            cid = f"{config['cid_prefix']}-{ymd}@slotanalyzer"
            artifact = FloorMapArtifact(
                category=category,
                label=config["label"],
                cid=cid,
                filename=email_name,
                email_bytes=mail_png,
                full_path=full_path,
                email_path=email_path,
                full_sha256=hashlib.sha256(full_png).hexdigest(),
                email_sha256=hashlib.sha256(mail_png).hexdigest(),
                full_width=marker_qa["width"],
                full_height=marker_qa["height"],
                email_width=email_width,
                email_height=email_height,
                full_size_bytes=len(full_png),
                email_size_bytes=len(mail_png),
                rank_machine_pairs=pairs,
                marker_overlap_count=marker_qa["marker_overlap_count"],
                marker_clipping_count=marker_qa["marker_clipping_count"],
            )
            results[category] = FloorMapCategoryResult(
                category=category,
                label=config["label"],
                status="OK",
                artifact=artifact,
            )
            manifest["categories"][category] = {
                "status": "OK",
                "label": config["label"],
                "cid": cid,
                "full_path": str(full_path),
                "email_path": str(email_path),
                "full_sha256": artifact.full_sha256,
                "email_sha256": artifact.email_sha256,
                "full_size_bytes": artifact.full_size_bytes,
                "email_size_bytes": artifact.email_size_bytes,
                "full_dimensions": [artifact.full_width, artifact.full_height],
                "email_dimensions": [artifact.email_width, artifact.email_height],
                "rank_machine_pairs": [list(item) for item in pairs],
                "marker_overlap_count": artifact.marker_overlap_count,
                "marker_clipping_count": artifact.marker_clipping_count,
            }
        except Exception as exc:
            error = _safe_error(exc)
            results[category] = FloorMapCategoryResult(
                category=category,
                label=config["label"],
                status="UNAVAILABLE",
                error=error,
            )
            manifest["categories"][category] = {
                "status": "UNAVAILABLE",
                "label": config["label"],
                "error": error,
            }

    try:
        manifest_path = output_dir / f"maruhan_top15_floor_{operation_date:%Y%m%d}_manifest.json"
        encoded = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        _atomic_write(manifest_path, encoded)
    except Exception:
        # Manifest is evidence only; inability to write it must not affect usable images.
        pass

    return results
