"""Pure calibration helpers shared by the RailGuard Studio editor and tests.

The editor works in normalized image coordinates, while detection and geometric
rules work in pixel coordinates.  A partially placed zone is never valid.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import yaml

from .config import get_zone
from .geometry import HazardZone, UNSAFE_SIDE_CONVENTION

LABELS = ("L0", "L1", "E0", "E1")


def empty_points() -> list[None]:
    return [None, None, None, None]


def validate_points(points: object, allow_partial: bool = True) -> list[list[float] | None]:
    if not isinstance(points, (tuple, list)) or len(points) != 4:
        raise ValueError("Exactly four endpoints [L0, L1, E0, E1] are required")
    clean: list[list[float] | None] = []
    for label, point in zip(LABELS, points):
        if point is None and allow_partial:
            clean.append(None)
            continue
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise ValueError(f"{label} must be an [x, y] pair")
        # Exclude booleans as accidental values and non-finite floating point numbers.
        if any(isinstance(v, bool) for v in point):
            raise ValueError(f"{label} must contain numeric coordinates")
        try:
            x, y = [float(v) for v in point]
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{label} must contain numeric coordinates") from exc
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y)):
            raise ValueError(f"{label} must be within normalized image coordinates [0, 1]")
        clean.append([round(x, 6), round(y, 6)])
    return clean


def config_from_points(points: object, width: int, height: int) -> dict:
    clean = validate_points(points, allow_partial=False)
    cfg = {"coordinates": "normalized", "zone": {
        "safety_line": clean[:2], "platform_edge": clean[2:],
        "unsafe_side": UNSAFE_SIDE_CONVENTION,
    }}
    # Validate geometry before allowing approval / saving / inference.
    get_zone(cfg, width, height)
    return cfg


def points_from_zone(zone: HazardZone, width: int, height: int) -> list[list[float]]:
    if width < 2 or height < 2:
        raise ValueError("Invalid image size")
    # Tự động cắt giao và nội suy lại điểm vào đúng biên ảnh [0, width-1] x [0, height-1]
    # nếu điểm bị tính toán vượt ra ngoài biên ảnh
    def _clip_segment_to_box(p1, p2, w, h):
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        dx = x2 - x1
        dy = y2 - y1
        pts_in = []
        # Kiểm tra nếu điểm gốc đã nằm trong hộp
        if 0 <= x1 <= w - 1 and 0 <= y1 <= h - 1:
            pts_in.append((x1, y1))
        if 0 <= x2 <= w - 1 and 0 <= y2 <= h - 1:
            pts_in.append((x2, y2))
        # Nếu thiếu điểm, tìm giao với 4 cạnh
        if len(pts_in) < 2 and (abs(dx) > 1e-7 or abs(dy) > 1e-7):
            for edge_x in (0.0, float(w - 1)):
                if abs(dx) > 1e-7:
                    t = (edge_x - x1) / dx
                    y_at = y1 + t * dy
                    if 0 <= y_at <= h - 1:
                        cand = (edge_x, y_at)
                        if not any(abs(cand[0]-p[0]) < 1e-3 and abs(cand[1]-p[1]) < 1e-3 for p in pts_in):
                            pts_in.append(cand)
            for edge_y in (0.0, float(h - 1)):
                if abs(dy) > 1e-7:
                    t = (edge_y - y1) / dy
                    x_at = x1 + t * dx
                    if 0 <= x_at <= w - 1:
                        cand = (x_at, edge_y)
                        if not any(abs(cand[0]-p[0]) < 1e-3 and abs(cand[1]-p[1]) < 1e-3 for p in pts_in):
                            pts_in.append(cand)
        if len(pts_in) >= 2:
            return pts_in[0], pts_in[1]
        # Fallback an toàn: clamp trực tiếp
        c1 = (max(0.0, min(float(w - 1), x1)), max(0.0, min(float(h - 1), y1)))
        c2 = (max(0.0, min(float(w - 1), x2)), max(0.0, min(float(h - 1), y2)))
        return c1, c2

    s_a, s_b = _clip_segment_to_box(zone.safety_line.a, zone.safety_line.b, width, height)
    e_a, e_b = _clip_segment_to_box(zone.platform_edge.a, zone.platform_edge.b, width, height)
    pts = (s_a, s_b, e_a, e_b)

    return validate_points([
        [max(0.0, min(1.0, float(x) / (width - 1))), max(0.0, min(1.0, float(y) / (height - 1)))]
        for x, y in pts
    ], allow_partial=False)


def points_from_config(cfg: dict, width: int, height: int) -> list[list[float]]:
    zone = get_zone(cfg, width, height)
    if zone is None:
        raise ValueError("Configuration contains no safety zone")
    return points_from_zone(zone, width, height)


def safe_camera_id(raw: str) -> str:
    """Allow only a single safe YAML basename; never allow path traversal."""
    name = raw.strip()
    if (not name or len(name) > 80 or
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or
            name in {".", ".."} or ".." in name):
        raise ValueError("Camera ID must be 1–80 letters, numbers, hyphens or underscores")
    return name


def calibration_yaml(points: object, width: int, height: int) -> str:
    return yaml.safe_dump(config_from_points(points, width, height), sort_keys=False)


def write_calibration(points: object, width: int, height: int, camera_id: str,
                      directory: str | Path = "configs", overwrite: bool = False) -> Path:
    text = calibration_yaml(points, width, height)
    path = Path(directory) / f"{safe_camera_id(camera_id)}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    # An existing calibration cannot be overwritten by an accidental click.
    with path.open("w" if overwrite else "x", encoding="utf-8") as file:
        file.write(text)
    return path
