"""Pure calibration helpers shared by the Streamlit image editor and tests.

The editor works in normalized image coordinates, while detection and geometric
rules work in pixel coordinates.  A partially placed zone is never valid.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import yaml

from .config import get_zone
from .geometry import HazardZone

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
    }}
    # Validate geometry before allowing approval / saving / inference.
    get_zone(cfg, width, height)
    return cfg


def points_from_zone(zone: HazardZone, width: int, height: int) -> list[list[float]]:
    if width < 2 or height < 2:
        raise ValueError("Invalid image size")
    pts = (zone.safety_line.a, zone.safety_line.b,
           zone.platform_edge.a, zone.platform_edge.b)
    # Numerical clipping is only for fitted lines whose ends are a fraction
    # outside the image; a totally out-of-frame zone must not be accepted.
    if any(x < -1 or y < -1 or x > width or y > height for x, y in pts):
        raise ValueError("Estimated line endpoints are outside the image")
    return validate_points([
        [max(0, min(1, x / (width - 1))), max(0, min(1, y / (height - 1)))]
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
