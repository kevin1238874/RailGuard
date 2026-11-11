"""Load scene geometry, never assume all sequences of a station share a camera."""
from __future__ import annotations
from pathlib import Path
import yaml
from .geometry import HazardZone, normalized_to_pixels


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def get_zone(config: dict, width: int, height: int) -> HazardZone | None:
    if "zone" not in config:
        return None
    zone = HazardZone.from_dict(config["zone"])
    units = config.get("coordinates", "normalized")
    if units == "normalized":
        points = (zone.safety_line.a, zone.safety_line.b,
                  zone.platform_edge.a, zone.platform_edge.b)
        if any(v < 0 or v > 1 for pair in points for v in pair):
            raise ValueError("Normalized line coordinates must lie in [0,1]")
        zone = normalized_to_pixels(zone, width, height)
    elif units != "pixels":
        raise ValueError("coordinates must be 'normalized' or 'pixels'")
    zone.polygon()  # early validation
    return zone
