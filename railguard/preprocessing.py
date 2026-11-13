"""Reproducible, composable M1 image and illumination preprocessing.

All public transforms accept and return three-channel ``uint8`` BGR images.
They are intentionally small and registered independently so experiments can
record an ordered, YAML/JSON-friendly list of steps instead of hiding image
changes inside M2 or M3.

Gamma convention
----------------
``output = 255 * (input / 255) ** (1 / gamma)`` (implemented as a LUT).
Consequently ``gamma > 1`` brightens mid-tones, ``0 < gamma < 1`` darkens
them, and the output is clipped to 0..255 before conversion to ``uint8``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from time import perf_counter
from typing import Any

import cv2
import numpy as np


Transform = Callable[..., np.ndarray]
PREPROCESSORS: dict[str, Transform] = {}


def register_preprocessor(name: str, *aliases: str) -> Callable[[Transform], Transform]:
    """Register a preprocessing function under a canonical name and aliases."""
    keys = (name, *aliases)

    def decorator(function: Transform) -> Transform:
        for key in keys:
            normalized = key.strip().lower()
            if not normalized or normalized in PREPROCESSORS:
                raise ValueError(f"Invalid or duplicate preprocessor name: {key!r}")
            PREPROCESSORS[normalized] = function
        return function

    return decorator


def _validate_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected a three-channel BGR image")
    if image.dtype != np.uint8:
        raise ValueError("expected uint8 image data")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must not be empty")
    return image


def _odd_kernel(value: int, name: str = "kernel_size") -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an odd integer")
    value = int(value)
    if value <= 0 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer")
    return value


def _color_space(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"lab", "ycrcb"}:
        raise ValueError("color_space must be 'lab' or 'ycrcb'")
    return normalized


@register_preprocessor("identity", "raw")
def identity(image: np.ndarray) -> np.ndarray:
    """Return an independent uint8 BGR copy without changing pixels."""
    return _validate_image(image).copy()


@register_preprocessor("gaussian_blur", "gaussian")
def gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 0.0) -> np.ndarray:
    """Blur all BGR channels together with a positive odd square kernel."""
    image = _validate_image(image)
    kernel_size = _odd_kernel(kernel_size)
    if float(sigma) < 0:
        raise ValueError("sigma must be non-negative")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), float(sigma))


@register_preprocessor("median_blur", "median")
def median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Median-filter BGR pixels with a positive odd square kernel."""
    image = _validate_image(image)
    kernel_size = _odd_kernel(kernel_size)
    return cv2.medianBlur(image, kernel_size)


def _replace_luminance(image: np.ndarray, operation: Callable[[np.ndarray], np.ndarray],
                       color_space: str) -> np.ndarray:
    image = _validate_image(image)
    color_space = _color_space(color_space)
    if color_space == "lab":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        converted[:, :, 0] = operation(converted[:, :, 0])
        return cv2.cvtColor(converted, cv2.COLOR_LAB2BGR)
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    converted[:, :, 0] = operation(converted[:, :, 0])
    return cv2.cvtColor(converted, cv2.COLOR_YCrCb2BGR)


@register_preprocessor("hist_eq", "histogram_equalization")
def hist_eq(image: np.ndarray, color_space: str = "ycrcb") -> np.ndarray:
    """Globally equalize luminance only; never equalize B, G, R separately."""
    return _replace_luminance(image, cv2.equalizeHist, color_space)


@register_preprocessor("clahe")
def clahe(image: np.ndarray, clip_limit: float = 2.0,
          tile_grid_size: Sequence[int] = (8, 8), color_space: str = "lab") -> np.ndarray:
    """Apply CLAHE only to LAB-L or YCrCb-Y luminance."""
    if float(clip_limit) <= 0:
        raise ValueError("clip_limit must be positive")
    if (not isinstance(tile_grid_size, Sequence) or len(tile_grid_size) != 2
            or any(isinstance(v, bool) or int(v) <= 0 for v in tile_grid_size)):
        raise ValueError("tile_grid_size must contain two positive integers")
    grid = tuple(int(v) for v in tile_grid_size)
    enhancer = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=grid)
    return _replace_luminance(image, enhancer.apply, color_space)


@register_preprocessor("gamma")
def gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """LUT gamma correction where values above one brighten mid-tones.

    The exact convention is ``255 * (I / 255) ** (1 / gamma)``.
    """
    image = _validate_image(image)
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be a finite positive number")
    values = np.arange(256, dtype=np.float64) / 255.0
    table = np.clip(np.rint(255.0 * np.power(values, 1.0 / gamma)), 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


DEFAULT_M1_CONFIG: dict[str, Any] = {
    "version": 1,
    "seed": 0,
    "gamma_convention": "output=255*(input/255)**(1/gamma); gamma>1 brightens",
    "branches": {
        "m2_boundary": [
            {"method": "clahe", "params": {"clip_limit": 2.0, "tile_grid_size": [8, 8],
                                              "color_space": "lab"}},
            {"method": "gaussian_blur", "params": {"kernel_size": 5, "sigma": 0.0}},
        ],
        "m3_person": [{"method": "identity", "params": {}}],
    },
}


def default_m1_config() -> dict[str, Any]:
    """Return an independent, YAML/JSON-serializable M1 configuration."""
    return deepcopy(DEFAULT_M1_CONFIG)


def _normalize_steps(steps: str | Sequence[str | Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(steps, str):
        raw_steps: Sequence[str | Mapping[str, Any]] = [s for s in steps.replace(",", "+").split("+") if s]
    elif isinstance(steps, Sequence):
        raw_steps = steps
    else:
        raise TypeError("pipeline must be a method string or ordered sequence")
    if not raw_steps:
        raise ValueError("pipeline must contain at least one step")

    normalized: list[dict[str, Any]] = []
    for item in raw_steps:
        if isinstance(item, str):
            method, params = item, {}
        elif isinstance(item, Mapping):
            method = item.get("method", item.get("name"))
            params = item.get("params", {})
        else:
            raise TypeError("each pipeline step must be a string or mapping")
        method = str(method).strip().lower() if method is not None else ""
        if method not in PREPROCESSORS:
            raise ValueError(f"Unknown preprocessing method {method!r}; available: {sorted(PREPROCESSORS)}")
        if not isinstance(params, Mapping):
            raise ValueError(f"params for {method!r} must be a mapping")
        normalized.append({"method": method, "params": dict(params)})
    return normalized


def resolve_pipeline(config: Mapping[str, Any], branch: str) -> list[dict[str, Any]]:
    """Resolve an ordered pipeline from a reproducible M1 branch config."""
    branches = config.get("branches")
    if not isinstance(branches, Mapping) or branch not in branches:
        raise ValueError(f"Missing M1 config branch: {branch}")
    return _normalize_steps(branches[branch])


def apply_pipeline(image: np.ndarray, steps: str | Sequence[str | Mapping[str, Any]],
                   *, return_metadata: bool = False) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Apply preprocessing steps in order and optionally return timing/config metadata."""
    output = _validate_image(image).copy()
    normalized = _normalize_steps(steps)
    timings: list[dict[str, Any]] = []
    started = perf_counter()
    for step in normalized:
        step_started = perf_counter()
        output = PREPROCESSORS[step["method"]](output, **step["params"])
        _validate_image(output)
        timings.append({**deepcopy(step), "runtime_ms": (perf_counter() - step_started) * 1000.0})
    metadata = {
        "pipeline": normalized,
        "steps": timings,
        "runtime_ms": (perf_counter() - started) * 1000.0,
        "shape": list(output.shape),
        "dtype": str(output.dtype),
    }
    return (output, metadata) if return_metadata else output


def compute_quality_flags(image: np.ndarray) -> dict[str, Any]:
    """Compute descriptive luminance flags without claiming highlight recovery.

    Dark/bright thresholds are Y <= 15 and Y >= 240; clipped highlights are
    Y == 255. Dynamic range is p95-p5, while contrast is luminance standard
    deviation. Values describe the supplied image only.
    """
    started = perf_counter()
    image = _validate_image(image)
    luminance = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    p5, p95 = np.percentile(luminance, [5, 95])
    return {
        "dark_pixel_ratio": float(np.mean(luminance <= 15)),
        "bright_pixel_ratio": float(np.mean(luminance >= 240)),
        "clipped_highlights_ratio": float(np.mean(luminance == 255)),
        "dynamic_range": float(p95 - p5),
        "contrast_std": float(np.std(luminance)),
        "luminance_p5": float(p5),
        "luminance_p95": float(p95),
        "runtime_ms": (perf_counter() - started) * 1000.0,
        "metadata": {"shape": list(image.shape), "dtype": str(image.dtype),
                     "color_order": "BGR", "highlight_recovery_claimed": False},
    }


def _fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)),
                                 max(1, round(image.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _histogram_panel(original: np.ndarray, processed: np.ndarray, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    plot_height = max(1, height - 24)
    for image, color in ((original, (100, 100, 100)), (processed, (0, 90, 220))):
        y = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        hist = cv2.calcHist([y], [0], None, [256], [0, 256]).reshape(-1)
        hist = hist / max(float(hist.max()), 1.0) * (plot_height - 1)
        points = np.column_stack((np.linspace(0, width - 1, 256).astype(np.int32),
                                  (plot_height - 1 - hist).astype(np.int32)))
        cv2.polylines(panel, [points], False, color, 1, cv2.LINE_AA)
    return panel


def make_contact_sheet(original: np.ndarray, processed: np.ndarray, runtime_ms: float | None = None,
                       panel_size: tuple[int, int] = (320, 240)) -> np.ndarray:
    """Create a 2x2 preview: original, processed, luminance histogram, edges/mask."""
    original = _validate_image(original)
    processed = _validate_image(processed)
    if original.shape != processed.shape:
        raise ValueError("original and processed images must have the same shape")
    width, height = (int(panel_size[0]), int(panel_size[1]))
    if width < 32 or height < 32:
        raise ValueError("panel_size must be at least 32x32")
    edges = cv2.Canny(cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY), 80, 160)
    edge_panel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    panels = [
        ("original", _fit_panel(original, width, height)),
        ("processed", _fit_panel(processed, width, height)),
        ("luminance histogram: original(gray) / processed(red)",
         _histogram_panel(original, processed, width, height)),
        ("edges / mask", _fit_panel(edge_panel, width, height)),
    ]
    title_height = 30
    rendered = []
    for title, panel in panels:
        canvas = np.full((height + title_height, width, 3), 255, dtype=np.uint8)
        canvas[title_height:] = panel
        cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (25, 25, 25), 1, cv2.LINE_AA)
        rendered.append(canvas)
    if runtime_ms is not None:
        cv2.putText(rendered[3], f"runtime: {float(runtime_ms):.3f} ms", (8, height + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
    return np.vstack((np.hstack(rendered[:2]), np.hstack(rendered[2:])))


__all__ = [
    "PREPROCESSORS", "DEFAULT_M1_CONFIG", "apply_pipeline", "clahe",
    "compute_quality_flags", "default_m1_config", "gamma", "gaussian_blur",
    "hist_eq", "identity", "make_contact_sheet", "median_blur",
    "register_preprocessor", "resolve_pipeline",
]
