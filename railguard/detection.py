"""Unified person detections and lazy Ultralytics adapters.

The module does not download or load a model at import time. Tests can inject a
small object exposing ``predict`` through ``model=``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import cv2
import numpy as np

from .preprocessing import gamma as apply_gamma, highlight_compress

Point = tuple[float, float]
Keypoint = tuple[float, float, float]
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _finite_bbox(bbox) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("bbox must contain x1, y1, x2, y2")
    value = tuple(float(item) for item in bbox)
    if not all(np.isfinite(value)) or value[2] < value[0] or value[3] < value[1]:
        raise ValueError("bbox must be finite and ordered x1 <= x2, y1 <= y2")
    return value


@dataclass
class Person:
    """Detector-independent person instance."""

    bbox: tuple[float, float, float, float]
    confidence: float = 1.0
    object_id: Optional[str] = None
    ankles: Optional[list[tuple[float, float, float]]] = None
    runtime_ms: float = 0.0
    detector: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    keypoints: Optional[list[Keypoint]] = None

    def __post_init__(self) -> None:
        self.bbox = _finite_bbox(self.bbox)
        self.confidence = float(self.confidence)
        self.runtime_ms = float(self.runtime_ms)
        if not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must lie in [0, 1]")
        if not np.isfinite(self.runtime_ms) or self.runtime_ms < 0:
            raise ValueError("runtime_ms must be finite and nonnegative")

    @property
    def bbox_bottom(self) -> Point:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    def foot_points(self, method: str = "bbox", min_conf: float = 0.35,
                    fallback: str = "bbox") -> tuple[list[Point], str]:
        """Return points and explicitly distinguish pose fallback/abstention."""
        aliases = {"bbox_bottom": "bbox", "pose_ankle": "pose"}
        method = aliases.get(method, method)
        if method not in {"bbox", "pose"}:
            raise ValueError("method must be bbox/bbox_bottom or pose/pose_ankle")
        if fallback not in {"bbox", "abstain"}:
            raise ValueError("fallback must be bbox or abstain")
        if not 0 <= min_conf <= 1:
            raise ValueError("min_conf must lie in [0, 1]")
        if method == "bbox":
            return [self.bbox_bottom], "bbox_bottom_center"

        x1, y1, x2, y2 = self.bbox
        reliable: list[Point] = []
        for ankle in self.ankles or []:
            if len(ankle) != 3:
                continue
            x, y, score = map(float, ankle)
            if (np.isfinite((x, y, score)).all() and score >= min_conf
                    and x > 0 and y > 0 and x1 <= x <= x2 and y1 <= y <= y2):
                reliable.append((x, y))
        if reliable:
            return reliable[:2], "two_ankles" if len(reliable) > 1 else "one_ankle"
        if fallback == "bbox":
            return [self.bbox_bottom], "bbox_fallback"
        return [], "abstain"


def _as_list(value):
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _scalar(value) -> float:
    value = _as_list(value)
    while isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("model returned an empty scalar")
        value = value[0]
    return float(value)


def _validate_imgsz(imgsz: int | tuple[int, int]) -> int | tuple[int, int]:
    if isinstance(imgsz, bool):
        raise ValueError("imgsz must be a positive integer or (height, width)")
    if isinstance(imgsz, int):
        if imgsz <= 0:
            raise ValueError("imgsz must be positive")
        return imgsz
    if (not isinstance(imgsz, (tuple, list)) or len(imgsz) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in imgsz)):
        raise ValueError("imgsz must be a positive integer or (height, width)")
    return tuple(imgsz)


def _resolve_weight_path(weights: str) -> str:
    """Resolve explicit or bundled weights independent of current working directory."""
    candidate = Path(weights)
    if candidate.is_file():
        return str(candidate)
    if not candidate.is_absolute():
        repo_candidate = _REPO_ROOT / candidate
        if repo_candidate.is_file():
            return str(repo_candidate)
    return weights


def _extract_box_xyxy(box) -> tuple[float, float, float, float]:
    bbox_raw = _as_list(box.xyxy)
    while (isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 1
           and isinstance(bbox_raw[0], (list, tuple))):
        bbox_raw = bbox_raw[0]
    return _finite_bbox(bbox_raw)


def _bbox_iou(box_a: tuple[float, float, float, float],
              box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return float(inter / union) if union > 0.0 else 0.0


def _match_det_and_pose(det_boxes: list[tuple[float, float, float, float]],
                        pose_boxes: list[tuple[float, float, float, float]],
                        min_iou: float = 0.35) -> tuple[dict[int, int], list[int], list[int]]:
    """Best-IoU 1-to-1 assignment between detection proposals and full-frame pose boxes."""
    candidates: list[tuple[float, int, int]] = []
    for d_idx, d_box in enumerate(det_boxes):
        for p_idx, p_box in enumerate(pose_boxes):
            iou = _bbox_iou(d_box, p_box)
            if iou >= min_iou:
                candidates.append((iou, d_idx, p_idx))
    candidates.sort(key=lambda item: item[0], reverse=True)

    det_to_pose: dict[int, int] = {}
    used_pose: set[int] = set()
    for _, d_idx, p_idx in candidates:
        if d_idx in det_to_pose or p_idx in used_pose:
            continue
        det_to_pose[d_idx] = p_idx
        used_pose.add(p_idx)

    unmatched_det = [i for i in range(len(det_boxes)) if i not in det_to_pose]
    unmatched_pose = [j for j in range(len(pose_boxes)) if j not in used_pose]
    return det_to_pose, unmatched_det, unmatched_pose


def _extract_keypoints(keypoints_obj, index: int,
                       offset_x: float = 0.0, offset_y: float = 0.0) -> list[Keypoint]:
    """Extract all available keypoints with image-space coordinates."""
    if keypoints_obj is None:
        return []
    xy_all = _as_list(getattr(keypoints_obj, "xy", None))
    if xy_all is None or index >= len(xy_all):
        return []
    points = xy_all[index]
    conf_all = _as_list(getattr(keypoints_obj, "conf", None))
    scores = (conf_all[index] if conf_all is not None and index < len(conf_all)
              else [1.0] * len(points))
    result: list[Keypoint] = []
    for k_idx, point in enumerate(points):
        score = float(scores[k_idx]) if k_idx < len(scores) else 0.0
        px, py = float(point[0]), float(point[1])
        if px > 0 or py > 0:
            px += offset_x
            py += offset_y
        result.append((px, py, score))
    return result


def _ankles_from_keypoints(kpts: list[Keypoint]) -> list[Keypoint]:
    return [kpts[k_idx] for k_idx in (15, 16) if k_idx < len(kpts)]


def _bbox_containment(box_a: tuple[float, float, float, float],
                      box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = min(area_a, area_b)
    return float(inter / denom) if denom > 0.0 else 0.0


def _luminance_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]


_GLARE_EXPOSURES = {"raw", "gamma08", "gamma06", "highlight_compress"}
_CASCADE_MODES = {"auto", "bbox_first", "dual_full"}


def _is_unverified_glare_sliver(
    bbox: tuple[float, float, float, float], conf: float, img_w: int,
    *, min_conf: float = 0.25, has_valid_ankles: bool = False,
) -> bool:
    """Reject narrow interior slivers while keeping valid border/cropped pedestrians."""
    bx1, by1, bx2, by2 = bbox
    bw = bx2 - bx1
    bh = by2 - by1
    if bh < 24.0 or bw < 8.0:
        return True
    if has_valid_ankles:
        return False
    touches_border = bx1 <= 2.0 or bx2 >= (float(img_w) - 2.0)
    is_sliver = (bw < 12.0) or ((bw / max(bh, 1.0)) < 0.16)
    if is_sliver and not touches_border and conf < 0.35:
        return True
    return conf < min_conf


def _check_box_glare(
    l_ch: np.ndarray, bbox: tuple[float, float, float, float], w: int, h: int,
) -> tuple[bool, float]:
    """Verify interior or surrounding halo glare (supports dark silhouettes backlit by bright halos)."""
    ix1, iy1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    ix2, iy2 = min(w, int(np.ceil(bbox[2]))), min(h, int(np.ceil(bbox[3])))
    if ix2 <= ix1 or iy2 <= iy1:
        return False, 0.0
    bright_in = float(np.mean(l_ch[iy1:iy2, ix1:ix2] >= 200))
    bw_i = max(1, ix2 - ix1)
    bh_i = max(1, iy2 - iy1)
    hx1 = max(0, ix1 - max(8, int(0.30 * bw_i)))
    hy1 = max(0, iy1 - max(8, int(0.15 * bh_i)))
    hx2 = min(w, ix2 + max(8, int(0.30 * bw_i)))
    hy2 = min(h, iy2 + max(8, int(0.15 * bh_i)))
    bright_halo = float(np.mean(l_ch[hy1:hy2, hx1:hx2] >= 200))
    if bright_in < 0.12 and (bright_halo < 0.18 or bright_in < 0.02):
        return False, 0.0
    return True, max(bright_in, bright_halo)


def _apply_glare_exposure(image: np.ndarray, mode: str) -> np.ndarray:
    """Apply localized glare-recovery exposure transform."""
    if mode == "raw":
        return image
    if mode == "gamma08":
        return apply_gamma(image, 0.8)
    if mode == "gamma06":
        return apply_gamma(image, 0.6)
    return highlight_compress(image)


def _extract_glare_rois(image_shape: tuple[int, ...], l_ch: np.ndarray,
                        max_rois: int = 2) -> list[tuple[int, int, int, int]]:
    """Locate up to max_rois compact glare-stress windows (<=30% area each) via integral maps."""
    h, w = image_shape[:2]
    if h < 64 or w < 64:
        return []
    if max(h, w) <= 320:
        if float(np.mean(l_ch >= 200)) >= 0.05:
            return [(0, 0, w, h)]
        return []

    scale = min(1.0, 256.0 / float(min(h, w)))
    sw = max(32, int(round(w * scale)))
    sh = max(32, int(round(h * scale)))
    small_l = cv2.resize(l_ch, (sw, sh), interpolation=cv2.INTER_AREA) if scale < 1.0 else l_ch

    bright = (small_l >= 200).astype(np.uint8)
    non_blown = (small_l < 235).astype(np.float32)
    halo_k = max(5, int(round(0.055 * min(sw, sh))) | 1)
    bright_halo = cv2.dilate(bright, np.ones((halo_k, halo_k), np.uint8))

    gx = cv2.Sobel(small_l, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small_l, cv2.CV_32F, 0, 1, ksize=3)
    edge_in_glare = ((np.hypot(gx, gy) >= 28.0) & (bright_halo > 0)).astype(np.float32)

    ii_bright = cv2.integral(bright.astype(np.float32))
    ii_non_blown = cv2.integral(non_blown)
    ii_edge = cv2.integral(edge_in_glare)

    def _box_sum(ii: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        return float(ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1])

    scored: list[tuple[float, tuple[int, int, int, int]]] = []
    for frac in (0.36, 0.45):
        ww = min(sw, max(24, int(round(sw * frac))))
        wh = min(sh, max(24, int(round(sh * frac))))
        if (ww * wh) > 0.30 * (sw * sh):
            continue
        step_x = max(8, ww // 4)
        step_y = max(8, wh // 4)
        xs = list(range(0, max(1, sw - ww + 1), step_x))
        if xs[-1] != sw - ww:
            xs.append(sw - ww)
        ys = list(range(0, max(1, sh - wh + 1), step_y))
        if ys[-1] != sh - wh:
            ys.append(sh - wh)
        area = float(ww * wh)
        for y1 in ys:
            y2 = y1 + wh
            for x1 in xs:
                x2 = x1 + ww
                b_frac = _box_sum(ii_bright, x1, y1, x2, y2) / area
                if b_frac < 0.12 or b_frac > 0.94:
                    continue
                nb_frac = _box_sum(ii_non_blown, x1, y1, x2, y2) / area
                if nb_frac < 0.10:
                    continue
                e_frac = _box_sum(ii_edge, x1, y1, x2, y2) / area
                if e_frac < 0.03:
                    continue
                balance = float(np.sqrt(b_frac * min(1.0 - b_frac + 0.15, nb_frac)))
                rx1 = int(round(x1 / scale))
                ry1 = int(round(y1 / scale))
                rx2 = min(w, int(round(x2 / scale)))
                ry2 = min(h, int(round(y2 / scale)))
                scored.append((e_frac * balance, (rx1, ry1, rx2, ry2)))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[int, int, int, int]] = []
    for _, roi in scored:
        roi_f = tuple(float(v) for v in roi)
        if any(_bbox_iou(roi_f, tuple(float(v) for v in prev)) > 0.25 for prev in selected):
            continue
        selected.append(roi)
        if len(selected) >= max_rois:
            break
    return selected


def _ankles_within_bbox(ankles: list[Keypoint], bbox: tuple[float, float, float, float],
                        min_conf: float = 0.25) -> bool:
    bx1, by1, bx2, by2 = bbox
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)
    for ax, ay, score in ankles:
        if score >= min_conf and ax > 0 and ay > 0:
            if (bx1 - 0.10 * bw) <= ax <= (bx2 + 0.10 * bw) and by1 <= ay <= (by2 + 0.08 * bh):
                return True
    return False


class YOLODetector:
    """B3 detection / A pose adapter for COCO person class 0.

    Supports direct pose, bbox, and detection-first crop-pose refinement with
    explicit coverage tracking so partial pose visibility is observable downstream.
    """

    def __init__(self, method: str = "bbox", weights: str | None = None,
                 confidence: float = 0.25, device: str | int | None = None, *,
                 imgsz: int | tuple[int, int] = 640, iou: float = 0.7,
                 conf: float | None = None, model=None,
                 enable_crop_pose: bool = True, det_model=None,
                 glare_recovery: bool = False,
                 glare_exposure: str = "highlight_compress",
                 cascade_mode: str = "auto"):
        if method not in {"bbox", "pose"}:
            raise ValueError("method must be bbox or pose")
        if conf is not None:
            confidence = conf
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("conf must be numeric")
        if not 0 <= float(confidence) <= 1:
            raise ValueError("conf must lie in [0, 1]")
        if isinstance(iou, bool) or not isinstance(iou, (int, float)) or not 0 < float(iou) <= 1:
            raise ValueError("iou must lie in (0, 1]")
        if weights is not None and (not isinstance(weights, str) or not weights.strip()):
            raise ValueError("weights must be a nonempty path or model name")
        if device is not None and not isinstance(device, (str, int)):
            raise ValueError("device must be None, an integer, or a nonempty string")
        if isinstance(device, str) and not device.strip():
            raise ValueError("device must not be empty")

        self.method = method
        self.weights = weights or ("yolo11n-pose.pt" if method == "pose" else "yolo11n.pt")
        self.confidence = float(confidence)
        self.imgsz = _validate_imgsz(imgsz)
        self.iou = float(iou)
        self.device = device
        self.enable_crop_pose = enable_crop_pose
        self.glare_recovery = bool(glare_recovery)
        exp_norm = str(glare_exposure).strip().lower()
        if exp_norm not in _GLARE_EXPOSURES:
            raise ValueError(f"glare_exposure must be one of {sorted(_GLARE_EXPOSURES)}")
        self.glare_exposure = exp_norm
        cas_norm = str(cascade_mode).strip().lower()
        if cas_norm not in _CASCADE_MODES:
            raise ValueError(f"cascade_mode must be one of {sorted(_CASCADE_MODES)}")
        self.cascade_mode = cas_norm
        self.last_glare_rois: list[tuple[int, int, int, int]] = []
        self._det_model = det_model
        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Ultralytics missing; install the optional 'vision' dependencies") from exc
            model = YOLO(_resolve_weight_path(self.weights))
            if self.method == "pose" and self._det_model is None:
                det_path = _resolve_weight_path("yolo11n.pt")
                if Path(det_path).is_file():
                    self._det_model = YOLO(det_path)
        self.model = model
        self.last_runtime_ms = 0.0
        self.last_proposals = 0
        self.last_pose_covered = 0
        self.last_coverage: dict[str, Any] = {
            "proposals": 0,
            "direct_pose_covered": 0,
            "crop_pose_recovered": 0,
            "glare_recovered": 0,
            "pose_covered": 0,
            "uncovered": 0,
            "final_people": 0,
            "detector_partial": False,
            "coverage_ratio": 1.0,
        }
        self.last_config: dict[str, Any] = {}

    def _build_pose_metadata(self, bbox: tuple[float, float, float, float],
                             ankles: list[Keypoint], image_w: int, image_h: int,
                             pose_source: str) -> dict[str, Any]:
        x1, y1, x2, y2 = bbox
        metadata = dict(self.last_config)
        metadata["legs_cropped"] = bool(y2 >= image_h - 1)
        metadata["bbox_clipped"] = bool(x1 <= 0 or y1 <= 0 or x2 >= image_w - 1 or y2 >= image_h - 1)
        metadata["missing_ankles"] = 2 - sum(
            int(x > 0 and y > 0 and s >= self.confidence) for x, y, s in ankles)
        metadata["pose_source"] = pose_source
        return metadata

    def _extract_crop_for_box(self, image: np.ndarray,
                              det_bbox: tuple[float, float, float, float]) -> tuple[np.ndarray, int, int] | None:
        image_h, image_w = image.shape[:2]
        bx1, by1, bx2, by2 = det_bbox
        bh = by2 - by1
        bw = bx2 - bx1
        pad_y = max(4, int(bh * 0.15))
        pad_x = max(4, int(bw * 0.15))
        cy1, cy2 = max(0, int(by1 - pad_y)), min(image_h, int(by2 + pad_y))
        cx1, cx2 = max(0, int(bx1 - pad_x)), min(image_w, int(bx2 + pad_x))
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
            return None
        return crop, cx1, cy1

    def _select_crop_pose_result(self, sub_out: Any, det_bbox: tuple[float, float, float, float],
                                 cx1: int, cy1: int) -> list[Keypoint]:
        sub_kpts_obj = getattr(sub_out, "keypoints", None)
        if sub_kpts_obj is None:
            return []
        sub_xy = _as_list(getattr(sub_kpts_obj, "xy", None))
        if not sub_xy:
            return []

        best_idx: int | None = None
        sub_boxes = getattr(sub_out, "boxes", None)
        if sub_boxes is not None and len(sub_boxes) > 0:
            best_iou = 0.0
            for s_idx, s_box in enumerate(sub_boxes):
                if hasattr(s_box, "cls") and int(_scalar(s_box.cls)) != 0:
                    continue
                if s_idx >= len(sub_xy):
                    continue
                sx1, sy1, sx2, sy2 = _extract_box_xyxy(s_box)
                shifted = (sx1 + cx1, sy1 + cy1, sx2 + cx1, sy2 + cy1)
                iou = _bbox_iou(det_bbox, shifted)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = s_idx
            if best_idx is None:
                return []
        else:
            best_idx = 0
        return _extract_keypoints(sub_kpts_obj, best_idx, offset_x=float(cx1), offset_y=float(cy1))

    def _run_crop_pose(self, image: np.ndarray,
                       det_bbox: tuple[float, float, float, float]) -> list[Keypoint]:
        """Run zoomed pose inference on an unmatched proposal crop and select best-IoU skeleton."""
        extracted = self._extract_crop_for_box(image, det_bbox)
        if extracted is None:
            return []
        crop, cx1, cy1 = extracted
        crop_opts: dict[str, Any] = {
            "source": crop, "verbose": False,
            "conf": max(0.15, self.confidence * 0.7), "imgsz": 256,
        }
        if self.device is not None:
            crop_opts["device"] = self.device
        sub_res = self.model.predict(**crop_opts)
        if not sub_res or len(sub_res) == 0:
            return []
        return self._select_crop_pose_result(sub_res[0], det_bbox, cx1, cy1)

    def _run_crop_pose_batch(
        self,
        items: list[tuple[np.ndarray, tuple[float, float, float, float], float, float]],
    ) -> list[list[Keypoint]]:
        """Run batched zoomed pose inference across proposal crops (falls back per-crop for mocks)."""
        results: list[list[Keypoint]] = [[] for _ in items]
        if not items:
            return results
        valid_slots: list[tuple[int, np.ndarray, tuple[float, float, float, float], int, int, float, float]] = []
        for idx, (src_img, bbox_global, ox, oy) in enumerate(items):
            local_box = (bbox_global[0] - ox, bbox_global[1] - oy, bbox_global[2] - ox, bbox_global[3] - oy)
            extracted = self._extract_crop_for_box(src_img, local_box)
            if extracted is None:
                continue
            crop, cx1, cy1 = extracted
            valid_slots.append((idx, crop, local_box, cx1, cy1, ox, oy))
        if not valid_slots:
            return results

        conf_pose = max(0.15, self.confidence * 0.7)
        crops_only = [slot[1] for slot in valid_slots]
        batch_outs: list[Any] | None = None
        if len(crops_only) > 1:
            opts: dict[str, Any] = {
                "source": crops_only, "verbose": False, "conf": conf_pose, "imgsz": 256,
            }
            if self.device is not None:
                opts["device"] = self.device
            try:
                res = self.model.predict(**opts)
                if res is not None and len(res) == len(crops_only):
                    batch_outs = list(res)
            except Exception:
                batch_outs = None

        if batch_outs is None:
            batch_outs = []
            for crop in crops_only:
                opts = {"source": crop, "verbose": False, "conf": conf_pose, "imgsz": 256}
                if self.device is not None:
                    opts["device"] = self.device
                res = self.model.predict(**opts)
                batch_outs.append(res[0] if (res is not None and len(res) > 0) else None)

        for (orig_idx, _, local_box, cx1, cy1, ox, oy), sub_out in zip(valid_slots, batch_outs):
            if sub_out is None:
                continue
            kpts_local = self._select_crop_pose_result(sub_out, local_box, cx1, cy1)
            if ox != 0.0 or oy != 0.0:
                kpts_local = [
                    (kx + ox, ky + oy, ks) if (kx > 0 or ky > 0) else (0.0, 0.0, ks)
                    for kx, ky, ks in kpts_local
                ]
            results[orig_idx] = kpts_local
        return results

    def _set_coverage(self, *, proposals: int, direct_pose_covered: int,
                      crop_pose_recovered: int, final_people: int,
                      detector_partial: bool, glare_recovered: int = 0,
                      glare_pose_covered: int = 0) -> None:
        pose_covered = direct_pose_covered + crop_pose_recovered + glare_pose_covered
        uncovered = max(0, proposals - final_people) if not self.enable_crop_pose else max(0, proposals - pose_covered)
        ratio = min(1.0, float(pose_covered / proposals)) if (proposals > 0 and self.method == "pose") else 1.0
        self.last_proposals = int(proposals)
        self.last_pose_covered = int(pose_covered)
        self.last_coverage = {
            "proposals": int(proposals),
            "direct_pose_covered": int(direct_pose_covered),
            "crop_pose_recovered": int(crop_pose_recovered),
            "glare_recovered": int(glare_recovered),
            "pose_covered": int(pose_covered),
            "uncovered": int(uncovered),
            "final_people": int(final_people),
            "detector_partial": bool(detector_partial),
            "coverage_ratio": ratio,
        }
        self.last_config["coverage"] = dict(self.last_coverage)

    def _predict_entries(
        self,
        model_obj,
        source: np.ndarray,
        *,
        conf: float,
        off_x: float = 0.0,
        off_y: float = 0.0,
    ) -> tuple[list[tuple[int, tuple[float, float, float, float], float]], Any]:
        opts: dict[str, Any] = {
            "source": source, "verbose": False, "conf": float(conf),
            "iou": self.iou, "imgsz": self.imgsz,
        }
        if self.device is not None:
            opts["device"] = self.device
        res = model_obj.predict(**opts)
        out = res[0] if (res is not None and len(res) > 0) else None
        raw_boxes = getattr(out, "boxes", None) if out is not None else None
        kpts_obj = getattr(out, "keypoints", None) if out is not None else None
        entries: list[tuple[int, tuple[float, float, float, float], float]] = []
        if raw_boxes is not None:
            for idx, box in enumerate(raw_boxes):
                if int(_scalar(box.cls)) != 0:
                    continue
                b_conf = max(0.0, min(1.0, float(_scalar(box.conf))))
                x1, y1, x2, y2 = _extract_box_xyxy(box)
                entries.append((idx, (x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y), b_conf))
        return entries, kpts_obj

    def _merge_glare_detections(self, image: np.ndarray, people: list[Person]) -> int:
        """Recover high-glare pedestrians on localized glare-stress ROIs."""
        self.last_glare_rois = []
        if not self.glare_recovery:
            return 0
        h, w = image.shape[:2]
        l_ch = _luminance_channel(image)
        rois = _extract_glare_rois(image.shape, l_ch, max_rois=2)
        if not rois:
            return 0
        self.last_glare_rois = list(rois)

        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        recovered = 0
        pose_scan_conf = max(0.18, self.confidence * 0.85)
        det_scan_conf = max(0.20, self.confidence * 0.85)

        for rx1, ry1, rx2, ry2 in rois:
            ox, oy = int(rx1), int(ry1)
            roi_bgr = bgr[ry1:ry2, rx1:rx2]
            if roi_bgr.size == 0 or roi_bgr.shape[0] < 16 or roi_bgr.shape[1] < 16:
                continue
            src = _apply_glare_exposure(roi_bgr, self.glare_exposure)
            p_entries, p_kpts_obj = self._predict_entries(
                self.model, src, conf=pose_scan_conf, off_x=float(ox), off_y=float(oy)
            )
            candidates: list[tuple[tuple[float, float, float, float], float, list[Keypoint], str]] = []
            if self.method == "pose" and self._det_model is not None:
                d_entries, _ = self._predict_entries(
                    self._det_model, src, conf=det_scan_conf, off_x=float(ox), off_y=float(oy)
                )
                d_boxes = [item[1] for item in d_entries]
                p_boxes = [item[1] for item in p_entries]
                det_to_pose, _, unmatched_pose = _match_det_and_pose(d_boxes, p_boxes)
                for d_idx, (_, d_box, d_conf) in enumerate(d_entries):
                    if d_idx in det_to_pose:
                        p_orig_idx, _, _ = p_entries[det_to_pose[d_idx]]
                        kpts = _extract_keypoints(p_kpts_obj, p_orig_idx, offset_x=float(ox), offset_y=float(oy))
                        src_tag = f"glare_{self.glare_exposure}_pose"
                    elif self.enable_crop_pose:
                        local_box = (d_box[0] - ox, d_box[1] - oy, d_box[2] - ox, d_box[3] - oy)
                        local_kpts = self._run_crop_pose(src, local_box)
                        kpts = [(kx + ox, ky + oy, ks) if (kx > 0 or ky > 0) else (0.0, 0.0, ks)
                                for kx, ky, ks in local_kpts]
                        src_tag = f"glare_{self.glare_exposure}_crop_pose" if kpts else f"glare_{self.glare_exposure}_det_only"
                    else:
                        kpts = []
                        src_tag = f"glare_{self.glare_exposure}_det_only"
                    candidates.append((d_box, d_conf, kpts, src_tag))
                for p_entry_idx in unmatched_pose:
                    p_orig_idx, p_box, p_conf = p_entries[p_entry_idx]
                    kpts = _extract_keypoints(p_kpts_obj, p_orig_idx, offset_x=float(ox), offset_y=float(oy))
                    candidates.append((p_box, p_conf, kpts, f"glare_{self.glare_exposure}_pose"))
            else:
                for p_orig_idx, p_box, p_conf in p_entries:
                    kpts = (
                        _extract_keypoints(p_kpts_obj, p_orig_idx, offset_x=float(ox), offset_y=float(oy))
                        if self.method == "pose" else []
                    )
                    src_tag = f"glare_{self.glare_exposure}_pose" if self.method == "pose" else f"glare_{self.glare_exposure}_det"
                    candidates.append((p_box, p_conf, kpts, src_tag))

            for cand_box, cand_conf, kpts, src_tag in candidates:
                if any(
                    _bbox_iou(cand_box, p.bbox) >= 0.35 or _bbox_containment(cand_box, p.bbox) >= 0.55
                    for p in people
                ):
                    continue
                has_glare, bright_frac = _check_box_glare(l_ch, cand_box, w, h)
                if not has_glare:
                    continue

                ankles: list[Keypoint] | None = None
                if self.method == "pose":
                    ankles = _ankles_from_keypoints(kpts)
                    has_ankles = _ankles_within_bbox(ankles, cand_box, min_conf=0.25)
                    if _is_unverified_glare_sliver(
                        cand_box, cand_conf, w,
                        min_conf=max(0.20, self.confidence * 0.85),
                        has_valid_ankles=has_ankles,
                    ):
                        continue
                    metadata = self._build_pose_metadata(cand_box, ankles, w, h, src_tag)
                else:
                    if _is_unverified_glare_sliver(
                        cand_box, cand_conf, w,
                        min_conf=max(0.20, self.confidence * 0.85),
                    ):
                        continue
                    metadata = dict(self.last_config)
                    metadata["pose_source"] = src_tag
                metadata["glare_recovered"] = True
                metadata["glare_bright_frac"] = round(bright_frac, 4)

                people.append(Person(
                    cand_box, cand_conf, ankles=ankles, runtime_ms=0.0,
                    detector="yolo_glare_recovered", metadata=metadata,
                    keypoints=(kpts or None) if self.method == "pose" else None,
                ))
                recovered += 1
        return recovered

    def _predict_bbox_first(self, image: np.ndarray) -> list[Person]:
        """Fast bbox-first -> glare ROI -> selective batched crop-pose cascade."""
        started = perf_counter()
        image_h, image_w = image.shape[:2]
        prop_model = self._det_model if (self.method == "pose" and self._det_model is not None) else self.model
        self.last_config = {
            "weights": self.weights, "conf": self.confidence, "iou": self.iou,
            "imgsz": self.imgsz, "device": self.device,
            "runtime_ms": 0.0, "cascade_mode": "bbox_first",
        }
        self.last_glare_rois = []

        # Stage 1: Single full-frame bbox proposal pass.
        base_entries, _ = self._predict_entries(prop_model, image, conf=self.confidence)
        # (bbox, conf, glare_recovered, bright_frac, src_img, ox, oy)
        proposals: list[tuple[tuple[float, float, float, float], float, bool, float, np.ndarray, float, float]] = [
            (bbox, b_conf, False, 0.0, image, 0.0, 0.0)
            for _, bbox, b_conf in base_entries
        ]
        primary_count = len(proposals)

        # Stage 2: Localized glare-stress ROI scan with exposure adaptation.
        glare_added = 0
        if self.glare_recovery:
            l_ch = _luminance_channel(image)
            rois = _extract_glare_rois(image.shape, l_ch, max_rois=2)
            self.last_glare_rois = list(rois)
            if rois:
                bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                glare_conf = max(0.20, self.confidence * 0.85)
                roi_imgsz = min(self.imgsz, 640) if isinstance(self.imgsz, int) else self.imgsz
                for rx1, ry1, rx2, ry2 in rois:
                    roi_bgr = bgr[ry1:ry2, rx1:rx2]
                    if roi_bgr.size == 0 or roi_bgr.shape[0] < 16 or roi_bgr.shape[1] < 16:
                        continue
                    src = _apply_glare_exposure(roi_bgr, self.glare_exposure)
                    roi_opts: dict[str, Any] = {
                        "source": src, "verbose": False, "conf": glare_conf,
                        "iou": self.iou, "imgsz": roi_imgsz,
                    }
                    if self.device is not None:
                        roi_opts["device"] = self.device
                    r_res = prop_model.predict(**roi_opts)
                    r_out = r_res[0] if (r_res is not None and len(r_res) > 0) else None
                    r_boxes = getattr(r_out, "boxes", None) if r_out is not None else None
                    if r_boxes is None:
                        continue
                    ox, oy = float(rx1), float(ry1)
                    for box in r_boxes:
                        if int(_scalar(box.cls)) != 0:
                            continue
                        cand_conf = max(0.0, min(1.0, float(_scalar(box.conf))))
                        bx1, by1, bx2, by2 = _extract_box_xyxy(box)
                        cand_box = (bx1 + ox, by1 + oy, bx2 + ox, by2 + oy)
                        if _is_unverified_glare_sliver(cand_box, cand_conf, image_w, min_conf=glare_conf):
                            continue
                        if any(
                            _bbox_iou(cand_box, p[0]) >= 0.35 or _bbox_containment(cand_box, p[0]) >= 0.55
                            for p in proposals
                        ):
                            continue
                        has_glare, bright_frac = _check_box_glare(l_ch, cand_box, image_w, image_h)
                        if not has_glare:
                            continue
                        proposals.append((cand_box, cand_conf, True, bright_frac, src, ox, oy))
                        glare_added += 1

        # Stage 3: Selective batched crop-pose on accepted proposals.
        kpts_per_prop: list[list[Keypoint]] = [[] for _ in proposals]
        if self.method == "pose" and self.enable_crop_pose and proposals:
            crop_items = [(p[4], p[0], p[5], p[6]) for p in proposals]
            kpts_per_prop = self._run_crop_pose_batch(crop_items)

        people: list[Person] = []
        crop_pose_cov = 0
        glare_pose_cov = 0
        for (bbox, conf_val, is_glare, bright_frac, _, _, _), kpts in zip(proposals, kpts_per_prop):
            if self.method == "pose":
                ankles = _ankles_from_keypoints(kpts)
                if kpts:
                    if is_glare:
                        pose_src = f"glare_{self.glare_exposure}_crop_pose"
                        glare_pose_cov += 1
                    else:
                        pose_src = "crop_pose"
                        crop_pose_cov += 1
                else:
                    pose_src = f"glare_{self.glare_exposure}_det_only" if is_glare else "det_only_no_pose"
                metadata = self._build_pose_metadata(bbox, ankles, image_w, image_h, pose_src)
                det_name = "yolo_glare_recovered" if is_glare else "yolo_pose_crop_enhanced"
            else:
                ankles = None
                metadata = dict(self.last_config)
                if is_glare:
                    metadata["pose_source"] = f"glare_{self.glare_exposure}_det"
                det_name = "yolo_glare_recovered" if is_glare else "yolo_detection"
            if is_glare:
                metadata["glare_recovered"] = True
                metadata["glare_bright_frac"] = round(bright_frac, 4)
            people.append(Person(
                bbox, conf_val, ankles=ankles, runtime_ms=0.0,
                detector=det_name, metadata=metadata,
                keypoints=(kpts or None) if self.method == "pose" else None,
            ))

        self.last_runtime_ms = (perf_counter() - started) * 1000
        self.last_config["runtime_ms"] = self.last_runtime_ms
        for person in people:
            person.runtime_ms = self.last_runtime_ms
            person.metadata["runtime_ms"] = self.last_runtime_ms
        pose_cov = crop_pose_cov + glare_pose_cov
        self._set_coverage(
            proposals=primary_count + glare_added,
            direct_pose_covered=0,
            crop_pose_recovered=crop_pose_cov,
            final_people=len(people),
            detector_partial=(self.method == "pose" and pose_cov < len(proposals)),
            glare_recovered=glare_added,
            glare_pose_covered=glare_pose_cov,
        )
        return people

    def predict(self, image: np.ndarray) -> list[Person]:
        if not isinstance(image, np.ndarray) or image.ndim not in {2, 3} or image.size == 0:
            raise ValueError("image must be a nonempty grayscale or color numpy array")
        use_bbox_first = (
            self.cascade_mode == "bbox_first"
            or (
                self.cascade_mode == "auto"
                and self.glare_recovery
                and self._det_model is not None
                and self.enable_crop_pose
            )
        )
        if use_bbox_first:
            return self._predict_bbox_first(image)
        opts: dict[str, Any] = {
            "source": image, "verbose": False, "conf": self.confidence,
            "iou": self.iou, "imgsz": self.imgsz,
        }
        if self.device is not None:
            opts["device"] = self.device
        started = perf_counter()
        outputs = self.model.predict(**opts)
        self.last_config = {
            "weights": self.weights, "conf": self.confidence, "iou": self.iou,
            "imgsz": self.imgsz, "device": self.device,
            "runtime_ms": 0.0,
        }
        image_h, image_w = image.shape[:2]
        output = outputs[0] if (outputs is not None and len(outputs) > 0) else None
        raw_boxes = getattr(output, "boxes", None) if output is not None else None
        keypoints = getattr(output, "keypoints", None) if output is not None else None

        pose_entries: list[tuple[int, tuple[float, float, float, float], float]] = []
        if raw_boxes is not None:
            for index, box in enumerate(raw_boxes):
                if int(_scalar(box.cls)) != 0:
                    continue
                conf_val = max(0.0, min(1.0, _scalar(box.conf)))
                pose_entries.append((index, _extract_box_xyxy(box), conf_val))

        det_proposals: list[tuple[tuple[float, float, float, float], float]] = []
        if self.method == "pose" and self._det_model is not None:
            det_opts: dict[str, Any] = {
                "source": image, "verbose": False, "conf": max(0.15, self.confidence * 0.75),
                "iou": self.iou, "imgsz": self.imgsz,
            }
            if self.device is not None:
                det_opts["device"] = self.device
            det_res = self._det_model.predict(**det_opts)
            if det_res and len(det_res) > 0 and getattr(det_res[0], "boxes", None) is not None:
                for box in det_res[0].boxes:
                    if int(_scalar(box.cls)) != 0:
                        continue
                    b_conf = max(0.0, min(1.0, float(_scalar(box.conf))))
                    if b_conf < self.confidence:
                        continue
                    det_proposals.append((_extract_box_xyxy(box), b_conf))

            det_boxes_only = [item[0] for item in det_proposals]
            pose_boxes_only = [item[1] for item in pose_entries]
            det_to_pose, unmatched_det, unmatched_pose = _match_det_and_pose(det_boxes_only, pose_boxes_only)

            people: list[Person] = []
            crop_recovered = 0
            if self.enable_crop_pose:
                for d_idx, (det_bbox, b_conf) in enumerate(det_proposals):
                    if d_idx in det_to_pose:
                        p_idx, _, _ = pose_entries[det_to_pose[d_idx]]
                        kpts = _extract_keypoints(keypoints, p_idx)
                        pose_source = "direct_pose"
                    else:
                        kpts = self._run_crop_pose(image, det_bbox)
                        if kpts:
                            pose_source = "crop_pose"
                            crop_recovered += 1
                        else:
                            pose_source = "det_only_no_pose"
                    ankles = _ankles_from_keypoints(kpts)
                    metadata = self._build_pose_metadata(det_bbox, ankles, image_w, image_h, pose_source)
                    people.append(Person(
                        det_bbox, b_conf, ankles=ankles,
                        runtime_ms=0.0, detector="yolo_pose_crop_enhanced",
                        metadata=metadata, keypoints=kpts or None,
                    ))
                # Preserve high-confidence direct pose detections not proposed by _det_model.
                for p_entry_idx in unmatched_pose:
                    orig_idx, p_bbox, p_conf = pose_entries[p_entry_idx]
                    kpts = _extract_keypoints(keypoints, orig_idx)
                    ankles = _ankles_from_keypoints(kpts)
                    metadata = self._build_pose_metadata(p_bbox, ankles, image_w, image_h, "direct_pose")
                    people.append(Person(
                        p_bbox, p_conf, ankles=ankles,
                        runtime_ms=0.0, detector="yolo_pose_crop_enhanced",
                        metadata=metadata, keypoints=kpts or None,
                    ))
                glare_added = self._merge_glare_detections(image, people)
                glare_pose_cov = (
                    sum(1 for p in people[-glare_added:] if p.keypoints)
                    if glare_added > 0 else 0
                )
                self.last_runtime_ms = (perf_counter() - started) * 1000
                self.last_config["runtime_ms"] = self.last_runtime_ms
                for person in people:
                    person.runtime_ms = self.last_runtime_ms
                    person.metadata["runtime_ms"] = self.last_runtime_ms
                total_proposals = len(det_proposals) + glare_added
                pose_covered = len(det_to_pose) + crop_recovered + glare_pose_cov
                self._set_coverage(
                    proposals=total_proposals,
                    direct_pose_covered=len(det_to_pose),
                    crop_pose_recovered=crop_recovered,
                    final_people=len(people),
                    detector_partial=pose_covered < total_proposals,
                    glare_recovered=glare_added,
                    glare_pose_covered=glare_pose_cov,
                )
                return people

            # enable_crop_pose=False (nocrop): return only direct pose people, flag partial coverage!
            for orig_idx, p_bbox, p_conf in pose_entries:
                kpts = _extract_keypoints(keypoints, orig_idx)
                ankles = _ankles_from_keypoints(kpts)
                metadata = self._build_pose_metadata(p_bbox, ankles, image_w, image_h, "direct_pose")
                people.append(Person(
                    p_bbox, p_conf, ankles=ankles,
                    runtime_ms=0.0, detector="yolo_pose",
                    metadata=metadata, keypoints=kpts or None,
                ))
            self.last_runtime_ms = (perf_counter() - started) * 1000
            self.last_config["runtime_ms"] = self.last_runtime_ms
            for person in people:
                person.runtime_ms = self.last_runtime_ms
                person.metadata["runtime_ms"] = self.last_runtime_ms
            self._set_coverage(
                proposals=len(det_proposals),
                direct_pose_covered=len(det_to_pose),
                crop_pose_recovered=0,
                final_people=len(people),
                detector_partial=len(unmatched_det) > 0,
            )
            return people

        # Standard single-model bbox or pose path (or mocked detector without det_model).
        people = []
        for orig_idx, bbox, conf_val in pose_entries:
            kpts: list[Keypoint] = []
            ankles = None
            metadata = dict(self.last_config)
            if self.method == "pose":
                kpts = _extract_keypoints(keypoints, orig_idx)
                ankles = _ankles_from_keypoints(kpts)
                metadata = self._build_pose_metadata(bbox, ankles, image_w, image_h, "direct_pose")
            people.append(Person(
                bbox, conf_val, ankles=ankles,
                runtime_ms=0.0,
                detector="yolo_pose" if self.method == "pose" else "yolo_detection",
                metadata=metadata,
                keypoints=kpts or None if self.method == "pose" else None,
            ))
        base_count = len(people)
        glare_added = self._merge_glare_detections(image, people)
        glare_pose_cov = (
            sum(1 for p in people[base_count:] if p.keypoints)
            if glare_added > 0 and self.method == "pose" else 0
        )
        self.last_runtime_ms = (perf_counter() - started) * 1000
        self.last_config["runtime_ms"] = self.last_runtime_ms
        for person in people:
            person.runtime_ms = self.last_runtime_ms
            person.metadata["runtime_ms"] = self.last_runtime_ms
        self._set_coverage(
            proposals=len(people),
            direct_pose_covered=base_count if self.method == "pose" else 0,
            crop_pose_recovered=0,
            final_people=len(people),
            detector_partial=False,
            glare_recovered=glare_added,
            glare_pose_covered=glare_pose_cov,
        )
        return people
