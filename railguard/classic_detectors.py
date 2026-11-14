"""Phase-P3 classical person baselines.

B0 is intentionally frame-level. It never creates a :class:`Person` or a
fabricated instance bounding box.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import cv2
import numpy as np

from .detection import Person


def _validate_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim not in {2, 3} or image.size == 0:
        raise ValueError("image must be a nonempty grayscale or color numpy array")
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8")
    return image


@dataclass(frozen=True)
class FrameLevelUnsafePrediction:
    unsafe: bool
    runtime_ms: float
    level: str = "frame"
    detector: str = "always_unsafe"


class AlwaysUnsafeBaseline:
    """B0 naive frame-level alert baseline; it has no instance semantics."""

    output_level = "frame"

    def predict(self, image: np.ndarray) -> FrameLevelUnsafePrediction:
        _validate_image(image)
        started = perf_counter()
        return FrameLevelUnsafePrediction(True, (perf_counter() - started) * 1000)

    predict_frame = predict


def _nms(boxes: list[list[int]], scores: list[float], score_threshold: float,
         nms_threshold: float) -> list[int]:
    if not boxes:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)
    if indices is None:
        return []
    return [int(item) for item in np.asarray(indices).reshape(-1)]


class HOGSVMDetector:
    """B1 OpenCV default-people HOG+SVM, with NMS and unified output."""

    def __init__(self, *, win_stride: tuple[int, int] = (8, 8),
                 padding: tuple[int, int] = (8, 8), scale: float = 1.05,
                 hit_threshold: float = 0.0, nms_threshold: float = 0.4,
                 min_confidence: float = 0.0, hog=None):
        if scale <= 1:
            raise ValueError("scale must be greater than 1")
        if not 0 <= nms_threshold <= 1:
            raise ValueError("nms_threshold must lie in [0, 1]")
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must lie in [0, 1]")
        self.win_stride, self.padding = tuple(win_stride), tuple(padding)
        self.scale, self.hit_threshold = float(scale), float(hit_threshold)
        self.nms_threshold, self.min_confidence = float(nms_threshold), float(min_confidence)
        if hog is None and not hasattr(cv2, "HOGDescriptor"):
            raise RuntimeError(
                "cv2.HOGDescriptor is unavailable in this OpenCV build (it was removed "
                "from OpenCV 5 core); install opencv-python<5 to run the B1 HOG+SVM baseline")
        self.hog = hog or cv2.HOGDescriptor()
        if hog is None:
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.last_runtime_ms = 0.0

    @staticmethod
    def _confidence(weight: float) -> float:
        # The SVM margin is bounded for ranking, not claimed as a probability.
        return float(1.0 / (1.0 + np.exp(-np.clip(weight, -50, 50))))

    def predict(self, image: np.ndarray) -> list[Person]:
        image = _validate_image(image)
        started = perf_counter()
        rects, weights = self.hog.detectMultiScale(
            image, hitThreshold=self.hit_threshold, winStride=self.win_stride,
            padding=self.padding, scale=self.scale)
        boxes: list[list[int]] = []
        scores: list[float] = []
        for (x, y, width, height), weight in zip(rects, np.asarray(weights).reshape(-1)):
            score = self._confidence(float(weight))
            if score >= self.min_confidence:
                boxes.append([int(x), int(y), int(width), int(height)])
                scores.append(score)
        keep = _nms(boxes, scores, self.min_confidence, self.nms_threshold)
        self.last_runtime_ms = (perf_counter() - started) * 1000
        return [Person(
            (x, y, x + width, y + height), scores[index],
            runtime_ms=self.last_runtime_ms, detector="hog_svm",
            metadata={"nms_iou": self.nms_threshold, "svm_margin_is_calibrated": False},
        ) for index in keep for x, y, width, height in [boxes[index]]]


@dataclass(frozen=True)
class ForegroundDiagnostics:
    frame_index: int
    foreground_ratio: float
    shadow_ratio: float
    large_train: bool
    camera_shake: bool
    moving_shadows: bool
    warmup: bool
    runtime_ms: float


class MOG2BlobDetector:
    """B2 foreground blobs for ordered, fixed-camera sequences only."""

    def __init__(self, *, history: int = 200, var_threshold: float = 16,
                 detect_shadows: bool = True, warmup: int = 5,
                 learning_rate: float = -1, morph_kernel: int = 3,
                 min_area: float = 250, max_area_ratio: float = 0.25,
                 min_aspect: float = 0.15, max_aspect: float = 1.4,
                 large_train_ratio: float = 0.35, camera_shake_ratio: float = 0.65,
                 shadow_ratio: float = 0.15, subtractor=None):
        if history <= 0 or warmup < 0:
            raise ValueError("history must be positive and warmup nonnegative")
        if morph_kernel <= 0 or morph_kernel % 2 == 0:
            raise ValueError("morph_kernel must be a positive odd integer")
        if min_area < 0 or not 0 < max_area_ratio <= 1:
            raise ValueError("invalid blob area filters")
        if not 0 < min_aspect <= max_aspect:
            raise ValueError("invalid aspect filters")
        if learning_rate != -1 and not 0 <= learning_rate <= 1:
            raise ValueError("learning_rate must be -1 or lie in [0, 1]")
        if any(not 0 <= value <= 1 for value in
               (large_train_ratio, camera_shake_ratio, shadow_ratio)):
            raise ValueError("diagnostic foreground ratios must lie in [0, 1]")
        self.warmup = int(warmup)
        self.learning_rate = float(learning_rate)
        self.morph_kernel = int(morph_kernel)
        self.min_area, self.max_area_ratio = float(min_area), float(max_area_ratio)
        self.min_aspect, self.max_aspect = float(min_aspect), float(max_aspect)
        self.large_train_ratio = float(large_train_ratio)
        self.camera_shake_ratio = float(camera_shake_ratio)
        self.shadow_ratio = float(shadow_ratio)
        self.subtractor = subtractor or cv2.createBackgroundSubtractorMOG2(
            history=int(history), varThreshold=float(var_threshold),
            detectShadows=bool(detect_shadows))
        self.last_diagnostics: list[ForegroundDiagnostics] = []

    def predict(self, image: np.ndarray):
        raise ValueError("MOG2 B2 rejects single-image mode; use predict_sequence with ordered frames")

    def predict_sequence(self, frames: Sequence[np.ndarray],
                         camera_ids: Sequence[str]) -> list[list[Person]]:
        if isinstance(frames, np.ndarray):
            raise ValueError("MOG2 B2 rejects single-image mode")
        frames = list(frames)
        camera_ids = ([camera_ids] * len(frames) if isinstance(camera_ids, str)
                      else list(camera_ids))
        if len(frames) < 2:
            raise ValueError("MOG2 requires an ordered sequence of at least two frames")
        if len(camera_ids) != len(frames):
            raise ValueError("camera_ids must align one-to-one with frames")
        if len(set(camera_ids)) != 1:
            raise ValueError("MOG2 requires one consistent camera_id")
        if any(camera_id is None or str(camera_id).strip() == "" for camera_id in camera_ids):
            raise ValueError("camera_id must be nonempty")
        first_shape = _validate_image(frames[0]).shape
        if any(_validate_image(frame).shape != first_shape for frame in frames):
            raise ValueError("all sequence frames must have the same shape")

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel))
        results: list[list[Person]] = []
        diagnostics: list[ForegroundDiagnostics] = []
        previous_gray = None
        for frame_index, frame in enumerate(frames):
            started = perf_counter()
            raw = self.subtractor.apply(frame, learningRate=self.learning_rate)
            shadow = raw == 127
            foreground = np.where(raw == 255, 255, 0).astype(np.uint8)
            foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
            foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
            ratio = float(np.count_nonzero(foreground) / foreground.size)
            shadows = float(np.count_nonzero(shadow) / shadow.size)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            global_change = (0.0 if previous_gray is None else
                             float(np.mean(cv2.absdiff(gray, previous_gray)) > 18))
            previous_gray = gray
            large_train = ratio >= self.large_train_ratio
            camera_shake = ratio >= self.camera_shake_ratio or global_change >= 0.75
            moving_shadows = shadows >= self.shadow_ratio
            people: list[Person] = []
            warmup = frame_index < self.warmup
            if not warmup and not large_train and not camera_shake:
                contours, _ = cv2.findContours(
                    foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                frame_area = float(frame.shape[0] * frame.shape[1])
                for contour in contours:
                    area = float(cv2.contourArea(contour))
                    x, y, width, height = cv2.boundingRect(contour)
                    aspect = width / max(height, 1)
                    if (area < self.min_area or area > self.max_area_ratio * frame_area
                            or not self.min_aspect <= aspect <= self.max_aspect):
                        continue
                    fill = area / max(width * height, 1)
                    people.append(Person(
                        (x, y, x + width, y + height), max(0.0, min(1.0, fill)),
                        detector="mog2_blob",
                        metadata={"camera_id": str(camera_ids[frame_index]),
                                  "foreground_ratio": ratio,
                                  "flags": {"large_train": large_train,
                                            "camera_shake": camera_shake,
                                            "moving_shadows": moving_shadows}},
                    ))
            elapsed = (perf_counter() - started) * 1000
            for person in people:
                person.runtime_ms = elapsed
            diagnostics.append(ForegroundDiagnostics(
                frame_index, ratio, shadows, large_train, camera_shake,
                moving_shadows, warmup, elapsed))
            results.append(people)
        self.last_diagnostics = diagnostics
        return results

    detect_sequence = predict_sequence


B0AlwaysUnsafe = AlwaysUnsafeBaseline
B1HOGSVM = HOGSVMDetector
B2MOG2 = MOG2BlobDetector
AlwaysUnsafe = AlwaysUnsafeBaseline
HOGDetector = HOGSVMDetector
MOG2Detector = MOG2BlobDetector
