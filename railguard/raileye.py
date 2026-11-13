"""RailEye3D unified annotations: one headerless CSV per sequence/frame.

Official schema:
  bboxes/<sequence>/<frame>.csv : TLx,TLy,BRx,BRy,labelName,objectId
  params/<sequence>/<frame>.csv : degree_of_occlusion,beyond_safety_line
The two files pair by *row order*. The parameter file has no object ID.
"""
from __future__ import annotations

import csv
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
import pandas as pd

SEQUENCE_RE = re.compile(r"^red-\d{2}-\d{2}-[A-Za-z0-9_-]+$")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".PNG", ".JPEG")
PERSON_LABELS = frozenset({"adult", "child", "person", "pedestrian"})


def _rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [row for row in csv.reader(handle) if row and any(cell.strip() for cell in row)]


def parse_bool(value: Any, *, location: str = "boolean value") -> bool:
    """Parse the boolean spellings used by RailEye3D and CSV round trips."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"{location}: invalid boolean {value!r}")


def read_frame(bbox_csv: str | Path, param_csv: str | Path | None = None) -> list[dict]:
    """Read a frame, strictly validating column counts and row-wise alignment."""
    bbox_csv = Path(bbox_csv)
    boxes = _rows(bbox_csv)
    params = _rows(Path(param_csv)) if param_csv is not None else None
    if params is not None and len(boxes) != len(params):
        raise ValueError(f"Annotation row count mismatch: {bbox_csv}: {len(boxes)} boxes vs {len(params)} params")
    objects: list[dict] = []
    for index, row in enumerate(boxes):
        if len(row) != 6:
            raise ValueError(f"{bbox_csv} row {index + 1}: expected 6 bbox columns, got {len(row)}")
        try:
            x1, y1, x2, y2 = map(float, row[:4])
        except ValueError as exc:
            raise ValueError(f"{bbox_csv} row {index + 1}: non-numeric coordinates") from exc
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError(f"{bbox_csv} row {index + 1}: invalid bbox {row[:4]}")
        label, object_id = row[4].strip(), row[5].strip()
        if not label or not object_id:
            raise ValueError(f"{bbox_csv} row {index + 1}: empty label or object ID")
        item = dict(x1=x1, y1=y1, x2=x2, y2=y2, label=label,
                    object_id=object_id, occlusion=None, beyond_safety_line=None)
        if params is not None:
            values = params[index]
            if len(values) != 2:
                raise ValueError(f"{param_csv} row {index + 1}: expected 2 columns")
            try:
                item["occlusion"] = float(values[0])
            except ValueError as exc:
                raise ValueError(f"{param_csv} row {index + 1}: invalid occlusion") from exc
            item["beyond_safety_line"] = parse_bool(
                values[1], location=f"{param_csv} row {index + 1}")
        objects.append(item)
    return objects


def locate_image(image_root: str | Path, sequence: str, frame_stem: str) -> Path | None:
    directory = Path(image_root) / sequence
    for ext in IMAGE_EXTENSIONS:
        image = directory / f"{frame_stem}{ext}"
        if image.is_file():
            return image
    return None


def _unified_root(path: str | Path) -> Path:
    path = Path(path)
    if (path / "anno" / "unified" / "bboxes").is_dir():
        return path / "anno" / "unified"
    if (path / "unified" / "bboxes").is_dir():
        return path / "unified"
    return path


def make_manifest(annotation_root: str | Path, image_root: str | Path | None = None,
                  sequences: list[str] | None = None, require_images: bool = True) -> pd.DataFrame:
    """One manifest row per object, plus one placeholder for a frame with zero objects.

    Missing image files are errors when require_images=True; never silently skip them.
    """
    root = _unified_root(annotation_root)
    boxes_dir, params_dir = root / "bboxes", root / "params"
    if not boxes_dir.is_dir():
        raise FileNotFoundError(f"Expected RailEye3D box directory: {boxes_dir}")
    if require_images and image_root is None:
        raise ValueError("--images is required unless --allow-missing-images is set")
    selected = set(sequences) if sequences else None
    seen: set[str] = set()
    records: list[dict] = []
    for directory in sorted(boxes_dir.iterdir()):
        if not directory.is_dir() or not SEQUENCE_RE.fullmatch(directory.name):
            continue
        sequence = directory.name
        if selected and sequence not in selected:
            continue
        seen.add(sequence)
        for box_path in sorted(directory.glob("*.csv")):
            if not box_path.stem.isdigit():
                raise ValueError(f"Non-numeric frame name: {box_path}")
            param_path = params_dir / sequence / box_path.name
            if not param_path.is_file():
                raise FileNotFoundError(f"Missing paired parameter file: {param_path}")
            image = locate_image(image_root, sequence, box_path.stem) if image_root else None
            if require_images and image is None:
                raise FileNotFoundError(f"Image not found for {sequence}/{box_path.stem} in {image_root}")
            common = dict(sequence=sequence, station="-".join(sequence.split("-")[:2]),
                          frame=int(box_path.stem), image=str(image.resolve()) if image else "",
                          bbox_file=str(box_path.resolve()), param_file=str(param_path.resolve()))
            objects = read_frame(box_path, param_path)
            if objects:
                records.extend({**common, "object_present": True, **obj} for obj in objects)
            else:
                records.append({**common, "object_present": False, "x1": None, "y1": None,
                                "x2": None, "y2": None, "label": "", "object_id": "",
                                "occlusion": None, "beyond_safety_line": None})
    if selected and selected - seen:
        raise ValueError(f"Sequences not present in annotation folder: {sorted(selected - seen)}")
    if not records:
        raise ValueError("No annotated frames found")
    return pd.DataFrame.from_records(records)


def summarize_manifest(df: pd.DataFrame) -> dict:
    present = df["object_present"].map(
        lambda value: False if pd.isna(value) else parse_bool(value, location="object_present"))
    person = df[present].copy()
    if not person.empty:
        person = person[person["label"].str.lower().isin(PERSON_LABELS)]
    if not person.empty:
        flags = person["beyond_safety_line"].dropna()
        positives = int(sum(parse_bool(value, location="beyond_safety_line") for value in flags))
    else:
        positives = 0
    return {"frames": int(df[["sequence", "frame"]].drop_duplicates().shape[0]),
            "sequences": sorted(df["sequence"].unique().tolist()),
            "stations": sorted(df["station"].unique().tolist()),
            "person_instances": len(person), "unsafe_person_instances": positives,
            "safe_person_instances": len(person) - positives,
            "labels": {str(k): int(v) for k, v in df.loc[present, "label"].value_counts().items()}}


def blocked_data_audit(reason: str, source: str | Path | None = None) -> dict:
    """Return an explicit zero-count audit without implying that data was tested."""
    return {
        "status": "BLOCKED_REAL_DATA",
        "reason": reason,
        "source": "" if source is None else str(source),
        "sequence_count": 0,
        "sequences": [],
        "station_count": 0,
        "stations": [],
        "camera_count": 0,
        "cameras": [],
        "frames": 0,
        "empty_frames": 0,
        "person_instances": 0,
        "safe_person_instances": 0,
        "unsafe_person_instances": 0,
        "labels": {},
        "occlusion": {"count": 0, "min": None, "max": None, "mean": None},
        "images": {
            "referenced": 0,
            "readable": 0,
            "missing": 0,
            "unreadable": 0,
            "sizes": {},
            "brightness_mean": None,
            "brightness_std": None,
            "brightness_min": None,
            "brightness_max": None,
        },
        "missing_files": {"images": 0, "bbox_csv": 0, "param_csv": 0},
    }


def audit_manifest(df: pd.DataFrame, source: str | Path | None = None) -> dict:
    """Audit counts and available image properties without inventing missing results."""
    if df.empty:
        return blocked_data_audit("Manifest is empty; no RailEye3D data is available.", source)

    required = {"sequence", "station", "frame", "object_present", "label",
                "occlusion", "beyond_safety_line", "image", "bbox_file", "param_file"}
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {missing_columns}")

    summary = summarize_manifest(df)
    frame_rows = df.drop_duplicates(["sequence", "frame"])
    present = df["object_present"].map(
        lambda value: False if pd.isna(value) else parse_bool(value, location="object_present"))
    people = df[present & df["label"].astype(str).str.lower().isin(PERSON_LABELS)]
    occlusion = pd.to_numeric(people["occlusion"], errors="coerce").dropna()

    missing_images = unreadable_images = 0
    sizes: Counter[str] = Counter()
    brightness: list[float] = []
    referenced_images = 0
    for value in frame_rows["image"]:
        if not isinstance(value, str) or not value.strip():
            missing_images += 1
            continue
        referenced_images += 1
        path = Path(value)
        if not path.is_file():
            missing_images += 1
            continue
        image = cv2.imread(str(path))
        if image is None:
            unreadable_images += 1
            continue
        height, width = image.shape[:2]
        sizes[f"{width}x{height}"] += 1
        brightness.append(float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()))

    def missing_path_count(column: str) -> int:
        return int(sum(not isinstance(value, str) or not value.strip() or not Path(value).is_file()
                       for value in frame_rows[column]))

    image_stats = {
        "referenced": referenced_images,
        "readable": int(sum(sizes.values())),
        "missing": missing_images,
        "unreadable": unreadable_images,
        "sizes": dict(sorted(sizes.items())),
        "brightness_mean": round(float(np.mean(brightness)), 4) if brightness else None,
        "brightness_std": round(float(np.std(brightness)), 4) if brightness else None,
        "brightness_min": round(float(np.min(brightness)), 4) if brightness else None,
        "brightness_max": round(float(np.max(brightness)), 4) if brightness else None,
    }
    real_data_ready = image_stats["readable"] > 0
    return {
        "status": "READY" if real_data_ready else "BLOCKED_REAL_DATA",
        "reason": "" if real_data_ready else "No readable real RGB images are referenced by the manifest.",
        "source": "" if source is None else str(source),
        "sequence_count": len(summary["sequences"]),
        "sequences": summary["sequences"],
        "station_count": len(summary["stations"]),
        "stations": summary["stations"],
        "camera_count": len(summary["sequences"]),
        "cameras": summary["sequences"],
        "frames": summary["frames"],
        "empty_frames": int((~present).groupby([df["sequence"], df["frame"]]).all().sum()),
        "person_instances": summary["person_instances"],
        "safe_person_instances": summary["safe_person_instances"],
        "unsafe_person_instances": summary["unsafe_person_instances"],
        "labels": summary["labels"],
        "occlusion": {
            "count": int(len(occlusion)),
            "min": float(occlusion.min()) if len(occlusion) else None,
            "max": float(occlusion.max()) if len(occlusion) else None,
            "mean": round(float(occlusion.mean()), 4) if len(occlusion) else None,
        },
        "images": image_stats,
        "missing_files": {
            "images": missing_images + unreadable_images,
            "bbox_csv": missing_path_count("bbox_file"),
            "param_csv": missing_path_count("param_file"),
        },
    }


def write_data_audit(summary: dict, json_path: str | Path,
                     csv_path: str | Path | None = None) -> tuple[Path, Path]:
    """Write the same audit as structured JSON and a flat metric/value CSV."""
    json_path = Path(json_path)
    csv_path = Path(csv_path) if csv_path is not None else json_path.with_suffix(".csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    rows: list[dict[str, Any]] = []

    def flatten(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                flatten(f"{prefix}.{key}" if prefix else str(key), item)
        elif isinstance(value, list):
            rows.append({"metric": prefix, "value": json.dumps(value, ensure_ascii=False)})
        else:
            rows.append({"metric": prefix, "value": value})

    flatten("", summary)
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(csv_path, index=False)
    return json_path, csv_path
