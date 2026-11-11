"""RailEye3D unified annotations: one headerless CSV per sequence/frame.

Official schema:
  bboxes/<sequence>/<frame>.csv : TLx,TLy,BRx,BRy,labelName,objectId
  params/<sequence>/<frame>.csv : degree_of_occlusion,beyond_safety_line
The two files pair by *row order*. The parameter file has no object ID.
"""
from __future__ import annotations

import csv
from pathlib import Path
import re
import pandas as pd

SEQUENCE_RE = re.compile(r"^red-\d{2}-\d{2}-[A-Za-z0-9_-]+$")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".PNG", ".JPEG")
PERSON_LABELS = frozenset({"adult", "child", "person", "pedestrian"})


def _rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [row for row in csv.reader(handle) if row and any(cell.strip() for cell in row)]


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
            flag = values[1].strip().lower()
            if flag not in {"true", "false", "1", "0"}:
                raise ValueError(f"{param_csv} row {index + 1}: invalid boolean {values[1]!r}")
            item["beyond_safety_line"] = flag in {"true", "1"}
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
    person = df[df["object_present"].astype(bool)].copy()
    if not person.empty:
        person = person[person["label"].str.lower().isin(PERSON_LABELS)]
    if not person.empty:
        flags = person["beyond_safety_line"].dropna()
        positives = int(sum(str(value).strip().lower() in {"true", "1"} for value in flags))
    else:
        positives = 0
    return {"frames": int(df[["sequence", "frame"]].drop_duplicates().shape[0]),
            "sequences": sorted(df["sequence"].unique().tolist()),
            "stations": sorted(df["station"].unique().tolist()),
            "person_instances": len(person), "unsafe_person_instances": positives,
            "safe_person_instances": len(person) - positives,
            "labels": {str(k): int(v) for k, v in df.loc[df["object_present"].astype(bool), "label"].value_counts().items()}}
