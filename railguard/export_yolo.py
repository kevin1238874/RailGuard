"""Optional supervised YOLO detection fine-tuning export; split checked first."""
from __future__ import annotations
from pathlib import Path
import shutil
import cv2
import pandas as pd
import yaml
from .raileye import PERSON_LABELS
from .splitting import assert_no_shared_sequences


def export_yolo(manifest: str | Path, destination: str | Path) -> dict:
    df = pd.read_csv(manifest)
    assert_no_shared_sequences(df)
    if not {"train", "val"}.issubset(set(df["split"])):
        raise ValueError("YOLO export requires nonempty train and val splits")
    target = Path(destination)
    counts = {"train": 0, "val": 0, "test": 0}
    for (sequence, frame), rows in df.groupby(["sequence", "frame"], sort=True):
        splits = set(rows["split"])
        if len(splits) != 1:
            raise ValueError(f"Inconsistent splits for frame: {sequence}/{frame}")
        split = splits.pop()
        if split not in counts:
            raise ValueError(f"Unexpected split: {split}")
        image_file = rows.iloc[0]["image"]
        if not isinstance(image_file, str) or not image_file:
            raise FileNotFoundError(f"Image missing in manifest: {sequence}/{frame}")
        image_path = Path(image_file)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        h, w = image.shape[:2]
        dest_image = target / "images" / split / f"{sequence}_{int(frame):05d}{image_path.suffix.lower()}"
        dest_labels = target / "labels" / split / f"{dest_image.stem}.txt"
        dest_image.parent.mkdir(parents=True, exist_ok=True)
        dest_labels.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, dest_image)
        labels = []
        for row in rows.itertuples(index=False):
            if not bool(row.object_present) or str(row.label).lower() not in PERSON_LABELS:
                continue
            x1, y1 = max(0, min(w, row.x1)), max(0, min(h, row.y1))
            x2, y2 = max(0, min(w, row.x2)), max(0, min(h, row.y2))
            if x2 <= x1 or y2 <= y1:
                continue
            labels.append(f"0 {(x1+x2)/(2*w):.6f} {(y1+y2)/(2*h):.6f} "
                          f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
        dest_labels.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        counts[split] += 1
    metadata = {"path": str(target.resolve()), "train": "images/train",
                "val": "images/val", "names": {0: "person"}}
    if counts["test"]:
        metadata["test"] = "images/test"
    target.mkdir(parents=True, exist_ok=True)
    (target / "data.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return counts


def train_yolo(data_yaml: str, weights: str = "yolo11n.pt", epochs: int = 30,
               image_size: int = 640, seed: int = 42, project: str = "runs"):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install: pip install -e '.[vision]'") from exc
    if not Path(data_yaml).is_file():
        raise FileNotFoundError(data_yaml)
    return YOLO(weights).train(data=data_yaml, epochs=epochs, imgsz=image_size,
                               seed=seed, project=project, exist_ok=False)
