from pathlib import Path
import cv2
import numpy as np
import pytest

# OpenCV 5 removed cv2.HOGDescriptor from core; B1 HOG+SVM needs opencv-python<5.
requires_hog = pytest.mark.skipif(
    not hasattr(cv2, "HOGDescriptor"),
    reason="cv2.HOGDescriptor unavailable (OpenCV 5); install opencv-python<5")


@pytest.fixture
def tiny_raileye(tmp_path: Path):
    """Three sequences, two people in one test frame, plus an empty frame."""
    root = tmp_path / "annotation"
    image_root = tmp_path / "images"
    sequences = ["red-00-00-front-left", "red-00-01-front-left", "red-04-00-front-left"]
    for seq in sequences:
        boxes_dir = root / "anno" / "unified" / "bboxes" / seq
        params_dir = root / "anno" / "unified" / "params" / seq
        images_dir = image_root / seq
        for directory in (boxes_dir, params_dir, images_dir):
            directory.mkdir(parents=True, exist_ok=True)
        (boxes_dir / "00000.csv").write_text(
            "20,45,40,80,adult,123\n50,10,70,40,adult,124\n", encoding="utf-8")
        (params_dir / "00000.csv").write_text("0,True\n25,False\n", encoding="utf-8")
        (boxes_dir / "00015.csv").write_text("", encoding="utf-8")
        (params_dir / "00015.csv").write_text("", encoding="utf-8")
        for stem in ("00000", "00015"):
            assert cv2.imwrite(str(images_dir / f"{stem}.png"), np.zeros((100, 100, 3), dtype=np.uint8))
    return root, image_root, sequences
