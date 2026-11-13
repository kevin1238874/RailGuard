# RailGuard 🚉

**RGB-based railway platform safety-zone monitoring** · **CS406 — Xử lý ảnh và Ứng dụng**

RailGuard investigates whether a person's feet enter a calibrated hazardous area **Z** between a platform safety line **L** and its edge **E**. Its focus is a reproducible comparison of an inexpensive bounding-box heuristic against a pose/keypoint approach, with open reporting of false alarms and missed unsafe cases.

> **Academic prototype only.** Not certified for train dispatch, passenger protection, or real-world railway decisions. No experimental accuracy is claimed here; performance must be measured on the original images using the recorded protocol.

## 1. Task and methods

| | Definition |
| --- | --- |
| Input | One RGB railway-platform image/frame |
| Geometry | Scene-specific safety line **L**, platform edge **E**, polygonal strip **Z** |
| Baseline | YOLO person bbox → bottom-center foot proxy → point-in-zone |
| Proposed comparison | YOLO Pose → valid COCO ankle keypoints → *any* ankle in zone |
| Output | Per-person SAFE / DANGER / UNKNOWN, annotated image, JSON |
| Ground truth | RailEye3D instance attribute `beyond_safety_line` |
| Evaluation | Unsafe-class precision, recall, F1; frame alert metrics; matched-box coverage, failure cases |

Pipeline: **frame → person detection/pose → foot points → calibrated Z → decision → visualization/evaluation**.

A secondary **classical image-processing** experiment (`--auto-boundary`) explores HSV paint segmentation, Canny/Hough, and orientation-invariant line consensus to suggest safety/edge lines, including steep or near-vertical camera views. It is a heuristic, **not a trustworthy source of geometry**: it can fail or select the wrong real-world structures. The updated web demo supports **Auto Detect → click/drag correction → explicit confirmation → save camera YAML → YOLO inference**. The editor now displays the image inside a **bounded viewport with independent zoom (+/− and mouse wheel), drag-to-pan mode and Fit**; browser zoom is not needed. Candidate ranking checks that the proposed painted line has platform-like texture on both sides and that the proposed outer edge separates the platform from a darker area. The selected painted line is recentered within the segmented stripe, rather than using either Hough-detected border of the paint. These are RGB heuristics, not guarantees: always inspect the zone manually. Use verified per-camera calibration as the primary experimental condition.

## 2. RailEye3D and data access

Source: [raileye3d/raileye3d_dataset](https://github.com/raileye3d/raileye3d_dataset). The dataset has **10 sequences at six Austrian railway stations**. The planned subset is **Stockerau (`red-00`)** and **Fels (`red-04`)**, provided there are enough sequences for the chosen split.

The official repo provides per-frame, headerless CSVs:

```text
anno/unified/bboxes/<sequence>/<frame>.csv
    TLx, TLy, BRx, BRy, labelName, objectId
anno/unified/params/<sequence>/<frame>.csv
    degree_of_occlusion, beyond_safety_line
```

Rows are matched **by their row order**, never by `objectId` (which does not exist in `params`). `beyond_safety_line` is a **binary label**—it does not provide physical line endpoints. You must calibrate L and E for each camera, using actual imagery. Only `adult`, `child`, `person`, `pedestrian` entries are scored as people; other labels are tracked in the audit.

**Important:** The original RGB images are **not shipped in the annotation repo**; request them from the author using the instructions in the [official README](https://github.com/raileye3d/raileye3d_dataset#raileye3d-dataset). Review their non-commercial licence before use. Do not commit raw images or downloaded weights to this repo.

Recommended local layout (adjust the image filename extension `.png` / `.jpg` / `.jpeg` / `.webp`):

```text
data/
  raw/
    annotations/anno/unified/{bboxes,params}/<sequence>/<frame>.csv
    images/<sequence>/<frame>.jpg
  processed/
configs/
  <sequence>.yaml
```

An external image layout can be accommodated by organizing symlinks/copies into `<images>/<sequence>/<frame>.<ext>`.

## 3. Installation

Python **3.10–3.12** recommended. For example, on Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[vision,demo,dev]"
python -m pytest -q
```

On Linux/macOS use `python3 -m venv .venv`, `source .venv/bin/activate`, then the same `pip install` command. If you only need to parse CSVs, run the baseline with provided ground-truth boxes, or run tests without model downloads, `pip install -e ".[dev]"` is sufficient. CPU inference is supported but slower. YOLO weights may download on first use; record the exact weights and version for your runs.

## 4. Reproducible workflow

### Step A — Parse & audit original annotations

```bash
railguard prepare \
  --annotations data/raw/annotations \
  --images data/raw/images \
  --out data/processed/manifest.csv
railguard audit --manifest data/processed/manifest.csv --out outputs/data_audit.json
```

The audit also writes `outputs/data_audit.csv`. If the manifest/dataset is absent, both files are still created with zero counts and `status: BLOCKED_REAL_DATA`.

For **annotations-only exploration**, omit `--images` and pass `--allow-missing-images`. Geometry evaluation still requires images. Parsing checks missing paired CSVs, row counts, bbox validity, and label format. Empty frames are retained.

### Step B — Select the no-training validation/test protocol

```bash
railguard split --manifest data/processed/manifest.csv \
  --split-mode pretrained_val_test \
  --calibration-frame red-04-01-front-left/00000 \
  --out data/processed/splits.csv
```

The main protocol uses pretrained COCO models without an internal training set: every `red-00-00*` sequence is validation and every `red-04-01*` sequence is test. Other sequences are marked `excluded`. Replace the calibration frame example with one or more frames selected in advance; these remain listed as `calibration_only` and are removed from evaluation. Fit/select thresholds on validation only and freeze test-camera YAML before scoring.

For optional detector fine-tuning, retain the independent whole-sequence train/validation/test mode:

```bash
railguard split --manifest data/processed/manifest.csv \
  --split-mode train_val_test \
  --test-sequence <EXISTING_TEST_SEQUENCE> \
  --val-sequence <EXISTING_VALIDATION_SEQUENCE> \
  --out data/processed/fine_tuning_splits.csv
```

This optional mode requires at least three independent sequences. **Never randomly distribute frames of one sequence** across splits.

### Step C — Calibrate each camera (primary method)

```bash
railguard calibrate \
  --image data/raw/images/red-00-00-front-left/00000.jpg \
  --out configs/red-00-00-front-left.yaml
```

In the desktop window click **L0, L1, E0, E1**, then `S` to save, `U` undo, `R` reset, or `Esc` cancel. The calibration persists **normalized coordinates** and is valid only for the camera/view used. **Review L/E visually**; a wrong line can produce a plausible but incorrect alert. `configs/example_manual.yaml` is intentionally synthetic—never evaluate RailEye3D with it without re-calibration.

### Step D — Test one image

```bash
# Geometry-only oracle using the official GT person boxes:
railguard predict --detector gt \
  --bbox-file data/raw/annotations/anno/unified/bboxes/red-00-00-front-left/00000.csv \
  --image data/raw/images/red-00-00-front-left/00000.jpg \
  --config configs/red-00-00-front-left.yaml \
  --out outputs/oracle.jpg

# Real model baseline (COCO person class 0):
railguard predict --detector yolo --weights yolo11n.pt \
  --image data/raw/images/red-00-00-front-left/00000.jpg \
  --config configs/red-00-00-front-left.yaml --out outputs/bbox.jpg

# Pose variant; abstain when ankles are missing:
railguard predict --detector pose --weights yolo11n-pose.pt \
  --pose-fallback abstain \
  --image data/raw/images/red-00-00-front-left/00000.jpg \
  --config configs/red-00-00-front-left.yaml --out outputs/pose.jpg
```

These commands write an annotated JPG and accompanying JSON. `gt` is an **oracle geometry-only sanity check**, **not** a deployable detector or a fair end-to-end comparison with YOLO.

### Step E — Held-out evaluation

```bash
railguard evaluate --manifest data/processed/splits.csv \
  --configs configs --split test --detector gt \
  --out outputs/oracle_eval

railguard evaluate --manifest data/processed/splits.csv \
  --configs configs --split test --detector yolo --weights yolo11n.pt \
  --out outputs/bbox_eval

railguard evaluate --manifest data/processed/splits.csv \
  --configs configs --split test --detector pose --weights yolo11n-pose.pt \
  --pose-fallback abstain --out outputs/pose_eval
```

Per run, results include `metrics.json` (instance- and frame-level alert metrics), `predictions.csv`, `failure_cases.csv`, and a few visualized frames. A true unsafe person missed by the detector or left UNKNOWN contributes a **false negative for unsafe recall**; an unmatched unsafe alert is a **false positive**. Conditional metrics (matched/determined instances only) are reported separately and **must not be presented as end-to-end performance**. The instance-level accuracy value is conditional on scored instances; a separate frame-level alert metric covers all evaluated frames.

To **isolate the foot-point heuristic** with the *same* pose-capable model, additionally run YOLO bbox evaluation with `--detector yolo --weights yolo11n-pose.pt` and compare against `--detector pose --weights yolo11n-pose.pt`; this controls for detector weights. The standard `yolo11n.pt` vs `yolo11n-pose.pt` comparison changes both model and foot-point method and must be interpreted as end-to-end systems, not a pure keypoint ablation.

### Step F — Optional detector fine-tuning

```bash
railguard export-yolo --manifest data/processed/splits.csv --out data/processed/yolo
railguard train --data data/processed/yolo/data.yaml --weights yolo11n.pt --epochs 30 --imgsz 640 --seed 42
```

This trains a **person detector**, not a new safety-line classifier or a pose network. It requires train/val images with box labels and does not fine-tune `yolo11n-pose.pt`. Training scripts must never access test annotations to tune parameters. The test split may be exported but must **not** be used for fitting.

### Step G — Interactive Web UI (**RailGuard Studio · HTML5/Canvas**)

```bash
# Install core + YOLO detection dependencies:
pip install -e ".[vision,dev]"

# Start RailGuard Studio web server (default port 8502):
python -m railguard.server
# or via CLI:
railguard studio --port 8502
```

Open **`http://localhost:8502`** in your browser and follow the interactive workflow:

1. **Upload an RGB station image (`JPG / PNG / WEBP`):** uploading a new image resets previous boundary points and detection results cleanly, waiting for your method choice.
2. **Choose line detection method (L & E):**
   - **1. LAB Color + Canny + Hough (`lab_canny_hough`):** adaptive LAB yellow/white paint mask + Canny curb edge detection with projective vanishing-point convergence. Auto-locks points to prevent accidental drag.
   - **2. Hybrid Region + RANSAC (`hybrid_region_evidence`):** combines HSV/LAB paint cues, RANSAC clustering + Hough candidates, ceiling-rafter filtering, and two-sided brightness drop verification. Auto-locks points.
   - **3. Manual Mode (`manual`):** unlocks direct click placement and dragging of the four boundary endpoints (`L0`, `L1`, `E0`, `E1`) on the HTML5 canvas.
3. **Select person detector (M3):** choose `YOLO11 Pose` (Crop-Pose multi-scale ankle detection for both near and distant passengers) or `YOLO11 Detection` (Bounding-Box bottom-center).
4. **Run Safety Check & Export:** click **Chạy Kiểm Tra An Toàn** to evaluate per-person `DANGER / SAFE / UNKNOWN` status, view annotated detections, and download the PNG overlay or JSON report.

Run `python -m pytest -q` after updating.

## 5. Implementation structure

```text
RailGuard/
├── railguard/
│   ├── raileye.py           # official CSV parsing, image lookup and audit
│   ├── geometry.py          # L / E / Z and point-in-polygon checks
│   ├── preprocessing.py     # M1 illumination/contrast/noise enhancements
│   ├── boundaries.py        # M2 LAB/HSV paint masks, Canny/Hough & VP alignment
│   ├── line_models.py       # RANSAC & collinear segment clustering
│   ├── calibration.py       # camera-specific line calibration helpers
│   ├── detection.py         # M3 YOLO bbox/pose adapters + Crop-Pose
│   ├── foot_refinement.py   # foot-point estimation & refinement
│   ├── classic_detectors.py # HOG+SVM & MOG2 baselines
│   ├── pipeline.py          # M4 spatial analysis and image overlays
│   ├── server.py            # lightweight HTTP backend for RailGuard Studio
│   ├── studio/
│   │   └── index.html       # HTML5/Canvas RailGuard Studio web UI
│   ├── ui_state.py          # UI session state and export helpers
│   ├── zone_editor.py       # coordinate normalization & YAML persistence
│   ├── splitting.py         # leakage checks and sequence-level split
│   ├── metrics.py           # IoU matching and binary metrics
│   ├── evaluate.py          # per-instance evaluation, failure export
│   ├── experiments.py       # E1–E5 benchmark runner
│   ├── export_yolo.py       # optional person detector data export/training
│   └── cli.py               # runnable CLI commands (including `studio`)
├── tests/                   # synthetic unit/integration tests
├── configs/                 # per-camera & M1 configuration templates
├── scripts/                 # synthetic smoke-test generator
├── docs/                    # protocol, methodology, experiments & progress
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 6. Experimental protocol & limitations

- **Primary comparison:** identical held-out frames, manual L/E calibration, same IoU matching threshold (default 0.5), and fixed weights, confidence and fallback policy.
- **Ablations:** geometry (manual vs auto-boundary), foot proxy (bbox center vs ankles), pose missing-keypoint fallback, detection confidence, IoU threshold.
- **Failure analysis:** occlusion (`degree_of_occlusion`), perspective, cropped feet, incorrect person boxes, station/camera shifts, boundary ambiguity, and violations near L/E.
- **Caveats:** class imbalance, ground-truth annotation noise, lack of physical camera calibration, boundary placement bias, and low confidence for heavily occluded subjects. Human line calibration may introduce bias: create it using designated calibration frames only and freeze it before test evaluation.
- **Metric caution:** a low violation prevalence makes recall/precision/F1 more meaningful than accuracy. Separate unsafe classification from detector localization when interpreting causes of errors.

A short synthetic-data smoke test is available (for **testing code only**, never as a real-data performance result):

```bash
python scripts/make_synthetic_demo.py --out data/synthetic
railguard prepare --annotations data/synthetic/annotations --images data/synthetic/images --out data/processed/synthetic.csv
railguard split --manifest data/processed/synthetic.csv --split-mode train_val_test --test-sequence red-04-00-front-left --val-sequence red-01-00-front-left --out data/processed/synthetic_split.csv
railguard evaluate --manifest data/processed/synthetic_split.csv --configs data/synthetic/configs --detector gt --out outputs/synthetic_check
```

## 7. Credits

- **Dataset:** [RailEye3D annotations](https://github.com/raileye3d/raileye3d_dataset), under the dataset authors' applicable research/non-commercial terms.
- **Original paper:** Wallner, M., Steininger, D., Widhalm, V., Schoerghuber, M., & Beleznai, C. (2021). *RGB-D Railway Platform Monitoring and Scene Understanding for Enhanced Passenger Safety.* ICPR International Workshops and Challenges. [DOI](https://doi.org/10.1007/978-3-030-68787-8_47).
- **Model framework:** [Ultralytics](https://docs.ultralytics.com/); check their weights/framework licence before distribution or commercial use.

**No original RailEye3D images, pretrained model weights or fabricated experiment scores are included.**
