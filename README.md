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

A secondary **classical image-processing** experiment (`--auto-boundary`) explores HSV paint segmentation, Canny/Hough, and orientation-invariant line consensus to suggest safety/edge lines, including steep or near-vertical camera views. It is a heuristic, **not a trustworthy source of geometry**: it can fail or select the wrong real-world structures. The updated web demo supports **Auto Detect → click/drag correction → explicit confirmation → save camera YAML → YOLO inference**. Use verified per-camera calibration as the primary experimental condition.

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

For **annotations-only exploration**, omit `--images` and pass `--allow-missing-images`. Geometry evaluation still requires images. Parsing checks missing paired CSVs, row counts, bbox validity, and label format. Empty frames are retained.

### Step B — Split by full **sequence**

```bash
railguard split --manifest data/processed/manifest.csv \
  --test-sequence red-04-00-front-left \
  --val-sequence <ANOTHER_EXISTING_SEQUENCE> \
  --out data/processed/splits.csv
```

Replace both example sequence names with **existing distinct sequences** from your manifest. At least **three independent sequences** are needed for train / val / test; if the chosen subset does not provide enough, extend it or explicitly report a different evaluation protocol. **Never randomly distribute frames of a single sequence** across splits. For cross-station experiments, keep stations disjoint when feasible and document this separately. Fit or select any thresholds **without inspecting the test split**.

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

### Step G — Interactive demo (click/drag calibration)

```bash
# First installation (includes Streamlit and YOLO):
pip install -e ".[vision,demo,dev]"
streamlit run app.py
```

Open the local Streamlit URL (usually `http://localhost:8501`) and follow the **four visible steps**:

1. **Upload an RGB image.** Optionally upload an existing camera YAML in the sidebar. The image and imported YAML are associated with the current view; changing either **invalidates the previous approval**.
2. **Auto Detect L/E** to get a provisional suggestion using `hybrid` (HSV paint cues + Canny/Hough line segments) or other selectable methods. If the algorithm cannot find a pair, it leaves existing points unchanged and displays a warning. You can also start manually with **Clear four points**.
3. **Edit directly on the image**: click **L0 / L1 / E0 / E1** and click the corresponding location, or **drag an existing colored point**. The yellow/orange line is **L (safety paint)**, blue is **E (platform edge)**, and the translucent polygon is **Z**. Four valid points are mandatory. This works for steep and near-vertical lines; no numeric coordinate entry is needed. **Review L and E visually**: an auto proposal is *not* a geometric ground truth and may select rails or highlights by mistake.
4. **Confirm, save, and infer**: check the verification box, enter a safe fixed-camera or RailEye3D sequence name, then **Download camera YAML** or **Save to local `configs/<sequence>.yaml`** (on the computer running Streamlit). Saving an existing YAML requires the explicit overwrite checkbox. Only after valid geometry and confirmation does **Run YOLO on this image** become enabled. It displays actual predictions and provides JSON export.

Any **new point edit or new Auto Detect** resets approval and clears stale inference results. Model loading may download weights when first invoked. The auto-detection candidate must **not** be treated as an automatically validated calibration, even if a polygon looks plausible. Camera-specific saved YAML can be used for other frames of that same **stationary camera**, and by the CLI for reproducible evaluation.

Streamlit's native **components v2** is used for direct manipulation (requires `streamlit>=1.56`; your existing `1.67` is compatible). No extra third-party canvas package is required. Run `python -m pytest -q` after updating.

## 5. Implementation structure

```text
RailGuard/
├── railguard/
│   ├── raileye.py        # official CSV parsing, image lookup and audit
│   ├── geometry.py       # L / E / Z and point-in-polygon checks
│   ├── boundaries.py     # classical HSV/Canny/Hough candidate lines
│   ├── calibration.py    # interactive camera-specific line calibration
│   ├── detection.py      # bbox/pose YOLO adapters and foot-point rules
│   ├── pipeline.py       # spatial analysis and image overlays
│   ├── splitting.py      # leakage checks and sequence-level split
│   ├── metrics.py        # IoU matching and binary metrics
│   ├── evaluate.py       # per-instance evaluation, failure export
│   ├── export_yolo.py    # optional person detector data export/training
│   └── cli.py            # runnable commands
├── app.py                # Streamlit app with real inference
├── tests/               # synthetic unit/integration tests
├── configs/             # per-camera calibration templates
├── scripts/             # synthetic smoke-test generator
├── docs/                # protocol and limitations
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
railguard split --manifest data/processed/synthetic.csv --test-sequence red-04-00-front-left --val-sequence red-01-00-front-left --out data/processed/synthetic_split.csv
railguard evaluate --manifest data/processed/synthetic_split.csv --configs data/synthetic/configs --detector gt --out outputs/synthetic_check
```

## 7. Credits

- **Dataset:** [RailEye3D annotations](https://github.com/raileye3d/raileye3d_dataset), under the dataset authors' applicable research/non-commercial terms.
- **Original paper:** Wallner, M., Steininger, D., Widhalm, V., Schoerghuber, M., & Beleznai, C. (2021). *RGB-D Railway Platform Monitoring and Scene Understanding for Enhanced Passenger Safety.* ICPR International Workshops and Challenges. [DOI](https://doi.org/10.1007/978-3-030-68787-8_47).
- **Model framework:** [Ultralytics](https://docs.ultralytics.com/); check their weights/framework licence before distribution or commercial use.

**No original RailEye3D images, pretrained model weights or fabricated experiment scores are included.**
