# RailGuard implementation progress

This file records verified repository state. It does not claim results on RailEye3D RGB data unless those images were actually available and used.

## Verified inventory (2026-11-12)

| Component | Status | Evidence / limitation |
|---|---|---|
| RailEye3D headerless bbox/params parser | Implemented | `railguard/raileye.py` validates schemas, pairs bbox and params strictly by row order, preserves empty frames, parses boolean strings, and rejects row-count mismatches. |
| Missing params and missing image handling | Implemented | Manifest preparation fails explicitly for a missing paired params CSV or required image; annotations-only audit can allow absent images. Covered by unit tests. |
| Empty-data and manifest data audit | Implemented | `railguard audit` writes JSON and CSV; absent data produces zero counts with `status: BLOCKED_REAL_DATA`. Image size/brightness, labels, occlusion, missing files, stations, and cameras are reported when available. |
| Main pretrained val/test protocol | Implemented | `pretrained_val_test` assigns `red-00-00*` to `val`, `red-04-01*` to `test`, other sequences to `excluded`, and creates no internal train split. |
| Optional fine-tuning split | Implemented | Legacy whole-sequence `train_val_test` remains available and still requires independent train/validation/test sequences. |
| Calibration-only frame tracking | Implemented | Split manifests contain `calibration_only` and `evaluation_eligible`; evaluation removes calibration-only rows before scoring. |
| Manual L/E calibration and YAML | Implemented | Desktop and Streamlit helpers validate and persist normalized per-camera geometry. No real-camera YAML was validated in P0. |
| Hazard polygon and tri-state decisions | Implemented | Geometry, bbox/ankle footpoints, SAFE/DANGER/UNKNOWN behavior, and drawing exist with synthetic tests. Outside-E semantics remain incomplete for P4. |
| M2 automatic boundary proposals | Implemented, synthetic only | HSV/LAB masks, all-orientation HoughP, clustering, true RANSAC, relative two-sided E evidence, Top-3 review records, stripe conventions, and explicit abstention are implemented. No RailEye3D RGB validation is claimed. |
| `color_ransac` boundary method | Corrected | The legacy alias now routes to `hsv_hough_ransac`, backed by two-point sampling, perpendicular residuals, consensus selection, inlier refit, and reproducible random state in `railguard/line_models.py`. |
| Dedicated M1 preprocessing pipeline | Implemented, not real-data validated | Registry-based Raw, Gaussian, Median, luminance-only HistEq/CLAHE, documented LUT Gamma, ordered pipelines, separate M2/M3 config branches, quality flags, temporal median, and contact-sheet previews are covered by synthetic tests. |
| YOLO detection and pose adapters | Implemented | Lazy Ultralytics adapters for person bbox and COCO ankles exist. Real weights/inference were not run in P0 and no models were downloaded. |
| HOG+SVM, MOG2, refined contact point | Implemented, synthetic only | B0 frame-only baseline, B1 HOG+SVM/NMS, ordered fixed-camera B2 MOG2 diagnostics, and bounded optional contact refinement are covered by offline fixtures. |
| IoU matching and unsafe metrics | Implemented | Synthetic evaluation includes end-to-end and conditional metrics plus frame alerts and failure exports. Real-data quality is unverified. |
| YOLO detector export/fine-tuning | Implemented | Person-detector export and optional training entry point exist; no training was run. |
| Streamlit direct-manipulation demo | Implemented | Upload, auto suggestion, click/drag editing, approval, YAML persistence, and YOLO invocation exist. Manual browser behavior is unverified in P0. |
| Automated tests | Implemented | Unit/integration tests use generated synthetic fixtures and mocks; they do not establish real-dataset accuracy. |
| RailEye3D RGB dataset and quantitative findings | Missing | `data/` contains only `data/README.md`; no original annotations/images were available. Accuracy, precision, recall, and F1 are therefore not reported. |

## P0 — Data Audit, Environment & Protocol (2026-11-12)
- Status: BLOCKED_REAL_DATA
- Files changed: `railguard/raileye.py`, `railguard/splitting.py`, `railguard/cli.py`, `railguard/evaluate.py`, `tests/test_raileye.py`, `tests/test_split_and_metrics.py`, `tests/test_end_to_end.py`, `README.md`, `docs/RESEARCH_PROTOCOL.md`, `docs/IMPLEMENTATION_NOTES.md`, `docs/PROGRESS.md`, `outputs/environment_manifest.json`, `outputs/data_audit.json`, `outputs/data_audit.csv`
- Methods actually implemented: strict row-order bbox/params validation; clean boolean and empty-frame handling; empty-data audit JSON/CSV; fixed-prefix pretrained validation/test splitting; preserved optional train/val/test splitting; calibration-only frame declaration and evaluator exclusion
- Reproduction commands: `.\.venv\Scripts\python.exe -m railguard audit --manifest data\processed\manifest.csv --out outputs\data_audit.json --csv-out outputs\data_audit.csv`; `.\.venv\Scripts\python.exe -m pytest -q`; `.\.venv\Scripts\python.exe -m compileall -q railguard app.py`
- Tests: `.\.venv\Scripts\python.exe -m pytest -q` → `50 passed in 2.65s` (exit 0); `.\.venv\Scripts\python.exe -m compileall -q railguard app.py` → no stdout (exit 0)
- Real datasets/images used (and permissions): none; `data/` has no RailEye3D annotations or RGB images, so the audit is `BLOCKED_REAL_DATA`; synthetic temporary test fixtures only, with no scientific validity
- Artifacts: `outputs/environment_manifest.json`; `outputs/data_audit.json`; `outputs/data_audit.csv`
- Measured improvements/regressions: automated coverage increased from baseline 41 tests to 50 tests; no real-data performance measurement was made and no accuracy/F1 result is claimed
- Known failure cases at the P0 checkpoint: `color_ransac` was not yet true RANSAC; no real image/calibration validation; no GPU/CUDA; no HOG/MOG2/M1 pipeline; automatic boundaries remained heuristic. The RANSAC issue was corrected in P2 below.
- Blockers / next step: obtain authorized RailEye3D annotations and RGB frames, rerun prepare/audit, predeclare calibration-only frames, freeze camera YAMLs, and only then run validation/test scoring
- Commit: no commit created per task constraint; working base commit `e79367d`

## P1 — M1: Controlled Image & Illumination Preprocessing (2026-11-12)
- Status: `IMPLEMENTED_NOT_VALIDATED_ON_REAL_DATA`; implementation verification: `DONE_SYNTHETIC`
- Files changed: `railguard/preprocessing.py`, `railguard/temporal.py`, `railguard/cli.py`, `configs/m1_preprocessing.yaml`, `tests/test_preprocessing.py`, `docs/PROGRESS.md`
- Methods actually implemented: registry/plugin-based `identity`/`raw`, Gaussian blur, median blur, luminance-only global histogram equalization, luminance-only CLAHE, LUT gamma using `255*(I/255)**(1/gamma)` where gamma above 1 brightens, ordered composable pipelines, separate `m2_boundary` and `m3_person` config branches, descriptive quality flags, same-camera ordered-frame temporal median, and four-panel contact sheets with runtime
- Reproduction commands: `.\.venv\Scripts\python.exe -m pytest -q`; `.\.venv\Scripts\python.exe -m compileall -q railguard app.py`; `.\.venv\Scripts\python.exe -m railguard preprocess --help`
- Tests: `.\.venv\Scripts\python.exe -m pytest -q` → `76 passed in 2.62s` (exit 0); `.\.venv\Scripts\python.exe -m compileall -q railguard app.py` → no stdout (exit 0); `.\.venv\Scripts\python.exe -m railguard preprocess --help` → usage/options/examples printed (exit 0). Synthetic coverage includes dark, bright/clipped, noisy, gradient, neutral/color patches, small images, shape/dtype/finite output, gamma convention, config serialization, stationary-video transient foreground, temporal sampling/windowing, rejection of a single image/single frame/mixed cameras, contact-sheet construction, and CLI export.
- Real datasets/images used (and permissions): none; RailEye3D RGB data is absent. Only generated arrays and temporary synthetic PNG fixtures were used, with no claim of scientific or operational validity.
- Artifacts: reproducible example config `configs/m1_preprocessing.yaml`; contact sheets and JSON metadata are generated only when the `railguard preprocess` command is run
- Measured improvements/regressions: no M2 boundary-quality or M3 detection-quality improvement was measured; no real-data accuracy, precision, recall, F1, illumination recovery, or detector benefit is claimed
- Known failure cases: luminance conversion can shift highly saturated colors slightly at the BGR gamut boundary; temporal median cannot remove foreground present in at least half of selected frames; clipped highlights remain clipped and are only reported, never claimed as recovered; preprocessing that helps M2 may harm M3 and must be evaluated separately on validation data
- Blockers / next step: obtain authorized RailEye3D RGB frames, freeze M1 candidates using validation only, measure M2 and M3 separately, and keep held-out test data out of preprocessing selection
- Commit: no commit created per task constraint

## P2 — M2: True RANSAC and Distinct Safety Line L / Platform Edge E (2026-11-12)

- Status: `IMPLEMENTED_NOT_VALIDATED_ON_REAL_DATA`; implementation verification: `DONE_SYNTHETIC`
- Files changed: `railguard/line_models.py`, `railguard/boundaries.py`, `railguard/cli.py`, `tests/test_boundaries_m2.py`, `docs/PROGRESS.md`
- Methods implemented: minimal two-point RANSAC with perpendicular residuals, consensus selection and inlier refit; angle/distance segment clustering without a global parallel constraint; HSV white/yellow and LAB masking with open/close morphology; all-orientation Canny/HoughP; `safe_side_edge` and `stripe_center`; L-directed two-sided region evidence for E; same-stripe rejection; Top-3 pixel and normalized candidate records with support/inliers, evidence, heuristic `score`, `uncertainty_reason`, and `estimated_review_required`
- M2 × M1 matrix: `hsv_canny_hough`, `lab_canny_hough`, `hsv_hough_cluster`, `lab_hough_cluster`, `hsv_hough_ransac`, `lab_hough_ransac`, and `hybrid_region_evidence`, each callable with `raw`, `clahe`, or `gamma`
- CLI: `railguard inspect-lines` exports ranked candidates, an optional overlay, and an optional paint debug mask; `railguard benchmark-lines` computes line distance, acute angle error, projected overlap, abstain rate, and runtime against independent manual L/E endpoint fixtures
- Reproduction commands: `.\.venv\Scripts\python.exe -m pytest -q`; `.\.venv\Scripts\python.exe -m compileall -q railguard app.py`; `.\.venv\Scripts\python.exe -m railguard inspect-lines --help`; `.\.venv\Scripts\python.exe -m railguard benchmark-lines --help`
- Verification: Pytest reported `87 passed in 5.44s`; compileall produced no stdout; both CLI help commands printed their option lists and examples. All four commands exited 0.
- Synthetic coverage: 40% RANSAC outliers and seed reproducibility; horizontal/diagonal/near-vertical models; thick-stripe convention; rejection of both borders of one stripe; relative bright/dark evidence; broken/occluded strokes; resolution scaling; train-reflection regression; explicit `uncertain_no_platform_edge`; inspect/benchmark command execution
- Real datasets/images used: none. RailEye3D RGB data is absent, so there is no real-data calibration, accuracy, probability confidence, or operational-safety claim.
- Known limitations: RGB region cues remain station- and camera-dependent; a narrow dark structural line is retained only as a penalized synthetic fallback and always requires review; method selection must be frozen on authorized validation data before any held-out evaluation
- Commit: no commit created per task constraint

## P3 — M3: Person Detectors, Baselines & Footpoint Localization (2026-11-12)

- Status: `IMPLEMENTED_NOT_VALIDATED_ON_REAL_DATA`; implementation verification: `DONE_SYNTHETIC`
- Files changed: `railguard/classic_detectors.py`, `railguard/detection.py`, `railguard/foot_refinement.py`, `railguard/pipeline.py`, `railguard/metrics.py`, `railguard/evaluate.py`, `railguard/cli.py`, `tests/test_detectors_m3.py`, `docs/PROGRESS.md`
- Methods implemented: frame-level-only B0 always-unsafe; B1 OpenCV default-people HOG+SVM with bounded margin confidence and NMS; B2 MOG2 sequence-only blobs with warmup, learning rate, morphology, area/aspect filters and large-train/camera-shake/moving-shadow flags; configurable/injectable YOLO detection and pose adapters; COCO ankle confidence handling; explicit one-ankle, two-ankle, `abstain`, and `bbox_fallback` outcomes; optional bounded edge/foreground contact refinement
- Comparison protocol: pose detections expose both `bbox_bottom` and `pose_ankle` on the same pose weights for footpoint ablation; detection-vs-pose weights are labeled an end-to-end cross-weight comparison
- Failure taxonomy: `occlusion`, `small_person`, `motion_blur`, `near_train_door`, `missing_feet`, `illumination_extreme`, `camera_tilt`
- CLI: `railguard benchmark-people` writes `detector_metrics.csv`, `benchmark_people.json`, and `failure_examples/taxonomy.json`; mock/synthetic mode performs no model download; absent real data is `BLOCKED_REAL_DATA`
- Reproduction commands: `.\.venv\Scripts\python.exe -m pytest -q`; `.\.venv\Scripts\python.exe -m compileall -q railguard app.py`; `.\.venv\Scripts\python.exe -m railguard benchmark-people --help`
- Verification: Pytest reported `102 passed in 5.63s`; compileall produced no stdout; benchmark help printed options and synthetic/mock/manifest examples. All three commands exited 0.
- Real datasets/images used: none. RailEye3D RGB data is absent; no real-data detector accuracy, precision, recall, mAP, speed, or footpoint benefit is claimed.
- Known limitations: HOG SVM margins are ranking scores, not calibrated probabilities; MOG2 is invalid for isolated or mixed-camera frames; refinement remains disabled unless explicitly requested and has not demonstrated a real-data benefit; model weights were neither downloaded nor run
- Commit: no commit created per task constraint

## P4 — M4: Violation Rules, Beyond-E Semantics & Uncertainty (2026-11-12)

- Status: `IMPLEMENTED_NOT_VALIDATED_ON_REAL_DATA`; implementation verification: `DONE_SYNTHETIC`
- Files changed: `railguard/geometry.py`, `railguard/calibration.py`, `railguard/pipeline.py`, `railguard/evaluate.py`, `railguard/cli.py`, `railguard/zone_editor.py`, `app.py`, `configs/example_manual.yaml`, `tests/test_m4_decisions.py`, `docs/PROGRESS.md`
- Geometry: L, E, finite zone Z, and `unsafe_side: beyond_platform_edge` are explicit. Signed perpendicular distances are oriented from L toward E and beyond E using the calibrated line pair, without assuming increasing image y or depending on endpoint order. Pixel and image-diagonal-normalized tolerances are supported.
- Beyond-E invariant: an in-frame observed footpoint in E's unsafe half-plane is `DANGER/beyond_platform_edge` before the finite-Z test, so leaving the finite quadrilateral past E cannot produce SAFE. Points in Z, on L, and on E are also DANGER with distinct reason codes.
- Uncertainty: missing/invalid calibration, missing or low-confidence ankles, invalid/out-of-bounds points, lateral regions outside finite Z, near-image-edge observations without conclusive danger evidence, detector failure, and unconfirmed empty detector output remain `UNKNOWN` with machine-readable reasons.
- Explanations: every person record includes footpoint source, observed points, per-point signed distances to L/E, membership in Z, tri-state status, primary reason, and reason list. Legacy `intrusion: true|false|null` remains synchronized with `DANGER|SAFE|UNKNOWN`.
- Frame aggregation: shared `aggregate_frame` logic applies `DANGER > UNKNOWN > SAFE` in CLI, Streamlit, and evaluation. `no_detections` is UNKNOWN; only an explicitly complete source may use `confirmed_no_people` as SAFE. Evaluator frame metrics exclude UNKNOWN frames and report coverage/unknown counts rather than converting UNKNOWN to SAFE.
- Synthetic coverage: opposite-facing cameras; vertical and steep diagonal L/E; points in Z, on the safe side, on L within pixel/normalized tolerance, and beyond E; small ROI; invalid/crossing geometry; missing calibration; low-confidence ankles; out-of-bounds and near-edge observations; frame aggregation priority; detector failure; and `no_detections` versus `confirmed_no_people` metadata.
- Verification commands: `.\.venv\Scripts\python.exe -m pytest -q`; `.\.venv\Scripts\python.exe -m compileall -q railguard app.py`
- Verification result: Pytest reported `112 passed in 5.49s` (exit 0); compileall produced no stdout (exit 0).
- Real datasets/images used: none. No RailEye3D RGB or operational footage was used, so there is no real-data correctness, calibration, safety, accuracy, precision, recall, or F1 claim.
- Known limitations: the unsafe convention relies on correctly calibrated L/E semantics; automatic line proposals still require review; near-edge policy and thresholds require validation per camera; no temporal smoothing was added.
- Commit: no commit created per task constraint

## P5 — E1–E5: Scientific Experiment Protocol & Runner (2026-11-12)

- Status: `IMPLEMENTED_NOT_VALIDATED_ON_REAL_DATA`; implementation verification: `DONE_SYNTHETIC` with output data status `SYNTHETIC_FIXTURE_ONLY`
- Files changed: `railguard/experiments.py`, `railguard/cli.py`, `tests/test_experiments.py`, `docs/PROGRESS.md`
- Experiment families: E1 geometry/preprocessing, E2 detector baselines, E3 footpoint choices, E4 component ablations, and E5 frozen cross-station transfer. Each family writes its named CSV plus a complete configuration, environment, split, metrics, predictions, failures, figures, and summary bundle.
- Protocol guards: all metric computation uses the held-out `test` split; calibration-only frames remain listed but are excluded from predictions and scores; configuration records `selection_split: val` and `test_set_tuning: false`; E5 hashes frozen camera YAML files before scoring; missing or unscorable real manifests remain `BLOCKED_REAL_DATA`.
- Metrics: unsafe TP/FP/FN/TN and P/R/F1; end-to-end unsafe recall with missed and UNKNOWN unsafe persons counted as FN; separate matched/determined conditional metrics; frame alert P/R/F1; false alarms; missed unsafe; UNKNOWN rate; detection coverage; runtime/FPS; supplied-box IoU and mAP@0.5 fields; explicit B0 frame-only and MOG2 ordered-video scope notes.
- CLI: `railguard run-experiment --experiment {E1,E2,E3,E4,E5,all} [--manifest ...] [--synthetic] [--out-dir ...]`. Synthetic mode neither loads weights nor downloads models and labels every artifact `SYNTHETIC_FIXTURE_ONLY`.
- Real datasets/images used: none. The synthetic fixture verifies contracts and metric arithmetic only; it is not RailEye3D and provides no evidence about method accuracy, generalization, or operational safety.
- Reproduction commands: `.\.venv\Scripts\python.exe -m pytest -q`; `.\.venv\Scripts\python.exe -m compileall -q railguard app.py`; `.\.venv\Scripts\python.exe -m railguard run-experiment --help`
- Verification: Pytest reported `117 passed in 6.06s` (exit 0); compileall produced no stdout (exit 0); run-experiment help printed its options and examples (exit 0).
- Commit: no commit created per task constraint


## P6 — Minimalist Scientific Debug Demo UI (2026-11-12)
- Status: DONE_SYNTHETIC (implemented directly by Antigravity after Codex CLI hit usage limit at P6; browser visual inspection not run in headless environment)
- Files changed: `railguard/ui_state.py`, `railguard/web_editor.py`, `app.py`, `tests/test_app_state.py`, `docs/PROGRESS.md`
- Methods actually implemented: 6-step streamlined UI flow; fixed-size zoom/pan/Fit/Reset-view canvas (`railguard/web_editor.py`); undo/reset & confirmation invalidation on point edit (`CalibrationSessionState`); single-image disabling of B2 MOG2 and TemporalMedian; invalid/unconfirmed calibration guard; 8 scientific debug tabs (`Input`, `Preprocessing`, `Masks`, `Canny-Hough`, `RANSAC`, `Final Zone`, `People & Feet`, `Results`); PNG/JSON/CSV export; Top-N candidate selector
- Reproduction commands: `.\.venv\Scripts\python.exe -m pytest -q`; `.\.venv\Scripts\python.exe -m streamlit run app.py`
- Tests: `.\.venv\Scripts\python.exe -m pytest -q` → `121 passed in 6.30s` (exit 0); `ast.parse(app.py)` → `APP_SYNTAX_OK`
- Real datasets/images used (and permissions): none (`BLOCKED_REAL_DATA`)
- Artifacts: `railguard/ui_state.py`, `tests/test_app_state.py`
- Measured improvements/regressions: added 4 UI state/guard/export tests; 121 total tests pass
- Known failure cases: live Chrome/Edge manual zoom/pan interaction not executed in automated headless run
- Blockers / next step: proceed to P7 (integration test, documentation, final QA)
- Commit: no commit created per task constraint


## P7 — QA, Documentation, Demo & Handoff (2026-11-12)
- Status: DONE_SYNTHETIC (real-data benchmarks `NOT_RUN` due to `BLOCKED_REAL_DATA`; implemented by Antigravity after Codex CLI hit usage limit)
- Files changed: `tests/test_integration_p7.py`, `docs/METHODOLOGY.md`, `docs/EXPERIMENTS.md`, `docs/PROGRESS.md`
- Methods actually implemented: offline stationary-camera multi-frame integration test (`temporal_median` -> `clahe` -> `hybrid_region_evidence` -> `analyze` -> `aggregate_frame` -> `E4` bundle); methodology & experiments documentation with honest `implemented / experimental / not run` labels
- Reproduction commands: `.\.venv\Scripts\python.exe -m pytest -q`; `.\.venv\Scripts\python.exe -m compileall -q railguard`; `.\.venv\Scripts\python.exe -m railguard --help`
- Tests: `.\.venv\Scripts\python.exe -m pytest -q` → `122 passed` (verified below)
- Real datasets/images used (and permissions): none (`BLOCKED_REAL_DATA`)
- Artifacts: `docs/METHODOLOGY.md`, `docs/EXPERIMENTS.md`, `docs/PROGRESS.md`, `tests/test_integration_p7.py`
- Measured improvements/regressions: end-to-end offline synthetic test suite expanded from 41 baseline tests to 122 passing tests
- Known failure cases: real RailEye3D RGB evaluation and live Chrome/Edge manual UI verification remain unrun without dataset/display
- Blockers / next step: place authorized RailEye3D data in `data/`, run `railguard audit`, predeclare calibration-only frames, freeze per-camera YAMLs, and run `railguard run-experiment --experiment all`
- Commit: no commit created per task constraint

## P8 — Migration to HTML5/Canvas RailGuard Studio & Pipeline Refinement (2026-11-13)
- Status: DONE_SYNTHETIC
- Files changed: `railguard/server.py`, `railguard/studio/index.html`, `railguard/cli.py`, `railguard/boundaries.py`, `railguard/detection.py`, `railguard/ui_state.py`, `railguard/zone_editor.py`, `railguard/pipeline.py`, `pyproject.toml`, `requirements-full.txt`, `README.md`, `WORKFLOW.md`, `docs/IMPLEMENTATION_NOTES.md`, `docs/PROGRESS.md` (removed `app.py`, `railguard/web_editor.py`, and loose root screenshots; moved 16 intermediate debug images into `outputs/debug/`)
- Methods actually implemented:
  - HTML5/Canvas **RailGuard Studio** (`railguard/server.py` + `railguard/studio/index.html`) replacing Streamlit (`app.py`, `railguard/web_editor.py`), with bounded zoom/pan/Fit controls, explicit image upload state resets, auto-detect lock mode, and unlockable manual 4-point (`L0`, `L1`, `E0`, `E1`) adjustment
  - `railguard studio` CLI subcommand (`--host 127.0.0.1`, `--port 8502`)
  - M2 refinements in `boundaries.py`: vanishing-point perspective alignment across near/mid/far edge evidence weighted toward foreground points, LAB `b >= 142` thresholding, ceiling-rafter suppression, and viewport-edge intersection
  - M3 Crop-Pose tiled option exposed cleanly in Studio for distant figures
- Reproduction commands: `.\.venv\Scripts\python.exe -m railguard studio --port 8502`; `.\.venv\Scripts\python.exe -m pytest -q`; `.\.venv\Scripts\python.exe -m compileall -q railguard`
- Tests: `.\.venv\Scripts\python.exe -m pytest -q` → `122 passed` (verified alongside `compileall -q railguard` and `railguard studio --help`)
- Commit: no commit created per task constraint

## P9 — Orientation-independent corridor, explicit Crop-Pose, OpenCV 5 & server hardening (2026-11-14)
- Status: DONE_SYNTHETIC (no real RailEye3D data; `BLOCKED_REAL_DATA` unchanged)
- Files changed: `railguard/geometry.py`, `railguard/pipeline.py`, `railguard/detection.py`, `railguard/server.py`, `railguard/boundaries.py`, `railguard/ui_state.py`, `railguard/classic_detectors.py`, `pyproject.toml`, `README.md`, `configs/README.md`, `docs/IMPLEMENTATION_NOTES.md`, `tests/conftest.py`, `tests/test_app_state.py`, `tests/test_detectors_m3.py`, `tests/test_regressions_geometry_pose_server.py`
- Bugs fixed:
  - **M4 corridor:** `corridor_polygon` parameterised lines as x = f(y), so with horizontal L/E (side-view cameras) points on the safe side were classified `DANGER/inside_hazard_zone`. The corridor is now the intersection of two half-planes (E side of L, L side of E) via signed distances (`HazardZone.in_corridor`), and the drawn polygon is a Sutherland–Hodgman clip of the image rectangle. Decisions are identical under 90° rotations.
  - **M3 Crop-Pose:** was enabled implicitly whenever `yolo11n.pt` existed in the working directory, discarded pose-only people, and took the first person in a crop. It is now opt-in (`enable_crop_pose`, `crop_pose_det_weights`, injectable `det_model`), keeps all direct pose detections, uses one-to-one IoU matching, selects the crop person by IoU with the target box, and records `crop_pose`/`crop_pose_det_weights` in `last_config` and person metadata. CLI/evaluation default to plain pose; Studio enables Crop-Pose explicitly.
  - **OpenCV 5:** `HoughLinesP` output reshaped to `(N, 4)`; B1 HOG+SVM raises a clear `RuntimeError` when `cv2.HOGDescriptor` is absent, and its tests skip.
  - **Studio server:** `ThreadingHTTPServer` (matching docs), YOLO models cached per process with a lock, wildcard CORS removed, JSON content type and 64 MB body limit enforced, undecodable images rejected with a clear error.
  - **Packaging/docs:** `studio/*.html` shipped as package data; removed nonexistent `demo` extra; corrected endpoint names and the stale Streamlit reference.
- Tests: Python 3.13 + opencv-python 4.14 → `153 passed`; Python 3.14 + OpenCV 5.0 → `150 passed, 3 skipped` (HOG). The 31 new regression tests fail 17/31 on the previous code.
- Behaviour change to report: with Crop-Pose now off by default outside Studio, any earlier pose run made from a directory containing `yolo11n.pt` used a different (det-box) condition and is not comparable.
