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


## P9 — Codex CLI Co-Designed M1–M4 Accuracy, Safety & Runtime Hardening (2026-11-14)
- Status: `DONE_VERIFIED_ON_4_USER_BENCHMARK_IMAGES` (Codex CLI `gpt-5.6-sol` review: `APPROVED` in `outputs/agent_runs/codex_workflow/round4_codex_final_review.md`; RailEye3D dataset remains absent in `data/`)
- Files changed: `railguard/boundaries.py`, `railguard/detection.py`, `railguard/foot_refinement.py`, `railguard/pipeline.py`, `railguard/server.py`, `tests/test_boundaries_m2.py`, `tests/test_detectors_m3.py`, `tests/test_m4_decisions.py`, `tests/test_studio_server.py`, `docs/PROGRESS.md`
- Methods & safety invariants implemented across Steps 1–6:
  - **M2 (`railguard/boundaries.py`)**: pre-VP orientation/trench verification + canonical vanishing-point horizon clipping (`_clip_zone_at_horizon`) preventing above-horizon X-crossings; overlapping clipped-span restriction in `_zone_evidence`; vectorized station sampling in `_evaluate_paint_band`, `_evaluate_trench_drop`, `_has_interior_trench`, and `_zone_evidence`.
  - **M3 (`railguard/detection.py`)**: cwd-independent weight resolution (`_resolve_weight_path`), best-IoU 1-to-1 det↔pose matching (`_match_det_and_pose`), target-IoU crop selection (`best_iou > 0.0`) returning empty keypoints when unmatched, accurate `crop_recovered` / `detector_partial` tracking, and offline `mock` guard preventing unintended weight loading.
  - **M3/M4 (`railguard/foot_refinement.py`, `railguard/pipeline.py`)**: structured `FootEvidence` with train-glass reflection suppression (`rejected_reflection` filtered in `analyze()` while retaining real ankle points), coherent-skeleton checks escalating ambiguous SAFE cases to `UNKNOWN (ambiguous_footpoint_evidence)`, and completeness-aware `aggregate_frame` priority (`DANGER > person UNKNOWN > detector partial/failure > calibration invalid > SAFE`).
  - **Studio parity (`railguard/server.py`)**: refactored `handle_auto_zone` and `handle_run_pipeline` endpoints supporting `preproc`/`preprocess`, `enable_crop_pose`, `refine_footpoints`, and `detector_coverage` diagnostics.
- Benchmark verification (`outputs/agent_runs/step6_final/step6_final_summary.md` & `step6_final_report.json`):
  - **M1 × M2 (84 runs across 4 benchmark images):** `75 accepted`, `9 abstained`, **`0 inverted`**, **`12/12` valid M1×M2 cells**, `59` golden-tolerance passes.
  - **M3 × M4 (24 runs across `yolo11n_bbox`, `yolo11n_pose_crop`, `yolo11n_pose_nocrop` × `refine={False,True}`):** **`0 False SAFE` frames** (`img3`: `2/2` `SAFE`/`DANGER` with reflection rejected; `img4` `yolo11n_pose_crop`: `3/3` `SAFE`/`SAFE`/`DANGER`; `img4` `yolo11n_pose_nocrop` partial `1/3`: escalated `UNKNOWN`).
  - **Runtime SLA:** `hybrid_region_evidence` `p50 = 3253.5 ms` (`<= 4000 ms`), `max = 4595.4 ms` (`<= 5000 ms`); non-hybrid `p50 = 1433.9 ms` (`<= 2000 ms`).
- Reproduction commands: `.\.venv\Scripts\pytest.exe -q`; `.\.venv\Scripts\python.exe outputs/agent_runs/eval_step.py step6_final`; `.\.venv\Scripts\python.exe -m railguard studio --port 8502`
- Artifacts: `outputs/agent_runs/step6_final/`, `outputs/agent_runs/codex_workflow/round1_codex_diagnosis.md`, `round2_codex_signoff.md`, `round3_codex_final_review.md`, `round4_codex_final_review.md`

## P10 — Sunlight Glare Preprocessing (M1), Glare-Verified M3 Recovery & Studio Isolation (2026-11-14)
- Status: `DONE_VERIFIED`
- Files changed: `railguard/preprocessing.py`, `railguard/detection.py`, `railguard/server.py`, `railguard/studio/index.html`, `tests/test_preprocessing.py`, `tests/test_detectors_m3.py`, `tests/test_studio_server.py`, `docs/PROGRESS.md`
- Methods implemented:
  - **M1 (`railguard/preprocessing.py`)**: added `highlight_compress` (alias `glare_suppress`) for LAB highlight compression + local detail boost + partial CLAHE blend, and `exposure_fusion` (alias `mertens_fusion`) with percentile scaling and flat-image guard.
  - **M3 (`railguard/detection.py`)**: added `glare_recovery: bool = False` on `YOLODetector` to rescan high-luminance glare regions (`highlight_compress` full frame + up to 2 compact bright windows `_extract_glare_rois`), filtering duplicates (`IoU >= 0.35` or containment `>= 0.55`), requiring overlap with blown glare (`bright_frac >= 0.15`), and enforcing pose-verified ankles inside bbox (`_ankles_within_bbox`) or high confidence to block sunlit metal/kiosk hallucinations.
  - **Studio & Server Isolation (`railguard/server.py`, `railguard/studio/index.html`)**: added `Tiền xử lý chống lóa nắng (M1)` selector (`auto` default, `highlight_compress`, `clahe`, `gamma`, `exposure_fusion`, `raw`); `_M3_GLARE_ONLY_PREPROCS` keeps M2 `/api/auto-zone` on `raw` when `auto`/`highlight_compress`/`exposure_fusion` is active while `auto` enables `glare_recovery=True` in M3 `/api/run-pipeline`.
- Verification: `.\.venv\Scripts\pytest.exe -q` → `143 passed`; `eval_step.py step6_final` → `84 runs`, `0 inverted`, `12/12` valid M1×M2 cells, `0 False SAFE`, `hybrid p50 = 3165.8 ms`; `glare_recovery=True` verified on all 4 benchmark images (`img1=0`, `img2=0`, `img3=2`, `img4=3` in both POSE & BBOX).

## P11 — RailEye3D Real-Dataset Evaluation, Backlit Cascade & M2 Anti-Inversion Hardening (2026-11-14)
- Status: `DONE_VERIFIED` (Codex CLI `gpt-5.6-sol` Round-2 review: **`APPROVE`** in `outputs/agent_runs/raileye_workflow/round2_codex_signoff.md`)
- Files changed: `railguard/boundaries.py`, `railguard/detection.py`, `railguard/foot_refinement.py`, `railguard/geometry.py`, `railguard/line_models.py`, `railguard/pipeline.py`, `railguard/preprocessing.py`, `railguard/server.py`, `railguard/studio/index.html`, `tests/test_boundaries_m2.py`, `tests/test_detectors_m3.py`, `tests/test_geometry.py`, `tests/test_m4_decisions.py`, `tests/test_preprocessing.py`, `tests/test_studio_server.py`, `docs/PROGRESS.md`
- Methods implemented across Steps 1–6:
  - **M3 (`railguard/detection.py`)**: localized glare-stress ROI extraction (`_extract_glare_rois`), exposure variants (`raw`, `gamma08`, `gamma06`, `highlight_compress`), and bbox-first -> glare ROI -> batched crop-pose cascade (`cascade_mode="bbox_first"`).
  - **M2 (`railguard/boundaries.py`)**: resolution-bounded line-pool capping (`_cap_line_pool(18, 20)` when `down_scale < 1.0`) with steep lower-edge preservation, fast-path early exit in `_evaluate_paint_band`, sunlit concrete coping recovery (`clear_curb_exception`), outward trench-drop metrics (`o_ext_d`, `med_ext`), and rejection of train-window reflections and asymmetric cross-trench traps (`_has_interior_trench`).
  - **M4 (`railguard/foot_refinement.py`, `railguard/pipeline.py`)**: per-frame cached `gray` and `edges` reused across person foot-refinement assessments.
  - **Studio (`railguard/server.py`)**: thread-safe detector/model instance caching (`_DETECTOR_CACHE` / `_get_detector`) keyed by effective config, auto-routing `preproc="auto"` to `imgsz=960`, `glare_recovery=True`, `glare_exposure="gamma06"`, `cascade_mode="bbox_first"`, and surfacing `glare_rois` & cascade metadata in `/api/run-pipeline`.
- Verification:
  - Unit suite: `.\.venv\Scripts\python.exe -m pytest -q` → `147 passed in 6.94s`.
  - Golden 4-image regression (`eval_step.py step6_final_v2`): `84 M2 runs`, `75 accepted`, `9 abstained`, **`0 inverted`**, `59` golden-tol passes, **`12/12` valid cells**, **`0 False SAFE`** (`hybrid p50/p95 = 2829.6 / 4019.2 ms`; `non_hybrid p50/p95 = 1370.1 / 2200.6 ms`).
  - RailEye3D `red-00-00-front-left` Holdout (`24` keyframes, `164` GT, `step6_holdout`): `cas_960_g06` `TP@35 = 97/164 (59.1%)` (`TP@50 = 95`, Prec `85.1%`, `bg=0, refl=0`), `GlareHi = 16/55` (vs baseline `pose_640_raw`: `84/164 = 51.2%`, `GlareHi = 11/55`), `Unsafe M4 = 26/27`, **`Safe->DNG = 0`**, Wall `p50/p95 = 917.2 / 1620.8 ms`.
  - RailEye3D `red-00-00-front-left` Full (`37` keyframes, `252` GT, `step6_all`, live `--m2`):
    - M2 `lab_canny_hough`: `37/37 tol_ok`, `abstained = 0`, **`inverted = 0`**, `med_max_err = 0.0289`, `p90_disp = 0.0257`, `p50/p95 = 631.1 / 902.5 ms`.
    - M2 `hybrid_region_evidence`: `37/37 tol_ok`, `abstained = 0`, **`inverted = 0`**, `med_max_err = 0.0273`, `p90_disp = 0.0225`, `p50/p95 = 922.8 / 1217.8 ms`.
    - M3+M4 `cas_960_g06`: `TP@35 = 154/252 (61.1%)` (`TP@50 = 146`, Prec `86.5%`, `bg=0, refl=0`), `GlareHi = 29/81` (vs baseline `pose_640_raw`: `135/252 = 53.6%`, `GlareHi = 19/81`), `Unsafe M4 = 39/41`, **`Safe->DNG = 0`**, Wall `p50/p95 = 610.2 / 869.6 ms`.
  - Live Studio daemon (`http://127.0.0.1:8502`): HTTP `GET /` → `200`, `POST /api/auto-zone` & `POST /api/run-pipeline` verified end-to-end on `img4_covered_3people.jpg` (`imgsz=960`, `glare_exposure=gamma06`, `cascade_mode=bbox_first`, `glare_rois=2`, `glare_recovered=1`, `frame.status=DANGER`).
- Commit: uncommitted per task constraint (`data/*` & `outputs/*` remain untracked).

## P12 — Train-on-Left Warm Granite Coping, Dark Station Floor & JPEG Ceiling-Above-VP Fix (2026-11-14)
- Status: `DONE_VERIFIED`
- Files changed: `railguard/boundaries.py`, `railguard/detection.py`, `railguard/preprocessing.py`, `railguard/server.py`, `railguard/studio/index.html`, `tests/test_boundaries_m2.py`, `tests/test_studio_server.py`, `docs/PROGRESS.md`
- Root causes fixed:
  - **Warm-stone background flood suppression (`railguard/boundaries.py` `paint_mask`)**: When `yellow` foreground fraction exceeds `0.23` across HSV or LAB on warm pinkish-beige coping stone, wide low-saturation/low-`b` background floods are suppressed while preserving saturated yellow or bright painted stripes.
  - **Dark-grey platform concourse floor (`railguard/boundaries.py` `_evaluate_paint_band`)**: `safe_bg_ok` distinguishes dark-grey platform floors (`safe_bg_abs_med >= 45.0`) from pitch-black track pits (`< 45.0`).
  - **Diverging cone guard & ceiling-above-VP Continuity (`railguard/boundaries.py` `estimate_zone`)**: Tightened steep upside-down diverging cone rejection (`* 0.85`), and refined the premature-crossing continuity check above `pre_vp_y` so a dark canopy/crowd gap immediately above VP (`sum(hits_seq[-3:]) < 2`) is not misclassified by bright white ceiling/wall panels higher up.
  - **Studio auto-resolution & M1 routing (`railguard/server.py`, `railguard/preprocessing.py`, `railguard/studio/index.html`)**: `preproc="auto"` routes `< 1280` frames to `imgsz=640, cascade_mode="dual_full"` and `>= 1280` to `imgsz=960, cascade_mode="bbox_first"`; `_pipeline_steps` supports bare `"gamma"` preset (`{"gamma": 0.65}`); `preprocSelect.onchange` avoids redundant `/api/auto-zone` triggers when effective M2 preproc (`raw`) is unchanged.
- Verification:
  - Unit suite: `.\.venv\Scripts\python.exe -m pytest -q` → `149 passed in 7.54s`.
  - Golden 4-image regression (`outputs/agent_runs/eval_step.py step7_final2`): `runs=84`, `accepted=75`, `abstained=9`, **`inverted=0`**, `passed_golden_tol=58`, **`cells_with_valid_method=12/12`**, **`False SAFE=0`**.
  - RailEye3D `red-00-00-front-left` Full M2 (`step7_final2_m2`, `37` keyframes): `lab_canny_hough` → `37/37 tol_ok`, `abstained=0`, `inverted=0`, `med_max_err=0.0289`, `p90_disp=0.0257`; `hybrid_region_evidence` → `37/37 tol_ok`, `abstained=0`, `inverted=0`, `med_max_err=0.0273`, `p90_disp=0.0225`.
  - Live Studio daemon (`http://127.0.0.1:8502`) smoke test on JPEG-encoded cropped train-on-left station photo (`scr[92:455, 412:627]`): `GET /` → `200`, `POST /api/auto-zone` (`lab_canny_hough` and `hybrid_region_evidence`, `preproc="auto"`) → `L=[0.754663, 1.0] -> [0.530739, 0.262431]`, `E=[0.119626, 1.0] -> [0.511749, 0.262431]` (`L` on right stripe, `E` on left gap); `POST /api/run-pipeline` (`yolo11n_pose`, `preproc="auto"`) → `frame.status='SAFE'`, `people=3`, `safe=3`, `danger=0`, `unknown=0`.
- Commit: uncommitted per task constraint (`data/*` & `outputs/*` remain untracked).

## Round 2B — Subway/Moscow Metro Platform Boundary Upgrade & Ablation (2026-11-14)
- Status: `DONE_VERIFIED`
- Files changed / added: `railguard/train_mask.py`, `railguard/boundaries.py`, `railguard/server.py`, `railguard/studio/index.html`, `tests/test_boundaries_m2.py`, `tests/test_train_mask.py`, `tests/test_studio_server.py`, `tests/test_subway_benchmark.py`, `outputs/agent_runs/subway_workflow/run_ablation.py`, `outputs/agent_runs/subway_workflow/round2_ablation.json`, `outputs/agent_runs/subway_workflow/round2_ablation.md`, `outputs/agent_runs/subway_workflow/round2_overlays/`, `docs/PROGRESS.md`
- Key changes:
  - **Lazy YOLO-Seg Challenger (`railguard/train_mask.py`, `railguard/boundaries.py`, `railguard/server.py`)**: `YoloSegTrainMaskProvider` (COCO `classes=[6]`, `retina_masks=True`) cached cleanly via `get_default_train_mask_provider()`; never imported or placed into `sys.modules` in `off` / `structural` default modes (`A1_no_torch=True`, `A2_no_torch=True`). Eroded train core suppresses interior train false lines while `train_boundary_confirmed` / `yolo_train_boundary_rescued` preserves valid platform/train seams.
  - **Structural & Moscow Metro Enhancements (`railguard/boundaries.py`)**: Color-role separation (`yellow` vs `white`), unbounded VP classification, stripe orientation consensus, repeated overhead roof-rib rejection (`moscow_4_overhead_ribbed_roof` `21/21` abstain), wide coping slab & dark-granite inset fallback (`_dark_inset_pass` + `dark_inset_glare_rescued`), and tightened false-pair/noise rejection (`dev_kievskaya` `7/7` abstain, `dev_arbatskaya` `0.0262` norm err).
- Verification:
  - Unit suite: `.\.venv\Scripts\python.exe -m pytest -q` → `159 passed in 7.91s`.
  - Golden 4-image regression (`outputs/agent_runs/eval_step.py round2b_final`): `runs=84`, `accepted=75`, `abstained=9`, **`inverted=0`**, **`passed_golden_tol=70`**, **`cells_with_valid_method=12/12`**, **`False SAFE=0`** (`hybrid p50/p95=3727.6/6165.4 ms`, `non_hybrid p50/p95=1883.0/3640.1 ms`).
  - RailEye3D `red-00-00-front-left` Full M2+M3+M4 (`outputs/agent_runs/eval_raileye.py round2b_final all --m2`, `37` keyframes, `252` GT):
    - `lab_canny_hough` (`structural`): **`37/37` tol_ok**, `abstained=0`, **`inverted=0`**, `med_max_err=0.0287`, `p90_disp=0.0278` (`p50/p95=717.1/960.9 ms`).
    - `hybrid_region_evidence` (`structural`): `34/37` tol_ok, `abstained=1`, **`inverted=0`**, `med_max_err=0.0162`, `p90_disp=0.3129` (`p50/p95=1029.5/1970.8 ms`).
  - Round 2B Approved Ablation (`outputs/agent_runs/subway_workflow/run_ablation.py --require-approved --with-stress --fresh`, 24 approved GT records / 280 matrix runs + 138 `SubwayDetectionAssistant` stress frames):
    - **A0 (Baseline)**: Golden 84 `58/75/0`, Critical 168 `top1=62, inv=21, false_acc=21`, Moscow dev/holdout/event top1 `7 / 1 / 11` (inversions `14 / 20 / 2`, dev `false_acc=3`).
    - **A1 (`off`)**: Golden 84 `72/77/0`, Critical 168 `top1=98, inv=17, false_acc=18`, Moscow dev/holdout/event top1 `10 / 1 / 11` (inversions `19 / 9 / 1`, `false_acc=0`).
    - **A2 (`structural`, default)**: Golden 84 `70/75/0`, Critical 168 `top1=114, inv=17, false_acc=0`, Moscow dev/holdout/event top1 `10 / 1 / 11` (inversions `19 / 9 / 1`, `false_acc=0`), `img3/raw/hsv_canny_hough` Rank 1 margin `+12.0847`, `moscow_1/raw/lab_canny_hough` Top-1, `moscow_4` `21/21` abstain, `dev_kievskaya` `7/7` abstain, stress `88/138` accepted (`p50/p95=1720.0/3076.1 ms`, determinism `True`). Runtime `p95` `0.9487×` A1 (`-191.7 ms`).
    - **A3 (`hybrid_yolo_seg`)**: Golden 84 `68/76/1`, Critical 168 `top1=112, top3=122, inv=8, false_acc=0`, Moscow dev/holdout/event top1 `10 / 1 / 11` (`inv=15/9/1`, overall precision `56.1%`, `pos_cov=75.0%` vs A2 `53.5%` / `80.2%`), reduces `moscow_2/raw` error `0.156 -> 0.095` and fixes `moscow_3/raw/hsv_canny_hough` (`0.5386 -> 0.0413`), stress `88/138` accepted (`16` frame diffs vs A2, `YOLO cold=676.2 ms`, warm `p50/p95=90.9/162.9 ms`, M2 `p50/p95=3044.3/8499.6 ms`, determinism `True`). Kept optional behind explicit `yolo_seg`/`hybrid_yolo_seg` selection (`structural` remains default due to Golden `inv=1` and `5.2pp` lower overall positive coverage).
    - **S1 (Reference SegNet)**: `104` ok / `6` 4K unsupported out of `138` stress frames (`cold=3481.5 ms`, warm `p50/p95=10931.1/16011.9 ms`).
  - Live Studio daemon (`http://127.0.0.1:8502`): `GET /` → `200`, `POST /api/auto-zone` tested on `moscow_1..4` across `off`/`structural`/`yolo_seg`/`hybrid_yolo_seg` (all `200`, `moscow_4` properly abstains in `structural` and falls back to `structural` when YOLO prior invalid), `POST /api/run-pipeline` on `moscow_1` → `200` (`DANGER`, `people=4, danger=2, safe=2`).

## Round 2C — Train-Side Pre-Suppression, E-Only Boundary Proposal & Two-Zone Foreground Anchors (2026-11-15)
- Status: `DONE_VERIFIED`
- Files changed / added: `railguard/boundaries.py`, `railguard/studio/index.html`, `tests/test_boundaries_m2.py`, `outputs/agent_runs/round2c_final/round2c_final_summary.md`, `outputs/agent_runs/subway_workflow/round2_ablation.md`, `outputs/agent_runs/subway_workflow/round2_ablation.json`, `docs/PROGRESS.md`
- Key changes (aligned with Codex contract):
  - **Idea 1 — YOLO Train-Side Pre-Suppression & E-Only Boundary Proposal (`railguard/boundaries.py`)**: Added `exclude_mask` pre-suppression in `_candidate_segments` (with extra safe-margin interior erosion `max(5, round(0.014 * d) | 1)` before zeroing Canny edges so platform-edge pixels near YOLO mask jitter stay intact), `_extract_train_boundary_candidates` silhouette RANSAC + narrow platformward edge-snap (`can_be_L=False, can_be_E=True`), and tightened `train_boundary_confirmed` (`e_train_boundary_p75_distance_px`, `e_train_fg_raw_overlap <= 0.30`). Preserves zero `torch`/`ultralytics` isolation in `off` and `structural` modes (`A1_no_torch=True`, `A2_no_torch=True`).
  - **Idea 2 — Two-Zone Mid + Near Foreground Anchors (`railguard/boundaries.py`)**: Added deferred 1-step IRLS weighted `foreground_refit` over `[0.45h, 0.96h]` in `_model_segments` (appended after primary `max_lines` slots and angle-clamped `<= 6.5 deg`), `_painted_stroke_candidates` foreground angular candidate (`applied_fg_angular` gated on `[0.35 deg, 5.0 deg]` and strict foreground residual improvement), and cleaned up `_vp_horizon_penalty` (`* 480.0`) + per-pair `seen_refined`/`rib_rejected_pairs` tracking in `estimate_zone`.
  - **Studio Stale Response Guard (`railguard/studio/index.html`)**: Added `autoZoneSeq` monotonically increasing request sequence check in `triggerAutoZone` so superseded `/api/auto-zone` responses never overwrite newer UI selections.
- Verification (post-final-edit):
  - Unit suite: `.\.venv\Scripts\python.exe -m pytest -q` → `161 passed in 11.94s`.
  - Golden 4-image regression (`outputs/agent_runs/eval_step.py round2c_final`): `runs=84`, `accepted=75`, `abstained=9`, **`inverted=0`**, **`passed_golden_tol=71`** (up from `70`), **`cells_with_valid_method=12/12`**, **`False SAFE=0`**.
  - RailEye3D `red-00-00-front-left` Full M2+M3+M4 (`outputs/agent_runs/eval_raileye.py round2c_final all --m2`, `37` keyframes, `252` GT):
    - `lab_canny_hough`: **`37/37` tol_ok**, `abstained=0`, **`inverted=0`**, `med_max_err=0.0287`, `p90_disp=0.0273` (`p50/p95=530.8/921.9 ms`).
    - `hybrid_region_evidence`: `33/37` tol_ok, `abstained=2`, **`inverted=0`**, `med_max_err=0.0252` (`p50/p95=787.0/1482.2 ms`).
  - Round 2C Approved Ablation (`outputs/agent_runs/subway_workflow/run_ablation.py --require-approved --with-stress --fresh`):
    - **A1 (`off`)**: Golden 84 `73/77/0`, Critical 168 `97/100/16/20`, All labeled `54.4% / 81.0% / 42`, `no_torch=True`.
    - **A2 (`structural`, default)**: Golden 84 **`71/75/0`**, Critical 168 **`115/118/16/0`**, Moscow dev/holdout/event `10 / 1 / 11` (`inv=18/7/1`), All labeled `54.0% / 80.2% / 42`, `p50/p95=1669.0/4353.9 ms` (`0.8063x` A1), `moscow_4` `21/21` abstain, `dev_kievskaya` `7/7` abstain, stress `86/138` accepted (`determinism=True`), `no_torch=True`.
    - **A3 (`hybrid_yolo_seg`)**: Golden 84 **`81/83/0`** (up from Round 2B `68/76/1`), Critical 168 **`126/135/7/0`** (up from `112/122/8/0`), Moscow dev/holdout/event `10 / 1 / 11` (`inv=13/7/1`, accepted `26/24/24`), All labeled **`57.4% / 82.9% / 28`** (up from `56.1% / 75.0% / 33`), `p50/p95=3093.1/9003.2 ms`, stress `87/138` accepted (`determinism=True`). All Spot Gates in `round2_ablation.md` passed (`golden_84_inv_eq_0=true`, `img3_hsv_margin=24.7139`).
  - Live Studio daemon (`http://127.0.0.1:8502`, `task-6003`): `GET /` → `200` (`autoZoneSeq` verified), `POST /api/auto-zone` smoke-tested on `moscow_3_train_left_orange_sill` (`raw` & `gamma` with `hybrid_yolo_seg` and `gamma` with `structural` — all `estimated_review_required`, `L ~ [0.88..0.91, 1.0] -> [0.55..0.57, ...]`, `E ~ [0.29, 1.0] -> [0.54..0.57, ...]`).




