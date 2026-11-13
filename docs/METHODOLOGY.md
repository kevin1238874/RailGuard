# RailGuard Methodology (CS406)

> **Status note:** Methods below are implemented and verified on synthetic fixtures and mocked detectors. Real RailEye3D RGB evaluation is `BLOCKED_REAL_DATA` until authorized images/annotations are provided in `data/`.

## Pipeline Overview

```text
RGB Frame / Fixed-Camera Sequence
  ├─ M1 Preprocessing (branch Z for boundaries, branch P for detectors)
  ├─ M2 Boundary Estimation (L: safety stripe edge, E: platform edge, Z: hazard strip)
  ├─ M3 Person Detection & Footpoint Localization (B0/B1/B2/B3/A + optional B refinement)
  └─ M4 Tri-State Spatial Decision (DANGER / SAFE / UNKNOWN + reason codes)
```

## Module Status

| Module | Method | Status | Reference |
|---|---|---|---|
| M1 | `identity`, `gaussian_blur`, `median_blur`, `hist_eq` (Y/L only), `clahe` (Y/L only), `gamma` (LUT) | Implemented (synthetic-verified) | Zuiderveld (1994) CLAHE; Gonzalez & Woods |
| M1 | `TemporalMedianBackground` (same-camera sequences only) | Implemented (synthetic-verified) | — |
| M2 | HSV/LAB masks, Canny, HoughLinesP, `line_clustering`, 2-point RANSAC (`fit_line_ransac`), two-sided E region evidence | Implemented (synthetic-verified) | Canny (1986); Matas et al. (2000) PPHT; Fischler & Bolles (1981) RANSAC |
| M3 | B0 Always-unsafe (frame-level), B1 HOG+SVM, B2 MOG2 (sequence-only) | Implemented (synthetic-verified) | Dalal & Triggs (2005) HOG; Zivkovic (2004) MOG2 |
| M3 | B3 YOLO11 Detection, A YOLO11 Pose (COCO ankles 15,16), `abstain` vs `bbox_fallback` | Implemented (mock-verified; real weights `NOT_RUN` on RailEye3D) | Ultralytics YOLO11 |
| M3 | B Bounded local footpoint refinement (`refine_foot_points`) | Experimental (synthetic-verified, off by default) | — |
| M4 | Orientation-independent signed-side rules, beyond-E `DANGER` invariant, `UNKNOWN` abstention | Implemented (synthetic-verified) | — |

## Key Invariants
1. **Safety stripe convention (`L`):** `safe_side_edge` (default) vs `stripe_center` (ablation).
2. **Beyond-`E` rule:** A footpoint past platform edge `E` toward the track is always `DANGER` (`beyond_platform_edge`), never `SAFE`.
3. **Uncertainty rule:** Missing calibration, invalid keypoints (`abstain`), or occluded/unobservable regions yield `UNKNOWN`, never `SAFE`.
