# Architecture and implementation notes

- `raileye.py`: strict official-format CSV parser and sequence/frame manifest. Record alignment is positional, not by object ID.
- `geometry.py` + `config.py`: normalized/pixel L and E lines; interpolated polygon Z. Coordinates are defined in the input image's native resolution. No metric camera calibration is assumed.
- `boundaries.py`: automatic platform line estimation via:
  - `lab_canny_hough` (LAB luminance/yellow-channel filtering with `b >= 142`, Canny edges, Hough segments, ceiling/roof rafter suppression, and vanishing-point perspective alignment across multiple edge points weighted toward foreground points), and
  - `hybrid_region_evidence` (region brightness drop + RANSAC consensus fitting with vanishing-point intersection).
  - Returns an *uncertain* status rather than inventing a zone if a valid pair is not found.
- `detection.py`: lazy Ultralytics YOLO dependency; COCO person class 0; bbox-bottom-center or ankle (COCO indices 15 and 16) foot references, with explicit fallback, plus an opt-in detect→crop→pose branch (`enable_crop_pose=True`, explicit `crop_pose_det_weights`) that keeps all direct pose detections and only adds people the pose model missed.
- `pipeline.py`: geometric classification with tri-state `True`/`False`/`None` and OpenCV visualization.
- `splitting.py`, `evaluate.py`, `metrics.py`: fixed-prefix pretrained val/test protocol, optional sequence-disjoint fine-tuning split, explicit calibration-only exclusions, IoU matching, false-positive/false-negative accounting, and CSV error export.
- `export_yolo.py`: optional **person detector** training export; frame-level negatives retained. No pose-network fine-tuning is provided.
- `zone_editor.py`: pure, tested normalization/validation and guarded per-camera YAML persistence.
- `ui_state.py`: framework-agnostic workspace state machine managing image uploads, auto-proposal lock states, manual point editing overrides, and stale prediction resets.
- `server.py` + `studio/index.html`: pure Python `ThreadingHTTPServer` (`railguard studio` / `python -m railguard.server`) serving the HTML5/Canvas **RailGuard Studio** frontend and same-origin JSON endpoints (`/api/auto-zone`, `/api/run-pipeline`; no CORS, 64 MB body limit, YOLO models cached per process). Replaces legacy Streamlit UI (`app.py`, `web_editor.py`).

> Model downloads and original image downloads are deliberately **not** performed by unit tests. The included synthetic PNGs are created only after running the smoke-test generator and have no scientific validity as a benchmark.


## Interactive RailGuard Studio & conservative auto-boundary estimation

- `railguard/studio/index.html` provides a dark-theme 3-column HTML5 Canvas workspace with bounded zoom/pan/Fit controls, explicit image upload state resets, auto-detect lock mode (preventing accidental edits to automated L/E lines), and unlockable manual 4-point (`L0`, `L1`, `E0`, `E1`) dragging.
- Auto-boundary algorithms extend detected near/mid/far edge evidence to full viewport boundaries via vanishing-point perspective alignment and multi-point weighted interpolation so partial corner gaps do not leave unclipped danger polygons.
- Candidate boundaries reject dark-track/bright-train reflections and overhead ceiling rafters, and require human verification or manual adjustment when an automatic method fails rather than emitting fabricated overlays.
