# Architecture and implementation notes

- `raileye.py`: strict official-format CSV parser and sequence/frame manifest. Record alignment is positional, not by object ID.
- `geometry.py` + `config.py`: normalized/pixel L and E lines; interpolated polygon Z. Coordinates are defined in the input image's native resolution. No metric camera calibration is assumed.
- `boundaries.py`: optional HSV yellow/white paint masks, Canny edges, Hough segments, orientation-independent consensus fit (handles steep and near-vertical lines). Returns an *uncertain* status rather than inventing a zone if a sufficiently parallel candidate pair is not found.
- `detection.py`: lazy Ultralytics YOLO dependency; COCO person class 0; bbox-bottom-center or ankle (COCO indices 15 and 16) foot references, with explicit fallback.
- `pipeline.py`: geometric classification with tri-state `True`/`False`/`None` and OpenCV visualization.
- `splitting.py`, `evaluate.py`, `metrics.py`: sequence-disjoint test split and IoU matching; false-positive/false-negative accounting and CSV error export.
- `export_yolo.py`: optional **person detector** training export; frame-level negatives retained. No pose-network fine-tuning is provided.
- `zone_editor.py`: pure, tested normalization/validation and guarded per-camera YAML persistence.
- `web_editor.py`: Streamlit components v2 DOM/SVG endpoint placement and dragging. Coordinates return as one-shot events. Uploaded image data is used only as an image source, not rendered as HTML.
- `app.py`: four-step image upload → automatic provisional zone suggestion or YAML import → direct correction and explicit approval → YAML download/local save → real YOLO inference (not mocked). Geometry edits reset approval and stale predictions.

> Model downloads and original image downloads are deliberately **not** performed by unit tests. The included synthetic PNGs are created only after running the smoke-test generator and have no scientific validity as a benchmark.
