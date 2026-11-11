# Reproducible research protocol (CS406)

## Experimental question

Does estimating the location of a person's feet from pose keypoints improve unsafe-area recognition over the simple bounding-box bottom-center heuristic on RailEye3D platform images?

## Data selection, splits and leakage

1. Obtain original RGB frames from the dataset authors under the applicable licence.
2. Parse `bboxes` and `params` by matching sequence and frame, then **pairing rows in the same order**.
3. Audit frame count, sequence IDs, class balance, unsafe fraction and image resolutions before splitting.
4. Use **whole-sequence** train/val/test partitions and save the resulting `splits.csv` before fitting any thresholds. Do not let frames from one sequence enter multiple subsets.
5. Calibrate a single L/E geometry per camera using separate calibration frames, **not** by optimizing over test labels or individual test frames. Freeze calibration files before test scoring.
6. If feasible, run an additional cross-station evaluation in which all sequences from a held-out station are test-only. Describe any overlap of camera/viewpoints across splits.

## Comparison table template

| Experiment | Detector | Foot-point rule | Zone | Test precision | Test recall | Test F1 |
| --- | --- | --- | --- | --- | --- | --- |
| GT-oracle diagnosis | GT boxes | bbox bottom center | manual | TBD | TBD | TBD |
| Baseline | YOLO detection | bbox bottom center | manual | TBD | TBD | TBD |
| Pose | YOLO pose | valid ankles, abstain | manual | TBD | TBD | TBD |
| Ablation | YOLO pose | ankles, bbox fallback | manual | TBD | TBD | TBD |
| Ablation | YOLO detection | bbox bottom center | automatic heuristic | TBD | TBD | TBD |

`TBD` is not a reported result. The oracle uses ground-truth boxes **at inference time** to isolate geometric errors. Never compare its result as though it were an independently deployed detector.

## Metric conventions

- Match predictions to annotated persons **one-to-one by descending IoU**, threshold default 0.5.
- A matched GT unsafe with DANGER → TP; unsafe with SAFE/UNKNOWN → FN.
- A matched GT safe with DANGER → FP; safe with SAFE → TN. Matched safe with UNKNOWN is excluded from the conditional classification metric.
- An unmatched GT unsafe → FN; an unmatched prediction with DANGER → FP.
- Undetected safe people and UNKNOWN safe detections do not create artificial TN.
- Also export the **conditional** classification metric on successfully matched, determined people, but do not present it as the end-to-end score.
- Metric `n` is the number of scored decisions, not the number of complete frame/person labels; interpretation requires coverage counters.
- Report true positives, false positives, false negatives, precision, recall, F1, **frame-level alert metrics**, detector coverage and number of abstentions along with the confusion counts. Also compare bbox and ankle rules using the same pose-capable weights to control for model differences.

## Failure analysis

Inspect `failure_cases.csv` and saved JPG previews for a diverse set of errors: occlusion, feet outside the image, unusual shoes and clothing, darkness, perspective distortion, truncated bodies, near-line cases, crowd overlaps, and inaccurate automatic line detection. Include at least three concrete sample cases with original frame IDs and explanations, if the obtained images permit it.

## Hyperparameter logging

For each experiment record: Python/environment, dataset version, train/val/test sequences, YOLO model and weights hash, `confidence`, pose-keypoint threshold, pose missing-keypoint fallback, bbox IoU threshold, L/E geometry files, OS/GPU/CPU, inference image sizes, seeds if training, and elapsed time/hardware.

## Known limitations

Camera-space polygons do not establish a physical distance in metres. The `beyond_safety_line` annotation and a hand-drawn polygon may differ on borderline persons. The Hough-based automatic zone is an experimental heuristic, not a robust platform-edge detector. The software is **not** intended to trigger safety-critical actions.
