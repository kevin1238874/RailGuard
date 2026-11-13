# RailGuard Experiments (E1–E5)

> **Data status:** `BLOCKED_REAL_DATA` — no authorized RailEye3D RGB dataset is present in `data/`. No real-data accuracy, precision, recall, or F1 is reported here.

## Experiment Suite

| ID | Question | Output CSV | Real-Data Status |
|---|---|---|---|
| **E1 — Geometry** | Impact of M1 (Raw/CLAHE/Gamma/TemporalMedian) and M2 (HSV/LAB, Hough/RANSAC) on L/E estimation | `e1_geometry.csv` | `NOT_RUN` (`BLOCKED_REAL_DATA`) |
| **E2 — Detectors** | Classical baselines (B0, B1 HOG+SVM, B2 MOG2) vs YOLO11n det/pose | `e2_detectors.csv` | `NOT_RUN` (`BLOCKED_REAL_DATA`) |
| **E3 — Footpoint** | BBox bottom-center vs pose ankles vs bounded refinement | `e3_footpoints.csv` | `NOT_RUN` (`BLOCKED_REAL_DATA`) |
| **E4 — Ablation** | Component-wise contribution on identical split | `e4_ablation.csv` | `NOT_RUN` (`BLOCKED_REAL_DATA`) |
| **E5 — Generalization** | Tuning on `red-00-00*` (Stockerau) transferred to frozen `red-04-01*` (Fels) | `e5_cross_station.csv` | `NOT_RUN` (`BLOCKED_REAL_DATA`) |

## Reproduction Commands (Offline / Synthetic Smoke)

```powershell
.\.venv\Scripts\python.exe -m railguard audit --manifest data\processed\manifest.csv --out outputs\data_audit.json --csv-out outputs\data_audit.csv
.\.venv\Scripts\python.exe -m railguard run-experiment --experiment all --synthetic --out-dir outputs\experiments
.\.venv\Scripts\python.exe -m pytest -q
```
