# Calibration files

Create **one YAML file named exactly `<sequence>.yaml` per camera/sequence**. For example,
`red-00-00-front-left.yaml`. Do not reuse a template without inspecting the relevant scene.

`L` is the safety line; `E` is the platform edge. The quadrilateral between them is `Z`.
Click L left, L right, E left, E right with `railguard calibrate`, or edit normalized
coordinates manually in `example_manual.yaml` then save it under the sequence name.

**Important:** The official RailEye3D `beyond_safety_line` field is a boolean supervision
label; it is *not* a coordinate from which a camera's line can be recovered.

Without calibration, prediction/evaluation fails explicitly. `--auto-boundary` permits an
unvalidated OpenCV heuristic for visual experimentation but must be reported separately
from manual calibration in experiments.
