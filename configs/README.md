# Calibration files

Create **one YAML file named exactly `<sequence>.yaml` per camera/sequence**. For example,
`red-00-00-front-left.yaml`. Do not reuse a template without inspecting the relevant scene.

`L` is the safety line; `E` is the platform edge. The quadrilateral between them is `Z`.
With RailGuard Studio (`railguard studio`, then open `http://localhost:8502`) upload an image and either click **Auto Detect** and correct
the suggested endpoints by dragging, or select L0/L1/E0/E1 and click directly on
the image. Review the zone, explicitly confirm, and **Save to local configs/** or
**Download camera YAML**. You can also use `railguard calibrate` in an OpenCV
desktop window or manually edit a normalized YAML. Camera image views must match.

**Important:** The official RailEye3D `beyond_safety_line` field is a boolean supervision
label; it is *not* a coordinate from which a camera's line can be recovered.

Without calibration, prediction/evaluation fails explicitly. `--auto-boundary` permits an
unvalidated OpenCV heuristic for visual experimentation but must be reported separately
from manual calibration in experiments.
