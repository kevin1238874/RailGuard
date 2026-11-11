"""RailGuard Streamlit demo: auto suggestion, direct manipulation, approval, YOLO."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import yaml

from railguard.boundaries import estimate_zone
from railguard.config import get_zone
from railguard.detection import YOLODetector
from railguard.pipeline import analyze, draw
from railguard.web_editor import render_zone_canvas
from railguard.zone_editor import (
    calibration_yaml, config_from_points, empty_points,
    points_from_config, points_from_zone, safe_camera_id,
    validate_points, write_calibration,
)

st.set_page_config(page_title="RailGuard · CS406", page_icon="🚉", layout="wide")
st.title("🚉 RailGuard")
st.caption("Railway platform risk-zone analysis · CS406 Xử lý ảnh và Ứng dụng")
st.warning("Academic prototype only. Auto-detected lines are suggestions, not ground truth. Review the zone before inference.")

with st.sidebar:
    st.header("Model inference")
    model_type = st.radio("Method", ["Bounding-box baseline", "Pose (ankle keypoints)"])
    confidence = st.slider("YOLO confidence", 0.05, 0.95, 0.25, 0.05)
    keypoint_mode = st.selectbox("If ankle keypoints are missing", ["BBox fallback", "Abstain / UNKNOWN"])
    st.caption("YOLO weights may be downloaded on first use. Internet access can be required.")
    st.divider()
    st.header("Camera calibration")
    calibration_file = st.file_uploader("Load existing camera YAML (optional)", type=["yaml", "yml"])
    st.caption("One calibration per fixed camera. Do not copy coordinates between different views.")

uploaded = st.file_uploader("Upload a platform image (RGB)", type=["png", "jpg", "jpeg", "webp"])
if uploaded is None:
    st.info("Upload an image to start. You can auto-detect candidate lines or place the four endpoints directly on the image.")
    st.stop()

raw = uploaded.getvalue()
image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
if image is None:
    st.error("Cannot decode the uploaded image")
    st.stop()
h, w = image.shape[:2]
if h < 32 or w < 32:
    st.error("Image must be at least 32×32 pixels")
    st.stop()

# A change to either the image or loaded YAML invalidates previous approval.
image_id = hashlib.sha256(raw).hexdigest()
yaml_bytes = calibration_file.getvalue() if calibration_file else b""
source_id = hashlib.sha256(raw + b"\x00calibration\x00" + yaml_bytes).hexdigest()
if st.session_state.get("rg_source_id") != source_id:
    st.session_state["rg_source_id"] = source_id
    st.session_state["rg_image_id"] = image_id
    st.session_state["rg_points"] = empty_points()
    st.session_state["rg_view"] = {"zoom": 1, "center": [.5, .5]}
    st.session_state["rg_generation"] = st.session_state.get("rg_generation", 0) + 1
    st.session_state["rg_zone_source"] = "manual"
    st.session_state["rg_approval"] = False
    st.session_state.pop("rg_inference", None)
    st.session_state["rg_import_error"] = None
    if yaml_bytes:
        try:
            cfg = yaml.safe_load(yaml_bytes)
            if not isinstance(cfg, dict):
                raise ValueError("YAML root must be a mapping")
            st.session_state["rg_points"] = points_from_config(cfg, w, h)
            st.session_state["rg_zone_source"] = "loaded YAML (requires re-confirmation)"
        except (ValueError, TypeError, KeyError, yaml.YAMLError) as err:
            st.session_state["rg_import_error"] = str(err)

if st.session_state["rg_import_error"]:
    st.error(f"Could not load calibration: {st.session_state['rg_import_error']}. You can still place points manually.")

st.subheader("1 · Propose or reset the safety zone")
auto_col, clear_col, method_col, color_col = st.columns([1.2, 1.1, 1.5, 1.1])
with method_col:
    boundary_method = st.selectbox("Auto method", ["hybrid", "color_hough", "canny_hough", "color_ransac"])
with color_col:
    paint_color = st.selectbox("Line paint", ["both", "white", "yellow"])
with auto_col:
    auto_clicked = st.button("✨ Auto Detect L/E", use_container_width=True)
with clear_col:
    clear_clicked = st.button("Clear four points", use_container_width=True)

if clear_clicked:
    st.session_state["rg_points"] = empty_points()
    st.session_state["rg_generation"] += 1
    st.session_state["rg_zone_source"] = "manual"
    st.session_state["rg_approval"] = False
    st.session_state.pop("rg_inference", None)
    st.session_state.pop("rg_auto_notice", None)

if auto_clicked:
    with st.spinner("Searching for painted lines and platform edges…"):
        suggestion = estimate_zone(image, method=boundary_method, color=paint_color)
    if suggestion.zone is None:
        st.session_state["rg_auto_notice"] = ("warning",
            f"Auto Detect could not find a reliable line pair ({suggestion.status}). "
            "Existing points were kept. Select and place the four points manually.")
    else:
        try:
            proposed = points_from_zone(suggestion.zone, w, h)
            config_from_points(proposed, w, h)
        except (ValueError, TypeError, KeyError) as err:
            st.session_state["rg_auto_notice"] = ("warning", f"Invalid auto proposal: {err}")
        else:
            st.session_state["rg_points"] = proposed
            st.session_state["rg_generation"] += 1
            st.session_state["rg_zone_source"] = f"AUTO proposal ({boundary_method}); review required"
            st.session_state["rg_approval"] = False
            st.session_state.pop("rg_inference", None)
            st.session_state["rg_auto_notice"] = ("warning",
                "Automatic geometry is provisional. Verify both L (paint) and E (edge); "
                "drag any misplaced endpoints before you approve it.")

if "rg_auto_notice" in st.session_state:
    kind, note = st.session_state["rg_auto_notice"]
    getattr(st, kind)(note)

st.subheader("2 · Click or drag the endpoints on the image")
st.caption("L0 → L1: safety paint line. E0 → E1: platform edge. Click a point label then click the image, or drag a colored marker. Move the points until the highlighted strip matches the actual hazardous platform area.")

# Fit the displayed image to a comfortable size; normalized coords stay the same.
max_width = 1250
scale = min(1.0, max_width / w)
preview = cv2.resize(image, (max(2, round(w * scale)), max(2, round(h * scale)))) if scale < 1 else image
ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 88])
if not ok:
    st.error("Cannot encode image preview")
    st.stop()
editor_result = render_zone_canvas(base64.b64encode(encoded.tobytes()).decode("ascii"),
                                   st.session_state["rg_points"],
                                   st.session_state["rg_generation"],
                                   st.session_state.get("rg_view"))
edit = editor_result.edit
if edit is not None:
    try:
        if not isinstance(edit, dict) or edit.get("generation") != st.session_state["rg_generation"]:
            raise ValueError("Outdated editor event ignored")
        revised = validate_points(edit.get("points"), allow_partial=True)
    except (TypeError, ValueError) as err:
        st.info(f"Editor event skipped: {err}")
    else:
        if revised != st.session_state["rg_points"]:
            st.session_state["rg_points"] = revised
            incoming_view = edit.get("view")
            if isinstance(incoming_view, dict):
                zoom = incoming_view.get("zoom")
                center = incoming_view.get("center")
                if (isinstance(zoom, (float, int)) and not isinstance(zoom, bool)
                    and 1 <= zoom <= 12 and isinstance(center, list) and len(center) == 2
                    and all(isinstance(v, (float, int)) and 0 <= v <= 1 for v in center)):
                    st.session_state["rg_view"] = {"zoom": float(zoom), "center": center}
            st.session_state["rg_generation"] += 1
            st.session_state["rg_zone_source"] = "manually edited"
            st.session_state["rg_approval"] = False
            st.session_state.pop("rg_inference", None)

points = st.session_state["rg_points"]
zone = None
cfg = None
error = None
if any(p is None for p in points):
    error = f"Place all four endpoints ({sum(p is not None for p in points)}/4 selected)."
else:
    try:
        cfg = config_from_points(points, w, h)
        zone = get_zone(cfg, w, h)
    except (ValueError, TypeError, KeyError) as err:
        error = str(err)

if error:
    st.warning(f"Calibration incomplete or invalid: {error}")
else:
    st.success("Valid four-point polygon. Check that the orange L line follows the safety paint and the blue E line follows the platform edge.")
st.caption("Current geometry: " + st.session_state["rg_zone_source"])

st.subheader("3 · Confirm and save this camera calibration")
confirmed = st.checkbox(
    "I visually checked both lines and the zone for this exact camera (not a guaranteed ground-truth label).",
    key="rg_approval", disabled=zone is None,
)
name = st.text_input("Camera / sequence ID (used for YAML filename)", value="camera-01",
                     help="Use RailEye3D sequence names such as red-00-00-front-left when known.")
try:
    camera_id = safe_camera_id(name)
    id_error = None
except ValueError as err:
    id_error = str(err)
    camera_id = "camera-01"
    st.error(id_error)

if zone is not None and confirmed and not id_error:
    yaml_text = calibration_yaml(points, w, h)
    dl, save = st.columns(2)
    with dl:
        st.download_button("⬇ Download camera YAML", yaml_text, file_name=f"{camera_id}.yaml",
                           mime="application/x-yaml", use_container_width=True)
    with save:
        overwrite = st.checkbox("Allow overwriting an existing camera file", value=False)
        if st.button("💾 Save to local configs/", use_container_width=True):
            try:
                output = write_calibration(points, w, h, camera_id,
                                           overwrite=overwrite)
            except FileExistsError:
                st.error(f"configs/{camera_id}.yaml already exists. Check overwrite explicitly if intended.")
            except OSError as err:
                st.error(f"Could not save file: {err}")
            else:
                st.success(f"Saved {output}. Use this YAML for other frames from the SAME fixed camera.")
    with st.expander("Show the YAML geometry"):
        st.code(yaml_text, language="yaml")
else:
    st.info("Confirm the proposed or manually edited geometry to enable saving and model inference.")

st.subheader("4 · Run real model inference")
if st.button("▶ Run YOLO on this image", type="primary", disabled=not confirmed or zone is None):
    try:
        with st.spinner("Loading YOLO and analyzing the image…"):
            pose = model_type.startswith("Pose")
            detector = YOLODetector("pose" if pose else "bbox", confidence=confidence)
            people = detector.predict(image)
            results, zone_used, status = analyze(image, people, zone,
                foot_method="pose" if pose else "bbox",
                pose_fallback="abstain" if keypoint_mode.startswith("Abstain") else "bbox")
            visual = draw(image, results, zone_used, status)
            png_ok, png = cv2.imencode(".png", visual)
            if not png_ok:
                raise RuntimeError("Cannot encode inference output")
            st.session_state["rg_inference"] = {
                "signature": [image_id, points, model_type, confidence, keypoint_mode],
                "image": png.tobytes(), "results": [x.as_dict() for x in results],
            }
    except Exception as err:
        st.error(f"Inference failed: {err}")

inference = st.session_state.get("rg_inference")
if inference and confirmed and inference["signature"] == [image_id, points, model_type, confidence, keypoint_mode]:
    st.image(inference["image"], caption="YOLO + calibrated risk zone (actual inference)",
             use_container_width=True)
    results = inference["results"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Detected persons", len(results))
    c2.metric("DANGER", sum(r["intrusion"] is True for r in results))
    c3.metric("UNKNOWN", sum(r["intrusion"] is None for r in results))
    st.dataframe(results, use_container_width=True)
    st.download_button("Download predictions JSON", json.dumps(results, indent=2),
                       file_name="railguard_results.json", mime="application/json")
