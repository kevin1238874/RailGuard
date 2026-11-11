"""Streamlit demo using real YOLO inference and explicit per-scene calibration."""
import json
import cv2
import numpy as np
import streamlit as st
import yaml
from railguard.config import get_zone
from railguard.detection import YOLODetector
from railguard.pipeline import analyze, draw

st.set_page_config(page_title="RailGuard • CS406", page_icon="🚉", layout="wide")
st.title("🚉 RailGuard")
st.caption("Railway platform risk-zone analysis • CS406 Image Processing and Applications")
st.warning("Research prototype only — not for railway operations. A safety boundary must be calibrated for each camera.")

with st.sidebar:
    st.header("Inference")
    model_type = st.radio("Method", ["Bounding-box baseline", "Pose (ankle keypoints)"])
    confidence = st.slider("YOLO confidence", 0.05, 0.95, 0.25, 0.05)
    keypoint_mode = st.selectbox("Missing ankle keypoints", ["BBox fallback", "Abstain / unknown"])
    st.caption("YOLO weights download on first use; internet access may be required.")
    st.header("Zone calibration")
    cfg_file = st.file_uploader("Optional YAML calibration file", type=["yaml", "yml"])
    if cfg_file:
        try:
            cfg = yaml.safe_load(cfg_file.getvalue())
            if not isinstance(cfg, dict):
                raise ValueError("YAML root must be a mapping")
        except Exception as err:
            st.error(f"Invalid calibration: {err}")
            cfg = {}
    else:
        st.caption("Example normalized coordinates below are NOT RailEye3D ground truth. Adjust all 4 endpoints from your actual image before inference.")
        l0 = (st.number_input("L left x", 0.0, 1.0, 0.0, 0.01),
              st.number_input("L left y", 0.0, 1.0, 0.58, 0.01))
        l1 = (st.number_input("L right x", 0.0, 1.0, 1.0, 0.01),
              st.number_input("L right y", 0.0, 1.0, 0.58, 0.01))
        e0 = (st.number_input("E left x", 0.0, 1.0, 0.0, 0.01),
              st.number_input("E left y", 0.0, 1.0, 0.73, 0.01))
        e1 = (st.number_input("E right x", 0.0, 1.0, 1.0, 0.01),
              st.number_input("E right y", 0.0, 1.0, 0.73, 0.01))
        cfg = {"coordinates": "normalized", "zone": {"safety_line": [list(l0), list(l1)],
                                                       "platform_edge": [list(e0), list(e1)]}}
        st.download_button("Save these calibration coordinates", yaml.safe_dump(cfg, sort_keys=False),
                           file_name="scene.yaml", mime="text/yaml")

uploaded = st.file_uploader("Upload a railway platform image", type=["png", "jpg", "jpeg", "webp"])
if uploaded is not None:
    image = cv2.imdecode(np.frombuffer(uploaded.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.error("Cannot decode this image")
    else:
        h, w = image.shape[:2]
        a, b = st.columns(2)
        a.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)
        try:
            zone = get_zone(cfg, w, h)
            b.image(cv2.cvtColor(draw(image, [], zone, "zone_preview"), cv2.COLOR_BGR2RGB),
                    caption="Calibrated L/E/Z (preview)", use_container_width=True)
        except Exception as exc:
            zone = None
            b.error(f"Invalid calibration: {exc}")
        confirmed = st.checkbox("I have reviewed the L/E zone on this camera (example coordinates are not ground truth)")
        if st.button("Run actual model inference", type="primary", disabled=not confirmed or zone is None):
            try:
                with st.spinner("Loading YOLO and processing image"):
                    pose = model_type.startswith("Pose")
                    detections = YOLODetector("pose" if pose else "bbox", confidence=confidence).predict(image)
                    results, zone, status = analyze(
                        image, detections, zone, foot_method="pose" if pose else "bbox",
                        pose_fallback="abstain" if keypoint_mode.startswith("Abstain") else "bbox")
                    visual = draw(image, results, zone, status)
                st.image(cv2.cvtColor(visual, cv2.COLOR_BGR2RGB),
                         caption="Actual inference output", use_container_width=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Detected people", len(results))
                c2.metric("Risk zone violations", sum(x.intrusion is True for x in results))
                c3.metric("Unknown", sum(x.intrusion is None for x in results))
                payload = [x.as_dict() for x in results]
                st.dataframe(payload, use_container_width=True)
                st.download_button("Download results JSON", json.dumps(payload, indent=2),
                                   file_name="railguard_results.json", mime="application/json")
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
