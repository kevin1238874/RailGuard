from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from railguard.train_mask import TrainMaskResult, YoloSegTrainMaskProvider


def test_train_mask_module_is_lazy_in_fresh_interpreter() -> None:
    code = (
        "import sys; import railguard.train_mask; import railguard.boundaries; "
        "assert 'torch' not in sys.modules; assert 'ultralytics' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_yolo_seg_provider_predict_contract_and_component_filtering() -> None:
    calls: list[dict] = []

    class FakeModel:
        def predict(self, **kwargs):
            calls.append(kwargs)
            masks = np.zeros((2, 100, 120), np.float32)
            masks[0, 20:70, 35:85] = 1.0
            masks[1, 0:2, 0:2] = 1.0  # below 0.2% of frame
            return [SimpleNamespace(masks=SimpleNamespace(data=masks))]

    image = np.zeros((100, 120, 3), np.uint8)
    provider = YoloSegTrainMaskProvider(model=FakeModel())
    result = provider.predict(image)
    assert isinstance(result, TrainMaskResult)
    assert result.mask.shape == image.shape[:2] and result.mask.dtype == np.bool_
    assert result.metadata["valid"] is True
    assert result.metadata["raw_instances"] == 2
    assert result.metadata["kept_components"] == 1
    assert calls[0]["classes"] == [6]
    assert calls[0]["retina_masks"] is True
    assert calls[0]["conf"] == 0.25 and calls[0]["imgsz"] == 640


def test_yolo_seg_provider_no_detection_fallback() -> None:
    model = SimpleNamespace(predict=lambda **_: [SimpleNamespace(masks=None)])
    result = YoloSegTrainMaskProvider(model=model).predict(np.zeros((80, 100, 3), np.uint8))
    assert result.metadata["valid"] is False
    assert result.metadata["fallback_reason"] == "no_train_detections"
    assert not result.mask.any()
