# RailGuard — Quy trình triển khai và nghiệm thu (CS406)

> **Tài liệu giao việc cho coding agent trên máy local**  
> Repo: `kevin1238874/RailGuard` · Baseline khi soạn: `09683b2` (11/11/2026).  
> **Trạng thái:** KẾ HOẠCH triển khai, **không** phải danh sách tính năng đã hoàn tất. Các kết quả thực nghiệm chỉ được ghi sau khi chạy trên dữ liệu thật.

## 0. Chỉ thị điều hành dành cho agent

**Mục tiêu:** Hoàn thiện một pipeline có thể giải thích và đánh giá khoa học về phát hiện người vượt vạch an toàn tại sân ga: `ảnh/video → M1 tiền xử lý → M2 tìm L/E và vùng Z → M3 phát hiện người/điểm chân → M4 phán đoán vi phạm → E1–E5 thực nghiệm + demo`.

1. **Đọc trước khi sửa:** `README.md`, `docs/RESEARCH_PROTOCOL.md`, `docs/IMPLEMENTATION_NOTES.md`, `railguard/{raileye,boundaries,geometry,config,calibration,zone_editor,web_editor,detection,pipeline,evaluate,metrics,splitting,cli,export_yolo}.py`, `app.py`, `tests/` và file này. In ra danh sách chức năng *đã có / thiếu / sai / chưa xác minh* trong `docs/PROGRESS.md`.
2. **Không viết lại từ đầu:** ưu tiên tái sử dụng các hàm sẵn có, thiết kế module rõ ràng, giữ CLI, định dạng annotation, file YAML và API hiện tại nếu có thể. Những thay đổi không tương thích phải có migration và test.
3. **Làm tuần tự** theo các pha P0 → P7 bên dưới; mỗi pha có deliverables, test và tiêu chí nghiệm thu. Chưa qua kiểm thử pha trước thì không đánh dấu pha sau là `DONE`.
4. **Trung thực về dữ liệu:** không có ảnh RailEye3D gốc thì chỉ kiểm thử bằng ảnh synthetic / dữ liệu người dùng cấp quyền và ghi `BLOCKED_REAL_DATA`. Không bịa accuracy, F1, tọa độ vạch hay ảnh ground truth. Không tự tải/đẩy ảnh có hạn chế bản quyền lên repo.
5. **Không có vùng tin cậy thì không đoán:** thiếu/không chắc L/E, mất keypoint cần thiết hoặc dữ liệu camera không phù hợp → trả `UNKNOWN`/`needs_review`, không mặc định `SAFE`.
6. **So sánh công bằng:** luôn giữ cùng bộ ảnh/sequence, độ phân giải hợp lệ, thiết lập đánh giá, IoU matching, và ghi seed + cấu hình. Không chọn tham số theo Test.
7. **Kiểm soát Git:** chạy `git status` trước khi sửa; không xóa dữ liệu/cấu hình riêng của người dùng; làm trên branch `feat/railguard-workflow` nếu phù hợp. Commit nhỏ theo pha (`feat(m1): ...`, `test(m2): ...`), không force-push/đổi visibility/chạm secrets; không tự push lên `main` nếu chưa được yêu cầu. Ghi commit SHA vào `docs/PROGRESS.md`.
8. Nếu thiếu dataset hoặc môi trường GPU, **vẫn hoàn thiện code, unit/integration tests, fixture synthetic, lệnh chạy và báo cáo blockers**, không giả vờ đã benchmark mô hình thật.

### Định nghĩa bài toán và thuật ngữ bất biến

- **Input:** RGB frame/chuỗi frame của camera cố định nhìn xuống sân ga; có thể chạy riêng ảnh đơn, nhưng thuật toán cần thời gian chỉ hoạt động với chuỗi ảnh.
- **L:** ranh giới an toàn của vạch sơn trắng/vàng trên nền sân ga. Cần **quy ước rõ mép vạch phía an toàn** (thay vì tùy tiện dùng tâm vạch sơn). Giữ hỗ trợ chế độ `stripe_center` như một ablation, không trộn nhãn/tiêu chí giữa hai định nghĩa.
- **E:** mép nền sân ga giáp khoảng trống/ray, là cấu trúc vật lý khác với vạch L.
- **Z:** vùng nguy hiểm phía E của L, tối thiểu là dải L–E. **Chân đã vượt E không được bị đánh dấu SAFE chỉ vì nằm ngoài đa giác dải L–E**; giải quyết bằng quy tắc phía nguy hiểm/signed side phù hợp, ghi thành test. Khi hình học không xác định rõ hướng an toàn → UNKNOWN.
- **Output:** từng người: `bbox`, detector/method, confidence, điểm chân + nguồn, `SAFE|DANGER|UNKNOWN`, lý do, vùng L/E/Z, debug overlays; toàn frame có cờ cảnh báo và metadata.
- **Ground truth:** trường `beyond_safety_line` của RailEye3D là *nhãn boolean*, không phải tọa độ L/E hay keypoint. Phải phân biệt rõ annotation thật với calibration thủ công.
- **Định hướng:** Ứng dụng nghiên cứu CS406, **không** được quảng bá như hệ thống bảo đảm an toàn đường sắt ngoài thực tế.

## 1. Hiện trạng xác minh từ repo (11/11/2026)

| Thành phần | Đang có | Phần cần hoàn thiện |
|---|---|---|
| Annotation/split/metrics | Parser CSV RailEye3D; sequence split train/val/test; IoU matching, metrics và failure export | Bổ sung protocol **không train** cho thực nghiệm chính; audit ảnh/video thực, số lượng class, tách nguồn, thử ảnh ngoài dataset |
| M1 | CLAHE tùy chọn cục bộ trong `boundaries.py`; Gaussian trước Canny | Pipeline M1 riêng; gamma/histogram equalization/median/temporal median; xử lý vùng quá sáng/tối và đánh giá tác động lên YOLO |
| M2 | HSV mask, morphology, Canny, HoughLinesP, nhóm near-collinear + `cv2.fitLine`, heuristic cặp L/E | LAB, RANSAC **thật**, so sánh phương pháp, chọn đúng mép vạch/phía E, nhiều ứng viên và debug hình trung gian |
| M3 | YOLO11 Nano detection, YOLO11 Nano Pose; bbox đáy + ankle COCO (15,16) | HOG+SVM, MOG2, khảo sát YOLO Pose S/M, imgsz/conf, image enhancement, refinement chân; đo detector riêng |
| M4 | Polygon `contains`, SAFE/DANGER/UNKNOWN, hình overlay | Xử lý ngoài mép E, điểm trên biên, thiếu điểm, phản hồi độ tin cậy, reason codes |
| UI | Streamlit, Auto Detect, click/kéo 4 điểm, zoom/pan/Fit, YAML per camera | Tối giản quy trình; hiển thị từng ảnh trung gian; so sánh song song; log hướng dẫn khi auto sai; không treo UI |
| Test | Synthetic unit/integration tests + GitHub Actions | Test edge cases, benchmark ảnh thật, kiểm chứng thời gian, ảnh sáng/tối, test không leak |

**Không được hiểu tên `color_ransac` hiện có là RANSAC đúng nghĩa.** `boundaries.py::_fit_segments` đang gom các Hough segments rồi `cv2.fitLine(..., cv2.DIST_L2)`, chưa có sample random + inlier mask + consensus/re-fit/outlier resistance. Phải xử lý đúng hoặc đổi tên để tránh báo cáo sai thuật toán.

## 2. Kiến trúc mong muốn (tránh xử lý trùng và tránh lệ thuộc UI)

```text
Input RGB image OR ordered fixed-camera frames
       │
       ├── M1 preprocessing (reproducible variants)
       │       ├── branch Z: image/color/edge processing to estimate L/E
       │       └── branch P: raw vs enhanced RGB for person detection (ablation)
       │
       ├── M2 line/edge estimation (candidate L, candidate E, diagnostics)
       │       └── validated/calibrated geometry: L, E, Z + safe-side metadata
       │
       ├── M3 person detector [HOG+SVM | MOG2(video) | YOLO det | YOLO pose]
       │       └── foot locator [bottom-center | ankles | refined contact point]
       │
       └── M4 spatial decision (DANGER/SAFE/UNKNOWN + reason)
               └── visual output, per-person JSON/CSV, experiment logs
```

- M1 xử lý cho M2 và M3 là **hai cấu hình riêng**; không áp cùng một biến đổi ảnh cho mọi mô hình rồi tuyên bố nó có lợi.
- Core API cần chạy độc lập UI, dùng chung bởi CLI, Streamlit và test. Sử dụng cấu hình YAML/JSON rõ seed, method, params, version.
- Thiết kế plugin/registry nhỏ cho preprocessing, boundary candidate, detector, footpoint; tránh viết một hàm dài chứa toàn bộ phương pháp.
- Giữ `railguard predict`, `evaluate`, `calibrate`, `prepare`, `audit` tương thích; nếu thêm lệnh, ưu tiên `preprocess`, `inspect-lines`, `benchmark-lines`, `benchmark-people`, `run-experiment`. Tất cả phải có `--help` và ví dụ chạy.
- Gợi ý module **mới** (không bắt buộc đúng tên): `preprocessing.py`, `temporal.py`, `line_models.py`, `edge_scoring.py`, `classic_detectors.py`, `foot_refinement.py`, `experiments.py`; tái sử dụng các module hiện tại.

## P0 — Kiểm kê dữ liệu, thiết lập môi trường và protocol

**Việc làm**

- [ ] Chạy `python -m pytest -q`; ghi baseline test, Python/torch/ultralytics/opencv/streamlit, OS, CPU/GPU, bộ nhớ; ghi commit hiện tại và `git status`.
- [ ] Kiểm tra file gốc, bbox, params và ảnh. CSV không header của RailEye3D:

  ```text
  anno/unified/bboxes/<sequence>/<frame>.csv
      TLx, TLy, BRx, BRy, labelName, objectId
  anno/unified/params/<sequence>/<frame>.csv
      degree_of_occlusion, beyond_safety_line
  ```

  Ghép **theo thứ tự dòng cùng sequence/frame**, không theo `objectId`; kiểm tra số dòng, ảnh mất, frame rỗng, label người và bool strings.
- [ ] Sinh `outputs/data_audit.json`, `outputs/data_audit.csv`: số sequence/frame/person, từng loại nhãn, dải occlusion, số vi phạm, image sizes, độ lệch ánh sáng, thiếu file, station/camera.
- [ ] Cố định **thực nghiệm chính không fine-tune**: pretrained COCO, không có tập train nội bộ. Dùng `red-00-00*` (Stockerau) làm validation/tuning và `red-04-01*` (Fels) làm test **nếu đúng sequence có ảnh và đủ nhãn**. Không mặc định chia train/val/test bằng CLI cũ (đang yêu cầu ≥3 sequences) cho protocol này; thêm `split_mode: pretrained_val_test` nhưng giữ mode cũ phục vụ fine-tuning tùy chọn.
- [ ] Calibration phải dùng frame được chọn trước **không nằm trong tập frame đánh giá**, không dựa vào nhãn `beyond_safety_line` của Test. Chọn đường cho camera Fels trên calibration-only frames, freeze YAML trước scoring; báo cáo rõ mức thao tác thủ công của test-camera.
- [ ] Lưu manifests phiên bản, IDs và hash/snapshot đủ để tái lập. Không commit RGB gốc, nhãn bị giới hạn quyền hoặc weight nặng.

**Nghiệm thu P0:** unit test vẫn xanh; có audit thật hoặc báo `BLOCKED_REAL_DATA`; có protocol val/test không train, split identities không trùng; README không tuyên bố đã chạy kết quả chưa có.

## P1 — M1: Tiền xử lý ảnh/ánh sáng có kiểm soát

**Phải triển khai thành các hàm độc lập, bật/tắt được:**

- [ ] `identity/raw` (đối chứng không xử lý).
- [ ] `GaussianBlur`, `MedianBlur`: hỗ trợ kernel hợp lệ, bảo toàn kích thước, không phá kênh màu.
- [ ] `global histogram equalization` (kênh độ sáng Y hoặc L, không equalize riêng từng kênh BGR gây sai màu).
- [ ] `CLAHE` (LAB-L hoặc YCrCb-Y, ghi `clipLimit`, `tileGridSize`).
- [ ] `gamma correction` (định nghĩa chính xác công thức và quy ước gamma làm sáng/tối; LUT, giới hạn pixel 0..255).
- [ ] `Temporal Median Background` cho frame liên tiếp **cùng camera** (window/sampling, dtype/shape, foreground contaminations); tuyệt đối không chạy temporal median trên ảnh đơn hay trộn camera.
- [ ] Báo cáo quality flags: pixel cực tối, cực sáng, clipped highlights, dynamic range, tương phản; dùng để chọn nhánh thử nghiệm **theo cấu hình Validation**, không tự học từ Test.
- [ ] Chỉ thêm giải pháp cao cấp (Retinex/guided filter) nếu baseline cho thấy nhu cầu; ghi lý do, tránh tăng phức tạp vô cớ.

**Ablation tối thiểu:** Raw / CLAHE / Gamma / HistEq / Gaussian / Median / CLAHE+Gaussian; thêm TemporalMedian cho video. Xuất contact sheet `original`, `processed`, histogram, mask/edges và thời gian chạy. Đánh giá **riêng M2 quality** và **M3 detection quality**, vì tăng tương phản có thể làm YOLO tệ hơn. Không tuyên bố khôi phục được chi tiết đã cháy trắng bão hòa.

**Nghiệm thu P1:** test ảnh tối, ảnh cháy sáng, noise, gradient, màu, ảnh kích thước nhỏ, video cố định, ảnh đơn bị từ chối temporal; không `NaN`, không sai shape/dtype; có config reproducible và preview.

## P2 — M2: Phân biệt đúng vạch an toàn L và mép sân ga E

### P2.1 Tìm candidate L — vạch sơn

- [ ] Thực nghiệm HSV mask cho trắng/vàng, LAB (đặc biệt kênh độ sáng/màu), adaptive threshold nếu cần; morphology open/close, loại nhiễu/điểm nhỏ.
- [ ] Canny trên ảnh gốc và ảnh đã lọc/CLAHE; HoughLinesP trích xuất các segment ở mọi hướng (kể cả gần thẳng đứng). Hiển thị segments/length/angle/response.
- [ ] `line clustering`: gom các segment đồng hướng/đồng tuyến có giới hạn góc và khoảng cách, chống trộn line của tàu với nền sân ga; không áp điều kiện song song cứng vì có phối cảnh.
- [ ] **RANSAC thật:** lấy mẫu tối thiểu, fit line, tính perpendicular residual, inlier mask, chọn consensus, refit trên inliers, kiểm soát `max_trials`, `residual_threshold`, `min_inlier_ratio`, `random_state`. Có unit test outlier 30–50% và kiểm thử reproducibility.
- [ ] Với vạch sơn dày, xác định **mép phía an toàn** hoặc `stripe_center` tùy config; không mặc định lấy một trong hai cạnh Hough hay tâm dải màu. Có debug mask chứng minh đường đặt theo định nghĩa được chốt.

### P2.2 Tìm candidate E — mép sân ga

- [ ] Không suy luận E chỉ từ hai đường song song sáng nhất. Xét Canny, Hough/RANSAC, ranh giới chuyển tiếp texture/brightness từ gạch nền sân ga sang khoảng tối vùng ray, liên tục theo không gian.
- [ ] Đánh giá mẫu điểm ở **hai bên** candidate E và phân biệt vùng sáng nền sân ga, vùng ray/tàu; không mặc định phía phải/trái theo trục `x`, có thể camera đảo hướng/đường chéo.
- [ ] Loại tình huống nhầm: hai biên của cùng vạch trắng, phản xạ trắng trên tàu, song cửa, đường ray, mép bậc thềm, đường rãnh lát nền.
- [ ] Không dùng threshold brightness cố định như một luật phổ quát; ghi ràng buộc thực nghiệm và kiểm thử bóng tối, ngược sáng, đêm, tàu che mép. Nếu không thấy E, trả `uncertain_no_platform_edge`.

### P2.3 Ghép L/E thành Z, lựa chọn ứng viên

- [ ] Mỗi candidate trả endpoint, support/inlier stats, evidence và diagnostics; lọc hợp lý bằng vị trí, hướng, khoảng cách theo phối cảnh, **hướng phía an toàn** và độ ổn định thời gian nếu có video.
- [ ] Không gán `confidence` chuẩn xác suất nếu chỉ có heuristic score; dùng `score/uncertainty_reason` và nhãn `estimated_review_required`.
- [ ] Có thể hiển thị Top-3 cặp L/E ứng viên để người dùng chọn; khi tất cả kém thì để người dùng click 4 điểm, **không** vẽ vùng bịa.
- [ ] Giao diện và dataset export dùng thống nhất tọa độ normalized vs pixel, xử lý đúng ảnh thay độ phân giải và không mở rộng polygon sang vùng vô nghĩa.

**Ma trận method M2 tối thiểu:** `HSV + Canny + HoughP`, `LAB + Canny + HoughP`, `HSV/LAB + HoughP + clustering`, `HSV/LAB + HoughP + RANSAC`, `hybrid + region evidence`, mỗi tổ hợp có `Raw`, `CLAHE`, `Gamma` tùy ablation. Báo cáo ablation riêng; không nhất thiết chọn tất cả cho demo mặc định.

**Nghiệm thu P2:** fixture ngang/chéo/dốc đứng, hai mép cùng vạch sơn, phản chiếu thân tàu, line bị đứt, occlusion, ảnh sáng/tối, thay size, xung đột candidates. So sánh với **L/E thủ công độc lập** trên một tập calibration/validation bằng line distance/angle/overlap; báo tỷ lệ abstain và thời gian xử lý. Không sử dụng annotation boolean như tọa độ đường.

## P3 — M3: Phát hiện người và điểm tiếp đất

### P3.1 Các detector/baselines bắt buộc

- [ ] **B0 Always-unsafe**: đối chứng ngây thơ ở **mức frame**; không giả vờ nó tự phát hiện bbox để so khớp instance.
- [ ] **B1 HOG + SVM (OpenCV pedestrian detector)**: frame độc lập; xuất bbox/confidence/NMS và thời gian.
- [ ] **B2 MOG2 + foreground blob**: chỉ dùng chuỗi frame camera cố định; cần warmup/learning rate/morphology/contours/blob filters, đánh dấu khi có tàu lớn, camera rung, bóng chuyển động. Không so sánh MOG2 trên chế độ ảnh đơn.
- [ ] **B3 YOLO Detection** (`yolo11n.pt` làm mốc): bbox bottom-center, log `conf`, `imgsz`, NMS IoU, precision/recall và tốc độ.
- [ ] **A YOLO Pose** (mốc `yolo11n-pose.pt`): bbox cùng ankle COCO 15,16, kiểm tra confidence từng cổ chân, thiếu 1/2 chân, chân bị cắt, bounding box thiếu. `abstain` và `bbox_fallback` phải được báo riêng.
- [ ] **B — refine contact point**: nghiên cứu cải thiện bằng ROI quanh đáy box/ankles, edges hoặc foreground/segment hỗ trợ; chỉ đưa vào pipeline chính khi kiểm thử chứng minh hữu ích, không suy đoán chân từ bóng hoặc nền ray một cách không kiểm soát.

### P3.2 Chẩn đoán YOLO Pose đang bỏ sót người

- [ ] Thêm tùy chọn inference `weights`, `imgsz`, `conf`, `iou`, `device` với validation. Model pose Nano **không đại diện** cho toàn bộ YOLO Pose.
- [ ] Trên cùng một tập **Validation**, đo detector coverage/precision/recall và mAP@0.5 (nếu GT box đầy đủ) của YOLO11n detect vs YOLO11n-pose; thêm YOLO11s-pose hoặc m-pose **khi máy đủ tài nguyên**. Ghi thời gian/VRAM/CPU/GPU và điều kiện cấu hình.
- [ ] Thử `imgsz` 640/960 (hoặc phù hợp cấu hình), `conf` 0.1/0.25/0.4, Raw vs CLAHE/Gamma. **Không chọn theo Test.** Đo FP riêng, không chỉ tối ưu Recall bằng cách giảm conf.
- [ ] Phân loại failure theo occlusion, cỡ người nhỏ, motion blur, người đứng sát cửa tàu, mất chân, quá sáng/tối, nghiêng góc camera. Dùng ảnh GT matched/unmatched để khoanh lỗi.
- [ ] Thử *detect→crop→pose* như nhánh nghiên cứu tùy chọn khi YOLO Pose trực tiếp vẫn kém: map keypoints về ảnh gốc, chống double counting, so sánh trên cùng detector / footpoint rule. Không coi đây là baseline bắt buộc nếu không khả thi.
- [ ] **Công bằng:** so sánh B3 bbox với pose ankle trên **cùng weights pose** để tách tác dụng footpoint; so sánh `yolo11n.pt` vs `yolo11n-pose.pt` chỉ là end-to-end systems, không kết luận keypoint cải thiện chỉ từ F1 của hai weight khác nhau.

**Nghiệm thu P3:** mọi detector trả format Person thống nhất, class person đúng, xử lý ảnh rỗng, nhiều người, missing keypoints, ảnh tối/sáng, test model mock không cần download weights; benchmark thực có file `detector_metrics.csv` và `failure_examples/` khi data sẵn có.

## P4 — M4: Quy tắc xác định vi phạm và bất định

- [ ] Xác định quy ước hình học của L (safe-side paint edge), E và hướng unsafe side từ camera calibration; không giả định `y` lớn nghĩa là gần ray.
- [ ] Với bbox bottom-center, ankles và refined contact point, tính quan hệ điểm với L/E; chi tiết giải thích ở output: nguồn điểm, điểm quan sát, distance tới L/E và reason.
- [ ] Xử lý `point-on-line`, sát ngưỡng, tolerance px/normalized, gần mép ảnh, ngoài E, vùng không quan sát được, calibration thiếu, keypoint invalid. **Vượt E không được SAFE**; thiếu bằng chứng → UNKNOWN.
- [ ] Nếu multi-person: trạng thái frame là DANGER khi có ít nhất một DANGER; nếu không có DANGER nhưng có UNKNOWN thì UNKNOWN; chỉ SAFE khi đủ dữ liệu quan sát và phương pháp không bỏ sót người một cách được biết đến. Phân biệt `no detections` với `no people` để không tuyên bố an toàn tuyệt đối.
- [ ] Nếu bổ sung temporal smoothing, phải log độ trễ, số frame tối thiểu và tác động đến false alarms; không bắt buộc giai đoạn đầu.
- [ ] Unit test geometry: hai camera hướng ngược nhau, line dốc đứng, chéo, chân trong/ngoài Z, vượt E, điểm ngay trên L, ROI nhỏ, poly sai, thiếu calibration, thiếu cổ chân.

**Nghiệm thu P4:** output nhất quán giữa CLI/Streamlit/evaluator; reason codes machine-readable; UNKNOWN không bị tính ngầm thành SAFE; hình vẽ và JSON cùng hệ tọa độ.

## P5 — E1–E5: Protocol thực nghiệm và phân tích khoa học

**Tuyệt đối không train trên test và không dùng test để chọn tham số.** Pretrained COCO là mặc định; fine-tuning YOLO chỉ là hướng mở rộng riêng, yêu cầu tập train độc lập, không được tự động bật.

| Nhóm | Câu hỏi khoa học | Cấu hình đối chứng & số đo | File đầu ra kỳ vọng |
|---|---|---|---|
| **E1 — Geometry** | Cải thiện sáng/tối và Canny/Hough/RANSAC có giúp xác định L/E không? | Raw/CLAHE/Gamma/TemporalMedian (video), HSV vs LAB, Hough vs RANSAC, manual vs auto; line/edge distance, angle, failure/abstain, runtime | `e1_geometry.csv`, ảnh mask/edges/candidates/overlay |
| **E2 — Người** | Baseline truyền thống vs YOLO/YOLO Pose nhận người ra sao? | B0 (frame only), B1 HOG+SVM, B2 MOG2 (video), B3 YOLO det, A pose; box P/R/mAP, coverage, runtime, occlusion | `e2_detectors.csv`, confusion/missed person examples |
| **E3 — Footpoint** | Điểm đáy box vs ankles vs refinement có tốt hơn? | Giữ cùng detector khi có thể; bbox vs 1/2 ankle, bbox fallback, abstain, refinement; distance to manual footpoint (nếu có) + violation F1 | `e3_footpoints.csv`, image comparison |
| **E4 — Ablation** | Thành phần nào thực sự đóng góp? | ROI, preprocessing, candidate geometry, pose ankle, refinement, final pipeline: tháo từng phần trong cùng split | `e4_ablation.csv` |
| **E5 — Generalization** | Tuning trên Stockerau chuyển sang Fels ra sao? | Chốt trên `red-00-00*`; test trên `red-04-01*` nếu có; camera-specific manual calibration phải freeze trước test; công bố shift và failure | `e5_cross_station.csv`, failures, config hash |

**Metrics cần có:** `TP, FP, FN, TN` theo quy tắc đã công bố; Precision, Recall, F1 **unsafe-class**, frame-level alert P/R/F1, false alarms, missed unsafe, UNKNOWN rate, detection coverage, FPS/time, + IoU/mAP@0.5 cho bbox detector (nếu GT hợp lệ). Chỉ báo Accuracy có thể gây hiểu sai khi class mất cân bằng. Một người unsafe bị bỏ sót/UNKNOWN phải được tính FN ở **end-to-end unsafe recall**; xuất thêm matched/determined conditional metrics nhưng không dùng thay end-to-end. B0/MOG2 có phạm vi đánh giá khác phải ghi rõ.

**Cách lưu mỗi experiment:**

```text
outputs/experiments/<experiment_id>/
  config.yaml                 # full config + method + parameters
  environment.json            # package versions, CPU/GPU, seed, model/weight hash
  split_manifest.csv          # sequence/frame identities + split
  metrics.json                # counts, coverage, P/R/F1, UNKNOWN, runtime
  predictions.csv             # frame, instance, GT, prediction, confidence, reason
  failures.csv                # FP/FN/unknown/missed persons with IDs
  figures/                    # original/preprocessed/masks/edges/lines/Z/detections
  summary.md                  # findings supported by measurements, limitations
```

**Không lấy test-frame để vẽ lại L/E tối ưu theo GT**: calibration frames của mỗi camera phải được liệt kê trước và tách khỏi score set; bất kỳ human adjustment nào trên Test là một condition khác, không báo như fully automatic generalization.

## P6 — Giao diện demo tối giản, tập trung vào debug khoa học

**Luồng duy nhất dễ hiểu:** `1. Upload ảnh hoặc chuỗi frame → 2. Chọn preprocessing → 3. Auto L/E (và sửa tay nếu cần) → 4. Confirm & save camera YAML → 5. Chọn phương pháp phát hiện người → 6. Run/compare/export`.

- [ ] Khung ảnh **có kích thước cố định trong trang**, zoom bằng `+`, `−`, wheel **chỉ ảnh**, kéo pan bằng chuột, `Fit`, `Reset view`; không làm zoom cả trình duyệt. Click/kéo L0/L1/E0/E1 ở mọi mức zoom phải giữ chuẩn normalized coords.
- [ ] Hiển thị **L (vạch sơn)**, **E (mép sân ga)**, Z, safe-side rõ và legend nhất quán. Đặc biệt có phương án chọn mép vạch phía an toàn thay cho tâm vạch.
- [ ] Auto hiển thị trạng thái (`candidate`, `needs_review`, `unknown`), evidence và nguyên nhân lỗi; cho xem Top-N candidates/ảnh trung gian; nếu không chắc chắn không ép chọn.
- [ ] Có các tab **Input / Preprocessing / Masks / Canny-Hough / RANSAC / Final Zone / People & Feet / Results**, chỉ hiện nội dung khả dụng. Không làm giao diện quá dài phải cuộn liên tục.
- [ ] Save/load YAML theo camera, cảnh báo khi đổi ảnh/góc camera, validate geometry, confirm lại khi chỉnh bất kỳ điểm nào; undo/reset điểm; không cho inference với calibration không hợp lệ.
- [ ] Cho chọn raw/CLAHE/gamma riêng cho boundary và detector, detector B1/B2/B3/A, confidence/weights/imgsz, footpoint logic. B2/TemporalMedian phải disable khi ảnh đơn.
- [ ] Kết quả có bảng người/footpoints/vi phạm/reason, download PNG/JSON/CSV và hiển thị phương pháp + tham số. Nếu thiếu weights/internet/model thì lỗi rõ, không fake output.
- [ ] Thử tối thiểu trên Chrome/Edge Windows; test browser zoom, CSS responsiveness, kéo ảnh khi đã zoom, thay ảnh, thay YAML, chiều ngang/hình đứng, bật/tắt sidebar.

**Nghiệm thu P6:** một người mới biết sử dụng demo có thể load ảnh và tự đặt L/E trong UI, kiểm tra ảnh trung gian và chạy một mô hình có sẵn mà không nhập 8 tọa độ; ảnh không tràn khung; các button không chạy sai trạng thái.

## P7 — QA, tài liệu, demo và bàn giao

- [ ] Test unit: parser row alignment + empty frames + missing params; LAB/HSV/CLAHE/gamma noise; RANSAC outlier; lines under 0–180°, false stripe, dark track, reflections; geometry and safe/unsafe; bbox/keypoint fallback; MOG2 temporal; CLI; app state.
- [ ] Test integration: tạo synthetic multi-frame with stationary camera; toàn pipeline `prepare→preprocess→lines→detector mock→decision→evaluate`; dùng fixture expected metrics, không cần Internet/model download.
- [ ] Test model thật riêng: sử dụng ảnh đã cấp quyền, tải weights hợp lệ; lưu checksum và version. Nếu không có weights/image, báo rõ `NOT_RUN`.
- [ ] Chạy `python -m pytest -q`, `python -m compileall -q railguard app.py`, `python -m railguard --help`, `python -m streamlit run app.py` (thử mở UI bằng browser). Nếu máy không cài dependency nào, ghi blocker thay vì báo pass.
- [ ] Kiểm tra đối chiếu cùng một ảnh giữa UI/CLI; đảm bảo không có leakage, file output đầy đủ, figure before/after, failure cases có giải thích.
- [ ] Cập nhật `README.md` installation/quick start/dataset access, `docs/METHODOLOGY.md`, `docs/EXPERIMENTS.md`, `docs/PROGRESS.md`; nguồn tham khảo HOG, Canny, Hough, RANSAC, CLAHE, MOG2, YOLO/RailEye3D. Đánh dấu methods **implemented / experimental / not run**, không ghi kết quả ảo.
- [ ] Dọn code chết, annotate source/weights/licences, tránh commit `.venv/`, `runs/`, raw dataset, model checkpoints, nội dung bí mật.

### Quy tắc dừng và báo cáo cho từng pha

Một pha chỉ được đánh `DONE` khi có **code + test đạt + demo/CLI được chạy thử nếu có thể + artifact/checkpoint đúng + tài liệu**. Nếu chỉ có code/test synthetic, ghi `IMPLEMENTED_NOT_VALIDATED_ON_REAL_DATA` thay vì `DONE_REAL_DATA`.

Mẫu cập nhật `docs/PROGRESS.md` sau **mỗi pha**:

```md
## P2 — Geometry (YYYY-MM-DD)
- Status: DONE_SYNTHETIC / DONE_REAL_DATA / BLOCKED_REAL_DATA / IN_PROGRESS
- Files changed: ...
- Methods actually implemented: ...
- Reproduction commands: ...
- Tests: <exact command> → <actual result>
- Real datasets/images used (and permissions): ...
- Artifacts: ...
- Measured improvements/regressions: ...
- Known failure cases: ...
- Blockers / next step: ...
- Commit: ...
```

## 3. Trình tự thực thi tối ưu cho agent

1. **Ngay:** P0 (audit + sửa protocol không-train) → P1 (đủ preprocessing) → P2 (RANSAC đúng, phân biệt L/E, metrics cho line). Đây là điểm yếu khoa học cấp thiết nhất.
2. **Tiếp:** P3 (diagnostics của YOLO Pose và HOG/MOG2, footpoint), P4 (ngữ nghĩa vùng nguy hiểm và UNKNOWN).
3. **Sau khi core đã đúng:** P5 (E1–E5 có đủ CSV/figures), P6 (UI và debug overlays), P7 (QA/docs). Được sửa lỗi UI sớm nếu cản việc kiểm thử P2/P3, nhưng không để UI thay thế thuật toán.
4. **Cổng dữ liệu:** bất kỳ test/dữ liệu thật nào chưa có → ghi blocker và hoàn thiện các phần mock/synthetic có thể kiểm thử; **không thay benchmark thật bằng synthetic**.

### Lệnh khởi động trên Windows PowerShell

```powershell
cd D:\FILE\UIT\Year3_Seme1\CS406_ImageProcessing\Project\RailGuard
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[vision,demo,dev]"
python -m pytest -q
python -m railguard --help
python -m streamlit run app.py
```

Nếu cần cài lại môi trường: `py -3.11 -m venv .venv`, kích hoạt rồi chạy lệnh pip ở trên. Mọi lệnh benchmark **mới** chỉ ghi là lệnh dùng được sau khi agent thực sự tạo CLI tương ứng.

## 4. Definition of Done toàn dự án

- [ ] Có dữ liệu thật đã được cấp quyền và có audit/manifest + protocol val/test rõ ràng; nếu không có, ghi `BLOCKED_REAL_DATA` và **không** tuyên bố nghiệm thu định lượng.
- [ ] M1 triển khai & được kiểm thử; ảnh sáng/tối có preview và kết quả ablation cho M2/M3.
- [ ] M2 có **Canny, Hough, HSV/LAB, Morphology, line grouping, RANSAC thực**, nhận L/E tách biệt, có debug và UNKNOWN khi sai/thiếu.
- [ ] M3 có **B0/B1 HOG+SVM/B2 MOG2/B3 YOLO Detection/A YOLO Pose**; B foot-refinement nếu đã chứng minh, hoặc báo lý do loại; kiểm tra/ghi đúng lý do YOLO Pose miss người.
- [ ] M4 có tri-state chính xác, các ca ngoài E và sát L; ảnh/JSON nhất quán.
- [ ] E1–E5 có config + số đo thật khi ảnh sẵn có; so sánh baseline công bằng, ablation, failure cases và generalization.
- [ ] UI dễ sử dụng, zoom/pan/click/drag mượt, có các stage preprocessing và debug, lưu calibration camera an toàn.
- [ ] Test automated xanh và có kiểm thử thủ công thực tế; hướng dẫn từ clean clone có thể chạy; README/phương pháp/kết quả nhất quán.

**Phạm vi không bắt buộc:** video realtime, huấn luyện lại YOLO, deep semantic segmentation, hệ thống cảnh báo an toàn thật, deployment cloud. Chỉ mở rộng sau khi core/đối chứng khoa học hoàn tất và có dữ liệu phù hợp.

---

**Câu bắt đầu dành cho agent:** *"Hãy đọc toàn bộ `WORKFLOW.md` và repo hiện tại, thực hiện P0 trước, tạo `docs/PROGRESS.md` rồi triển khai tuần tự P1–P7. Không được xem một phương pháp đã hoàn thành nếu chưa có code + test + kết quả tương ứng; ưu tiên giải quyết M1/M2 và kiểm chứng YOLO Pose trên validation. Báo rõ blockers thay vì tự đoán."*