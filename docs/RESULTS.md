# RailGuard: Báo cáo kết quả thực nghiệm

> **CS406 · Xử lý ảnh và Ứng dụng.** Toàn bộ số liệu trong file này được đo thật trên dữ liệu RailEye3D, ngày 22–24/11/2026, code `railguard/` ở commit `606bf23`.
> File chi tiết về lỗi vùng an toàn: [FAILURE_ANALYSIS_TEST_GT_AUTO.md](FAILURE_ANALYSIS_TEST_GT_AUTO.md).

## 0. Tóm tắt

- **Bài toán:** từ một ảnh RGB sân ga, hệ thống (1) tự xác định vùng nguy hiểm giữa vạch an toàn **L** và mép sân ga **E** bằng các kỹ thuật xử lý ảnh cổ điển, (2) phát hiện người bằng YOLO11, (3) xác định điểm chân và phán mỗi người là **SAFE / DANGER / UNKNOWN**.
- **Câu hỏi chính:** lấy điểm chân bằng **keypoint mắt cá (YOLO Pose)** có tốt hơn **tâm cạnh đáy bbox** không?
- **Quy trình:** chọn ngưỡng trên val (ga Stockerau), chốt lại, rồi chấm trên test (ga Fels). Val chọn `conf` 0.50 và imgsz 1280 (mặc định: 0.25 và 640).
- **Kết quả trên tập test** (273 frame, 1039 người, 385 người vi phạm):

| Cấu hình | Precision | Recall | **F1** (ngưỡng chọn trên val) | F1 (ngưỡng mặc định) |
|---|---|---|---|---|
| YOLO + tâm đáy bbox | 0.810 | 0.608 | **0.694** | 0.653 |
| YOLO + mắt cá (không thấy → UNKNOWN) | 0.787 | 0.577 | **0.666** | 0.645 |
| YOLO + mắt cá (không thấy → tâm đáy bbox) | 0.765 | 0.634 | **0.693** | 0.658 |
| *Tham chiếu: box ground truth + tâm đáy* | 0.939 | 0.803 | *0.866* | – |

- **Kết luận:**
  1. Chọn ngưỡng trên val làm F1 test của bbox tăng **+0.042** và của mắt cá có fallback tăng **+0.035**, chủ yếu nhờ giảm báo động sai (FP của bbox: 96 → 55).
  2. Mắt cá và tâm đáy bbox **ngang nhau** ở kết quả cuối (F1 0.693 so với 0.694). Trên những người nhìn thấy mắt cá, mắt cá nhỉnh hơn một chút (F1 0.918 so với 0.906 trên cùng 675 người).
  3. Nguồn lỗi lớn nhất là **detector bỏ sót người bị che khuất**, **bóng người phản chiếu trên thân tàu bị phát hiện thành người**, và **vùng an toàn tự dò bị sai lúc sân ga đông người**.
  4. Ở mức frame ("frame này có người vi phạm không?"), hệ thống đạt **F1 0.96–0.97, precision 0.99**.

- **Kiểm tra chéo** (mục 6.5): đổi vai hai ga, chọn ngưỡng trên Fels rồi chấm trên Stockerau. Cả ba cách đạt F1 **0.935** (mặc định 0.909). Kết luận chính vẫn dựa vào chiều Stockerau → Fels, vì Stockerau chỉ có 41 người vi phạm.

## 1. Hệ thống

```
Ảnh RGB ──► M1 Tiền xử lý ──► M2 Dò L/E (vùng nguy hiểm Z) ──┐
        └──────────────────► M3 YOLO11: người + điểm chân ───┴─► M4 Phán SAFE / DANGER / UNKNOWN
```

| Module | Nội dung | Code |
|---|---|---|
| M1 | CLAHE, gamma, lọc nhiễu, xử lý chói nắng | `railguard/preprocessing.py` |
| M2 | Mặt nạ màu HSV/LAB cho vạch sơn, Canny + Hough, RANSAC, điểm tụ phối cảnh, kiểm tra hai phía của mép sân ga (`hybrid_region_evidence`) | `railguard/boundaries.py`, `line_models.py` |
| M3 | YOLO11n phát hiện người; YOLO11n-pose chạy trên từng người để lấy mắt cá (COCO keypoint 15, 16) | `railguard/detection.py` |
| M4 | Chân vượt L → DANGER; chân vượt E → luôn DANGER; thiếu vùng hoặc thiếu điểm chân → UNKNOWN, **không bao giờ mặc định SAFE** | `railguard/pipeline.py`, `geometry.py` |

Ba cách xác định điểm chân được so sánh:
- **bbox:** tâm cạnh đáy của box người.
- **ankle-abstain:** điểm mắt cá; nếu không thấy mắt cá thì trả UNKNOWN.
- **ankle-fallback:** điểm mắt cá; nếu không thấy thì dùng tâm đáy bbox.

## 2. Dữ liệu và protocol

| | Val | Test |
|---|---|---|
| Ga / sequence | Stockerau / `red-00-00-front-left` | Fels / `red-04-01-front-left` |
| Số frame có annotation | 37 | 273 |
| Số người (GT) | 252 | 1039 |
| Người vi phạm (`beyond_safety_line = true`) | 41 (16%) | 385 (37%) |
| Độ phân giải | 1088 × 1920 (ảnh dọc) | 1088 × 1920 |

- Annotation: repo chính thức `raileye3d/raileye3d_dataset` (commit `ce1b2cf`). Ảnh: bộ ảnh do tác giả dataset cấp. Audit không thiếu file nào (`outputs/data_audit.json`).
- **Không train.** Dùng weights COCO pretrained. Val và test là **hai ga khác nhau**, nên kết quả test phản ánh khả năng tổng quát sang ga mới.
- **Chiều chính:** val = Stockerau, test = Fels. **Chiều ngược** (mục 6.5) đổi vai hai ga, chỉ để kiểm tra chéo. Từ đây trở đi, "val" và "test" không ghi thêm gì là theo chiều chính.
- Nhãn `beyond_safety_line` trên val dùng để chấm điểm và chọn ngưỡng; trên test **chỉ dùng để chấm điểm**. Vị trí L/E do hệ thống tự dò trên từng ảnh, không dùng nhãn.

## 3. Cấu hình

| Tham số | Giá trị |
|---|---|
| Dò vùng | `--auto-boundary --boundary-method hybrid_region_evidence`, quy ước L = mép phía an toàn của dải sơn (`safe_side_edge`, mặc định) |
| Tiền xử lý M1 | **Tắt** (ảnh gốc) cho cả M2 và M3; M1 và chống chói chỉ bật trong Studio, chưa được đánh giá định lượng |
| Detector | `yolo11n.pt` (sha256 `0ebbc80d…`), pose: `yolo11n-pose.pt` (sha256 `869e83fc…`) |
| imgsz / conf | **1280 / 0.50**, chọn trên val (mục 5.2); mặc định là 640 / 0.25 |
| Ngưỡng tin cậy mắt cá | 0.35 (mặc định; val không phân biệt được các giá trị đã thử) |
| NMS IoU | 0.7 (mặc định, không quét) |
| Ngưỡng IoU ghép với GT | 0.5 |
| Môi trường | Windows 11, Intel Core i7-12650H, chạy CPU (không dùng GPU), Python 3.11.7, ultralytics 8.4.215, torch 2.15.0+cpu, OpenCV 4.14.0 |
| Thời gian chạy | Một tiến trình (các lượt mặc định): khoảng 1.5–3 giây/frame, phần lớn dành cho bước dò L/E. Chọn ngưỡng và test với 4 tiến trình song song: hai vòng quét val 287 + 286 giây, test 224 giây (trong đó dò L/E cho 273 frame mất 74 giây). Chiều ngược: quét Fels 1907 giây, chấm Stockerau 17 giây (dùng lại L/E và phát hiện đã lưu) |

## 4. Cách tính điểm

- Ghép người dự đoán với người GT một-một theo IoU giảm dần, ngưỡng 0.5.
- **TP:** người vi phạm được phán DANGER.
- **FN:** người vi phạm bị phán SAFE hoặc UNKNOWN, **bị detector bỏ sót**, hoặc nằm trong frame không dò được vùng.
- **FP:** người không vi phạm bị phán DANGER, **hoặc** box dự đoán không khớp GT nào nhưng bị phán DANGER.
- **End-to-end** (chỉ số chính): tính tất cả các trường hợp trên. **Conditional**: chỉ tính những người detector bắt được và có phán đoán; dùng để tách lỗi của detector khỏi lỗi phán đoán.
- **Frame-level:** frame có ít nhất một người vi phạm thì GT = cảnh báo. Frame bị phán UNKNOWN được loại khỏi phép tính.

## 5. Kết quả trên val (Stockerau)

### 5.1. Các lượt với `conf` mặc định (0.25)

| Cấu hình | imgsz | TP | FP | FN | P | R | **F1** | Detector bắt được |
|---|---|---|---|---|---|---|---|---|
| Box GT + bbox (tham chiếu) | – | 39 | 0 | 2 | 1.000 | 0.951 | **0.975** | 100% |
| YOLO + bbox | 640 | 35 | 1 | 6 | 0.972 | 0.854 | **0.909** | 52.0% |
| YOLO + bbox | 1280 | 36 | 1 | 5 | 0.973 | 0.878 | **0.923** | 56.1% |
| YOLO + ankle-abstain | 640 | 35 | 1 | 6 | 0.972 | 0.854 | **0.909** | 52.0% |
| YOLO + ankle-abstain | 1280 | 36 | 3 | 5 | 0.923 | 0.878 | **0.900** | 56.9% |
| YOLO + ankle-fallback | 640 | 35 | 1 | 6 | 0.972 | 0.854 | **0.909** | 52.0% |
| YOLO-pose làm detector + bbox | 640 | 3 | 0 | 38 | 1.000 | 0.073 | **0.136** | 18.3% |

Nhận xét:
- Vùng an toàn tự dò hoạt động tốt trên val: box GT đạt F1 0.975, chỉ 1/37 frame không dò được vùng.
- Ở cấu hình mặc định, detector chỉ bắt được khoảng 52% số người. Gần như toàn bộ người bị bỏ sót là **người bị che khuất trên 25%**: bị che 0% thì bỏ sót 1/110, bị che trên 75% thì bỏ sót 35/36. Trong số đó rất ít người vi phạm, nên F1 vẫn đạt khoảng 0.91.
- Val quá nhỏ (41 người vi phạm). Chênh lệch giữa các cấu hình chỉ 1–2 người, **không đủ để phân biệt** bbox với ankle, hay 640 với 1280.
- **Model pose dùng làm detector trực tiếp thì gần như không dùng được** (recall 18%). Kết quả này phù hợp với thiết kế của M3: tìm người bằng `yolo11n.pt` trước, rồi mới chạy pose trên từng người.

### 5.2. Chọn ngưỡng trên val

**Quy tắc** (đặt trước khi quét): lấy cấu hình có F1 end-to-end cao nhất. Nếu hoà, lấy cấu hình đổi ít tham số nhất so với mặc định, rồi gần giá trị mặc định nhất. Mỗi cách lấy điểm chân được chọn riêng, trên cùng một lưới: `conf` từ 0.10 đến 0.80, imgsz 640 và 1280; với mắt cá, thêm ngưỡng tin cậy 0.20 / 0.35 / 0.50.

F1 trên val (ngưỡng mắt cá 0.35):

| `conf` | bbox 640 | bbox 1280 | abstain 640 | abstain 1280 | fallback 640 | fallback 1280 |
|---|---|---|---|---|---|---|
| 0.10 | 0.897 | 0.878 | 0.886 | 0.889 | 0.886 | 0.791 |
| 0.15 | 0.897 | 0.889 | 0.886 | 0.889 | 0.886 | 0.847 |
| 0.25 | 0.909 | 0.923 | 0.909 | 0.900 | 0.909 | 0.900 |
| 0.35 | 0.909 | 0.923 | 0.909 | 0.900 | 0.909 | 0.900 |
| **0.50** | 0.921 | **0.935** | 0.921 | **0.935** | 0.921 | **0.935** |
| 0.60 | 0.921 | 0.921 | 0.921 | 0.935 | 0.921 | 0.935 |
| 0.70 | 0.829 | 0.829 | 0.829 | 0.935 | 0.829 | 0.935 |
| 0.80 | 0.093 | 0.509 | 0.093 | 0.656 | 0.093 | 0.656 |

- Cả ba cách đều chọn **imgsz 1280, `conf` 0.50**, với TP 36, FP 0, FN 5. Ngưỡng mắt cá giữ 0.35, vì 0.20 và 0.50 cho cùng F1.
- Vòng quét đầu chỉ tới `conf` 0.50. Vì giá trị được chọn nằm ở mép lưới, 0.60–0.80 được thêm vào, **chỉ trên val và trước lượt test**. Lựa chọn không đổi.
- imgsz 1280 cũng nằm ở mép lưới, vì chỉ thử 640 và 1280. Kích thước lớn hơn chưa được thử.
- So với mặc định, cấu hình được chọn chỉ hơn **1 TP và bớt 1 FP**. Với mắt cá có 9 cấu hình hoà điểm. Val quá nhỏ để coi đây là giá trị tối ưu chắc chắn.

## 6. Kết quả trên test (Fels): kết quả chính, và kiểm tra chéo

Các bảng dưới dùng ngưỡng đã chốt ở mục 5.2 (imgsz 1280, `conf` 0.50), trừ khi ghi khác. Dòng "Box GT" không dùng YOLO nên không phụ thuộc ngưỡng.

### 6.1. Bảng chính

| Cấu hình | TP | FP | FN | Precision | Recall | **F1** |
|---|---|---|---|---|---|---|
| *Box GT + bbox (tham chiếu, chỉ đo lỗi của vùng an toàn)* | 309 | 20 | 76 | 0.939 | 0.803 | *0.866* |
| YOLO + bbox | 234 | 55 | 151 | 0.810 | 0.608 | **0.694** |
| YOLO + ankle-abstain | 222 | 60 | 163 | 0.787 | 0.577 | **0.666** |
| YOLO + ankle-fallback | 244 | 75 | 141 | 0.765 | 0.634 | **0.693** |

Để so sánh, các lượt với ngưỡng mặc định (imgsz 640, `conf` 0.25):

| Cấu hình | TP | FP | FN | Precision | Recall | **F1** |
|---|---|---|---|---|---|---|
| YOLO + bbox | 233 | 96 | 152 | 0.708 | 0.605 | **0.653** |
| YOLO + ankle-abstain | 226 | 90 | 159 | 0.715 | 0.587 | **0.645** |
| YOLO + ankle-fallback | 246 | 117 | 139 | 0.678 | 0.639 | **0.658** |
| YOLO-pose làm detector + bbox | 211 | 79 | 174 | 0.728 | 0.548 | **0.625** |

### 6.2. Tác động của việc chọn ngưỡng

Khoảng tin cậy 95% được tính bằng bootstrap theo khối 10 frame liên tiếp (5000 lần, so sánh cặp trên cùng frame).

| So sánh | Chênh F1 | Khoảng tin cậy 95% |
|---|---|---|
| bbox: đã chọn − mặc định | **+0.042** | +0.021 đến +0.063 |
| ankle-fallback: đã chọn − mặc định | **+0.035** | +0.018 đến +0.052 |
| ankle-abstain: đã chọn − mặc định | +0.021 | 0.000 đến +0.040 |
| ankle-fallback − bbox (cùng ngưỡng đã chọn) | −0.001 | −0.015 đến +0.012 |

- Cải thiện đến từ **ít báo động sai hơn**: FP của bbox từ 96 xuống 55, của fallback từ 117 xuống 75. Recall gần như giữ nguyên.
- Ở ankle-abstain, khoảng tin cậy chạm 0 nên mức tăng chưa chắc chắn.
- Bbox và ankle-fallback **không phân biệt được** ở kết quả cuối.

### 6.3. Detector và kết quả conditional

| Cấu hình | Người tìm được / 972 | Box thừa | UNKNOWN | Conditional P | Conditional R | **Conditional F1** |
|---|---|---|---|---|---|---|
| Box GT + bbox | 972 (100%) | 0 | 14 | 0.939 | 0.896 | **0.917** |
| YOLO + bbox | 772 (79.4%) | 57 | 13 | 0.929 | 0.880 | **0.903** |
| YOLO + ankle-abstain | 781 (80.3%) | 78 | 98 | 0.925 | 0.910 | **0.917** |
| YOLO + ankle-fallback | 781 (80.3%) | 78 | 14 | 0.931 | 0.897 | **0.914** |

972 là số người GT nằm trong các frame có vùng an toàn; 67 người còn lại thuộc 15 frame không dò được vùng. So với mặc định, số box thừa của bbox giảm từ 139 xuống 57, trong khi số người tìm được gần như không đổi (768 → 772).

### 6.4. Cảnh báo theo frame

| Cấu hình | Số frame được chấm | TP | TN | FP | FN | Precision | Recall | **F1** | Frame UNKNOWN |
|---|---|---|---|---|---|---|---|---|---|
| Box GT + bbox | 258 | 155 | 92 | 1 | 10 | 0.994 | 0.939 | **0.966** | 15 |
| YOLO + bbox | 252 | 147 | 92 | 1 | 12 | 0.993 | 0.925 | **0.958** | 21 |
| YOLO + ankle-abstain | 234 | 149 | 76 | 1 | 8 | 0.993 | 0.949 | **0.971** | 39 |
| YOLO + ankle-fallback | 253 | 149 | 92 | 1 | 11 | 0.993 | 0.931 | **0.961** | 20 |

Ở mức frame, hệ thống gần như không báo động sai (1 FP trên 93 frame an toàn được chấm, riêng ankle-abstain là 77). Ankle-abstain có F1 frame cao nhất, nhưng phải bỏ qua 39 frame UNKNOWN, nên không so sánh trực tiếp được với các dòng khác.

Chọn ngưỡng theo F1 từng người làm F1 theo frame **giảm nhẹ** so với mặc định: bbox 0.961 → 0.958, abstain 0.984 → 0.971, fallback 0.968 → 0.961.

### 6.5. Kiểm tra chéo: chọn ngưỡng trên Fels, chấm trên Stockerau

Lưới và quy tắc chọn giống hệt mục 5.2, chỉ đổi vai hai ga. Ngưỡng được chốt trước khi chấm trên Stockerau.

**F1 trên Fels, lúc này làm val** (ngưỡng mắt cá 0.35):

| `conf` | bbox 640 | bbox 1280 | abstain 640 | abstain 1280 | fallback 640 | fallback 1280 |
|---|---|---|---|---|---|---|
| 0.10 | 0.630 | 0.615 | 0.636 | 0.641 | 0.638 | 0.623 |
| 0.15 | 0.641 | 0.644 | 0.639 | 0.643 | 0.644 | 0.635 |
| 0.25 | 0.653 | 0.678 | 0.645 | 0.657 | 0.658 | 0.678 |
| 0.35 | 0.659 | 0.690 | 0.656 | 0.665 | 0.667 | 0.691 |
| 0.50 | 0.670 | **0.694** | 0.662 | 0.666 | 0.675 | 0.693 |
| 0.60 | 0.658 | 0.690 | 0.653 | 0.669 | 0.663 | 0.696 |
| 0.70 | 0.613 | 0.687 | 0.621 | **0.682** | 0.632 | **0.708** |
| 0.80 | 0.453 | 0.586 | 0.519 | 0.657 | 0.524 | 0.677 |

**Ngưỡng được chọn:**

| Cách | imgsz | `conf` | Ngưỡng mắt cá | F1 val (Fels) | Mặc định |
|---|---|---|---|---|---|
| bbox | 1280 | 0.50 | – | 0.694 | 0.653 |
| ankle-abstain | 1280 | 0.70 | 0.20 | 0.684 | 0.645 |
| ankle-fallback | 1280 | 0.70 | 0.35 | 0.708 | 0.658 |

**Kết quả trên Stockerau, lúc này làm test** (37 frame, 41 người vi phạm):

| Cách | TP | FP | FN | Precision | Recall | **F1** | F1 mặc định |
|---|---|---|---|---|---|---|---|
| bbox | 36 | 0 | 5 | 1.000 | 0.878 | **0.935** | 0.909 |
| ankle-abstain | 36 | 0 | 5 | 1.000 | 0.878 | **0.935** | 0.909 |
| ankle-fallback | 36 | 0 | 5 | 1.000 | 0.878 | **0.935** | 0.909 |

**So sánh hai chiều** (F1 test):

| Cách | Stockerau → **Fels** | Fels → **Stockerau** |
|---|---|---|
| bbox | 0.694 (`conf` 0.50) | 0.935 (`conf` 0.50) |
| ankle-abstain | 0.666 (`conf` 0.50) | 0.935 (`conf` 0.70) |
| ankle-fallback | 0.693 (`conf` 0.50) | 0.935 (`conf` 0.70) |

- **Bbox chọn cùng một bộ ngưỡng ở cả hai chiều** (1280, 0.50). Hai cách dùng mắt cá chọn `conf` khác nhau (0.50 và 0.70).
- Cả hai chiều đều chọn imgsz 1280 và `conf` cao hơn mặc định, và F1 test đều tăng. Ở Stockerau, mức tăng chỉ là **thêm 1 TP và bớt 1 FP** (+0.026, khoảng tin cậy 95% từ 0.000 đến +0.071), nên chưa chắc chắn.
- Ba cách cho **cùng TP, FP, FN** trên Stockerau: 41 người vi phạm không đủ để phân biệt chúng.
- **0.935 không có nghĩa chiều này tốt hơn.** Hai chiều chấm trên hai ga khác nhau, và Stockerau là cảnh dễ hơn (box GT đạt 0.975, so với 0.866 ở Fels). Không nên lấy trung bình hai số.
- **Stockerau không phải tập test sạch:** chính lưới này đã được quét trên Stockerau ở mục 5.2, và 0.935 là F1 cao nhất của lưới đó.
- F1 theo frame trên Stockerau là 0.986 (35 TP, 0 FP, 1 FN) nhưng ít ý nghĩa, vì cả 36 frame được chấm đều có người vi phạm.

## 7. Phân tích

### 7.1. Mắt cá so với tâm đáy bbox (câu hỏi chính)

Bộ tìm người ở các lượt bbox và ankle là **cùng một model `yolo11n.pt`**. Vì vậy có thể so sánh trên đúng cùng một tập người: 772 người mà cả hai lượt đều phát hiện (ngưỡng đã chọn).

| So với YOLO + bbox | ankle-abstain | ankle-fallback |
|---|---|---|
| Cả hai cùng đúng | 629 | 706 |
| Ankle sửa được lỗi của bbox | **7** | **7** |
| Ankle làm sai chỗ bbox đúng | 2 | 2 |
| Ankle trả UNKNOWN trong khi bbox đúng | 78 (57 an toàn, 21 vi phạm) | 1 |

Trên 675 người mà cả hai cách (bbox và ankle-abstain) đều đưa ra phán đoán:

| Điểm chân | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| Tâm đáy bbox | 213 | 18 | 26 | 0.922 | 0.891 | 0.906 |
| Mắt cá | 218 | 18 | 21 | 0.924 | 0.912 | **0.918** |

**Diễn giải:**
- Mắt cá giúp ở nhóm khó: **người đứng sát vạch, đang bước, hoặc đứng nghiêng**. Ở những người này, tâm đáy bbox còn nằm phía an toàn trong khi một bàn chân đã qua vạch.
- Nhóm này rất nhỏ (cải thiện ròng 4–5 người trên 772). Lượt ankle-fallback có thêm 10 TP so với bbox, nhưng cũng có thêm 20 FP, đều từ box thừa. Hai mặt bù trừ nhau, nên F1 cuối **ngang nhau**.
- Ở ngưỡng mặc định, ankle-fallback nhỉnh hơn bbox +0.005. Chênh lệch đó không còn khi chọn ngưỡng cho cả hai, nên **không có cơ sở để khẳng định mắt cá tốt hơn**.
- Ở chế độ abstain, khoảng 13% số người tìm được bị trả UNKNOWN (98/781), phần lớn vì không thấy mắt cá (bị che, chân ra ngoài khung). Điều này làm giảm recall end-to-end. Fallback về bbox giải quyết được vấn đề này.

### 7.2. Phân rã lỗi end-to-end

**Nguồn gốc FN** (người vi phạm bị bỏ qua):

| Nguồn | Box GT | YOLO + bbox | ankle-abstain | ankle-fallback |
|---|---|---|---|---|
| Frame không dò được vùng (M2) | 38 | 38 | 38 | 38 |
| Detector bỏ sót (M3) | 0 | 80 | 74 | 74 |
| Phán SAFE sai (điểm chân hoặc vạch lệch) | 36 | 32 | 22 | 28 |
| UNKNOWN | 2 | 1 | 29 | 1 |
| **Tổng FN** | **76** | **151** | **163** | **141** |

**Nguồn gốc FP** (báo động sai):

| Nguồn | YOLO + bbox | ankle-abstain | ankle-fallback |
|---|---|---|---|
| Người không vi phạm bị phán DANGER | 18 | 18 | 18 |
| Box thừa nằm **bên kia mép E** (trên thân tàu) | **31** | **38** | **48** |
| Box thừa nằm trong vùng L–E | 6 | 4 | 9 |
| **Tổng FP** | **55** | **60** | **75** |

Ở ngưỡng mặc định, box thừa bên kia mép E của bbox là 59. Cấu hình đã chọn loại được gần một nửa, nhưng đây vẫn là nguồn FP lớn nhất.

### 7.3. Ba nguồn lỗi chính

1. **Detector bỏ sót người bị che khuất (M3).** Trên test, YOLO + bbox bỏ sót 200/972 người. Tỉ lệ bỏ sót tăng mạnh theo mức che khuất:

   Bảng dưới tính trên toàn bộ 1039 người GT; người trong frame không có vùng không bị tính là bỏ sót.

   | Mức che khuất | Số người | Bị bỏ sót |
   |---|---|---|
   | 0% | 595 | 3 (1%) |
   | 1–25% | 121 | 7 (6%) |
   | 26–50% | 194 | 77 (40%) |
   | 51–75% | 102 | 87 (85%) |
   | > 75% | 27 | 26 (96%) |

   Lúc hành khách lên xuống tàu, người đứng chồng lên nhau ngay sát cửa tàu, cũng chính là nơi có nhiều người vi phạm nhất.

2. **Bóng phản chiếu trên thân tàu (M3 cộng M4).** Thân tàu inox và kính cửa phản chiếu hành khách. YOLO phát hiện những bóng này thành người. Vì chúng nằm bên kia mép E, luật "vượt E thì luôn DANGER" phán chúng là vi phạm. Đây là **nguồn FP lớn nhất: 31/55 FP của YOLO + bbox**, xem ảnh minh hoạ ở mục 8. Một số box bên kia E cũng có thể do E dò lệch. Chẩn đoán trên test: nếu loại được toàn bộ các box này, precision của YOLO + bbox sẽ từ 0.810 lên khoảng 0.91 và F1 từ 0.694 lên khoảng 0.73.

3. **Vùng an toàn tự dò sai (M2).** Ngay cả khi dùng box GT, F1 cũng chỉ đạt 0.866. Phân tích chi tiết ở [FAILURE_ANALYSIS_TEST_GT_AUTO.md](FAILURE_ANALYSIS_TEST_GT_AUTO.md):
   - 15 frame không tìm được mép E vì đám đông che mép (38 FN);
   - khoảng 33 frame có L/E lệch lớn: L dính mép ảnh, L/E nhảy sang thân tàu, hoặc xoay sai góc;
   - người đứng sát vạch bị phán SAFE: dù L đã đặt ở mép phía an toàn của dải sơn, nhãn vẫn coi những người cách L vài chục pixel về phía sân ga là vi phạm (nhãn bảo thủ, hoặc L tự dò bị rung).

   Camera cố định, nên **dùng một cặp L/E ổn định theo thời gian** (trung vị qua các frame) có thể nâng F1 của box GT từ 0.866 lên khoảng 0.93. Mô phỏng này được tính trên chính tập test, chỉ để định hướng.

### 7.4. Từ val sang test (tổng quát sang ga mới)

| | Val (Stockerau) | Test (Fels) |
|---|---|---|
| F1, box GT | 0.975 | 0.866 |
| F1, YOLO + bbox (ngưỡng đã chọn) | 0.935 | 0.694 |
| F1, YOLO + bbox (ngưỡng mặc định) | 0.909 | 0.653 |
| Detector bắt được (ngưỡng đã chọn) | 51% | 79% |
| Tỉ lệ người vi phạm | 16% | 37% |

F1 trên val ở ngưỡng đã chọn là điểm lạc quan, vì ngưỡng được chọn trên chính val. F1 giảm mạnh khi sang ga mới, dù detector ở Fels còn bắt được nhiều người hơn ở Stockerau. Nguyên nhân có thể là **cảnh ở Fels đông và phức tạp hơn**: nhiều người sát cửa tàu, nhiều phản chiếu, mép sân ga hay bị che. Vạch sơn và mép sân ga ở Fels cũng có hình dạng khác Stockerau, nên các heuristic M2 vốn được tinh chỉnh trên ảnh Stockerau và ảnh mẫu khác chuyển sang kém hơn.

## 8. Ca lỗi tiêu biểu

Các ca dưới đây lấy từ các lượt `conf` mặc định (0.25). Ảnh nằm trong thư mục `failures/` của từng lượt, được tạo bằng `scripts/render_failures.py`. Quy ước: viền nét đứt là GT (tím = vi phạm, trắng = an toàn), viền liền là dự đoán (đỏ = DANGER, xanh = SAFE), đường cam là L, đường xanh dương là E.

| Frame | Lượt | Hiện tượng | Nguyên nhân |
|---|---|---|---|
| Fels 04065 | YOLO + bbox | 3 FP "DANGER, no GT" trên thân tàu, 1 FN | YOLO phát hiện **bóng người phản chiếu** trên cửa kính tàu; người thật phía sau bị che nên bị bỏ sót |
| Fels 01590 | Box GT | 6/6 người vi phạm thành FN | Đám đông lên tàu **che kín mép sân ga**, M2 không tìm được E nên cả frame không có vùng |
| Fels 02610 | Box GT | 3 người an toàn bị báo DANGER | M2 đặt **L trùng mép trái ảnh**, cả sân ga thành vùng nguy hiểm |
| Fels 01650 | Box GT | 3 người vi phạm ở xa bị phán SAFE | L/E đúng, nhưng chân nằm **sát vạch** (dưới 30 px) phía an toàn; nhãn coi giẫm lên dải vàng là vi phạm |
| Stockerau 00420 | YOLO + bbox (1280) | 1 FN + 1 FP | Hai người đứng sát nhau bị YOLO **gộp thành một box**, không khớp GT nào |

## 9. Hạn chế và các yếu tố ảnh hưởng độ tin cậy

1. **Mỗi split chỉ có một camera.** Kết quả chỉ đại diện cho hai cảnh cụ thể, không phải cho sân ga nói chung.
2. **Val nhỏ** (41 người vi phạm). Ngưỡng được chọn chỉ hơn mặc định 1 TP và bớt 1 FP trên val, nên không chắc là tối ưu. Ngoài ra, M2 từng được chỉnh trên ảnh Stockerau (xem [PROGRESS.md](PROGRESS.md)), nên điểm val có thể lạc quan.
3. **Test đã bị xem trước khi chọn ngưỡng.** Các lượt mặc định trên test có trước khi chọn ngưỡng, và ankle-fallback cùng YOLO-pose làm detector được chạy thêm sau khi đã xem các lượt test đầu. Ngưỡng chỉ được chọn bằng điểm val, và cấu hình đã chốt được chạy trên test một lần. Dù vậy Fels không còn là tập hoàn toàn mới; muốn có con số độc lập cần thêm camera hoặc ga khác. Ở chiều ngược (mục 6.5), Stockerau cũng đã được quét và M2 từng được chỉnh trên ga này.
4. **Các phân tích "nếu sửa thì F1 bao nhiêu"** (mục 7.3) được tính trên test, chỉ để định hướng, không phải kết quả.
5. **Khoảng tin cậy ở mục 6.2 chỉ phản ánh một camera**, và không tính đến độ bất định của việc chọn ngưỡng trên val. Các chênh lệch nhỏ khác (dưới khoảng 0.01 F1) chưa được kiểm định.
6. **Định nghĩa vạch:** nhãn `beyond_safety_line` do người gán, không có toạ độ vạch. Nhãn có vẻ bảo thủ hơn quy tắc "chân vượt mép phía an toàn của dải sơn" mà hệ thống dùng.
7. **Thời gian chạy** đo trên CPU, có chạy song song nhiều tác vụ, nên chỉ mang tính tham khảo. **NMS IoU và dung sai quanh L** không được quét.
8. Đây là **prototype học thuật**, không dùng cho hệ thống an toàn thực tế.

## 10. Kết luận và hướng phát triển

**Kết luận.**
- Pipeline tự động hoàn toàn (xử lý ảnh cổ điển cho vùng an toàn, YOLO cho người) đạt F1 cho người vi phạm khoảng **0.69 trên ga Fels, khác với ga dùng để chọn ngưỡng**, và F1 cảnh báo theo frame khoảng **0.96–0.97**, precision 0.99.
- Chọn ngưỡng trên val (nâng `conf` và imgsz) cải thiện F1 khoảng **+0.04** cho bbox và ankle-fallback, nhờ giảm báo động sai. Khi đổi vai hai ga, F1 trên Stockerau tăng từ 0.909 lên 0.935, và bbox chọn lại đúng bộ ngưỡng cũ.
- Mắt cá chân cải thiện nhẹ độ chính xác ở người sát vạch, nhưng kết quả cuối **ngang với bbox**. Bbox đơn giản và có precision cao hơn (0.810 so với 0.765); **mắt cá kèm fallback** có recall cao hơn (0.634 so với 0.608). Cách xác định điểm chân **không phải yếu tố quyết định** kết quả.
- Các yếu tố quyết định là **người bị che khuất**, **phản chiếu trên thân tàu** và **độ ổn định của vùng an toàn**.

**Hướng phát triển**, xếp theo mức tác động ước tính:
1. Loại các phát hiện nằm trên vùng thân tàu, ví dụ bằng mask tàu `yolo11n-seg` đã có sẵn trong `train_mask.py`, hoặc bỏ box có chân nằm xa bên kia E.
2. Ổn định L/E theo thời gian, vì camera cố định: lấy trung vị trên nhiều frame và giữ L/E cũ khi frame hiện tại không dò được.
3. Thêm khoảng dung sai δ quanh L về phía sân ga và chọn δ trên val.
4. Dùng detector lớn hơn (`yolo11s/m`) hoặc fine-tune để giảm bỏ sót người bị che.
5. Đánh giá thêm các camera và ga khác trong RailEye3D.

## 11. Tái lập

```powershell
# Dữ liệu
railguard prepare --annotations data/annotations/raileye3d_dataset --images data `
  --sequences red-00-00-front-left red-04-01-front-left --out data/processed/manifest.csv
railguard audit --manifest data/processed/manifest.csv --out outputs/data_audit.json
railguard split --manifest data/processed/manifest.csv --split-mode pretrained_val_test --out data/processed/splits.csv

# Một lượt đánh giá (đổi --split val|test, --detector gt|yolo|pose, --pose-fallback abstain|bbox)
railguard evaluate --manifest data/processed/splits.csv --configs configs --split test `
  --detector pose --weights yolo11n-pose.pt --pose-fallback bbox `
  --auto-boundary --boundary-method hybrid_region_evidence --out outputs/test_pose_fallback_auto

# Chọn ngưỡng trên val, chốt vào selection.json, rồi chấm test
python scripts/select_thresholds.py val
python scripts/select_thresholds.py test

# Chiều ngược: chọn ngưỡng trên Fels, chấm trên Stockerau
python scripts/select_thresholds.py val --swap
python scripts/select_thresholds.py test --swap

# Ảnh các frame lỗi, phân tích vùng an toàn
python scripts/render_failures.py --run-dir outputs/test_pose_fallback_auto
python scripts/analyze_boundary_errors.py --split test
```

| Thư mục kết quả | Cấu hình |
|---|---|
| `outputs/threshold_selection/test/{bbox,ankle_abstain,ankle_fallback}` | **Kết quả chính**: ngưỡng chọn trên val |
| `outputs/threshold_selection/{val_sweep.csv,selection.json}` | Bảng quét val và ngưỡng đã chốt (`*_grid1`: vòng quét đầu) |
| `outputs/threshold_selection_swap/` | Chiều ngược (mục 6.5), cấu trúc giống `threshold_selection/`, kèm `splits_swap.csv` |
| `outputs/{val,test}_gt_auto` | Box GT + bbox |
| `outputs/{val,test}_yolo_auto` | YOLO + bbox, ngưỡng mặc định (val có thêm `_1280`) |
| `outputs/{val,test}_pose_auto` | YOLO + ankle-abstain (val có thêm `_1280`) |
| `outputs/{val,test}_pose_fallback_auto` | YOLO + ankle-fallback |
| `outputs/{val,test}_yolo_poseweights_auto` | YOLO-pose làm detector + bbox |

Các thư mục `*_auto` dùng `conf` 0.25 và imgsz 640 (riêng `_1280` dùng imgsz 1280). Mỗi thư mục kết quả gồm `metrics.json`, `predictions.csv` và `failure_cases.csv`. Ảnh lỗi `failures/` chỉ có ở các lượt `gt`, `yolo` và `pose` (`*_auto`, `*_auto_1280`); chưa vẽ cho ankle-fallback, YOLO-pose làm detector và các lượt đã chọn ngưỡng. Thư mục `outputs/` không được commit.
