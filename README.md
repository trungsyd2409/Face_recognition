# Face Detection & Recognition (Project nhỏ Computer Vision)

Project nhận diện khuôn mặt: phát hiện mặt bằng OpenCV, nhận diện "là ai" bằng DeepFace
(dùng pretrained model, không cần tự train).

**Cập nhật:** chế độ webcam (`main_webcam.py`) giờ đã chuyển từ nhận diện danh tính
sang **nhận diện cảm xúc** (emotion recognition) bằng `DeepFace.analyze()` - hiện đầy đủ
7 loại cảm xúc: Vui (happy), Buồn (sad), Giận (angry), Ngạc nhiên (surprise), Sợ hãi
(fear), Ghê tởm (disgust), Bình thường (neutral). Khung quanh mặt đổi màu theo cảm xúc
và hiện kèm % độ tin cậy. Chế độ ảnh/video tĩnh (`main_static.py`) vẫn giữ nguyên tính
năng nhận diện danh tính như cũ.

**Cập nhật thêm:** chế độ webcam giờ nhận diện thêm **bàn tay** bằng MediaPipe
(`HandLandmarker`) - vẽ khung xương/khớp ngón tay lên hình và nhận diện vài cử chỉ cơ
bản: nắm tay, xòe tay, thumbs up, hoặc đếm số ngón đang giơ.

## Cấu trúc project

```
face_recognition_project/
├── requirements.txt      # danh sách thư viện cần cài
├── utils.py              # các hàm dùng chung (detect, recognize/emotion, log)
├── model3d.py            # đọc file .obj/.fbx + rasteriser 3D thuần numpy
├── palm_ar.py            # dựng hệ trục lòng bàn tay + vẽ hologram đứng trên tay
├── models_3d/            # bỏ file model 3D của bạn vào đây (.obj, .fbx, .glb...)
├── main_webcam.py        # chạy qua webcam thời gian thực - NHẬN DIỆN CẢM XÚC
├── main_static.py        # chạy trên 1 file ảnh hoặc video có sẵn - nhận diện danh tính
├── known_faces/          # bỏ ảnh mẫu (người muốn nhận diện) vào đây - dùng cho main_static.py
├── models/               # chứa 2 file model: YuNet (mặt) + hand_landmarker.task (tay)
├── recognition_log.csv   # tự động tạo khi chạy main_static.py, ghi lại ai được nhận diện lúc nào
├── emotion_log.csv       # tự động tạo khi chạy main_webcam.py, ghi lại cảm xúc + thời gian
└── hand_log.csv          # tự động tạo khi chạy main_webcam.py, ghi lại cử chỉ tay + thời gian
```

## Bước 1: Cài Python và tạo virtual environment

Yêu cầu: Python 3.9–3.11 (DeepFace/TensorFlow chưa hỗ trợ tốt Python quá mới).

Mở terminal trong VS Code, tại thư mục `face_recognition_project`, chạy:

```bash
python -m venv venv
```

Kích hoạt venv:
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

## Bước 2: Cài thư viện

```bash
pip install -r requirements.txt
```

Lưu ý: `deepface` sẽ tự cài kèm `tensorflow` (khá nặng, ~500MB-1GB), quá trình cài
có thể mất vài phút. Đảm bảo máy còn đủ dung lượng và kết nối mạng ổn định.

## Bước 3: Thêm ảnh mẫu (người bạn muốn nhận diện)

Vào thư mục `known_faces/`, đọc file `HUONG_DAN.txt` rồi bỏ ảnh của bạn (và bạn bè
nếu muốn) vào đó. Đặt tên file = tên người, ví dụ `An.jpg`.

## Bước 3.5: Tải các model cần thiết (chỉ cần chạy 1 lần)

```bash
python download_model.py
```

Script này tải **2 file model** về thư mục `models/`:
1. `face_detection_yunet_2026may.onnx` (~230KB) - phát hiện khuôn mặt.
2. `hand_landmarker.task` (vài MB) - phát hiện bàn tay + cử chỉ (MediaPipe).

Cần chạy đúng 1 lần duy nhất trước khi chạy `main_webcam.py` hoặc `main_static.py`
lần đầu tiên.

> **Vì sao cần bước này?** Từ OpenCV 5.0 trở đi, `CascadeClassifier` (cách phát
> hiện khuôn mặt kiểu cũ - Haar Cascade) đã bị chuyển sang module `contrib` riêng,
> không còn có sẵn trong `opencv-python` mặc định nữa. Project này dùng
> `FaceDetectorYN` (model YuNet) thay thế — đây cũng là cách OpenCV chính thức
> khuyến nghị, cho kết quả detect nhanh và chính xác hơn Haar Cascade cũ. Tương
> tự, MediaPipe bản mới cũng yêu cầu tải riêng 1 file model `.task` cho việc nhận
> diện bàn tay (API cũ `mediapipe.solutions.hands` không cần tải file này nhưng
> đã bị loại bỏ khỏi các bản MediaPipe mới).

Nếu script báo lỗi tải hoặc file quá nhỏ (do GitHub dùng Git LFS, hoặc mạng chặn),
mở link được in ra bằng trình duyệt, tải file về thủ công, rồi bỏ vào thư mục
`models/` (giữ đúng tên file như trong thông báo).

## Bước 4: Chạy với webcam (thời gian thực) - nhận diện cảm xúc

```bash
python main_webcam.py
```

- Lần chạy đầu tiên, DeepFace sẽ tự tải model phân tích cảm xúc về máy (cần internet,
  chỉ tải 1 lần duy nhất). Không cần ảnh mẫu trong `known_faces/` cho chế độ này.
- Cửa sổ video hiện lên, khung quanh mặt đổi màu theo cảm xúc và hiện tên cảm xúc
  kèm % độ tin cậy, ví dụ `Vui (87%)`. 7 loại cảm xúc: Vui (xanh lá), Buồn (xanh
  dương), Giận (đỏ), Ngạc nhiên (vàng), Sợ hãi (tím), Ghê tởm (xanh rêu), Bình
  thường (xám).
- Nếu có bàn tay trong khung hình, sẽ hiện thêm khung xương/khớp ngón tay + tên
  cử chỉ (vd. `Right: Xoe tay`, `Left: Thumbs up`, `Right: 2 ngon tay`).
- Nhấn phím `q` để thoát.

### Hologram đứng trên lòng bàn tay

Xoè bàn tay ra trước camera, model 3D trong `models_3d/` sẽ đứng ngay trên lòng
bàn tay bạn, nghiêng và xoay theo tay, có bóng đổ, vòng sáng dưới chân và vạch
quét kiểu hologram. Hai tay trong khung hình thì mỗi tay một hologram.

| Cử chỉ / phím | Tác dụng |
|---|---|
| Xoè bàn tay ra trước camera | Hologram hiện lên trên lòng bàn tay |
| Nghiêng / xoay bàn tay | Model nghiêng xoay theo |
| Nắm tay lại | Tắt hologram của tay đó |
| `n` | Đổi sang model kế tiếp |
| `[` `]` | Thu nhỏ / phóng to hologram |
| `w` `s` `h` `q` | Khung dây · tự xoay · ẩn hướng dẫn · thoát |

**Toán đằng sau** (`palm_ar.py`): chỉ cần 3 landmark là dựng được cả mặt phẳng
lòng bàn tay trong không gian 3D, vì MediaPipe trả về cả toạ độ `z` tương đối:

```
u = chuẩn hoá(gốc ngón út − gốc ngón trỏ)        # trục ngang lòng bàn tay
f = chuẩn hoá(trung điểm 2 gốc ngón − cổ tay)    # trục dọc theo ngón tay
n = chuẩn hoá(f × u)                             # pháp tuyến lòng bàn tay
v = n × u                                        # trục còn lại trong mặt phẳng
```

Model được đặt vào hệ trục này, đáy chạm mặt phẳng lòng bàn tay. Một chi tiết
đáng chú ý: nếu dựng model thẳng đúng theo pháp tuyến `n` thì khi xoè tay đối
diện camera, `n` chĩa thẳng vào ống kính nên ta nhìn model từ nóc xuống, trông
bẹt dí. Vì vậy trục đứng thực tế là pha trộn `up = (1−lean)·n + lean·f` với
`lean ≈ 0.7` — model vẫn bám theo tay nhưng luôn nhìn thấy khối.

Phép chiếu ở chế độ này là **chiếu trực giao yếu** (lấy thẳng thành phần x, y
của điểm 3D, z chỉ dùng để sắp xếp độ sâu) — với vật nhỏ nằm gọn trên bàn tay
thì gần như không khác chiếu phối cảnh đầy đủ mà đơn giản hơn nhiều.

**Các bước render trong `model3d.py`** (đúng quy trình đồ hoạ 3D cơ bản):

1. Đọc `.obj` → mảng đỉnh + mảng mặt tam giác
2. Chuẩn hoá: dời tâm về gốc toạ độ, thu về bán kính 1
3. Đặt vào hệ trục bàn tay (hoặc xoay/scale ở chế độ xem thường)
4. Chiếu xuống 2D
5. Cull mặt sau: bỏ mặt quay lưng về camera (xét dấu diện tích tam giác đã chiếu)
6. Sắp xếp độ sâu: vẽ mặt xa trước, mặt gần sau (thuật toán "painter")
7. Tô màu theo định luật Lambert (mặt hướng về nguồn sáng thì sáng hơn)

Ba bước cuối nằm trong hàm dùng chung `draw_faces`, nhận vào mảng đỉnh đã chiếu
sẵn.

**Định dạng model:** `.obj` đọc trực tiếp. `.fbx/.glb/.gltf/.stl/.ply` sẽ được
tự convert sang `.obj` nếu máy có **assimp CLI** hoặc **Blender** trong PATH
(kết quả lưu lại nên chỉ convert 1 lần). Không có công cụ nào thì tự export
bằng Blender (`File > Export > Wavefront .obj`), Unity (package *FBX Exporter*),
hoặc web `imagetostl.com` / `convert3d.org`.

Model trên ~4000 mặt được tự giảm bớt bằng vertex clustering để giữ tốc độ thời
gian thực (đo trong sandbox: ~18 ms/frame cho 2 tay với model ~1300 mặt).

Tinh chỉnh ở `__init__` của `PalmHologram` trong `palm_ar.py`: `size_ratio` (cỡ
model so với bàn tay), `lean` (độ ngả), `hover` (nhấc lên khỏi tay), `spin_speed`
(tốc độ tự xoay), `color`, `alpha`, `scanlines`, `edges`.

Nếu webcam của bạn không phải camera số 0 (máy có nhiều camera), sửa dòng
`cv2.VideoCapture(0)` trong `main_webcam.py` thành `1`, `2`,...

## Bước 5: Chạy với ảnh hoặc video có sẵn - nhận diện danh tính

```bash
python main_static.py duong_dan_toi_anh.jpg
python main_static.py duong_dan_toi_video.mp4
```

Chế độ này vẫn nhận diện "là ai" như cũ (so khớp với ảnh trong `known_faces/`).
Kết quả sẽ được lưu ra file mới cùng thư mục, tên dạng `output_<tên file gốc>`.

## Bước 6: Xem log kết quả

- `emotion_log.csv` (tạo khi chạy `main_webcam.py`): ghi lại thời gian + cảm xúc
  nhận diện được qua webcam.
- `hand_log.csv` (tạo khi chạy `main_webcam.py`): ghi lại thời gian + cử chỉ tay
  nhận diện được qua webcam.
- `recognition_log.csv` (tạo khi chạy `main_static.py`): ghi lại thời gian + tên
  người được nhận diện (chỉ ghi khi nhận diện được, không ghi "Unknown").

Mở các file này bằng Excel hoặc VS Code để xem.

## Xử lý lỗi thường gặp

- **`AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`**: bạn đang
  dùng OpenCV 5.x — xem giải thích ở Bước 3.5, đảm bảo đã chạy `python download_model.py`
  và file model đã có trong `models/`.
- **`cv2.error: ... The function is not implemented. Rebuild the library with
  Windows, GTK+ 2.x or Cocoa support`** (lỗi ở `cv2.imshow`): trong venv đang có
  bản OpenCV không kèm giao diện (`opencv-python-headless`, thường bị cài kèm bởi
  thư viện khác và ghi đè lên bản thường). Sửa bằng:
  ```bash
  pip uninstall -y opencv-python-headless opencv-python opencv-contrib-python
  pip install opencv-python
  ```
  Sau đó `pip list | findstr opencv` chỉ nên còn đúng 1 dòng `opencv-python`.
- **"Không mở được webcam"**: kiểm tra ứng dụng khác có đang dùng camera không,
  hoặc cấp quyền camera cho terminal/VS Code trong Settings của hệ điều hành.
- **Cài `deepface`/`tensorflow` bị lỗi**: thử nâng cấp pip trước
  (`pip install --upgrade pip`), hoặc dùng Python 3.10/3.11 thay vì bản quá mới.
- **Chạy chậm/giật khi dùng webcam**: bình thường, vì nhận diện khuôn mặt (DeepFace)
  nặng hơn nhiều so với phát hiện khuôn mặt (OpenCV). Có thể tăng số trong
  `RECOGNIZE_EVERY_N_FRAMES` ở `main_webcam.py` để chạy mượt hơn (đổi lại nhận diện
  ít thường xuyên hơn).
- **Toàn ra "Unknown"**: kiểm tra ảnh trong `known_faces/` có rõ mặt không, có đúng
  định dạng .jpg/.png không.
- **`FileNotFoundError: Chưa có file model nhận diện bàn tay`**: chưa chạy (hoặc
  chạy chưa xong) `python download_model.py` — kiểm tra thư mục `models/` đã có
  file `hand_landmarker.task` chưa.
- **Cài `mediapipe` bị lỗi / báo không tương thích**: MediaPipe hiện chỉ hỗ trợ
  tốt trên Python 3.9–3.12 (64-bit); nếu đang dùng Python quá mới hoặc bản 32-bit,
  hãy đổi sang Python 3.10/3.11 64-bit.

## Hướng phát triển thêm (nếu muốn làm portfolio)

- Thử đổi model nhận diện của DeepFace (VGG-Face, Facenet, ArcFace...) để so sánh
  độ chính xác — chỉnh tham số `model_name` trong `DeepFace.find()`.
- Làm giao diện đơn giản bằng Streamlit thay vì cửa sổ OpenCV.
- Tự train 1 model classification nhỏ (ví dụ CNN với Keras) trên tập ảnh khuôn mặt
  của vài người, so sánh với cách dùng pretrained model ở đây.
