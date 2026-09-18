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
├── slime_effect.py       # hiệu ứng slime (gel) giữa ngón cái và ngón trỏ - metaball
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

### Hiệu ứng slime giữa 2 ngón

Ngón cái và ngón trỏ của mỗi tay biến thành 2 khối gel dính nhau (kỹ thuật
metaball trong `slime_effect.py`):

- 2 ngón gần nhau → dính thành 1 khối liền.
- Kéo ra xa → sợi slime thắt eo lại, võng xuống theo trọng lực và rung nhẹ.
- Kéo quá xa → sợi **đứt**, sinh vài giọt slime rơi xuống khung hình.
- Bề mặt có khúc xạ nhẹ (ảnh nền bị bẻ cong), viền sáng và đốm sáng phản chiếu.

Chỉnh nhanh ở đầu class `SlimeEffect` trong `slime_effect.py`: `gel_color` (màu),
`break_ratio` (kéo bao xa thì đứt), `blob_ratio` (độ to của khối gel),
`render_scale` (giảm xuống 0.4 nếu máy yếu).

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
