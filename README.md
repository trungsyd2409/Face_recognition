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
├── particles.py          # hệ hạt: vệt lửa bắn ra từ đầu ngón tay
├── magic_circle.py       # vòng ma pháp phát sáng tự xoay ở đầu ngón tay
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

### Vòng ma pháp + vệt lửa ở đầu ngón tay

Mỗi đầu ngón **đang giơ** có một vòng ma pháp phát sáng tự xoay, đồng thời liên
tục bắn ra các hạt sáng tắt dần và để lại vệt. Gập ngón nào thì ngón đó tắt cả
vòng lẫn hạt; nắm tay lại là tắt hết.

**Ngón cái được để trống hoàn toàn** — không vòng, không hạt (kể cả lớp hạt rải
quanh bàn tay cũng bỏ qua các landmark của ngón cái). Trọng lực
mặc định hướng **lên trên** nên hạt bốc lên như tàn lửa (đổi dấu `gravity` trong
`particles.py` nếu muốn hạt rơi xuống). Vung tay càng nhanh thì hạt sinh ra càng nhiều và bắn càng mạnh theo hướng
vung. Chụm ngón cái + trỏ thì các hạt bị hút về điểm chụm, xoáy tròn và gom lại
thành một quả cầu sáng trong tay.

| Cử chỉ / phím | Tác dụng |
|---|---|
| Vung tay nhanh | Hạt sinh nhiều hơn, bắn mạnh theo hướng vung |
| Xoè 4 ngón trỏ / giữa / áp út / út | Các vòng nhỏ gộp thành 1 vòng lớn giữa lòng bàn tay, cả bàn tay ửng sáng |
| Cả 2 tay cùng mở vòng lớn | Chùm đỏ nối 2 tâm vòng, hạt đỏ toả ra hai bên, vùng quanh chùm bị đảo màu (âm bản) |
| Chụm ngón cái + trỏ | Hút hạt về, xoáy tròn thành quả cầu |
| `c` | Đổi bảng màu: lửa → băng → độc → tím (vòng ma pháp đổi theo) |
| `x` | Bật/tắt vòng ma pháp |
| `g` / `t` | Bật tắt trọng lực / vệt sáng |
| `space` / `q` | Xoá hết hạt · thoát |

**Vòng ma pháp** (`magic_circle.py`) màu đỏ, gồm 3 lớp đồng tâm quay với tốc độ
và chiều khác nhau — đó chính là thứ tạo cảm giác "đang chạy phép": vòng ngoài
kèm vạch chia và chấm sáng quay thuận, các cung đứt đoạn ở giữa quay ngược và
nhanh hơn, ngôi sao 5 cánh bên trong quay nhanh nhất. Năm ngón lệch pha nhau nên
không quay trùng nhịp.

**Gộp thành vòng lớn:** xoè đủ 4 ngón trỏ / giữa / áp út / út (ngón cái duỗi hay
cụp đều không tính) thì 5 vòng nhỏ trôi về tâm lòng bàn tay và
mờ đi, đồng thời một vòng lớn hiện ra giữa lòng bàn tay, quay chậm hơn cho ra
dáng "đại phép". Gập bớt ngón thì chạy ngược lại. Quá trình mất đúng **0,1 giây**
(`merge_time`) — tính theo đồng hồ hệ thống chứ không theo số frame, nên máy
nhanh hay chậm thì hiệu ứng vẫn diễn ra đúng chừng ấy lâu.

Trạng thái vòng do một **máy trạng thái** quyết định, không bám theo kết quả
nhận diện từng frame — MediaPipe luôn rung nhẹ ở ngón áp út và ngón út khi bàn
tay hơi nghiêng, nếu bám thẳng vào đó thì vòng lớn co bóp liên tục. Luật chuyển
trạng thái là **bất đối xứng** (vào dễ, ra khó):

- NHỎ → LỚN: ngay khi đủ 4 ngón xoè
- LỚN → NHỎ: phải thoả **cả hai** — đã ở trạng thái lớn ít nhất `min_big_time`
  (1 giây) **và** điều kiện thiếu ngón kéo dài liên tục ít nhất `exit_delay`
  (0,35 giây); chỉ cần một frame đọc lại đủ ngón là đồng hồ chờ này reset

Biến `merge` (0..1) chỉ còn là biến chạy hoạt hình đi theo trạng thái. Vòng lớn
cũng **không có nhịp phồng xẹp** như vòng nhỏ (`pulse=0`), vì ở cỡ lớn cùng biên
độ đó nhìn thành nhấp nháy chứ không còn là nhịp thở.

Đo trong sandbox (mô phỏng 30 fps, 25–30% số frame bị đọc hụt một ngón): diện
tích vòng lớn dao động 0,7% — phần còn lại là do hoa văn quay, không phải co
bóp. Gập ngón thật và giữ nguyên thì vòng tách sau ~0,47 giây.

Khi vòng lớn mở, hệ hạt chuyển từ bắn ở đầu ngón sang **rải đều khắp bàn tay**:
mỗi hạt được đặt tại một landmark ngẫu nhiên của bàn tay (trừ các điểm thuộc
ngón cái) rồi lệch đi một chút, nên hạt bám đúng hình bàn tay thay vì một hình chữ nhật bao quanh.
Nhóm hạt này gần như không chịu trọng lực (hệ số riêng 0.1, nếu không chúng bốc
lên và rời khỏi tay), bay chậm và mờ hơn hẳn — bàn tay ửng sáng mà không át màu
đỏ của vòng ma pháp. Chỉnh bằng `palm_glow_rate` và `palm_glow_dim` trong
`ParticleSystem`, `big_ratio` và `merge_time` trong `MagicCircles`.

**Chùm nối 2 tay:** khi cả hai bàn tay cùng mở vòng lớn, một chùm sáng đỏ nối
thẳng 2 tâm vòng — gồm một dải sáng hai lớp (lõi mảnh sáng, vỏ dày mờ), vài khối
hạt to trôi dọc theo nó, **hạt đỏ liên tục toả ra hai bên** (bắn theo phương
vuông góc với chùm — `ParticleSystem.emit_link`, nhóm hạt này giữ màu đỏ cố định
thay vì màu theo bảng màu), và **vùng không gian quanh chùm bị đảo màu** thành
âm bản.

Vùng đảo màu là một hình bầu dục ôm lấy đoạn nối, nhưng **chừa ra một hành lang
dọc giữa** cho chùm sáng chạy qua. Phải chừa vì cảnh webcam thường tối: đảo màu
xong vùng đó sáng trắng, mà chùm lại vẽ theo kiểu cộng ánh sáng — cộng lên nền
đã sáng thì cháy trắng và mất hẳn màu đỏ. Chừa hành lang thì chùm vẫn chạy trên
nền tối và giữ màu, còn hai dải âm bản nằm hai bên. Chỉnh bằng `invert_ratio`,
`invert_gap` và `invert_feather`. Các khối được rải cách đều rồi cùng trôi theo thời
gian (vị trí lấy phần lẻ của `t`) nên nhìn như dòng năng lượng chảy giữa hai
tay; khối ở giữa to hơn khối ở hai đầu nên chùm phình ở giữa. Chỉnh bằng
`link_dots` và `link_speed`.

Ba lớp được **vẽ sẵn một lần** vào 3 ảnh mẫu 256×256 nét dày; mỗi frame chỉ xoay
và thu nhỏ chúng bằng `cv2.warpAffine` rồi dán vào đầu ngón. Vẽ trực tiếp từng
nét ở cỡ nhỏ cho ra nét mảnh và nhợt; vẽ sẵn ở cỡ lớn rồi thu nhỏ thì nét đặc,
mượt và nhanh hơn. Việc làm mờ và cộng ánh sáng chỉ chạy trong hình chữ nhật bao
quanh các vòng, không phải cả khung hình — tiết kiệm ~8 ms/frame.

**Hệ hạt** (`particles.py`):

1. **Sinh hạt** — mỗi frame, tại mỗi đầu ngón sinh vài hạt. Vận tốc ban đầu lấy
   từ chính vận tốc của đầu ngón (hiệu vị trí so với frame trước), nên hạt bay
   theo hướng bạn vung tay.
2. **Vật lý** — mọi thuộc tính của hạt là **mảng numpy**, không phải danh sách
   object, nên cả nghìn hạt chỉ tốn vài phép tính mảng mỗi frame:
   `vận tốc += trọng lực` → `vận tốc *= cản` → `vị trí += vận tốc` → `tuổi -= 1`.
   Trọng lực mang giá trị âm (trục y của ảnh hướng xuống) nên hạt bay lên.
3. **Hút khi chụm ngón** — lực hút giảm theo `1/r` (có chặn khoảng cách tối
   thiểu, nếu không hạt sát tâm sẽ nhận lực quá lớn và bị bắn văng đi), cộng
   thêm thành phần vuông góc để hạt xoáy quanh tâm. Hạt vào gần thì bị hãm lại
   và sống lâu hơn, tạo thành quả cầu.
4. **Vẽ** — không vẽ từng hạt bằng `cv2.circle`. Màu các hạt được cộng dồn vào
   một ảnh đệm độ phân giải thấp bằng `np.add.at`, làm mờ 2 lần (bán kính nhỏ
   cho lõi, bán kính lớn cho quầng), rồi **cộng** vào khung hình (additive
   blending) như ánh sáng thật. Ảnh đệm được giữ lại một phần qua các frame nên
   hạt kéo theo vệt mờ dần.

Màn hình chỉ có hình webcam và hiệu ứng hạt — không hiện chữ hướng dẫn nào.

**Về tốc độ:** ba chỗ nặng nhất đều là phép trên toàn khung hình, và cả ba đều
được làm bằng hàm số nguyên của OpenCV thay vì đổi sang float trong numpy:

| Việc | numpy float | OpenCV uint8 |
|---|---|---|
| Pha vùng đảo màu | ~13,7 ms | ~1,3 ms |
| Cộng ánh sáng hạt vào khung hình | ~12,6 ms | ~6,8 ms |
| Cộng ánh sáng vòng ma pháp | — | gộp trong 11,5 ms của `circles.draw` |

Sai lệch màu trung bình giữa hai cách chưa tới 0,2/255 (mắt không thấy được).
Tổng: ~19 ms/frame với 2 tay, đủ cả hạt, vòng lớn, chùm nối và vùng đảo màu.

Tinh chỉnh ở `__init__` của `ParticleSystem`: `max_particles`, `spawn_per_tip`,
`speed_spawn` (độ nhạy với tốc độ vung tay), `speed_scale` (nhân vận tốc ban
đầu, mặc định 1.5), `gravity` (âm = bay lên), `drag`, `life_range`,
`pull_strength`, `trail` (độ dài vệt), `brightness` (độ chói — hạ xuống nếu thấy loá), `render_scale` (giảm nếu máy yếu).

Tinh chỉnh vòng ma pháp ở `__init__` của `MagicCircles`: `radius_ratio` (cỡ
vòng), `spin_speed` (tốc độ quay), `ticks` / `arcs` (số vạch chia và số cung),
`color` (mặc định đỏ), `glow` (độ chói), `blur` (độ toả sáng quanh nét).

**Về việc nhận biết ngón đang giơ** (`utils.get_fingers_up`): cách phổ biến là
so toạ độ y của đầu ngón với khớp giữa, nhưng cách đó chỉ đúng khi bàn tay dựng
thẳng — hơi nghiêng tay là ngón trỏ bị đọc nhầm thành gập, làm hiệu ứng chớp
tắt. Ở đây dùng **so khoảng cách tới cổ tay**: đầu ngón duỗi ra thì xa cổ tay
hơn hẳn khớp giữa, đúng với mọi hướng đặt tay. Thêm lớp `FingerHold` giữ trạng
thái ngón 4 frame trước khi cho là đã gập, để vài frame nhận diện lỗi không làm
hiệu ứng nhấp nháy.

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
