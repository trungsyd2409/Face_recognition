"""
utils.py
Các hàm dùng chung cho cả webcam và ảnh/video tĩnh:
- detect_faces: tìm vị trí khuôn mặt trong 1 frame (dùng FaceDetectorYN - model YuNet)
- recognize_face: so khớp 1 khuôn mặt đã crop với ảnh trong known_faces/ (dùng DeepFace) - dùng cho main_static.py
- log_recognition: ghi lại kết quả nhận diện danh tính vào file CSV
- detect_emotion: phân tích cảm xúc (7 loại: angry, disgust, fear, happy, sad, surprise,
  neutral) của 1 khuôn mặt đã crop (dùng DeepFace.analyze) - dùng cho main_webcam.py
- log_emotion: ghi lại kết quả nhận diện cảm xúc vào file CSV riêng
- detect_hands: phát hiện bàn tay + nhận diện cử chỉ cơ bản trong 1 frame (dùng
  MediaPipe Tasks API - HandLandmarker) - dùng cho main_webcam.py
- draw_hand_landmarks: vẽ khung xương/khớp ngón tay lên frame
- log_hand_gesture: ghi lại kết quả nhận diện cử chỉ tay vào file CSV riêng
- get_two_hand_quad_points: khi có đủ 2 tay (trái + phải), tính 4 điểm góc tứ giác
  nối 1 cặp đầu ngón tay bất kỳ (mặc định: ngón cái + ngón trỏ) của 2 tay
- invert_quad_region: đảo màu (âm bản) vùng ảnh nằm trong 1 tứ giác
- zero_color_channel_in_quad: đặt 1 kênh màu (r/g/b) về 0 cho vùng ảnh trong 1 tứ giác
- draw_quad_outline: vẽ đường viền khép kín nối các điểm của tứ giác
- is_pinching: kiểm tra 1 bàn tay có đang "chụm" ngón cái + ngón trỏ lại (chạm nhau)
  hay không
- apply_quad_color_effect: áp 1 hiệu ứng màu (đảo màu, hoặc bỏ 1 kênh r/g/b) lên
  vùng ảnh nằm trong 1 tứ giác

Lưu ý: từ OpenCV 5.0, CascadeClassifier (Haar Cascade) đã bị chuyển sang module
contrib riêng, không còn có sẵn trong opencv-python mặc định. Vì vậy project này
dùng FaceDetectorYN - 1 model deep learning nhỏ gọn (YuNet), có sẵn trong
opencv-python bản thường và cho kết quả chính xác hơn Haar Cascade cũ.

Lưu ý về MediaPipe: từ bản mới, API cũ `mediapipe.solutions.hands` đã bị loại bỏ
hoàn toàn, thay bằng Tasks API (`mediapipe.tasks.python.vision.HandLandmarker`),
cần 1 file model `.task` tải riêng (khác với model YuNet ở trên).

Chạy `python download_model.py` một lần trước khi dùng để tải các file model này về.
"""

import os
import csv
import time
from datetime import datetime

import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2026may.onnx")
HAND_MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")

KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
LOG_FILE = os.path.join(BASE_DIR, "recognition_log.csv")
EMOTION_LOG_FILE = os.path.join(BASE_DIR, "emotion_log.csv")
HAND_LOG_FILE = os.path.join(BASE_DIR, "hand_log.csv")

_face_detector = None  # khởi tạo 1 lần duy nhất, dùng lại cho các lần detect sau


def _get_face_detector(width, height):
    global _face_detector

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            "Chưa có file model nhận diện khuôn mặt.\n"
            "Hãy chạy lệnh sau 1 lần trước khi dùng: python download_model.py"
        )

    if _face_detector is None:
        _face_detector = cv2.FaceDetectorYN.create(
            MODEL_PATH, "", (width, height),
            score_threshold=0.6, nms_threshold=0.3, top_k=5000,
        )
    else:
        _face_detector.setInputSize((width, height))

    return _face_detector


def detect_faces(frame):
    """
    Phát hiện khuôn mặt trong 1 frame (ảnh BGR từ OpenCV).
    Trả về danh sách các box dạng (x, y, w, h).
    """
    height, width = frame.shape[:2]
    detector = _get_face_detector(width, height)

    _, faces = detector.detect(frame)

    boxes = []
    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            # Toạ độ đôi khi âm nhẹ ở gần biên ảnh, giới hạn lại cho an toàn
            x, y = max(x, 0), max(y, 0)
            boxes.append((x, y, w, h))

    return boxes


def recognize_face(face_img):
    """
    Nhận vào 1 ảnh khuôn mặt đã crop (numpy array, BGR).
    So khớp với các ảnh trong thư mục known_faces/ bằng DeepFace.
    Trả về tên người (lấy từ tên file ảnh) nếu khớp, ngược lại trả về "Unknown".
    """
    # Nếu ảnh crop rỗng (box lỗi) hoặc chưa có ảnh mẫu nào thì bỏ qua
    if face_img.size == 0:
        return "Unknown"
    if not os.path.isdir(KNOWN_FACES_DIR) or not os.listdir(KNOWN_FACES_DIR):
        return "Unknown"

    try:
        results = DeepFace.find(
            img_path=face_img,
            db_path=KNOWN_FACES_DIR,
            enforce_detection=False,   # không bắt buộc phải detect lại (đã crop sẵn)
            silent=True,
        )
        # DeepFace.find trả về list các DataFrame (1 DataFrame cho mỗi khuôn mặt tìm thấy trong img_path)
        if len(results) > 0 and len(results[0]) > 0:
            best_match = results[0].iloc[0]
            identity_path = best_match["identity"]
            name = os.path.splitext(os.path.basename(identity_path))[0]
            return name
    except Exception as e:
        print(f"[Lỗi khi nhận diện]: {e}")

    return "Unknown"


def log_recognition(name):
    """Ghi lại tên người + thời gian nhận diện vào file CSV."""
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "name"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name])


def detect_emotion(face_img):
    """
    Nhận vào 1 ảnh khuôn mặt đã crop (numpy array, BGR).
    Phân tích cảm xúc bằng DeepFace.analyze() - trả về cảm xúc chiếm ưu thế nhất
    trong 7 loại: angry, disgust, fear, happy, sad, surprise, neutral.

    Trả về tuple (emotion, confidence):
    - emotion: tên cảm xúc (str), hoặc "Unknown" nếu không phân tích được.
    - confidence: độ tin cậy (%) của cảm xúc đó, dạng float (0-100).
    """
    if face_img.size == 0:
        return "Unknown", 0.0

    try:
        results = DeepFace.analyze(
            img_path=face_img,
            actions=["emotion"],
            enforce_detection=False,  # không bắt buộc phải detect lại (đã crop sẵn)
            detector_backend="skip",  # bỏ qua detect lại (opencv/haarcascade không có sẵn), dùng thẳng ảnh đã crop
            silent=True,
        )
        # DeepFace.analyze trả về list các dict (1 dict cho mỗi khuôn mặt tìm thấy)
        if isinstance(results, list) and len(results) > 0:
            result = results[0]
            dominant_emotion = result["dominant_emotion"]
            confidence = result["emotion"][dominant_emotion]
            return dominant_emotion, confidence
    except Exception as e:
        print(f"[Lỗi khi phân tích cảm xúc]: {e}")

    return "Unknown", 0.0


def log_emotion(emotion):
    """Ghi lại cảm xúc + thời gian nhận diện vào file CSV riêng (emotion_log.csv)."""
    file_exists = os.path.isfile(EMOTION_LOG_FILE)
    with open(EMOTION_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "emotion"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), emotion])


# ==================== Nhận diện bàn tay (MediaPipe Tasks - HandLandmarker) ====================

_hand_landmarker = None  # khởi tạo 1 lần duy nhất, dùng lại cho các lần detect sau
_last_hand_timestamp_ms = 0  # HandLandmarker (chế độ VIDEO) yêu cầu timestamp tăng dần

# Danh sách các cặp điểm (start, end) nối thành khung xương bàn tay, dùng để tự vẽ
# (bản MediaPipe mới không còn kèm sẵn hàm vẽ draw_landmarks như bản cũ)
_HAND_CONNECTIONS = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS

# Chỉ số landmark đầu ngón tay (tip) theo thứ tự: thumb, index, middle, ring, pinky
_FINGER_TIP_IDS = [4, 8, 12, 16, 20]


def _get_hand_landmarker():
    global _hand_landmarker

    if not os.path.isfile(HAND_MODEL_PATH):
        raise FileNotFoundError(
            "Chưa có file model nhận diện bàn tay.\n"
            "Hãy chạy lệnh sau 1 lần trước khi dùng: python download_model.py"
        )

    if _hand_landmarker is None:
        options = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)

    return _hand_landmarker


def detect_hands(frame):
    """
    Phát hiện bàn tay trong 1 frame (ảnh BGR từ OpenCV) bằng MediaPipe HandLandmarker.
    Trả về danh sách dict, mỗi dict ứng với 1 bàn tay:
        {"landmarks": list 21 điểm landmark (toạ độ chuẩn hoá 0-1, có .x/.y/.z),
         "handedness": "Left" hoặc "Right",
         "gesture": tên cử chỉ nhận diện được}
    """
    global _last_hand_timestamp_ms

    landmarker = _get_hand_landmarker()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # timestamp (ms) phải tăng dần qua từng lần gọi ở chế độ VIDEO
    timestamp_ms = int(time.time() * 1000)
    if timestamp_ms <= _last_hand_timestamp_ms:
        timestamp_ms = _last_hand_timestamp_ms + 1
    _last_hand_timestamp_ms = timestamp_ms

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    hands_info = []
    if result.hand_landmarks and result.handedness:
        for hand_landmarks, handedness_categories in zip(
            result.hand_landmarks, result.handedness
        ):
            handedness = handedness_categories[0].category_name  # "Left" hoặc "Right"
            gesture = _classify_gesture(hand_landmarks, handedness)
            hands_info.append({
                "landmarks": hand_landmarks,
                "handedness": handedness,
                "gesture": gesture,
            })

    return hands_info


def draw_hand_landmarks(frame, hand_landmarks):
    """Vẽ các điểm khớp (landmarks) + đường nối các ngón tay lên frame."""
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for connection in _HAND_CONNECTIONS:
        cv2.line(frame, points[connection.start], points[connection.end], (255, 255, 255), 2)

    for point in points:
        cv2.circle(frame, point, 4, (0, 200, 0), -1)


def _classify_gesture(landmarks, handedness):
    """
    Nhận vào list 21 điểm landmark (toạ độ chuẩn hoá 0-1, có .x/.y) của 1 bàn tay
    + handedness ("Left"/"Right"). Trả về tên cử chỉ cơ bản: "Nam tay" (fist),
    "Xoe tay" (open palm), "Thumbs up", hoặc "N ngon tay" (đang giơ N ngón).

    Cách làm: với mỗi ngón, so sánh vị trí đầu ngón (tip) với 1 khớp gần đó -
    nếu tip "vươn ra xa hơn" thì coi là ngón đang giơ (extended).
    """
    fingers_up = []

    # Ngón cái: so sánh toạ độ x giữa đầu ngón và khớp IP, hướng so sánh phụ
    # thuộc vào đây là tay trái hay tay phải.
    thumb_tip = landmarks[_FINGER_TIP_IDS[0]]
    thumb_ip = landmarks[_FINGER_TIP_IDS[0] - 1]
    if handedness == "Right":
        fingers_up.append(1 if thumb_tip.x < thumb_ip.x else 0)
    else:
        fingers_up.append(1 if thumb_tip.x > thumb_ip.x else 0)

    # 4 ngón còn lại (index, middle, ring, pinky): ngón đang giơ nếu đầu ngón
    # (tip) nằm cao hơn (toạ độ y nhỏ hơn) khớp giữa (pip).
    for tip_id in _FINGER_TIP_IDS[1:]:
        tip = landmarks[tip_id]
        pip = landmarks[tip_id - 2]
        fingers_up.append(1 if tip.y < pip.y else 0)

    total_up = sum(fingers_up)

    if total_up == 0:
        return "Nam tay"
    if total_up == 5:
        return "Xoe tay"
    if fingers_up == [1, 0, 0, 0, 0]:
        return "Thumbs up"
    return f"{total_up} ngon tay"


def log_hand_gesture(gesture):
    """Ghi lại cử chỉ tay + thời gian nhận diện vào file CSV riêng (hand_log.csv)."""
    file_exists = os.path.isfile(HAND_LOG_FILE)
    with open(HAND_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "gesture"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), gesture])


# ==================== Tứ giác 2 tay + hiệu ứng màu vùng bên trong ====================

# Chỉ số landmark đầu các ngón tay (tip), dùng để ghép cặp tạo tứ giác giữa 2 tay
THUMB_TIP_ID = 4
INDEX_TIP_ID = 8
MIDDLE_TIP_ID = 12
RING_TIP_ID = 16
PINKY_TIP_ID = 20


def _landmark_to_pixel(landmark, width, height):
    """Đổi 1 điểm landmark (toạ độ chuẩn hoá 0-1) sang toạ độ pixel (x, y)."""
    return (int(landmark.x * width), int(landmark.y * height))


def get_two_hand_quad_points(left_landmarks, right_landmarks, width, height,
                              tip_id_a=INDEX_TIP_ID, tip_id_b=THUMB_TIP_ID):
    """
    Khi có đủ landmark của 2 tay (trái + phải), trả về 4 điểm (pixel) tạo thành
    1 tứ giác dựa trên 2 đầu ngón tay `tip_id_a` và `tip_id_b`, theo thứ tự:
        đầu ngón A tay trái -> đầu ngón B tay trái ->
        đầu ngón B tay phải -> đầu ngón A tay phải
    (rồi khép kín lại về điểm đầu tiên).

    Mặc định tip_id_a = ngón trỏ, tip_id_b = ngón cái - tương ứng tứ giác
    "ngón cái - ngón trỏ" giữa 2 tay (tính năng gốc). Truyền các cặp ID khác
    (INDEX_TIP_ID/MIDDLE_TIP_ID, MIDDLE_TIP_ID/RING_TIP_ID, RING_TIP_ID/PINKY_TIP_ID)
    để lấy tứ giác giữa các cặp ngón còn lại.
    """
    left_a = _landmark_to_pixel(left_landmarks[tip_id_a], width, height)
    left_b = _landmark_to_pixel(left_landmarks[tip_id_b], width, height)
    right_b = _landmark_to_pixel(right_landmarks[tip_id_b], width, height)
    right_a = _landmark_to_pixel(right_landmarks[tip_id_a], width, height)

    return [left_a, left_b, right_b, right_a]


def _quad_mask(frame, quad_points):
    """Tạo mask (ảnh xám 0/255) đánh dấu vùng bên trong tứ giác `quad_points`."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.array([quad_points], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)
    return mask.astype(bool)


def invert_quad_region(frame, quad_points):
    """
    Đảo màu (invert - giống hiệu ứng "âm bản") toàn bộ vùng ảnh nằm bên trong
    tứ giác `quad_points` (danh sách 4 điểm pixel (x, y)). Vẽ trực tiếp lên `frame`.
    """
    mask_bool = _quad_mask(frame, quad_points)
    frame[mask_bool] = 255 - frame[mask_bool]


# Tên kênh màu (r/g/b, không phân biệt hoa thường) -> chỉ số kênh trong ảnh BGR
# của OpenCV (kênh 0 = B, 1 = G, 2 = R).
_CHANNEL_NAME_TO_BGR_INDEX = {"b": 0, "g": 1, "r": 2}


def zero_color_channel_in_quad(frame, quad_points, channel):
    """
    Đặt kênh màu `channel` ("r", "g", hoặc "b") về 0 cho toàn bộ vùng ảnh nằm
    bên trong tứ giác `quad_points`. Vẽ trực tiếp lên `frame`.

    Lưu ý: ảnh của OpenCV lưu theo thứ tự kênh BGR (không phải RGB), hàm này tự
    quy đổi tên kênh r/g/b sang đúng chỉ số kênh tương ứng.
    """
    channel_index = _CHANNEL_NAME_TO_BGR_INDEX[channel.lower()]
    mask_bool = _quad_mask(frame, quad_points)
    frame[mask_bool, channel_index] = 0


def draw_quad_outline(frame, quad_points, color=(255, 255, 255), thickness=2):
    """Vẽ đường viền khép kín nối lần lượt các điểm trong `quad_points`."""
    pts = np.array([quad_points], dtype=np.int32)
    cv2.polylines(frame, pts, isClosed=True, color=color, thickness=thickness)


# ==================== Cử chỉ "chụm ngón" (pinch) 1 tay + hiệu ứng theo vùng tứ giác ====================

def _landmark_distance(landmark_a, landmark_b):
    """Khoảng cách Euclid giữa 2 điểm landmark (toạ độ chuẩn hoá 0-1)."""
    return ((landmark_a.x - landmark_b.x) ** 2 + (landmark_a.y - landmark_b.y) ** 2) ** 0.5


def is_pinching(landmarks, ratio_threshold=0.4):
    """
    Kiểm tra ngón cái và ngón trỏ của 1 bàn tay có đang chạm nhau không (cử chỉ
    "chụm ngón" - pinch). So sánh khoảng cách giữa 2 đầu ngón với 1 khoảng cách
    tham chiếu trên chính bàn tay đó (cổ tay - gốc ngón giữa), để không bị ảnh
    hưởng bởi việc tay ở gần hay xa camera.
    """
    thumb_tip = landmarks[THUMB_TIP_ID]
    index_tip = landmarks[INDEX_TIP_ID]
    wrist = landmarks[0]
    middle_mcp = landmarks[9]  # gốc ngón giữa, dùng làm mốc đo "kích thước" bàn tay

    hand_size = _landmark_distance(wrist, middle_mcp)
    if hand_size == 0:
        return False

    pinch_distance = _landmark_distance(thumb_tip, index_tip)
    return (pinch_distance / hand_size) < ratio_threshold


# Chuỗi hiệu ứng lặp vòng khi chụm ngón: đảo màu -> bỏ đỏ -> bỏ xanh lá -> bỏ
# xanh dương -> quay lại đảo màu, ...
COLOR_EFFECT_CYCLE = ["invert", "r0", "g0", "b0"]


def apply_quad_color_effect(frame, quad_points, effect):
    """
    Áp 1 hiệu ứng màu (1 phần tử của COLOR_EFFECT_CYCLE) lên vùng ảnh nằm bên
    trong tứ giác `quad_points` - KHÔNG áp cho toàn bộ khung hình. Vẽ trực tiếp
    lên `frame`.
    """
    if effect == "invert":
        invert_quad_region(frame, quad_points)
    elif effect in ("r0", "g0", "b0"):
        zero_color_channel_in_quad(frame, quad_points, effect[0])  # "r0"->"r", v.v.
