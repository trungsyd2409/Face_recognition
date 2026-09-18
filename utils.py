"""
utils.py
Các hàm dùng chung cho cả webcam và ảnh/video tĩnh:
- detect_faces: tìm vị trí khuôn mặt trong 1 frame (dùng FaceDetectorYN - model YuNet)
- recognize_face: so khớp 1 khuôn mặt đã crop với ảnh trong known_faces/ (dùng DeepFace) - dùng cho main_static.py
- log_recognition: ghi lại kết quả nhận diện danh tính vào file CSV
- detect_hands: phát hiện bàn tay + nhận diện cử chỉ cơ bản trong 1 frame (dùng
  MediaPipe Tasks API - HandLandmarker) - dùng cho main_webcam.py
- draw_hand_landmarks: vẽ khung xương/khớp ngón tay lên frame
- log_hand_gesture: ghi lại kết quả nhận diện cử chỉ tay vào file CSV riêng
- get_fingers_up: trả về trạng thái giơ/gập của 5 ngón của 1 bàn tay
- FingerHold: giữ trạng thái ngón thêm vài frame để hiệu ứng không chớp tắt
- is_pinching: kiểm tra 1 bàn tay có đang "chụm" ngón cái + ngón trỏ lại (chạm
  nhau) hay không - dùng làm cử chỉ xoay model 3D

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
                # [thumb, index, middle, ring, pinky] - 1 = đang giơ, 0 = gập
                "fingers_up": get_fingers_up(hand_landmarks, handedness),
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


def get_fingers_up(landmarks, handedness=None):
    """
    Trả về list 5 phần tử [thumb, index, middle, ring, pinky], mỗi phần tử là
    1 (ngón đang giơ) hoặc 0 (ngón đang gập).

    Cách làm: SO SÁNH KHOẢNG CÁCH TỚI CỔ TAY. Ngón duỗi ra thì đầu ngón (tip)
    nằm xa cổ tay hơn hẳn khớp giữa (pip); ngón gập lại thì đầu ngón cụp vào
    nên khoảng cách đó ngắn lại.

    Cách cũ (so sánh toạ độ y: tip cao hơn pip thì coi là giơ) chỉ đúng khi bàn
    tay dựng thẳng đứng - hơi nghiêng tay hoặc chĩa ngón sang ngang là ngón bị
    đọc nhầm thành gập, gây hiện tượng hiệu ứng ở ngón trỏ chớp tắt liên tục.
    Cách so khoảng cách này không phụ thuộc vào hướng đặt tay.

    Tham số `handedness` không còn cần thiết, giữ lại cho tương thích ngược.
    """
    wrist = landmarks[0]
    fingers_up = []

    # Ngón cái: so khoảng cách từ gốc ngón trỏ (5) tới đầu ngón cái (4) và tới
    # khớp IP (3). Ngón cái duỗi ra thì đầu ngón xa gốc ngón trỏ hơn.
    index_mcp = landmarks[5]
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    fingers_up.append(
        1 if _landmark_distance(index_mcp, thumb_tip) >
             _landmark_distance(index_mcp, thumb_ip) * 1.10 else 0
    )

    # 4 ngón còn lại: đầu ngón (tip) phải xa cổ tay hơn khớp giữa (pip)
    for tip_id in (8, 12, 16, 20):
        tip = landmarks[tip_id]
        pip = landmarks[tip_id - 2]
        fingers_up.append(
            1 if _landmark_distance(wrist, tip) >
                 _landmark_distance(wrist, pip) * 1.08 else 0
        )

    return fingers_up


class FingerHold:
    """
    Giữ trạng thái ngón tay thêm vài frame trước khi cho là đã gập.

    MediaPipe thỉnh thoảng nhận diện lệch 1-2 frame, khiến 1 ngón (hay gặp nhất
    là ngón trỏ khi nó che lấp ngón khác) bị đọc thành gập rồi giơ lại ngay -
    hiệu ứng gắn trên ngón đó sẽ chớp tắt khó chịu. Lớp này nhớ lần cuối mỗi
    ngón được thấy là đang giơ, và chỉ tắt sau `hold` frame liên tiếp không thấy.
    """

    def __init__(self, hold=4):
        self.hold = hold
        self._counters = {}

    def apply(self, hands_info):
        """Sửa trực tiếp trường `fingers_up` của từng bàn tay trong danh sách."""
        seen = set()
        for hand in hands_info:
            handedness = hand["handedness"]
            seen.add(handedness)
            counters = self._counters.setdefault(handedness, [0] * 5)

            held = []
            for i, up in enumerate(hand["fingers_up"]):
                if up:
                    counters[i] = self.hold
                elif counters[i] > 0:
                    counters[i] -= 1
                held.append(1 if counters[i] > 0 else 0)
            hand["fingers_up"] = held

        for handedness in list(self._counters):
            if handedness not in seen:
                self._counters.pop(handedness, None)
        return hands_info


def _classify_gesture(landmarks, handedness):
    """
    Nhận vào list 21 điểm landmark (toạ độ chuẩn hoá 0-1, có .x/.y) của 1 bàn tay
    + handedness ("Left"/"Right"). Trả về tên cử chỉ cơ bản: "Nam tay" (fist),
    "Xoe tay" (open palm), "Thumbs up", hoặc "N ngon tay" (đang giơ N ngón).

    Việc xác định ngón nào đang giơ do get_fingers_up() đảm nhận.
    """
    fingers_up = get_fingers_up(landmarks, handedness)

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


# ==================== Cử chỉ "chụm ngón" (pinch) ====================

# Chỉ số landmark đầu ngón cái / ngón trỏ (theo chuẩn MediaPipe Hands)
THUMB_TIP_ID = 4
INDEX_TIP_ID = 8


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
