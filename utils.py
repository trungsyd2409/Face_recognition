"""
utils.py
Các hàm dùng chung cho cả webcam và ảnh/video tĩnh:
- detect_faces: tìm vị trí khuôn mặt trong 1 frame (dùng FaceDetectorYN - model YuNet)
- recognize_face: so khớp 1 khuôn mặt đã crop với ảnh trong known_faces/ (dùng DeepFace)
- log_recognition: ghi lại kết quả nhận diện vào file CSV

Lưu ý: từ OpenCV 5.0, CascadeClassifier (Haar Cascade) đã bị chuyển sang module
contrib riêng, không còn có sẵn trong opencv-python mặc định. Vì vậy project này
dùng FaceDetectorYN - 1 model deep learning nhỏ gọn (YuNet), có sẵn trong
opencv-python bản thường và cho kết quả chính xác hơn Haar Cascade cũ.
Chạy `python download_model.py` một lần trước khi dùng để tải file model này về.
"""

import os
import csv
from datetime import datetime

import cv2
from deepface import DeepFace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2026may.onnx")

KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
LOG_FILE = os.path.join(BASE_DIR, "recognition_log.csv")

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
