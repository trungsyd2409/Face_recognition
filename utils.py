"""
utils.py
Các hàm dùng chung cho cả webcam và ảnh/video tĩnh:
- detect_faces: tìm vị trí khuôn mặt trong 1 frame (dùng OpenCV Haar Cascade)
- recognize_face: so khớp 1 khuôn mặt đã crop với ảnh trong known_faces/ (dùng DeepFace)
- log_recognition: ghi lại kết quả nhận diện vào file CSV
"""

import os
import csv
from datetime import datetime

import cv2
from deepface import DeepFace

# Bộ phát hiện khuôn mặt có sẵn trong OpenCV, không cần train, chạy nhanh, đủ dùng cho project nhỏ
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

KNOWN_FACES_DIR = "known_faces"   # thư mục chứa ảnh mẫu, đặt tên file = tên người
LOG_FILE = "recognition_log.csv"  # file log kết quả nhận diện


def detect_faces(frame):
    """
    Phát hiện khuôn mặt trong 1 frame (ảnh BGR từ OpenCV).
    Trả về danh sách các box dạng (x, y, w, h).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    return faces


def recognize_face(face_img):
    """
    Nhận vào 1 ảnh khuôn mặt đã crop (numpy array, BGR).
    So khớp với các ảnh trong thư mục known_faces/ bằng DeepFace.
    Trả về tên người (lấy từ tên file ảnh) nếu khớp, ngược lại trả về "Unknown".
    """
    # Nếu chưa có ảnh mẫu nào thì không thể nhận diện được ai
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
