"""
download_model.py
Tải các file model pretrained cần thiết về máy:
1. YuNet (face_detection_yunet) - phát hiện khuôn mặt, dùng cho OpenCV 5.x
2. HandLandmarker (hand_landmarker.task) - phát hiện bàn tay + cử chỉ, dùng MediaPipe Tasks API

Chỉ cần chạy 1 LẦN DUY NHẤT trước khi dùng main_webcam.py / main_static.py.

Cách chạy:
    python download_model.py
"""

import os
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

FACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2026may.onnx"
)
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2026may.onnx")

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")


def download_model(name, url, path, min_size_kb):
    """Tải 1 file model về `path` nếu chưa có hoặc file cũ có vẻ bị lỗi (quá nhỏ)."""
    if os.path.isfile(path) and os.path.getsize(path) / 1024 > min_size_kb:
        print(f"[{name}] Model đã có sẵn tại: {path}")
        return

    print(f"[{name}] Đang tải model...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        print(f"[{name}] Lỗi khi tải: {e}")
        print(f"[{name}] Hãy thử tải thủ công bằng trình duyệt tại link sau, rồi bỏ vào thư mục models/:")
        print(url)
        return

    size_kb = os.path.getsize(path) / 1024
    if size_kb < min_size_kb:
        # File quá nhỏ nghĩa là tải nhầm phải file "con trỏ" (vd. git-lfs), không phải file model thật
        print(f"[{name}] Cảnh báo: file tải về có vẻ không đúng (dung lượng quá nhỏ: {size_kb:.0f} KB).")
        print(f"[{name}] Hãy mở link sau bằng trình duyệt, tải file về, rồi bỏ vào thư mục models/:")
        print(url)
    else:
        print(f"[{name}] Tải xong! Model đã lưu tại: {path} ({size_kb:.0f} KB)")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    download_model("Face - YuNet", FACE_MODEL_URL, FACE_MODEL_PATH, min_size_kb=100)
    download_model("Hand - HandLandmarker", HAND_MODEL_URL, HAND_MODEL_PATH, min_size_kb=1000)


if __name__ == "__main__":
    main()
