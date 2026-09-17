"""
download_model.py
Tải file model YuNet (pretrained face detector, dùng cho OpenCV 5.x) về máy.
Chỉ cần chạy 1 LẦN DUY NHẤT trước khi dùng main_webcam.py / main_static.py.

Cách chạy:
    python download_model.py
"""

import os
import urllib.request

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2026may.onnx"
)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2026may.onnx")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.isfile(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100_000:
        print(f"Model đã có sẵn tại: {MODEL_PATH}")
        return

    print("Đang tải model nhận diện khuôn mặt (YuNet, khoảng 230KB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        print(f"Lỗi khi tải: {e}")
        print(f"Hãy thử tải thủ công bằng trình duyệt tại link sau, rồi bỏ vào thư mục models/:")
        print(MODEL_URL)
        return

    size_kb = os.path.getsize(MODEL_PATH) / 1024
    if size_kb < 100:
        # File quá nhỏ nghĩa là tải nhầm phải file "con trỏ" git-lfs, không phải file model thật
        print("Cảnh báo: file tải về có vẻ không đúng (dung lượng quá nhỏ).")
        print("Hãy mở link sau bằng trình duyệt, tải file .onnx về, rồi bỏ vào thư mục models/:")
        print(MODEL_URL)
    else:
        print(f"Tải xong! Model đã lưu tại: {MODEL_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
