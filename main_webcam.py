"""
main_webcam.py
Chạy face detection + emotion recognition thời gian thực qua webcam.

Dùng DeepFace.analyze() để phân tích cảm xúc khuôn mặt: angry, disgust, fear,
happy, sad, surprise, neutral. Khung quanh mặt sẽ đổi màu và hiện tên cảm xúc
(kèm % độ tin cậy) tương ứng với cảm xúc chiếm ưu thế.

Cách chạy:
    python main_webcam.py

Nhấn 'q' để thoát.
"""

import cv2
from utils import detect_faces, detect_emotion, log_emotion

# Không phân tích cảm xúc (DeepFace) ở mọi frame vì sẽ rất chậm/lag.
# Chỉ chạy phân tích mỗi N frame, các frame còn lại dùng lại kết quả gần nhất.
RECOGNIZE_EVERY_N_FRAMES = 15

# Màu khung (BGR) theo từng loại cảm xúc
EMOTION_COLORS = {
    "happy": (0, 200, 0),        # xanh lá
    "sad": (255, 120, 0),        # xanh dương
    "angry": (0, 0, 255),        # đỏ
    "surprise": (0, 220, 255),   # vàng
    "fear": (200, 0, 200),       # tím
    "disgust": (0, 128, 100),    # xanh rêu
    "neutral": (180, 180, 180),  # xám
    "Unknown": (255, 255, 255),  # trắng
}

# Nhãn hiển thị tiếng Việt (không dấu để tránh lỗi font khi vẽ bằng cv2.putText)
EMOTION_LABELS_VI = {
    "happy": "Vui",
    "sad": "Buon",
    "angry": "Gian",
    "surprise": "Ngac nhien",
    "fear": "So hai",
    "disgust": "Ghe tom",
    "neutral": "Binh thuong",
    "Unknown": "Khong xac dinh",
}


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam. Kiểm tra lại thiết bị hoặc quyền truy cập camera.")
        return

    frame_count = 0
    last_results = {}  # cache (emotion, confidence), key = vị trí xấp xỉ của khuôn mặt

    print("Đang chạy webcam - nhận diện cảm xúc... Nhấn 'q' trong cửa sổ video để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được frame từ webcam.")
            break

        faces = detect_faces(frame)
        frame_count += 1

        for (x, y, w, h) in faces:
            pos_key = (x // 50, y // 50)  # gộp các vị trí gần nhau lại để cache ổn định hơn

            if frame_count % RECOGNIZE_EVERY_N_FRAMES == 0:
                face_img = frame[y:y + h, x:x + w]
                emotion, confidence = detect_emotion(face_img)
                last_results[pos_key] = (emotion, confidence)
                if emotion != "Unknown":
                    log_emotion(emotion)
            else:
                emotion, confidence = last_results.get(pos_key, ("Detecting...", 0.0))

            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            label = EMOTION_LABELS_VI.get(emotion, emotion)
            if emotion in EMOTION_COLORS and emotion != "Unknown":
                text = f"{label} ({confidence:.0f}%)"
            else:
                text = label

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

        cv2.imshow("Nhan dien cam xuc - nhan 'q' de thoat", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
