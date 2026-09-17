"""
main_webcam.py
Chạy face detection + recognition thời gian thực qua webcam.

Cách chạy:
    python main_webcam.py

Nhấn 'q' để thoát.
"""

import cv2
from utils import detect_faces, recognize_face, log_recognition

# Không nhận diện (DeepFace) ở mọi frame vì sẽ rất chậm/lag.
# Chỉ chạy nhận diện mỗi N frame, các frame còn lại dùng lại kết quả gần nhất.
RECOGNIZE_EVERY_N_FRAMES = 15


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam. Kiểm tra lại thiết bị hoặc quyền truy cập camera.")
        return

    frame_count = 0
    last_labels = {}  # cache tên đã nhận diện, key = vị trí xấp xỉ của khuôn mặt

    print("Đang chạy webcam... Nhấn 'q' trong cửa sổ video để thoát.")

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
                label = recognize_face(face_img)
                last_labels[pos_key] = label
                if label != "Unknown":
                    log_recognition(label)
            else:
                label = last_labels.get(pos_key, "Detecting...")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

        cv2.imshow("Face Detection & Recognition - nhan 'q' de thoat", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
