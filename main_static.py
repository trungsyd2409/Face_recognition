"""
main_static.py
Chạy face detection + recognition trên 1 file ảnh hoặc video có sẵn (không cần webcam).

Cách chạy:
    python main_static.py duong_dan_file.jpg
    python main_static.py duong_dan_file.mp4

Kết quả sẽ được lưu ra file mới với tiền tố "output_".
"""

import os
import sys

import cv2
from utils import detect_faces, recognize_face, log_recognition


def process_image(path):
    frame = cv2.imread(path)
    if frame is None:
        print(f"Không đọc được ảnh: {path}")
        return

    faces = detect_faces(frame)
    print(f"Tìm thấy {len(faces)} khuôn mặt.")

    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        label = recognize_face(face_img)
        log_recognition(label)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

    out_path = f"output_{os.path.basename(path)}"
    cv2.imwrite(out_path, frame)
    print(f"Đã lưu kết quả vào: {out_path}")


def process_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Không đọc được video: {path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = f"output_{os.path.basename(path)}"
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_count = 0
    last_labels = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)
        frame_count += 1

        for (x, y, w, h) in faces:
            pos_key = (x // 50, y // 50)

            # Video xử lý offline nên không cần tối ưu tốc độ nhiều như webcam,
            # nhưng vẫn giới hạn để chạy nhanh hơn với video dài
            if frame_count % 10 == 0:
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

        out.write(frame)

    cap.release()
    out.release()
    print(f"Đã lưu kết quả vào: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python main_static.py duong_dan_file.jpg (hoặc .mp4)")
        sys.exit(1)

    input_path = sys.argv[1]

    if not os.path.isfile(input_path):
        print(f"Không tìm thấy file: {input_path}")
        sys.exit(1)

    ext = os.path.splitext(input_path)[1].lower()

    if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        process_image(input_path)
    elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
        process_video(input_path)
    else:
        print(f"Định dạng file '{ext}' chưa được hỗ trợ.")
