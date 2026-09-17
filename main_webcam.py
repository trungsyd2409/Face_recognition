"""
main_webcam.py
Chạy face detection + emotion recognition + hand detection thời gian thực qua webcam.

- Khuôn mặt: dùng DeepFace.analyze() để phân tích cảm xúc (angry, disgust, fear,
  happy, sad, surprise, neutral). Khung quanh mặt đổi màu và hiện tên cảm xúc
  (kèm % độ tin cậy) tương ứng với cảm xúc chiếm ưu thế.
- Bàn tay: dùng MediaPipe Hands để phát hiện bàn tay, vẽ khung xương/khớp ngón
  tay, và nhận diện vài cử chỉ cơ bản: nắm tay, xòe tay, thumbs up, hoặc đếm
  số ngón đang giơ.
- Khi cả 2 tay (trái + phải) cùng xuất hiện trong khung hình: với mỗi cặp ngón
  tay liền kề, vẽ 1 tứ giác nối đầu ngón A tay trái -> đầu ngón B tay trái ->
  đầu ngón B tay phải -> đầu ngón A tay phải (khép kín), rồi áp 1 hiệu ứng màu
  lên vùng ảnh bên trong tứ giác đó - giống hiệu ứng tạo 1 "khung ảnh" bằng 2 tay:
    - Ngón cái - ngón trỏ:     đảo màu (invert / âm bản)
    - Ngón trỏ - ngón giữa:    bỏ kênh đỏ   (R = 0)
    - Ngón giữa - ngón áp út:  bỏ kênh xanh lá (G = 0)
    - Ngón áp út - ngón út:    bỏ kênh xanh dương (B = 0)

Cách chạy:
    python main_webcam.py

Nhấn 'q' để thoát.
"""

import cv2
from utils import (
    detect_faces, detect_emotion, log_emotion,
    detect_hands, draw_hand_landmarks, log_hand_gesture,
    get_two_hand_quad_points, invert_quad_region, draw_quad_outline,
    zero_color_channel_in_quad,
    THUMB_TIP_ID, INDEX_TIP_ID, MIDDLE_TIP_ID, RING_TIP_ID, PINKY_TIP_ID,
)

# Không phân tích cảm xúc (DeepFace) ở mọi frame vì sẽ rất chậm/lag.
# Chỉ chạy phân tích mỗi N frame, các frame còn lại dùng lại kết quả gần nhất.
# (Nhận diện tay bằng MediaPipe nhẹ hơn nhiều nên vẫn chạy mỗi frame để mượt.)
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

HAND_TEXT_COLOR = (255, 255, 255)  # trắng


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam. Kiểm tra lại thiết bị hoặc quyền truy cập camera.")
        return

    frame_count = 0
    last_results = {}  # cache (emotion, confidence) cho mặt, key = vị trí xấp xỉ

    print("Đang chạy webcam - nhận diện cảm xúc + bàn tay... Nhấn 'q' trong cửa sổ video để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được frame từ webcam.")
            break

        frame_count += 1

        # ---- Nhận diện khuôn mặt + cảm xúc ----
        faces = detect_faces(frame)

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

        # ---- Nhận diện bàn tay + cử chỉ ----
        hands_info = detect_hands(frame)
        frame_h, frame_w = frame.shape[:2]

        left_hand = next((h for h in hands_info if h["handedness"] == "Left"), None)
        right_hand = next((h for h in hands_info if h["handedness"] == "Right"), None)

        # Khi có đủ 2 tay: áp hiệu ứng màu lên 4 vùng tứ giác (mỗi vùng ứng với
        # 1 cặp ngón liền kề) TRƯỚC, rồi mới vẽ khung xương/nhãn của từng tay đè
        # lên trên, để chúng luôn hiện rõ dù nằm trong hay ngoài các vùng đó.
        if left_hand and right_hand:
            left_lm = left_hand["landmarks"]
            right_lm = right_hand["landmarks"]

            # Ngón cái - ngón trỏ: đảo màu (invert / âm bản)
            quad_thumb_index = get_two_hand_quad_points(
                left_lm, right_lm, frame_w, frame_h, INDEX_TIP_ID, THUMB_TIP_ID,
            )
            invert_quad_region(frame, quad_thumb_index)
            draw_quad_outline(frame, quad_thumb_index)

            # Ngón trỏ - ngón giữa: bỏ kênh đỏ (R = 0)
            quad_index_middle = get_two_hand_quad_points(
                left_lm, right_lm, frame_w, frame_h, INDEX_TIP_ID, MIDDLE_TIP_ID,
            )
            zero_color_channel_in_quad(frame, quad_index_middle, "r")
            draw_quad_outline(frame, quad_index_middle)

            # Ngón giữa - ngón áp út: bỏ kênh xanh lá (G = 0)
            quad_middle_ring = get_two_hand_quad_points(
                left_lm, right_lm, frame_w, frame_h, MIDDLE_TIP_ID, RING_TIP_ID,
            )
            zero_color_channel_in_quad(frame, quad_middle_ring, "g")
            draw_quad_outline(frame, quad_middle_ring)

            # Ngón áp út - ngón út: bỏ kênh xanh dương (B = 0)
            quad_ring_pinky = get_two_hand_quad_points(
                left_lm, right_lm, frame_w, frame_h, RING_TIP_ID, PINKY_TIP_ID,
            )
            zero_color_channel_in_quad(frame, quad_ring_pinky, "b")
            draw_quad_outline(frame, quad_ring_pinky)

        for hand in hands_info:
            draw_hand_landmarks(frame, hand["landmarks"])

            # Ghi log cử chỉ tay theo cùng nhịp với nhận diện cảm xúc, tránh ghi
            # log ở mọi frame (sẽ tạo quá nhiều dòng trùng lặp).
            if frame_count % RECOGNIZE_EVERY_N_FRAMES == 0:
                log_hand_gesture(hand["gesture"])

            # Hiện tên tay (Left/Right) + cử chỉ ngay phía trên cổ tay
            wrist = hand["landmarks"][0]
            text_x = int(wrist.x * frame_w)
            text_y = max(int(wrist.y * frame_h) - 20, 20)
            hand_text = f'{hand["handedness"]}: {hand["gesture"]}'
            cv2.putText(
                frame, hand_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, HAND_TEXT_COLOR, 2,
            )

        cv2.imshow("Nhan dien cam xuc & ban tay - nhan 'q' de thoat", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
