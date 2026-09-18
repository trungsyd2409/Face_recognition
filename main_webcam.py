"""
main_webcam.py
Chạy hand detection + hiệu ứng slime thời gian thực qua webcam.

- Bàn tay: dùng MediaPipe HandLandmarker để phát hiện bàn tay và nhận diện vài
  cử chỉ cơ bản: nắm tay, xòe tay, thumbs up, hoặc đếm số ngón đang giơ (KHÔNG
  vẽ khung xương/khớp đầy đủ của bàn tay).
- Hiệu ứng slime (xem slime_effect.py): ngón cái và ngón trỏ của mỗi tay biến
  thành 2 khối gel dính nhau (kỹ thuật metaball):
    * 2 ngón gần nhau  -> dính thành 1 khối liền
    * kéo ra xa        -> sợi slime thắt eo lại, võng xuống và rung nhẹ
    * kéo quá xa       -> sợi ĐỨT, sinh vài giọt slime rơi xuống
  Bề mặt có khúc xạ nhẹ, viền sáng và đốm sáng phản chiếu cho giống gel bóng.
- Cử chỉ tay được ghi log định kỳ vào hand_log.csv.

Cách chạy:
    python main_webcam.py

Nhấn 'q' để thoát.
"""

import cv2
from utils import detect_hands, log_hand_gesture
from slime_effect import SlimeEffect

# Chỉ ghi log cử chỉ tay mỗi N frame để tránh ghi quá nhiều dòng trùng lặp.
RECOGNIZE_EVERY_N_FRAMES = 15

HAND_TEXT_COLOR = (255, 255, 255)  # trắng


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam. Kiểm tra lại thiết bị hoặc quyền truy cập camera.")
        return

    frame_count = 0
    slime = SlimeEffect()

    print("Đang chạy webcam - hiệu ứng slime... Nhấn 'q' trong cửa sổ video để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được frame từ webcam.")
            break

        frame_count += 1

        # ---- Nhận diện bàn tay + cử chỉ ----
        hands_info = detect_hands(frame)
        frame_h, frame_w = frame.shape[:2]

        # ---- Vẽ slime giữa ngón cái và ngón trỏ của từng tay ----
        slime.update_and_draw(frame, hands_info)

        for hand in hands_info:
            # Ghi log cử chỉ tay theo 1 nhịp cố định, tránh ghi log ở mọi frame
            # (sẽ tạo quá nhiều dòng trùng lặp).
            if frame_count % RECOGNIZE_EVERY_N_FRAMES == 0:
                log_hand_gesture(hand["gesture"])

            # Hiện tên tay (Left/Right) + cử chỉ ngay phía trên cổ tay
            wrist = hand["landmarks"][0]
            text_x = int(wrist.x * frame_w)
            text_y = max(int(wrist.y * frame_h) - 20, 20)
            hand_text = f'{hand["handedness"]}: {hand["gesture"]}'
            # cv2.putText(
            #     frame, hand_text, (text_x, text_y),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, HAND_TEXT_COLOR, 2,
            # )

        cv2.imshow("Hieu ung slime - nhan 'q' de thoat", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
