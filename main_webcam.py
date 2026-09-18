"""
main_webcam.py
VỆT LỬA + VÒNG MA PHÁP Ở ĐẦU NGÓN TAY.

- Vòng ma pháp ĐỎ phát sáng tự xoay ở mỗi đầu ngón (magic_circle.py)
- Giơ đủ 5 ngón: 5 vòng nhỏ gộp thành 1 vòng lớn giữa lòng bàn tay, đồng thời
  hạt rải đều khắp bàn tay làm cả bàn tay ửng sáng
- Hệ hạt bắn ra từ đầu ngón, tắt dần và để lại vệt sáng (particles.py)

CẢ HAI CHỈ HIỆN Ở NHỮNG NGÓN ĐANG GIƠ - gập ngón nào thì ngón đó tắt, nắm tay
lại thì tắt hết.

- Vung tay nhanh  -> hạt sinh ra nhiều hơn và bắn mạnh theo hướng vung
- Trọng lực hướng LÊN -> hạt bốc lên như tàn lửa (đổi dấu `gravity` nếu muốn rơi)
- Chụm ngón cái + trỏ -> các hạt bị hút về điểm chụm và xoáy tròn quanh đó
- Cả 2 tay trong khung hình đều bắn hạt

Chi tiết cách sinh hạt, vật lý và cách vẽ phát sáng nằm trong particles.py.

Phím tắt:
    q : thoát                 c : đổi bảng màu (lửa / băng / độc / tím)
    g : bật/tắt trọng lực     t : bật/tắt vệt sáng
    x : bật/tắt vòng ma pháp  space : xoá hết hạt

Màn hình chỉ có hình webcam + hiệu ứng hạt, không hiện chữ hướng dẫn nào.

Cách chạy:
    python main_webcam.py
"""

import cv2

from utils import detect_hands, log_hand_gesture, FingerHold
from particles import ParticleSystem
from magic_circle import MagicCircles

# Chỉ ghi log cử chỉ tay mỗi N frame để tránh ghi quá nhiều dòng trùng lặp.
RECOGNIZE_EVERY_N_FRAMES = 15

FINGER_DOT_COLOR = (80, 80, 80)


def draw_fingertips(frame, hands_info):
    """Chấm mờ ở 5 đầu ngón để thấy chương trình có đang bám được tay không."""
    height, width = frame.shape[:2]
    for hand in hands_info:
        for tip_id in (4, 8, 12, 16, 20):
            lm = hand["landmarks"][tip_id]
            cv2.circle(frame, (int(lm.x * width), int(lm.y * height)), 3,
                       FINGER_DOT_COLOR, -1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam. Kiểm tra lại thiết bị hoặc quyền truy cập camera.")
        return

    particles = ParticleSystem()
    circles = MagicCircles()
    # Giữ trạng thái ngón thêm vài frame: MediaPipe thỉnh thoảng đọc nhầm 1-2
    # frame làm hiệu ứng ở ngón đó (hay gặp nhất là ngón trỏ) chớp tắt
    finger_hold = FingerHold(hold=4)
    trail_saved = particles.trail
    frame_count = 0

    print("Đang chạy webcam - vệt lửa từ đầu ngón tay. Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được frame từ webcam.")
            break

        frame_count += 1
        frame_h, frame_w = frame.shape[:2]

        # ---- Nhận diện bàn tay ----
        hands_info = finger_hold.apply(detect_hands(frame))

        # ---- Vòng ma pháp: cập nhật mức "gộp thành vòng lớn" TRƯỚC khi sinh
        # hạt, để hệ hạt biết bàn tay nào đang mở vòng lớn mà rải hạt khắp tay
        circles.update_merge(hands_info)
        for hand in hands_info:
            hand["palm_glow"] = circles.merge_amount(hand["handedness"])

        # ---- Sinh hạt + cập nhật vật lý + vẽ ----
        particles.update(hands_info, frame_w, frame_h)
        draw_fingertips(frame, hands_info)
        particles.draw(frame)
        circles.draw(frame, hands_info)     # vòng ma pháp vẽ đè lên trên lớp hạt

        # Ghi log cử chỉ tay theo 1 nhịp cố định
        if frame_count % RECOGNIZE_EVERY_N_FRAMES == 0:
            for hand in hands_info:
                log_hand_gesture(hand["gesture"])

        cv2.imshow("Vet lua tu dau ngon tay - nhan 'q' de thoat", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            print("Bảng màu hạt:", particles.next_theme())
        elif key == ord("g"):
            particles.gravity_on = not particles.gravity_on
        elif key == ord("t"):
            particles.trail = 0.0 if particles.trail else trail_saved
        elif key == ord("x"):
            circles.toggle()
        elif key == ord(" "):
            particles.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
