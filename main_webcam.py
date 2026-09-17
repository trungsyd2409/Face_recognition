"""
main_webcam.py
Chạy hand detection + nhận diện cử chỉ thời gian thực qua webcam.

- Bàn tay: dùng MediaPipe Hands để phát hiện bàn tay và nhận diện vài cử chỉ
  cơ bản: nắm tay, xòe tay, thumbs up, hoặc đếm số ngón đang giơ (KHÔNG vẽ
  khung xương/khớp đầy đủ của bàn tay).
- Khi cả 2 tay (trái + phải) cùng xuất hiện trong khung hình: vẽ 1 tứ giác nối
  đầu ngón trỏ tay trái -> đầu ngón cái tay trái -> đầu ngón cái tay phải ->
  đầu ngón trỏ tay phải (khép kín) cùng 4 điểm đánh dấu ở các đỉnh, và áp 1
  hiệu ứng màu lên vùng ảnh BÊN TRONG tứ giác đó (không ảnh hưởng phần còn lại
  của khung hình) - giống hiệu ứng tạo 1 "khung ảnh" bằng 2 tay.
- Hiệu ứng áp lên vùng tứ giác đó đổi mỗi khi ngón cái và ngón trỏ của MỘT tay
  (trái hoặc phải) chạm vào nhau (cử chỉ "chụm ngón" - pinch), xoay vòng qua
  13 hiệu ứng (xem COLOR_EFFECT_CYCLE trong utils.py): đảo màu -> bỏ đỏ -> bỏ
  xanh lá -> bỏ xanh dương -> pixelate -> nhiễu hạt -> xoáy -> sóng nước ->
  blur -> cạnh viền -> heatmap nhiệt -> grayscale -> sepia -> (quay lại đảo màu)

Cách chạy:
    python main_webcam.py

Nhấn 'q' để thoát.
"""

import cv2
from utils import (
    detect_hands, log_hand_gesture,
    get_two_hand_quad_points, draw_quad_outline, draw_quad_vertices,
    is_pinching, apply_quad_color_effect, COLOR_EFFECT_CYCLE,
)

# Chỉ ghi log cử chỉ tay mỗi N frame để tránh ghi quá nhiều dòng trùng lặp.
RECOGNIZE_EVERY_N_FRAMES = 15

HAND_TEXT_COLOR = (255, 255, 255)  # trắng


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam. Kiểm tra lại thiết bị hoặc quyền truy cập camera.")
        return

    frame_count = 0

    # Trạng thái hiệu ứng màu áp lên vùng tứ giác 2 tay, đổi mỗi khi phát hiện
    # cử chỉ "chụm ngón" (pinch) MỚI ở 1 trong 2 tay. Bắt đầu ở "invert" (âm
    # bản) - đúng bước đầu tiên trong vòng lặp hiệu ứng.
    current_effect_index = 0
    was_pinching = {"Left": False, "Right": False}

    print("Đang chạy webcam - nhận diện bàn tay... Nhấn 'q' trong cửa sổ video để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được frame từ webcam.")
            break

        frame_count += 1

        # ---- Nhận diện bàn tay + cử chỉ ----
        hands_info = detect_hands(frame)
        frame_h, frame_w = frame.shape[:2]

        left_hand = next(
            (h for h in hands_info if h["handedness"] == "Left"), None)
        right_hand = next(
            (h for h in hands_info if h["handedness"] == "Right"), None)

        # ---- Cử chỉ "chụm ngón" (pinch) của từng tay -> xoay vòng hiệu ứng ----
        # Cập nhật TRƯỚC khi vẽ tứ giác, để nếu vừa chụm ở đúng frame này thì
        # tứ giác hiển thị luôn đúng hiệu ứng mới ngay lập tức.
        # Chỉ đổi hiệu ứng ở đúng thời điểm 2 ngón VỪA chạm nhau (cạnh lên),
        # không đổi liên tục trong lúc vẫn đang giữ chụm.
        for hand in hands_info:
            handedness = hand["handedness"]
            pinching_now = is_pinching(hand["landmarks"])
            if pinching_now and not was_pinching.get(handedness, False):
                current_effect_index = (
                    current_effect_index + 1) % len(COLOR_EFFECT_CYCLE)
            was_pinching[handedness] = pinching_now

        # Khi có đủ 2 tay: áp hiệu ứng màu hiện tại lên vùng tứ giác tạo bởi 4
        # đầu ngón tay TRƯỚC, rồi mới vẽ nhãn của từng tay đè lên trên, để
        # chúng luôn hiện rõ dù nằm trong hay ngoài vùng đó.
        if left_hand and right_hand:
            quad_points = get_two_hand_quad_points(
                left_hand["landmarks"], right_hand["landmarks"], frame_w, frame_h,
            )
            current_effect = COLOR_EFFECT_CYCLE[current_effect_index]
            apply_quad_color_effect(frame, quad_points, current_effect)
            draw_quad_outline(frame, quad_points)
            draw_quad_vertices(frame, quad_points)

            # Hiện tên hiệu ứng hiện tại ngay phía trên tứ giác
            quad_top_point = min(quad_points, key=lambda p: p[1])
            # cv2.putText(
            #     frame, f"Hieu ung: {current_effect}",
            #     (quad_top_point[0], max(quad_top_point[1] - 15, 20)),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            # )

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

        cv2.imshow("Nhan dien ban tay - nhan 'q' de thoat", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
