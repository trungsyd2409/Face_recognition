"""
main_webcam.py
TÚM VÀ KÉO HÌNH WEBCAM BẰNG TAY (liquid warp).

Chụm ngón cái + ngón trỏ là "túm" lấy hình ảnh tại đúng chỗ đó, kéo tay đi thì
mảng ảnh quanh chỗ túm bị lôi theo đầu ngón như kéo một tấm cao su. Nhả ngón ra
thì vết méo tan dần, ảnh đàn hồi về hình cũ.

Tay để bình thường (không chụm) thì màn hình hiện y nguyên hình webcam, không
méo chút nào. Hai tay chụm cùng lúc thì mỗi tay túm một chỗ, kéo 2 hướng khác
nhau được.

Chi tiết cách dựng trường dịch chuyển và dùng cv2.remap nằm trong liquid_warp.py.

Phím tắt:
    q : thoát                 r : xoá biến dạng, ảnh về nguyên trạng ngay
    e : đổi độ đàn hồi (vết méo tan nhanh / giữ lâu)
    [ ] : thu nhỏ / mở rộng vùng ảnh bị kéo theo
    m : lật ảnh như soi gương (bật/tắt)

Cách chạy:
    python main_webcam.py
"""

import cv2

from utils import detect_hands, log_hand_gesture, FingerHold
from liquid_warp import LiquidWarp

# Chỉ ghi log cử chỉ tay mỗi N frame để tránh ghi quá nhiều dòng trùng lặp.
RECOGNIZE_EVERY_N_FRAMES = 15

# 2 mức đàn hồi, đổi qua lại bằng phím 'e': tan nhanh <-> giữ vết lâu
DECAY_LEVELS = [0.88, 0.97]


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam. Kiểm tra lại thiết bị hoặc quyền truy cập camera.")
        return

    warp = LiquidWarp()
    # Giữ trạng thái ngón thêm vài frame: MediaPipe thỉnh thoảng đọc nhầm 1-2
    # frame, làm cử chỉ bị nhảy qua lại giữa 2 kiểu biến dạng
    finger_hold = FingerHold(hold=4)

    decay_index = 0
    mirror = True
    frame_count = 0

    print("Đang chạy webcam - chụm 2 ngón rồi kéo để túm hình. Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được frame từ webcam.")
            break

        frame_count += 1
        if mirror:
            # Lật ngang cho giống soi gương - kéo tay sang phải thì ảnh cũng
            # bị kéo sang phải, đỡ bị ngược cảm giác
            frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]

        # ---- Nhận diện bàn tay ----
        hands_info = finger_hold.apply(detect_hands(frame))

        # ---- Cập nhật các cú túm rồi áp biến dạng lên khung hình ----
        warp.update(hands_info, frame_w, frame_h)
        frame = warp.apply(frame)

        # Ghi log cử chỉ tay theo 1 nhịp cố định
        if frame_count % RECOGNIZE_EVERY_N_FRAMES == 0:
            for hand in hands_info:
                log_hand_gesture(hand["gesture"])

        cv2.imshow("Tum va keo hinh bang tay - nhan 'q' de thoat", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            warp.reset()
        elif key == ord("e"):
            decay_index = (decay_index + 1) % len(DECAY_LEVELS)
            warp.decay = DECAY_LEVELS[decay_index]
            print("Độ đàn hồi (decay):", warp.decay)
        elif key == ord("["):
            warp.radius_ratio = max(1.0, warp.radius_ratio - 0.3)
        elif key == ord("]"):
            warp.radius_ratio = min(6.0, warp.radius_ratio + 0.3)
        elif key == ord("m"):
            mirror = not mirror
            warp.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
