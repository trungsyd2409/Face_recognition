"""
main_webcam.py
KÉO HÌNH WEBCAM NHƯ KÉO MỘT TẤM VẢI (liquid warp).

Hình webcam được coi như tấm vải trải trên mặt bàn. Chụm ngón cái + ngón trỏ là
túm lấy tấm vải tại đúng chỗ đó; kéo tay đi thì cả tấm bị lôi theo - chỗ túm đi
nhiều nhất, càng xa đi càng ít, 4 mép khung hình đứng yên như vải bị ghim đinh.

Nhả tay ra thì vải NẰM YÊN ở chỗ mới, không đàn hồi về. Lần kéo sau túm vào tấm
vải đang nhăn sẵn và kéo tiếp, các nếp nhăn chồng lên nhau. Bấm 'r' để trải
phẳng lại.

Tay để bình thường (không chụm) thì không có gì thay đổi. Hai tay chụm cùng lúc
thì mỗi tay túm một chỗ, kéo hai hướng khác nhau được.

Chi tiết cách dựng trường dịch chuyển và dùng cv2.remap nằm trong liquid_warp.py.

Phím tắt:
    q : thoát                 r : trải phẳng lại tấm vải
    [ ] : thu hẹp / mở rộng tầm ảnh hưởng của cú túm
    m : lật ảnh như soi gương (bật/tắt)

Cách chạy:
    python main_webcam.py
"""

import cv2

from utils import detect_hands, log_hand_gesture, FingerHold
from liquid_warp import LiquidWarp

# Chỉ ghi log cử chỉ tay mỗi N frame để tránh ghi quá nhiều dòng trùng lặp.
RECOGNIZE_EVERY_N_FRAMES = 15

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam. Kiểm tra lại thiết bị hoặc quyền truy cập camera.")
        return

    warp = LiquidWarp()
    # Giữ trạng thái ngón thêm vài frame: MediaPipe thỉnh thoảng đọc nhầm 1-2
    # frame, làm cử chỉ bị nhảy qua lại giữa 2 kiểu biến dạng
    finger_hold = FingerHold(hold=4)

    mirror = True
    frame_count = 0

    print("Đang chạy webcam - chụm 2 ngón rồi kéo tấm vải. 'r' trải phẳng, 'q' thoát.")

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

        # ---- Kéo tấm vải theo tay rồi áp biến dạng lên khung hình ----
        warp.update(hands_info, frame_w, frame_h)
        frame = warp.apply(frame)

        # Ghi log cử chỉ tay theo 1 nhịp cố định
        if frame_count % RECOGNIZE_EVERY_N_FRAMES == 0:
            for hand in hands_info:
                log_hand_gesture(hand["gesture"])

        cv2.imshow("Keo tam vai bang tay - nhan 'q' de thoat", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            warp.reset()
        elif key == ord("["):
            warp.radius_ratio = max(0.8, warp.radius_ratio - 0.3)
        elif key == ord("]"):
            warp.radius_ratio = min(8.0, warp.radius_ratio + 0.3)
        elif key == ord("m"):
            mirror = not mirror
            warp.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
