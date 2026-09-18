"""
main_webcam.py
HOLOGRAM TRÊN LÒNG BÀN TAY: đặt 1 model 3D đứng ngay trên lòng bàn tay bạn qua
webcam, model nghiêng và xoay theo tay - kiểu hologram trong phim khoa học viễn
tưởng.

Cách hoạt động (chi tiết trong palm_ar.py và model3d.py):
- MediaPipe HandLandmarker cho 21 điểm landmark của bàn tay, có cả toạ độ z.
- Từ 3 điểm cổ tay / gốc ngón trỏ / gốc ngón út dựng được 1 hệ trục vuông góc
  gắn với lòng bàn tay, trong đó có PHÁP TUYẾN của mặt phẳng lòng bàn tay.
- Model 3D (đọc từ file .obj trong models_3d/) được đặt vào hệ trục đó: đáy
  model chạm mặt phẳng lòng bàn tay, trục đứng của model trùng pháp tuyến.
- Model được chiếu xuống 2D rồi tô bằng numpy thuần (cull mặt sau, sắp xếp độ
  sâu, tô sáng Lambert), kèm bóng đổ, vòng sáng dưới chân và vạch quét hologram.

Điều khiển:
- Xoè bàn tay ra trước camera  -> hologram hiện lên trên lòng bàn tay
- Nghiêng / xoay bàn tay       -> model nghiêng xoay theo
- Nắm tay lại                  -> tắt hologram của tay đó
- Cả 2 tay trong khung hình    -> mỗi tay 1 hologram

Phím tắt:
    q : thoát                 n : đổi model kế tiếp
    w : bật/tắt khung dây     s : bật/tắt tự xoay
    [ / ] : thu nhỏ / phóng to hologram
    h : ẩn/hiện bảng hướng dẫn

Cách chạy:
    python main_webcam.py
"""

import os

import cv2

import model3d
from utils import detect_hands, log_hand_gesture
from palm_ar import PalmHologram

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models_3d")

# Chỉ ghi log cử chỉ tay mỗi N frame để tránh ghi quá nhiều dòng trùng lặp.
RECOGNIZE_EVERY_N_FRAMES = 15

HOLOGRAM_COLOR = (255, 220, 120)   # màu hologram (BGR) - xanh lơ sáng
HUD_COLOR = (255, 255, 255)
FINGER_DOT_COLOR = (0, 200, 255)


def load_model_list():
    """Tìm các file model trong models_3d/. Chưa có file nào thì dùng khối lập phương."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    files = model3d.list_model_files(MODELS_DIR)
    if not files:
        print(f"Chưa có model nào trong '{MODELS_DIR}'.")
        print("Hãy bỏ file .obj (hoặc .fbx/.glb/.stl) vào thư mục đó rồi chạy lại.")
        print("Tạm thời hiển thị khối lập phương dựng sẵn.")
    return files


def get_mesh(files, index, cache):
    """Lấy mesh thứ `index` (đọc file lần đầu, các lần sau lấy từ cache)."""
    if not files:
        if "cube" not in cache:
            cache["cube"] = model3d.make_cube()
        return cache["cube"]

    path = files[index % len(files)]
    if path not in cache:
        try:
            print(f"Đang đọc model: {os.path.basename(path)} ...")
            cache[path] = model3d.load_mesh(path)
            print(f"  -> {cache[path]}")
        except Exception as error:
            print(f"[Lỗi đọc model] {os.path.basename(path)}: {error}")
            cache[path] = model3d.make_cube()
    return cache[path]


def draw_fingertips(frame, hands_info):
    """Chấm nhỏ ở 5 đầu ngón để thấy rõ chương trình đang bám được tay hay chưa."""
    height, width = frame.shape[:2]
    for hand in hands_info:
        for tip_id in (4, 8, 12, 16, 20):
            lm = hand["landmarks"][tip_id]
            cv2.circle(frame, (int(lm.x * width), int(lm.y * height)), 5,
                       FINGER_DOT_COLOR, -1)


def draw_hud(frame, mesh, hologram, hands_count, show_help):
    """Hiện tên model, số tay đang bám được và danh sách phím tắt."""
    lines = [
        f"Model: {mesh.name}  ({mesh.face_count} mat)",
        f"Hologram tren long ban tay  |  {hands_count} tay  |  co {hologram.size_ratio:.2f}",
    ]
    if show_help:
        lines += [
            "Xoe ban tay ra truoc camera = hologram hien len",
            "Nghieng/xoay tay = model nghieng theo | nam tay = tat",
            "q thoat  n doi model  w khung day  s tu xoay  [ ] co  h an bang nay",
        ]

    for i, text in enumerate(lines):
        y = 28 + i * 24
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    HUD_COLOR, 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam. Kiểm tra lại thiết bị hoặc quyền truy cập camera.")
        return

    model_files = load_model_list()
    mesh_cache = {}
    hologram = PalmHologram(color=HOLOGRAM_COLOR)
    model_index = 0
    wireframe = False
    show_help = True
    frame_count = 0

    print("Đang chạy webcam - hologram trên lòng bàn tay. Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được frame từ webcam.")
            break

        frame_count += 1

        # ---- Nhận diện bàn tay ----
        hands_info = detect_hands(frame)
        mesh = get_mesh(model_files, model_index, mesh_cache)

        # ---- Vẽ hologram lên lòng bàn tay của từng tay ----
        hologram.step()
        for hand in hands_info:
            hologram.draw(frame, mesh, hand, wireframe=wireframe)

        draw_fingertips(frame, hands_info)
        draw_hud(frame, mesh, hologram, len(hands_info), show_help)

        # Ghi log cử chỉ tay theo 1 nhịp cố định
        if frame_count % RECOGNIZE_EVERY_N_FRAMES == 0:
            for hand in hands_info:
                log_hand_gesture(hand["gesture"])

        cv2.imshow("Hologram tren long ban tay - nhan 'q' de thoat", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            model_index = (model_index + 1) % max(len(model_files), 1)
            hologram.reset()
        elif key == ord("w"):
            wireframe = not wireframe
        elif key == ord("s"):
            hologram.auto_spin = not hologram.auto_spin
        elif key == ord("["):
            hologram.size_ratio = max(0.15, hologram.size_ratio - 0.05)
        elif key == ord("]"):
            hologram.size_ratio = min(2.0, hologram.size_ratio + 0.05)
        elif key == ord("h"):
            show_help = not show_help

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
