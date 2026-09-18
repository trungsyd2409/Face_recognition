"""
main_webcam.py
Xem và điều khiển MODEL 3D bằng cử chỉ tay qua webcam.

Model 3D được đọc từ thư mục `models_3d/` (file .obj; file .fbx/.glb/.stl... sẽ
được tự động convert sang .obj nếu máy có assimp hoặc Blender), rồi tự chiếu và
vẽ đè lên hình webcam bằng numpy thuần - xem model3d.py để biết các bước chiếu
phối cảnh, cull mặt sau, sắp xếp độ sâu và tô sáng.

Điều khiển bằng tay (xem hand_control.py):
    - Chụm ngón cái + trỏ rồi kéo tay   -> xoay model
    - 2 tay đưa xa / lại gần nhau       -> phóng to / thu nhỏ
    - Nắm tay rồi di chuyển             -> kéo model đi trong khung hình
    - Giơ N ngón (1 tay, giữ yên 1 nhịp)-> đổi sang model thứ N

Phím tắt:
    q : thoát            r : đặt lại góc nhìn      w : bật/tắt khung dây
    n : model kế tiếp    s : bật/tắt tự xoay       h : ẩn/hiện bảng hướng dẫn

Cách chạy:
    python main_webcam.py
"""

import os

import cv2

import model3d
from utils import detect_hands, log_hand_gesture
from hand_control import ModelController

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models_3d")

# Chỉ ghi log cử chỉ tay mỗi N frame để tránh ghi quá nhiều dòng trùng lặp.
RECOGNIZE_EVERY_N_FRAMES = 15

MODEL_COLOR = (210, 170, 90)     # màu model (BGR) - xanh ngọc nhạt
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


def draw_hud(frame, mesh, controller, show_help):
    """Hiện tên model, cử chỉ đang nhận và danh sách phím tắt."""
    lines = [
        f"Model: {mesh.name}  ({mesh.face_count} mat)",
        f"Cu chi: {controller.action}   Zoom: {controller.scale:.2f}x",
    ]
    if show_help:
        lines += [
            "Chum ngon + keo = xoay | 2 tay xa/gan = zoom",
            "Nam tay = di chuyen | gio N ngon = doi model",
            "q thoat  r reset  w khung day  n model sau  s tu xoay  h an bang nay",
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
    controller = ModelController()
    wireframe = False
    show_help = True
    frame_count = 0

    print("Đang chạy webcam - xem model 3D bằng cử chỉ tay. Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được frame từ webcam.")
            break

        frame_count += 1
        frame_h, frame_w = frame.shape[:2]

        # ---- Nhận diện bàn tay + đọc cử chỉ ----
        hands_info = detect_hands(frame)
        controller.update(hands_info, frame_w, frame_h,
                          model_count=max(len(model_files), 1))

        # ---- Vẽ model 3D đè lên hình webcam ----
        mesh = get_mesh(model_files, controller.model_index, mesh_cache)
        model3d.render_mesh(
            frame, mesh,
            yaw=controller.yaw, pitch=controller.pitch, scale=controller.scale,
            offset=tuple(controller.offset), base_color=MODEL_COLOR,
            wireframe=wireframe,
        )

        draw_fingertips(frame, hands_info)
        draw_hud(frame, mesh, controller, show_help)

        # Ghi log cử chỉ tay theo 1 nhịp cố định
        if frame_count % RECOGNIZE_EVERY_N_FRAMES == 0:
            for hand in hands_info:
                log_hand_gesture(hand["gesture"])

        cv2.imshow("Model 3D dieu khien bang tay - nhan 'q' de thoat", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            controller.reset_view(keep_model=True)
        elif key == ord("w"):
            wireframe = not wireframe
        elif key == ord("n"):
            controller.next_model(max(len(model_files), 1))
        elif key == ord("s"):
            controller.auto_spin = not controller.auto_spin
        elif key == ord("h"):
            show_help = not show_help

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
