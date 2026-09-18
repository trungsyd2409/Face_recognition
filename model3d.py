"""
model3d.py
Đọc file model 3D (.obj) và vẽ nó lên frame webcam bằng NUMPY THUẦN (không cần
OpenGL / thư viện 3D nào).

Toàn bộ quy trình render nằm ở đây, đi đúng các bước cơ bản của đồ hoạ 3D:

1. Đọc file .obj  -> mảng đỉnh (N x 3) + mảng mặt tam giác (M x 3 chỉ số đỉnh)
2. Chuẩn hoá      -> dời tâm model về gốc toạ độ, thu về bán kính 1 (để model
                     nào cũng hiện ra với kích thước hợp lý)
3. Biến đổi       -> xoay (ma trận xoay quanh trục Y rồi trục X), phóng to/thu nhỏ
4. Chiếu phối cảnh-> (x, y, z) 3D  ->  (u, v) 2D trên màn hình:
                        u = cx + f * x / z ,  v = cy - f * y / z
                     vật càng xa (z lớn) thì trông càng nhỏ
5. Cull mặt sau   -> bỏ các mặt quay lưng về phía camera (xét dấu diện tích của
                     tam giác sau khi chiếu) - bớt được khoảng nửa số mặt
6. Sắp xếp độ sâu -> vẽ mặt xa trước, mặt gần sau (thuật toán "painter") để mặt
                     gần che mặt xa, thay cho z-buffer
7. Tô màu         -> độ sáng mỗi mặt tính theo định luật Lambert: sáng nhất khi
                     mặt hướng thẳng về phía nguồn sáng

Ba bước cuối (5, 6, 7) nằm trong hàm dùng chung `draw_faces`, nhận vào mảng đỉnh
đã chiếu sẵn - nhờ vậy chế độ hologram trên lòng bàn tay (palm_ar.py) dùng lại
được y nguyên, chỉ khác cách tính toạ độ chiếu.

Ngoài ra:
- File .fbx được tự động convert sang .obj (dùng assimp CLI hoặc Blender nếu có
  trong máy), vì Python không đọc thẳng FBX được.
- Model quá nhiều mặt sẽ được giảm bớt bằng vertex clustering, nếu không vòng
  lặp vẽ từng mặt trong Python sẽ quá chậm.
"""

import os
import shutil
import subprocess

import cv2
import numpy as np

# Vượt quá số mặt này thì model sẽ được giảm bớt để còn vẽ kịp thời gian thực
DEFAULT_MAX_FACES = 4000

SUPPORTED_DIRECT = (".obj",)
SUPPORTED_CONVERT = (".fbx", ".dae", ".glb", ".gltf", ".stl", ".ply")


# ============================== Đọc / chuẩn bị model ==============================

class Mesh:
    """1 model 3D đã sẵn sàng để vẽ: đỉnh đã chuẩn hoá + các mặt tam giác."""

    def __init__(self, name, vertices, faces):
        self.name = name
        self.vertices = vertices.astype(np.float32)   # (N, 3)
        self.faces = faces.astype(np.int32)           # (M, 3)

    @property
    def face_count(self):
        return len(self.faces)

    def __repr__(self):
        return f"<Mesh {self.name}: {len(self.vertices)} đỉnh, {len(self.faces)} mặt>"


def load_obj(path):
    """
    Đọc file Wavefront .obj -> (vertices, faces).

    Chỉ lấy dòng `v` (đỉnh) và `f` (mặt); bỏ qua vật liệu, toạ độ texture,
    normal... vì renderer này chỉ cần hình khối. Mặt có nhiều hơn 3 đỉnh được
    cắt thành các tam giác (fan triangulation).
    """
    vertices = []
    faces = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                # Mỗi đỉnh của mặt có dạng "12", "12/3", "12/3/4" hoặc "12//4"
                idx = []
                for token in line.split()[1:]:
                    vertex_index = token.split("/")[0]
                    if vertex_index:
                        idx.append(int(vertex_index))
                # Chỉ số âm trong .obj là đếm ngược từ cuối danh sách đỉnh
                idx = [i - 1 if i > 0 else len(vertices) + i for i in idx]
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])

    if not vertices or not faces:
        raise ValueError(f"File không có đỉnh hoặc mặt nào đọc được: {path}")

    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def convert_to_obj(path, output_dir=None):
    """
    Convert 1 file model (.fbx, .glb, .stl...) sang .obj rồi trả về đường dẫn
    file .obj đó. Nếu file .obj đã được convert từ trước thì dùng lại luôn.

    Thử lần lượt: assimp CLI -> Blender (chế độ nền). Không có công cụ nào thì
    báo lỗi kèm hướng dẫn.
    """
    output_dir = output_dir or os.path.dirname(os.path.abspath(path))
    base = os.path.splitext(os.path.basename(path))[0]
    obj_path = os.path.join(output_dir, base + ".converted.obj")

    if os.path.isfile(obj_path) and os.path.getmtime(obj_path) >= os.path.getmtime(path):
        return obj_path

    assimp = shutil.which("assimp")
    if assimp:
        subprocess.run([assimp, "export", path, obj_path],
                       check=True, capture_output=True)
        return obj_path

    blender = shutil.which("blender")
    if blender:
        script = (
            "import bpy, sys;"
            "bpy.ops.wm.read_factory_settings(use_empty=True);"
            f"bpy.ops.import_scene.fbx(filepath=r'{path}')"
            if path.lower().endswith(".fbx") else
            "import bpy, sys;"
            "bpy.ops.wm.read_factory_settings(use_empty=True);"
            f"bpy.ops.wm.obj_import(filepath=r'{path}')"
        )
        script += f";bpy.ops.wm.obj_export(filepath=r'{obj_path}')"
        subprocess.run([blender, "--background", "--python-expr", script],
                       check=True, capture_output=True)
        return obj_path

    raise RuntimeError(
        f"Không convert được '{os.path.basename(path)}' sang .obj vì máy chưa có "
        "công cụ convert.\nCách xử lý (chọn 1):\n"
        "  - Cài assimp CLI và thêm vào PATH, rồi chạy lại\n"
        "  - Cài Blender và thêm vào PATH, rồi chạy lại\n"
        "  - Tự export model sang .obj (Blender: File > Export > Wavefront .obj,\n"
        "    Unity: package FBX Exporter, hoặc web imagetostl.com / convert3d.org)"
    )


def normalize_vertices(vertices):
    """Dời tâm model về gốc toạ độ và thu nhỏ sao cho bán kính lớn nhất = 1."""
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    centered = vertices - center
    radius = float(np.linalg.norm(centered, axis=1).max())
    if radius < 1e-6:
        radius = 1.0
    return centered / radius


def decimate(vertices, faces, max_faces=DEFAULT_MAX_FACES):
    """
    Giảm số mặt bằng VERTEX CLUSTERING: chia không gian thành lưới ô vuông, mọi
    đỉnh rơi vào cùng 1 ô được gộp thành 1 đỉnh (lấy trung bình). Mặt nào có 2-3
    đỉnh gộp chung thì bị bẹp và bị loại.

    Cách này thô hơn các thuật toán giảm mặt xịn (quadric error metrics) nhưng
    chạy rất nhanh, chỉ vài dòng numpy, và đủ tốt để xem model thời gian thực.
    """
    if len(faces) <= max_faces:
        return vertices, faces

    # Dò dần độ mịn của lưới cho tới khi số mặt xuống dưới ngưỡng
    grid = 64
    for _ in range(12):
        keys = np.floor((vertices + 1.0) / 2.0 * grid).astype(np.int64)
        keys = np.clip(keys, 0, grid - 1)
        flat = (keys[:, 0] * grid + keys[:, 1]) * grid + keys[:, 2]

        unique_keys, inverse = np.unique(flat, return_inverse=True)
        new_faces = inverse[faces]
        # Bỏ tam giác bị bẹp (có 2 đỉnh trùng nhau sau khi gộp)
        keep = ((new_faces[:, 0] != new_faces[:, 1]) &
                (new_faces[:, 1] != new_faces[:, 2]) &
                (new_faces[:, 0] != new_faces[:, 2]))
        new_faces = new_faces[keep]

        if len(new_faces) <= max_faces or grid <= 8:
            # Toạ độ đỉnh mới = trung bình các đỉnh cũ trong cùng ô
            sums = np.zeros((len(unique_keys), 3), dtype=np.float64)
            counts = np.zeros(len(unique_keys), dtype=np.float64)
            np.add.at(sums, inverse, vertices)
            np.add.at(counts, inverse, 1.0)
            new_vertices = (sums / counts[:, None]).astype(np.float32)
            # Bỏ mặt trùng lặp
            new_faces = np.unique(np.sort(new_faces, axis=1), axis=0)
            return new_vertices, new_faces.astype(np.int32)

        grid = max(8, int(grid * 0.75))

    return vertices, faces


def load_mesh(path, max_faces=DEFAULT_MAX_FACES):
    """Đọc 1 file model bất kỳ (convert nếu cần) -> đối tượng Mesh sẵn sàng vẽ."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_DIRECT:
        path_obj = convert_to_obj(path)
    else:
        path_obj = path

    vertices, faces = load_obj(path_obj)
    vertices = normalize_vertices(vertices)
    vertices, faces = decimate(vertices, faces, max_faces)
    return Mesh(os.path.splitext(os.path.basename(path))[0], vertices, faces)


def make_cube():
    """Khối lập phương dựng sẵn trong code - dùng khi chưa có file model nào."""
    v = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float32)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                  [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]], dtype=np.int32)
    return Mesh("cube (mac dinh)", normalize_vertices(v), f)


def list_model_files(folder):
    """Liệt kê các file model trong thư mục (bỏ qua file .obj do chính mình convert)."""
    if not os.path.isdir(folder):
        return []
    valid = SUPPORTED_DIRECT + SUPPORTED_CONVERT
    files = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(".converted.obj"):
            continue
        if os.path.splitext(name)[1].lower() in valid:
            files.append(os.path.join(folder, name))
    return files


# ==================================== Render ====================================

def rotation_matrix(yaw, pitch, roll=0.0):
    """Ma trận xoay: quay quanh trục Y (yaw), rồi trục X (pitch), rồi trục Z (roll)."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cx, sx = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(roll), np.sin(roll)

    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ rx @ ry


def draw_faces(frame, mesh, points, base_color=(210, 170, 90), wireframe=False,
               light_dir=(0.4, 0.6, -0.7), alpha=1.0, cull=True, face_alpha=None):
    """
    Rasteriser dùng chung cho mọi chế độ hiển thị.

    `points` là mảng (N, 3): mỗi đỉnh đã được chiếu sẵn thành (x_màn_hình,
    y_màn_hình, độ_sâu). Hàm này lo 3 bước cuối của pipeline:
        - cull mặt sau (xét dấu diện tích tam giác đã chiếu)
        - sắp xếp theo độ sâu, vẽ mặt xa trước
        - tô màu theo định luật Lambert

    Trả về số mặt thực sự được vẽ.
    """
    height, width = frame.shape[:2]
    tri = points[mesh.faces][:, :, :2]         # (M, 3, 2)

    ax = tri[:, 1, 0] - tri[:, 0, 0]
    ay = tri[:, 1, 1] - tri[:, 0, 1]
    bx = tri[:, 2, 0] - tri[:, 0, 0]
    by = tri[:, 2, 1] - tri[:, 0, 1]
    area = ax * by - ay * bx
    visible = (area < 0) if (cull and not wireframe) else np.ones(len(tri), dtype=bool)

    tri_min = tri.min(axis=1)
    tri_max = tri.max(axis=1)
    visible &= ((tri_max[:, 0] >= 0) & (tri_min[:, 0] < width) &
                (tri_max[:, 1] >= 0) & (tri_min[:, 1] < height))
    if not visible.any():
        return 0

    faces = mesh.faces[visible]
    tri = tri[visible]

    depth = points[:, 2][faces].mean(axis=1)
    order = np.argsort(-depth)                 # xa -> gần
    faces = faces[order]
    tri = tri[order]

    # Pháp tuyến tính trong hệ toạ độ có trục y hướng LÊN (màn hình thì y hướng
    # xuống, nên lật dấu y trước khi tính)
    flip = np.array([1.0, -1.0, 1.0], dtype=np.float32)
    p0 = points[faces[:, 0]] * flip
    p1 = points[faces[:, 1]] * flip
    p2 = points[faces[:, 2]] * flip
    normals = np.cross(p1 - p0, p2 - p0)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)

    light = np.array(light_dir, dtype=np.float32)
    light /= np.linalg.norm(light)
    lambert = np.abs(normals @ light) if wireframe else np.clip(normals @ light, 0.0, 1.0)
    shade = 0.26 + 0.74 * lambert              # 0.26 = ánh sáng môi trường

    base = np.array(base_color, dtype=np.float32)
    colors = np.clip(shade[:, None] * base, 0, 255).astype(np.int32)

    blend = alpha < 0.999
    target = frame.copy() if blend else frame
    tri_int = np.round(tri).astype(np.int32)

    if wireframe:
        cv2.polylines(target, tri_int, isClosed=True,
                      color=tuple(int(c) for c in base), thickness=1, lineType=cv2.LINE_AA)
    else:
        for polygon, color in zip(tri_int, colors):
            cv2.fillConvexPoly(target, polygon,
                               (int(color[0]), int(color[1]), int(color[2])),
                               lineType=cv2.LINE_AA)

    if blend:
        cv2.addWeighted(target, alpha, frame, 1.0 - alpha, 0, dst=frame)

    return len(tri)


def render_mesh(frame, mesh, yaw=0.0, pitch=0.0, roll=0.0, scale=1.0,
                offset=(0, 0), base_color=(210, 170, 90), wireframe=False,
                camera_distance=3.2, light_dir=(0.4, 0.6, -0.7), alpha=1.0):
    """
    Vẽ `mesh` đè lên `frame` (ảnh BGR của OpenCV). Vẽ trực tiếp lên frame.

    yaw/pitch/roll : góc xoay (radian)
    scale          : hệ số phóng to/thu nhỏ
    offset         : dời model trên màn hình, tính bằng pixel (dx, dy)
    base_color     : màu cơ bản của model (BGR)
    wireframe      : True = chỉ vẽ khung dây, False = tô mặt kín
    alpha          : độ mờ khi chồng lên hình webcam (1.0 = đục hoàn toàn)

    Trả về số mặt thực sự được vẽ (tiện để hiện lên màn hình).
    """
    height, width = frame.shape[:2]
    cx = width / 2.0 + offset[0]
    cy = height / 2.0 + offset[1]
    focal = 0.9 * min(width, height)

    # --- Bước 3: xoay + phóng to/thu nhỏ ---
    rot = rotation_matrix(yaw, pitch, roll)
    points = (mesh.vertices * scale) @ rot.T
    z = points[:, 2] + camera_distance
    z = np.maximum(z, 0.05)             # tránh chia cho 0 với đỉnh sát camera

    # --- Bước 4: chiếu phối cảnh về toạ độ màn hình ---
    u = cx + focal * points[:, 0] / z
    v = cy - focal * points[:, 1] / z

    # Gom lại thành (x_màn_hình, y_màn_hình, độ_sâu) rồi giao cho rasteriser chung
    points = np.stack([u, v, z], axis=1)
    return draw_faces(frame, mesh, points, base_color=base_color,
                      wireframe=wireframe, light_dir=light_dir, alpha=alpha)
