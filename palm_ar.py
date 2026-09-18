"""
palm_ar.py
Chế độ AR: đặt 1 model 3D ĐỨNG TRÊN LÒNG BÀN TAY và xoay theo tay, giống
hologram trong phim.

Ý tưởng toán học: chỉ cần 3 điểm landmark là dựng được cả mặt phẳng lòng bàn tay
trong không gian 3D (MediaPipe trả về cả toạ độ z tương đối, không chỉ x, y):

    cổ tay (0) --- gốc ngón trỏ (5) --- gốc ngón út (17)

Từ 3 điểm đó dựng 1 hệ trục vuông góc gắn với bàn tay:

    u  = hướng NGANG lòng bàn tay   = chuẩn hoá(gốc ngón út - gốc ngón trỏ)
    f  = hướng DỌC theo ngón tay    = chuẩn hoá(trung điểm 2 gốc ngón - cổ tay)
    n  = PHÁP TUYẾN lòng bàn tay    = chuẩn hoá(f x u)
    v  = trục còn lại               = n x u  (cùng hướng với f)

Lưu ý quan trọng về hướng "đứng" của model: nếu cho model dựng thẳng theo đúng
pháp tuyến n thì khi bạn xoè tay đối diện camera, n chĩa thẳng vào ống kính -
ta nhìn model từ trên nóc xuống nên nó trông bẹt dí. Vì vậy trục đứng thực tế
là PHA TRỘN giữa pháp tuyến n và hướng ngón tay f:

    up = chuẩn hoá( (1 - lean) * n  +  lean * f )

với lean ~ 0.7. Model vẫn đứng trên lòng bàn tay và vẫn nghiêng/xoay theo tay,
nhưng hơi ngả về phía các ngón nên luôn nhìn thấy được khối 3D của nó.

Model được đặt vào hệ trục này: trục đứng của model trùng với n, đáy model chạm
mặt phẳng lòng bàn tay. Tay bạn nghiêng/xoay thì u, v, n xoay theo, nên model
trông như thật sự nằm trên tay.

Phép chiếu ở đây là CHIẾU TRỰC GIAO YẾU (weak perspective): lấy thẳng thành phần
(x, y) của điểm 3D trong không gian pixel, còn thành phần z chỉ dùng để sắp xếp
độ sâu. Với vật nhỏ nằm gọn trên bàn tay thì cách này cho kết quả gần như không
khác chiếu phối cảnh đầy đủ, mà đơn giản hơn nhiều.

Vì landmark rung nhẹ qua từng frame, hệ trục và vị trí được LÀM MƯỢT (lọc thông
thấp) trước khi vẽ, nếu không hologram sẽ giật liên tục.
"""

import numpy as np
import cv2

from model3d import draw_faces

WRIST_ID = 0
INDEX_MCP_ID = 5
MIDDLE_MCP_ID = 9
PINKY_MCP_ID = 17


def _normalize(vector):
    length = float(np.linalg.norm(vector))
    return vector / length if length > 1e-8 else vector


def palm_basis(landmarks, width, height):
    """
    Dựng hệ trục gắn với lòng bàn tay từ các landmark.

    Trả về dict gồm:
        origin : tâm lòng bàn tay, toạ độ "pixel 3D" (x, y, z)
        u, v   : 2 trục nằm TRONG mặt phẳng lòng bàn tay
        n      : pháp tuyến lòng bàn tay (hướng ra khỏi lòng bàn tay)
        size   : bề ngang lòng bàn tay tính bằng pixel (dùng để chọn cỡ model)

    MediaPipe trả z theo cùng thang với x, nên nhân z với `width` để cả 3 trục
    cùng đơn vị pixel.
    """
    def to_3d(landmark):
        return np.array([landmark.x * width, landmark.y * height, landmark.z * width],
                        dtype=np.float32)

    wrist = to_3d(landmarks[WRIST_ID])
    index_mcp = to_3d(landmarks[INDEX_MCP_ID])
    pinky_mcp = to_3d(landmarks[PINKY_MCP_ID])
    middle_mcp = to_3d(landmarks[MIDDLE_MCP_ID])

    u = _normalize(pinky_mcp - index_mcp)                       # ngang lòng bàn tay
    forward = _normalize((index_mcp + pinky_mcp) / 2.0 - wrist)  # dọc theo ngón tay
    n = _normalize(np.cross(forward, u))                         # pháp tuyến

    # Pháp tuyến phải hướng về phía camera (z càng nhỏ càng gần camera trong
    # hệ toạ độ của MediaPipe), nếu không model sẽ mọc xuyên qua bàn tay.
    if n[2] > 0:
        n = -n
    v = _normalize(np.cross(n, u))

    # Tâm đặt model: giữa lòng bàn tay, hơi lệch về phía các ngón
    # v tính từ tích có hướng có thể ngược chiều ngón tay -> lật lại cho chắc
    if float(np.dot(v, forward)) < 0:
        v = -v

    origin = (wrist + index_mcp + pinky_mcp + middle_mcp) / 4.0
    size = float(np.linalg.norm(pinky_mcp - index_mcp))

    return {"origin": origin, "u": u, "v": v, "n": n, "size": max(size, 1.0)}


class PalmHologram:
    """
    Vẽ model 3D đứng trên lòng bàn tay, có làm mượt chuyển động, bóng đổ, vòng
    sáng dưới chân và hiệu ứng quét ngang kiểu hologram.

    Tham số:
        size_ratio  : cỡ model so với bề ngang lòng bàn tay
        smooth      : độ mượt của hệ trục (0-1, càng nhỏ càng mượt/càng trễ)
        spin_speed  : tốc độ tự xoay quanh trục đứng (radian mỗi frame)
        lean        : 0 = dựng đúng theo pháp tuyến lòng bàn tay (nhìn từ nóc
                      xuống, trông bẹt), 1 = dựng theo hướng ngón tay. ~0.7 cho
                      hình khối rõ nhất mà vẫn bám theo tay
        hover       : nhấc model lên khỏi lòng bàn tay 1 chút cho giống hologram
        edges       : vẽ thêm đường viền các cạnh cho ra chất "hình chiếu"
        color       : màu hologram (BGR)
        alpha       : độ trong suốt khi chồng lên hình webcam
    """

    def __init__(self, size_ratio=0.55, smooth=0.35, spin_speed=0.035,
                 color=(255, 220, 120), alpha=0.88, scanlines=True,
                 lean=0.7, hover=0.12, edges=True):
        self.size_ratio = size_ratio
        self.smooth = smooth
        self.spin_speed = spin_speed
        self.color = color
        self.alpha = alpha
        self.scanlines = scanlines
        self.lean = lean
        self.hover = hover
        self.edges = edges
        self.auto_spin = True

        self._state = {}     # "Left"/"Right" -> hệ trục đã làm mượt
        self._spin = 0.0

    # ------------------------------------------------------------------ public

    def step(self):
        """Gọi 1 lần mỗi frame để cập nhật góc tự xoay."""
        if self.auto_spin:
            self._spin += self.spin_speed

    def draw(self, frame, mesh, hand, scale=1.0, wireframe=False, up_axis="y"):
        """
        Vẽ `mesh` đứng trên lòng bàn tay của `hand` (1 phần tử trong kết quả
        của utils.detect_hands). Trả về True nếu có vẽ.

        Nắm tay lại (không ngón nào giơ) thì hologram tắt - coi như "nắm lấy" nó.
        """
        if sum(hand.get("fingers_up", [1] * 5)) == 0:
            return False

        height, width = frame.shape[:2]
        basis = self._smoothed_basis(hand, width, height)

        size = self.size_ratio * basis["size"] * scale
        u, v, n, origin = basis["u"], basis["v"], basis["n"], basis["origin"]

        # Bóng đổ + vòng sáng nằm trong ĐÚNG mặt phẳng lòng bàn tay
        self._draw_shadow(frame, origin, u, v, size)
        self._draw_ring(frame, origin, u, v, n, size)

        # Trục đứng của model: pha giữa pháp tuyến và hướng ngón tay (xem
        # phần giải thích ở đầu file)
        up = _normalize((1.0 - self.lean) * n + self.lean * v)
        # 2 trục vuông góc với trục đứng, dùng để model tự xoay quanh trục đó
        side = _normalize(np.cross(up, n) if abs(float(np.dot(up, n))) < 0.99
                          else np.cross(up, u))
        depth_axis = _normalize(np.cross(up, side))

        cos_a, sin_a = np.cos(self._spin), np.sin(self._spin)
        axis_x = side * cos_a + depth_axis * sin_a
        axis_z = -side * sin_a + depth_axis * cos_a

        # Đưa đỉnh model từ hệ toạ độ riêng của nó sang hệ trục bàn tay:
        #   điểm = tâm + size * (x*axis_x + y*up + z*axis_z)
        # với y là trục đứng của model (đã dời để đáy model nằm trên tay).
        vertices = self._oriented_vertices(mesh, up_axis)
        base_point = origin + n * (size * self.hover)
        points = base_point + size * (np.outer(vertices[:, 0], axis_x)
                                      + np.outer(vertices[:, 1], up)
                                      + np.outer(vertices[:, 2], axis_z))

        box = self._bounding_box(points, frame.shape)
        before = frame[box[1]:box[3], box[0]:box[2]].copy() if box else None

        drawn = draw_faces(frame, mesh, points, base_color=self.color,
                           wireframe=wireframe, alpha=self.alpha)
        if self.edges and drawn and not wireframe:
            # Viền cạnh mờ chồng lên mặt đã tô -> trông giống hình chiếu sáng
            draw_faces(frame, mesh, points, base_color=self.color,
                       wireframe=True, alpha=0.30)

        if self.scanlines and drawn and box is not None:
            self._draw_scanlines(frame, box, before)
        return drawn > 0

    def reset(self):
        self._state.clear()
        self._spin = 0.0

    # --------------------------------------------------------------- internals

    def _smoothed_basis(self, hand, width, height):
        """Làm mượt hệ trục qua các frame để hologram không rung theo landmark."""
        key = hand["handedness"]
        current = palm_basis(hand["landmarks"], width, height)
        previous = self._state.get(key)

        if previous is None:
            self._state[key] = current
            return current

        k = self.smooth
        merged = {}
        for field in ("origin", "u", "v", "n"):
            merged[field] = previous[field] + (current[field] - previous[field]) * k
        for field in ("u", "v", "n"):
            merged[field] = _normalize(merged[field])
        merged["size"] = previous["size"] + (current["size"] - previous["size"]) * k

        self._state[key] = merged
        return merged

    @staticmethod
    def _oriented_vertices(mesh, up_axis):
        """
        Xoay trục đứng của model về trục y, rồi dời lên sao cho đáy model nằm
        đúng trên mặt phẳng lòng bàn tay (điểm thấp nhất có y = 0).
        """
        v = mesh.vertices
        if up_axis == "z":
            v = np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)
        elif up_axis == "x":
            v = np.stack([v[:, 1], v[:, 0], v[:, 2]], axis=1)
        return np.stack([v[:, 0], v[:, 1] - v[:, 1].min(), v[:, 2]], axis=1)

    def _plane_circle(self, origin, u, v, radius, steps=28):
        """Các điểm của 1 đường tròn nằm trong mặt phẳng lòng bàn tay (đã chiếu 2D)."""
        angles = np.linspace(0, 2 * np.pi, steps, endpoint=False)
        pts = (origin[None, :]
               + np.outer(np.cos(angles), u) * radius
               + np.outer(np.sin(angles), v) * radius)
        return np.round(pts[:, :2]).astype(np.int32)

    def _draw_shadow(self, frame, origin, u, v, size):
        """Bóng mờ dưới chân model - giúp mắt tin rằng model đang "đứng" trên tay."""
        polygon = self._plane_circle(origin, u, v, size * 0.75)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], (20, 20, 20))
        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, dst=frame)

    def _draw_ring(self, frame, origin, u, v, n, size):
        """Vòng sáng đặt trên lòng bàn tay, kiểu bệ chiếu hologram."""
        for radius, thickness in ((size * 1.05, 2), (size * 0.82, 1)):
            polygon = self._plane_circle(origin, u, v, radius)
            cv2.polylines(frame, [polygon], isClosed=True, color=self.color,
                          thickness=thickness, lineType=cv2.LINE_AA)

    @staticmethod
    def _bounding_box(points, frame_shape):
        """Hình chữ nhật bao quanh model sau khi chiếu, đã cắt theo khung hình."""
        x1 = int(max(points[:, 0].min(), 0))
        x2 = int(min(points[:, 0].max() + 1, frame_shape[1]))
        y1 = int(max(points[:, 1].min(), 0))
        y2 = int(min(points[:, 1].max() + 1, frame_shape[0]))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return (x1, y1, x2, y2)

    def _draw_scanlines(self, frame, box, before):
        """
        Vạch quét ngang mờ chạy trên model, cho giống hình chiếu hologram.

        Chỉ tối đi ĐÚNG những pixel mà model vừa vẽ đè lên (so sánh vùng ảnh
        trước và sau khi vẽ), nếu không sẽ thấy cả 1 khung chữ nhật kẻ sọc.
        """
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        mask = np.any(roi != before, axis=2)

        stripes = roi[::3]
        stripe_mask = mask[::3]
        stripes[stripe_mask] = (stripes[stripe_mask] * 0.78).astype(roi.dtype)
