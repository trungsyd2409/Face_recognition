"""
liquid_warp.py
Coi hình webcam như một TẤM VẢI trải trên mặt bàn: chụm ngón cái + ngón trỏ là
túm lấy tấm vải tại đúng chỗ đó, kéo tay đi thì cả tấm bị lôi theo - chỗ túm đi
nhiều nhất, càng xa đi càng ít, 4 mép khung hình đứng yên như vải bị ghim đinh.

Nhả tay ra thì vải NẰM YÊN ở chỗ mới (không đàn hồi về như cao su). Lần kéo sau
túm vào tấm vải đang nhăn sẵn và kéo tiếp, các nếp nhăn chồng lên nhau. Bấm 'r'
để trải phẳng lại tấm vải.

---------------------------------------------------------------------------
TRƯỜNG DỊCH CHUYỂN

Trạng thái của tấm vải được lưu bằng 1 trường dịch chuyển D: với mỗi điểm ảnh q
trên màn hình, D(q) nói "điểm này lấy màu từ chỗ nào trong ảnh webcam gốc":

    ảnh_hiển_thị(q) = ảnh_gốc(q + D(q))        <- đúng cái cv2.remap làm

CHỒNG BIẾN DẠNG (phần cốt lõi, và cũng là chỗ dễ làm sai nhất)

Mỗi frame, tay dịch đi một đoạn `delta`. Biến dạng mới KHÔNG phải là cộng thêm
`-delta*w` vào D cũ - làm vậy thì nếp nhăn cũ đứng yên tại chỗ trong khi lẽ ra
chúng phải bị kéo đi cùng tấm vải. Phải HỢP hai phép biến dạng:

    u(q)     = q - delta * w(q)          <- phép kéo của riêng frame này
    D_mới(q) = (u(q) - q) + D_cũ(u(q))
             = -delta * w(q) + D_cũ(u(q))

Nghĩa là: trước khi cộng phần kéo mới, trường cũ phải được LẤY MẪU LẠI tại u(q)
- chính là remap bản thân trường dịch chuyển. Nhờ vậy nếp nhăn có sẵn cũng trôi
theo tay, đúng như kéo một tấm vải đã nhăn.

TRỌNG SỐ w(q) - quyết định "cảm giác vải"

    w(q) = f(khoảng cách từ q tới đầu ngón) * b(khoảng cách từ q tới mép khung)

    f(d) = 1 / (1 + (d/r)^2)   giảm dần nhưng có ĐUÔI DÀI, nên cả tấm vải đều
                               bị lôi theo ít nhiều, không có ranh giới cứng
                               kiểu "trong thì méo, ngoài thì không"
    b     = 0 ngay tại mép khung hình và tăng mượt vào trong  -> 4 cạnh bị ghim

Tại đúng đầu ngón f = 1, nên mẩu vải bạn túm bám sát đầu ngón.

Trường được tính ở độ phân giải THẤP (mặc định 1/4) rồi mới phóng to: biến dạng
vốn trơn nên không cần chi tiết, mà tính ở 1/4 độ phân giải thì rẻ hơn 16 lần.
Khi tấm vải đang phẳng, apply() trả thẳng ảnh gốc về, không chạy remap.
"""

import cv2
import numpy as np

from utils import is_pinching

THUMB_TIP_ID = 4
INDEX_TIP_ID = 8
PALM_ID = 9          # gốc ngón giữa - dùng để đo cỡ bàn tay
WRIST_ID = 0


class LiquidWarp:
    """
    Túm và kéo hình webcam như kéo một tấm vải.

    Tham số:
        field_scale  : độ phân giải của trường dịch chuyển so với khung hình
        radius_ratio : tầm ảnh hưởng của cú túm, tính theo cỡ bàn tay. Đây là
                       khoảng cách mà mức kéo giảm còn một nửa, KHÔNG phải mép
                       cứng - ngoài khoảng đó vải vẫn bị lôi theo, ít dần
        edge_margin  : bề rộng dải ghim ở mép khung hình (theo tỉ lệ cạnh ngắn)
        max_step     : chặn quãng tay dịch trong 1 frame (pixel), tránh
                       MediaPipe nhảy điểm làm vải giật đột ngột
    """

    def __init__(self, field_scale=0.25, radius_ratio=2.8, edge_margin=0.10,
                 max_step=70.0):
        self.field_scale = field_scale
        self.radius_ratio = radius_ratio
        self.edge_margin = edge_margin
        self.max_step = max_step

        self._dx = None          # trường dịch chuyển (độ phân giải thấp)
        self._dy = None
        self._grid_x = None      # lưới toạ độ ở độ phân giải thấp
        self._grid_y = None
        self._edge = None        # hệ số ghim mép, dựng 1 lần
        self._base_x = None      # lưới toạ độ ở độ phân giải gốc
        self._base_y = None

        # Các cú túm đang giữ: tay -> {"point": vị trí ngón frame trước, "radius"}
        self._grabs = {}

    # ------------------------------------------------------------------ public

    def update(self, hands_info, width, height):
        """Đọc cử chỉ chụm ngón và kéo tấm vải theo tay."""
        self._ensure_field(width, height)
        scale = self.field_scale

        pinching_now = set()
        for hand in hands_info:
            handedness = hand["handedness"]
            landmarks = hand["landmarks"]
            if not is_pinching(landmarks):
                continue

            # Điểm túm = điểm giữa 2 đầu ngón cái và trỏ
            point = np.array([
                (landmarks[THUMB_TIP_ID].x + landmarks[INDEX_TIP_ID].x) / 2 * width,
                (landmarks[THUMB_TIP_ID].y + landmarks[INDEX_TIP_ID].y) / 2 * height,
            ], dtype=np.float32)

            palm = np.array([landmarks[PALM_ID].x * width,
                             landmarks[PALM_ID].y * height], dtype=np.float32)
            wrist = np.array([landmarks[WRIST_ID].x * width,
                              landmarks[WRIST_ID].y * height], dtype=np.float32)
            hand_size = float(np.linalg.norm(palm - wrist))
            if hand_size < 12:
                continue

            pinching_now.add(handedness)
            grab = self._grabs.get(handedness)
            if grab is None:
                # Vừa túm vào vải: chưa kéo gì, chỉ ghi lại vị trí xuất phát
                self._grabs[handedness] = {
                    "point": point.copy(),
                    "radius": self.radius_ratio * hand_size,
                }
                continue

            delta = point - grab["point"]
            grab["point"] = point.copy()

            step = float(np.linalg.norm(delta))
            if step < 0.3:
                continue                     # tay đứng yên
            if step > self.max_step:         # nhận diện nhảy điểm -> chặn lại
                delta = delta / step * self.max_step

            self._drag(point * scale, grab["radius"] * scale, delta * scale)

        # Nhả ngón: chỉ cần quên cú túm đi, tấm vải giữ nguyên hình dạng
        for handedness in list(self._grabs):
            if handedness not in pinching_now:
                self._grabs.pop(handedness)

    def apply(self, frame):
        """Áp trường dịch chuyển lên frame bằng cv2.remap. Trả về ảnh đã méo."""
        if self._dx is None:
            return frame

        # Vải đang phẳng -> trả thẳng ảnh gốc, khỏi remap
        limit = 0.25 * self.field_scale
        if float(np.abs(self._dx).max()) < limit and float(np.abs(self._dy).max()) < limit:
            return frame

        height, width = frame.shape[:2]
        factor = 1.0 / self.field_scale
        dx = cv2.resize(self._dx, (width, height), interpolation=cv2.INTER_LINEAR) * factor
        dy = cv2.resize(self._dy, (width, height), interpolation=cv2.INTER_LINEAR) * factor

        map_x = cv2.add(self._base_x, dx)
        map_y = cv2.add(self._base_y, dy)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    def reset(self):
        """Trải phẳng lại tấm vải."""
        if self._dx is not None:
            self._dx[:] = 0.0
            self._dy[:] = 0.0
        self._grabs.clear()

    @property
    def grabbing(self):
        """Đang có cú túm nào không."""
        return len(self._grabs) > 0

    # --------------------------------------------------------------- internals

    def _ensure_field(self, width, height):
        """Cấp phát trường, các lưới toạ độ và hệ số ghim mép (chỉ làm 1 lần)."""
        small_w = max(8, int(width * self.field_scale))
        small_h = max(8, int(height * self.field_scale))

        if self._dx is None or self._dx.shape != (small_h, small_w):
            self._dx = np.zeros((small_h, small_w), dtype=np.float32)
            self._dy = np.zeros((small_h, small_w), dtype=np.float32)
            self._grid_x, self._grid_y = np.meshgrid(
                np.arange(small_w, dtype=np.float32),
                np.arange(small_h, dtype=np.float32))

            # Hệ số ghim mép: 0 ngay tại mép, tăng mượt vào trong theo hàm
            # smoothstep t^2*(3-2t) - nhờ vậy 4 cạnh khung hình đứng yên hẳn,
            # giống tấm vải bị ghim đinh, và không có bậc thang ở chỗ chuyển tiếp
            margin = max(self.edge_margin * min(small_w, small_h), 1.0)
            dist_x = np.minimum(self._grid_x, small_w - 1 - self._grid_x)
            dist_y = np.minimum(self._grid_y, small_h - 1 - self._grid_y)
            t = np.clip(np.minimum(dist_x, dist_y) / margin, 0.0, 1.0)
            self._edge = (t * t * (3.0 - 2.0 * t)).astype(np.float32)

        if self._base_x is None or self._base_x.shape != (height, width):
            self._base_x, self._base_y = np.meshgrid(
                np.arange(width, dtype=np.float32),
                np.arange(height, dtype=np.float32))

    def _drag(self, center, radius, delta):
        """
        Kéo tấm vải một bước: tay vừa dịch `delta`, chỗ túm ở `center`.
        Mọi tham số đều tính theo toạ độ của trường (độ phân giải thấp).
        """
        radius = max(radius, 2.0)

        # Trọng số: đuôi dài 1/(1 + (d/r)^2), nhân với hệ số ghim mép
        gx = self._grid_x - center[0]
        gy = self._grid_y - center[1]
        weight = 1.0 / (1.0 + (gx * gx + gy * gy) / (radius * radius))
        weight *= self._edge

        # u(q) = q - delta * w(q): toạ độ cần lấy mẫu trường cũ
        ux = self._grid_x - weight * float(delta[0])
        uy = self._grid_y - weight * float(delta[1])

        # Trường cũ trôi theo tấm vải: lấy mẫu lại D_cũ tại u(q)
        moved_dx = cv2.remap(self._dx, ux, uy, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        moved_dy = cv2.remap(self._dy, ux, uy, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

        # D_mới(q) = -delta*w(q) + D_cũ(u(q))
        self._dx = moved_dx - weight * float(delta[0])
        self._dy = moved_dy - weight * float(delta[1])
