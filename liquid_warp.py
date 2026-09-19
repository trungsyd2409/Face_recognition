"""
liquid_warp.py
Kéo giãn / bóp méo chính hình ảnh webcam bằng tay, như kéo một tấm cao su.

Nguyên lý: mỗi frame ta dựng một TRƯỜNG DỊCH CHUYỂN (displacement field) - với
mỗi điểm ảnh, trường này nói "điểm này phải lấy màu từ chỗ nào trong ảnh gốc".
Đưa trường đó cho cv2.remap là xong:

    map_x(x, y) = x + dx(x, y)
    map_y(x, y) = y + dy(x, y)
    ảnh_méo = cv2.remap(ảnh_gốc, map_x, map_y)

Bàn tay tạo ra các "cọ" (brush) tác động lên trường này, mỗi cử chỉ một kiểu:

    Chụm ngón cái + trỏ rồi kéo  -> KÉO: ảnh bị lôi theo tay như cao su
    Xoè cả bàn tay               -> PHÌNH: ảnh nở phồng ra khỏi tâm bàn tay
    Nắm tay                      -> NÉN: ảnh bị hút co vào tâm bàn tay
    Giơ đúng 2 ngón (trỏ + giữa) -> XOÁY: ảnh xoay tròn quanh tâm bàn tay

Ảnh hưởng của mỗi cọ giảm dần từ tâm ra mép theo hàm (1 - (d/r)^2)^2 - bằng 1 ở
tâm, bằng 0 đúng tại mép, nên chỉ cần tính trong một ô vuông nhỏ quanh bàn tay
chứ không phải cả khung hình.

Hai điểm khiến hiệu ứng có cảm giác "chất lỏng" chứ không phải kính lúp:

1. Trường dịch chuyển được CỘNG DỒN qua các frame, nên kéo tay đi một đoạn dài
   thì ảnh bị kéo dài theo cả quãng đường đó, không chỉ theo vị trí hiện tại.
2. Mỗi frame trường lại được nhân với `decay` (< 1), nên khi bỏ tay ra, ảnh từ
   từ đàn hồi trở về hình dạng ban đầu thay vì bật lại ngay.

Lưu ý về 3 cử chỉ TĨNH (phình, nén, xoáy): chúng tác động đều đặn mỗi frame, nên
nếu cộng thẳng vào trường thì giữ tay yên vài giây là ảnh bị bóp nát. Vì vậy
phần đóng góp của chúng được nhân với (1 - decay): cộng dồn theo cấp số nhân,
trường sẽ HỘI TỤ về đúng mức biến dạng mong muốn rồi dừng ở đó, dù bạn giữ tay
bao lâu. Riêng cử chỉ KÉO thì vẫn cộng thẳng, vì nó vốn bị giới hạn bởi quãng
đường tay bạn di chuyển.

Trường được tính ở độ phân giải THẤP (mặc định 1/4) rồi mới phóng to: biến dạng
vốn trơn và rộng nên không cần chi tiết, mà tính ở 1/4 độ phân giải thì rẻ hơn
16 lần.
"""

import cv2
import numpy as np

from utils import is_pinching

PALM_ID = 9          # gốc ngón giữa - dùng làm "tâm bàn tay"
WRIST_ID = 0


class LiquidWarp:
    """
    Kéo giãn / bóp méo hình webcam theo cử chỉ tay.

    Tham số:
        field_scale   : độ phân giải của trường dịch chuyển so với khung hình
        decay         : mỗi frame trường còn lại bao nhiêu (đàn hồi về hình cũ).
                        Càng gần 1 thì vết méo càng lâu tan
        radius_ratio  : bán kính vùng ảnh hưởng, theo kích thước bàn tay
        grab_strength : độ mạnh của cử chỉ kéo (chụm ngón)
        bulge_strength: mức phình / nén ở trạng thái ổn định (0.35 = ±35% cỡ)
        swirl_strength: góc xoáy ở trạng thái ổn định (radian, tại tâm bàn tay)
        max_shift     : giới hạn dịch chuyển tối đa của 1 điểm ảnh (pixel),
                        tránh kéo quá đà làm ảnh gập chồng lên nhau
    """

    def __init__(self, field_scale=0.25, decay=0.9, radius_ratio=2.6,
                 grab_strength=1.35, bulge_strength=0.35, swirl_strength=0.8,
                 max_shift=90.0):
        self.field_scale = field_scale
        self.decay = decay
        self.radius_ratio = radius_ratio
        self.grab_strength = grab_strength
        self.bulge_strength = bulge_strength
        self.swirl_strength = swirl_strength
        self.max_shift = max_shift

        self._dx = None          # trường dịch chuyển (độ phân giải thấp)
        self._dy = None
        self._base_x = None      # lưới toạ độ gốc, dựng 1 lần
        self._base_y = None
        self._prev_palm = {}     # tay -> vị trí tâm bàn tay ở frame trước
        self.action = {}         # tay -> tên cử chỉ đang tác động (để tiện gỡ lỗi)

    # ------------------------------------------------------------------ public

    def update(self, hands_info, width, height):
        """Đọc cử chỉ của từng bàn tay và cộng thêm biến dạng vào trường."""
        self._ensure_field(width, height)

        # Đàn hồi: mỗi frame vết méo cũ nhạt bớt đi một chút
        self._dx *= self.decay
        self._dy *= self.decay

        scale = self.field_scale
        self.action = {}
        seen = set()

        for hand in hands_info:
            handedness = hand["handedness"]
            seen.add(handedness)
            landmarks = hand["landmarks"]
            fingers_up = hand.get("fingers_up", [1] * 5)

            palm = np.array([landmarks[PALM_ID].x * width,
                             landmarks[PALM_ID].y * height], dtype=np.float32)
            wrist = np.array([landmarks[WRIST_ID].x * width,
                              landmarks[WRIST_ID].y * height], dtype=np.float32)
            hand_size = float(np.linalg.norm(palm - wrist))
            if hand_size < 12:
                continue

            previous = self._prev_palm.get(handedness)
            self._prev_palm[handedness] = palm
            velocity = palm - previous if previous is not None else np.zeros(2, np.float32)

            radius = self.radius_ratio * hand_size * scale
            center = palm * scale
            total_up = sum(fingers_up)

            # Thứ tự kiểm tra rất quan trọng: NẮM TAY phải xét TRƯỚC chụm ngón,
            # vì lúc nắm tay thì ngón cái cũng nằm sát ngón trỏ nên is_pinching
            # vẫn trả về True - xét sau thì nắm tay luôn bị hiểu nhầm là kéo.
            if total_up == 0:
                # NÉN: hút ảnh co vào tâm bàn tay
                self._add_radial(center, radius, -self.bulge_strength)
                self.action[handedness] = "nen"
            elif is_pinching(landmarks):
                # KÉO: ảnh bị lôi theo hướng tay vừa đi
                self._add_drag(center, radius, velocity * self.grab_strength * scale)
                self.action[handedness] = "keo"
            elif fingers_up[1] and fingers_up[2] and total_up <= 2:
                # XOÁY: xoay ảnh quanh tâm bàn tay
                self._add_swirl(center, radius, self.swirl_strength)
                self.action[handedness] = "xoay"
            elif total_up >= 4:
                # PHÌNH: đẩy ảnh nở ra khỏi tâm bàn tay
                self._add_radial(center, radius, self.bulge_strength)
                self.action[handedness] = "phinh"

        for handedness in list(self._prev_palm):
            if handedness not in seen:
                self._prev_palm.pop(handedness, None)

        # Chặn biên độ để ảnh không bị kéo quá đà, gập chồng lên chính nó
        limit = self.max_shift * self.field_scale
        np.clip(self._dx, -limit, limit, out=self._dx)
        np.clip(self._dy, -limit, limit, out=self._dy)

    def apply(self, frame):
        """Áp trường dịch chuyển lên frame bằng cv2.remap. Trả về ảnh đã méo."""
        if self._dx is None:
            return frame

        height, width = frame.shape[:2]
        # Trường tính ở độ phân giải thấp -> phóng to lại, nhân hệ số cho đúng
        # đơn vị pixel của ảnh gốc
        factor = 1.0 / self.field_scale
        dx = cv2.resize(self._dx, (width, height), interpolation=cv2.INTER_LINEAR) * factor
        dy = cv2.resize(self._dy, (width, height), interpolation=cv2.INTER_LINEAR) * factor

        map_x = cv2.add(self._base_x, dx)
        map_y = cv2.add(self._base_y, dy)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    def reset(self):
        """Xoá sạch biến dạng, ảnh trở lại bình thường ngay."""
        if self._dx is not None:
            self._dx[:] = 0.0
            self._dy[:] = 0.0
        self._prev_palm.clear()

    # --------------------------------------------------------------- internals

    def _ensure_field(self, width, height):
        """Cấp phát trường dịch chuyển + lưới toạ độ gốc (chỉ làm 1 lần)."""
        small_w = max(8, int(width * self.field_scale))
        small_h = max(8, int(height * self.field_scale))
        if self._dx is None or self._dx.shape != (small_h, small_w):
            self._dx = np.zeros((small_h, small_w), dtype=np.float32)
            self._dy = np.zeros((small_h, small_w), dtype=np.float32)
        if self._base_x is None or self._base_x.shape != (height, width):
            self._base_x, self._base_y = np.meshgrid(
                np.arange(width, dtype=np.float32),
                np.arange(height, dtype=np.float32))

    def _window(self, center, radius):
        """
        Ô vuông quanh 1 cọ + trọng số giảm dần từ tâm ra mép.

        Trọng số dùng (1 - (d/r)^2)^2: bằng 1 ở tâm, bằng 0 ĐÚNG tại mép (khác
        hàm Gauss vốn không bao giờ về 0), nhờ vậy chỉ cần tính trong ô vuông
        này, ngoài ra không ảnh hưởng gì - vừa nhanh vừa không để lại vệt.
        """
        height, width = self._dx.shape
        radius = max(radius, 3.0)
        x0 = int(max(center[0] - radius, 0))
        y0 = int(max(center[1] - radius, 0))
        x1 = int(min(center[0] + radius + 1, width))
        y1 = int(min(center[1] + radius + 1, height))
        if x1 <= x0 or y1 <= y0:
            return None

        gx = np.arange(x0, x1, dtype=np.float32)[None, :] - center[0]
        gy = np.arange(y0, y1, dtype=np.float32)[:, None] - center[1]
        d2 = (gx ** 2 + gy ** 2) / (radius ** 2)
        weight = np.clip(1.0 - d2, 0.0, 1.0) ** 2
        return (slice(y0, y1), slice(x0, x1)), gx, gy, weight

    def _add_drag(self, center, radius, shift):
        """KÉO: dịch cả vùng theo vector `shift` (ảnh bị lôi theo tay)."""
        window = self._window(center, radius)
        if window is None:
            return
        (ys, xs), _, _, weight = window
        # remap lấy màu từ toạ độ nguồn, nên muốn ảnh dịch THEO tay thì trường
        # phải trỏ ngược lại hướng tay đi
        self._dx[ys, xs] -= weight * float(shift[0])
        self._dy[ys, xs] -= weight * float(shift[1])

    def _add_radial(self, center, radius, strength):
        """PHÌNH (strength > 0) hoặc NÉN (strength < 0) quanh tâm bàn tay."""
        window = self._window(center, radius)
        if window is None:
            return
        (ys, xs), gx, gy, weight = window
        step = self._static_step()
        self._dx[ys, xs] -= step * weight * strength * gx
        self._dy[ys, xs] -= step * weight * strength * gy

    def _add_swirl(self, center, radius, angle):
        """XOÁY: xoay ảnh quanh tâm bàn tay, góc xoay giảm dần ra mép."""
        window = self._window(center, radius)
        if window is None:
            return
        (ys, xs), gx, gy, weight = window
        theta = angle * weight * self._static_step()
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Vị trí mới của điểm sau khi xoay, trừ đi vị trí cũ = độ dịch chuyển
        self._dx[ys, xs] += (cos_t * gx - sin_t * gy) - gx
        self._dy[ys, xs] += (sin_t * gx + cos_t * gy) - gy

    def _static_step(self):
        """
        Hệ số cho các cử chỉ TĨNH (phình / nén / xoáy).

        Mỗi frame trường bị nhân với `decay`, nên nếu mỗi frame cộng thêm
        (1 - decay) lần mức mong muốn thì tổng cấp số nhân đúng bằng mức đó:
            (1-d) * (1 + d + d^2 + ...) = (1-d) / (1-d) = 1
        Nhờ vậy giữ tay yên bao lâu thì ảnh cũng chỉ méo tới đúng mức đã đặt
        rồi dừng, thay vì méo mãi cho tới lúc nát ảnh.
        """
        return max(1.0 - self.decay, 1e-3)
