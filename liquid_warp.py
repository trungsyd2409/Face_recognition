"""
liquid_warp.py
"Túm" lấy hình webcam bằng cách chụm ngón cái + ngón trỏ, rồi kéo đi - mảng ảnh
quanh chỗ túm bị lôi theo đầu ngón như kéo một tấm cao su.

Chỉ có DUY NHẤT cử chỉ này tác động lên hình. Tay để bình thường (không chụm)
thì màn hình hiện y nguyên hình webcam, không méo chút nào.

---------------------------------------------------------------------------
Cách làm: mỗi frame dựng một TRƯỜNG DỊCH CHUYỂN (displacement field) - với mỗi
điểm ảnh, trường này nói "điểm này phải lấy màu từ chỗ nào trong ảnh gốc". Đưa
trường đó cho cv2.remap là xong:

    map_x(x, y) = x + dx(x, y)
    map_y(x, y) = y + dy(x, y)
    ảnh_méo = cv2.remap(ảnh_gốc, map_x, map_y)

ĐIỂM NEO - thứ làm cho ảnh dính đúng vào ngón tay:

    Lúc bạn vừa chụm 2 ngón, chương trình ghi lại ĐIỂM NEO = chỗ ngón tay đang
    chụm. Sau đó, mỗi frame:

        độ lệch  = vị trí ngón hiện tại - điểm neo
        dx(p)    = -độ_lệch.x * trọng_số(p)
        dy(p)    = -độ_lệch.y * trọng_số(p)

    với trọng số bằng 1 ngay tại đầu ngón và giảm dần ra mép vùng ảnh hưởng.
    Tại đúng đầu ngón (trọng số = 1), remap lấy màu từ điểm neo - nghĩa là mẩu
    ảnh bạn túm lúc đầu luôn bám chặt vào đầu ngón, dù bạn kéo đi đâu.

    (Cách làm sai thường gặp: mỗi frame cộng thêm 1 chút dịch chuyển theo vận
    tốc tay. Khi đó ảnh bị "trét" dần theo đường tay đi chứ chỗ túm không dính
    đúng vào ngón, và kéo đi kéo lại thì biến dạng cứ cộng dồn mãi.)

Trọng số dùng (1 - (d/r)^2)^2: bằng 1 ở tâm, bằng ĐÚNG 0 tại mép (khác hàm
Gauss vốn không bao giờ về 0), nên ngoài vùng ảnh hưởng ảnh đứng yên tuyệt đối
và chỉ cần tính trong một ô vuông nhỏ quanh bàn tay.

NHẢ NGÓN RA: phần biến dạng của cú túm đó được chuyển sang "trường dư", và
trường dư này mỗi frame lại nhân với `decay` (< 1) nên vết méo tan dần trong
khoảng nửa giây - ảnh đàn hồi về hình cũ chứ không bật lại đột ngột.

Trường được tính ở độ phân giải THẤP (mặc định 1/4) rồi mới phóng to: biến dạng
vốn trơn và rộng nên không cần chi tiết, mà tính ở 1/4 độ phân giải thì rẻ hơn
16 lần. Khi không còn biến dạng nào đáng kể, hàm apply() trả thẳng ảnh gốc về,
không chạy remap - vừa nhanh vừa đảm bảo ảnh sắc nét đúng nguyên bản.
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
    Kéo giãn hình webcam bằng cử chỉ chụm ngón rồi kéo.

    Tham số:
        field_scale  : độ phân giải của trường dịch chuyển so với khung hình
        decay        : sau khi nhả ngón, mỗi frame vết méo còn lại bao nhiêu
                       (càng gần 1 thì càng lâu tan)
        radius_ratio : bán kính vùng bị kéo theo, tính theo cỡ bàn tay
        max_shift    : giới hạn quãng kéo của 1 cú túm (pixel) - kéo quá xa thì
                       ảnh gập chồng lên chính nó, trông như rách
    """

    def __init__(self, field_scale=0.25, decay=0.88, radius_ratio=2.6,
                 max_shift=260.0):
        self.field_scale = field_scale
        self.decay = decay
        self.radius_ratio = radius_ratio
        self.max_shift = max_shift

        self._dx = None          # trường tổng (độ phân giải thấp)
        self._dy = None
        self._res_dx = None      # trường dư: phần còn lại sau khi nhả ngón
        self._res_dy = None
        self._base_x = None      # lưới toạ độ gốc, dựng 1 lần
        self._base_y = None

        # Các cú túm đang giữ: tay -> {"anchor": điểm neo, "point": vị trí ngón
        # hiện tại, "radius": bán kính vùng ảnh hưởng}
        self._grabs = {}

    # ------------------------------------------------------------------ public

    def update(self, hands_info, width, height):
        """Cập nhật các cú túm theo cử chỉ tay và dựng lại trường dịch chuyển."""
        self._ensure_field(width, height)
        scale = self.field_scale

        # Vết méo cũ (của những cú túm đã nhả) nhạt dần đi
        self._res_dx *= self.decay
        self._res_dy *= self.decay

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
                # Vừa chụm ngón -> ghi điểm neo và chốt bán kính vùng ảnh hưởng
                self._grabs[handedness] = {
                    "anchor": point.copy(),
                    "point": point,
                    "radius": self.radius_ratio * hand_size,
                }
            else:
                grab["point"] = point

        # Tay nào vừa nhả ngón (hoặc biến mất) -> đẩy biến dạng sang trường dư
        for handedness in list(self._grabs):
            if handedness not in pinching_now:
                grab = self._grabs.pop(handedness)
                self._paint_grab(self._res_dx, self._res_dy, grab, scale)

        # Trường tổng = các cú túm đang giữ + trường dư đang tan dần
        self._dx[:] = self._res_dx
        self._dy[:] = self._res_dy
        for grab in self._grabs.values():
            self._paint_grab(self._dx, self._dy, grab, scale)

    def apply(self, frame):
        """Áp trường dịch chuyển lên frame bằng cv2.remap. Trả về ảnh đã méo."""
        if self._dx is None:
            return frame

        # Không có biến dạng đáng kể -> trả thẳng ảnh gốc, khỏi remap
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
        """Xoá sạch biến dạng, ảnh trở lại bình thường ngay."""
        if self._dx is not None:
            self._dx[:] = 0.0
            self._dy[:] = 0.0
            self._res_dx[:] = 0.0
            self._res_dy[:] = 0.0
        self._grabs.clear()

    @property
    def grabbing(self):
        """Đang có cú túm nào không (tiện để gỡ lỗi / hiện chỉ báo)."""
        return len(self._grabs) > 0

    # --------------------------------------------------------------- internals

    def _ensure_field(self, width, height):
        """Cấp phát các trường + lưới toạ độ gốc (chỉ làm 1 lần)."""
        small_w = max(8, int(width * self.field_scale))
        small_h = max(8, int(height * self.field_scale))
        if self._dx is None or self._dx.shape != (small_h, small_w):
            self._dx = np.zeros((small_h, small_w), dtype=np.float32)
            self._dy = np.zeros((small_h, small_w), dtype=np.float32)
            self._res_dx = np.zeros((small_h, small_w), dtype=np.float32)
            self._res_dy = np.zeros((small_h, small_w), dtype=np.float32)
        if self._base_x is None or self._base_x.shape != (height, width):
            self._base_x, self._base_y = np.meshgrid(
                np.arange(width, dtype=np.float32),
                np.arange(height, dtype=np.float32))

    def _paint_grab(self, dx_field, dy_field, grab, scale):
        """
        Cộng biến dạng của 1 cú túm vào trường.

        Vùng ảnh hưởng đặt tại VỊ TRÍ NGÓN HIỆN TẠI (không phải điểm neo), nên
        cả "bọng" biến dạng di chuyển theo tay bạn.
        """
        offset = grab["point"] - grab["anchor"]
        length = float(np.linalg.norm(offset))
        if length > self.max_shift:          # kéo quá xa thì chặn lại
            offset = offset / length * self.max_shift

        center = grab["point"] * scale
        radius = max(grab["radius"] * scale, 3.0)

        height, width = dx_field.shape
        x0 = int(max(center[0] - radius, 0))
        y0 = int(max(center[1] - radius, 0))
        x1 = int(min(center[0] + radius + 1, width))
        y1 = int(min(center[1] + radius + 1, height))
        if x1 <= x0 or y1 <= y0:
            return

        gx = np.arange(x0, x1, dtype=np.float32)[None, :] - center[0]
        gy = np.arange(y0, y1, dtype=np.float32)[:, None] - center[1]
        weight = np.clip(1.0 - (gx ** 2 + gy ** 2) / (radius ** 2), 0.0, 1.0) ** 2

        # remap lấy màu từ toạ độ nguồn, nên muốn ảnh dịch THEO tay thì trường
        # phải trỏ ngược lại hướng tay đã đi
        dx_field[y0:y1, x0:x1] -= weight * float(offset[0]) * scale
        dy_field[y0:y1, x0:x1] -= weight * float(offset[1]) * scale
