"""
magic_circle.py
Vòng ma pháp phát sáng tự xoay ở từng đầu ngón tay (kiểu vòng phép trong anime).

Mỗi vòng gồm 3 LỚP vẽ sẵn, mỗi lớp quay với tốc độ và chiều khác nhau - đó chính
là thứ tạo cảm giác "đang chạy phép":

    lớp ngoài  : vòng tròn dày + các vạch chia + chấm sáng   (quay thuận)
    lớp giữa   : các cung đứt đoạn                            (quay ngược, nhanh hơn)
    lớp trong  : ngôi sao 5 cánh trong 1 vòng tròn nhỏ        (quay thuận, nhanh nhất)

CÁCH VẼ - dùng "sprite" thay vì vẽ lại từng frame:
    3 lớp trên được vẽ SẴN 1 lần duy nhất vào 3 ảnh mẫu (template) kích thước
    lớn, nét dày. Mỗi frame chỉ việc XOAY + THU NHỎ ảnh mẫu bằng cv2.warpAffine
    rồi dán vào đúng vị trí đầu ngón tay.

    Làm vậy vì vòng khá nhỏ trên màn hình: nếu vẽ trực tiếp từng nét bằng
    cv2.line / cv2.ellipse ở cỡ nhỏ thì nét mảnh, răng cưa và nhợt nhạt. Vẽ sẵn
    ở cỡ lớn rồi thu nhỏ cho ra nét đặc, mượt và đậm hơn hẳn - lại nhanh hơn vì
    mỗi frame chỉ còn 3 phép warp thay vì mấy chục lệnh vẽ.

Ảnh kết quả được làm mờ nhẹ rồi CỘNG vào khung hình (additive blending) nên vòng
trông như phát sáng thật. Việc làm mờ + cộng chỉ chạy trong HÌNH CHỮ NHẬT BAO
QUANH các vòng chứ không phải cả khung hình - các vòng rất nhỏ so với khung hình
nên cách này nhanh hơn khoảng 8 ms/frame.

Vòng chỉ hiện ở những ngón ĐANG GIƠ.

CHÙM NỐI 2 TAY: khi CẢ HAI bàn tay cùng mở vòng lớn, một chùm sáng đỏ nối thẳng
2 tâm vòng: 1 dải sáng chạy giữa 2 tâm, trên đó có những khối hạt to trôi qua
lại. Chùm chỉ hiện khi cả 2 tay đã gộp xong (mức gộp của tay yếu hơn quyết định
độ đậm của chùm).

Ngón CÁI không có vòng (và cũng không bắn hạt) - để trống hẳn.

GỘP THÀNH VÒNG LỚN: khi 1 bàn tay xoè cả 4 ngón trỏ / giữa / áp út / út (ngón
cái duỗi hay cụp đều được), các vòng nhỏ sẽ trôi dần về
tâm lòng bàn tay, mờ đi, đồng thời 1 vòng LỚN hiện ra ngay giữa lòng bàn tay
(quay chậm hơn cho ra dáng "đại phép"). Gập bớt ngón nào thì quá trình chạy
ngược lại. Chuyển động này được nội suy dần qua vài frame (biến `merge` chạy từ
0 tới 1) nên nhìn mượt chứ không nhảy phựt. Thời gian gộp/tách tính theo GIÂY
THẬT (đồng hồ hệ thống) chứ không theo số frame, nên máy nhanh hay chậm thì
hiệu ứng vẫn diễn ra đúng chừng ấy lâu.
"""

import time

import cv2
import numpy as np

# Ngón có vòng ma pháp: trỏ, giữa, áp út, út - KHÔNG có ngón cái.
# Mỗi phần tử là (chỉ số ngón trong fingers_up, chỉ số landmark của đầu ngón).
CIRCLE_FINGERS = [(1, 8), (2, 12), (3, 16), (4, 20)]

# Các ngón phải cùng xoè thì mới gộp thành vòng lớn: tất cả TRỪ ngón cái
MERGE_FINGERS = (1, 2, 3, 4)
WRIST_ID = 0
MIDDLE_MCP_ID = 9

TEMPLATE_SIZE = 256          # cỡ ảnh mẫu (vẽ 1 lần, sau đó chỉ xoay + thu nhỏ)


def _build_templates(ticks=12, arcs=5):
    """
    Vẽ sẵn 3 lớp của vòng ma pháp thành 3 ảnh mẫu (float 0-1), nét dày để khi
    thu nhỏ vẫn đặc và rõ.
    """
    size = TEMPLATE_SIZE
    center = (size // 2, size // 2)
    radius = size // 2 - 12

    outer = np.zeros((size, size), dtype=np.float32)
    middle = np.zeros((size, size), dtype=np.float32)
    inner = np.zeros((size, size), dtype=np.float32)

    # --- Lớp ngoài: 2 vòng tròn dày + vạch chia + chấm sáng ---
    cv2.circle(outer, center, radius, 1.0, 7, cv2.LINE_AA)
    cv2.circle(outer, center, int(radius * 0.86), 0.85, 4, cv2.LINE_AA)
    for k in range(ticks):
        angle = 2 * np.pi * k / ticks
        p1 = (int(center[0] + np.cos(angle) * radius * 0.86),
              int(center[1] + np.sin(angle) * radius * 0.86))
        p2 = (int(center[0] + np.cos(angle) * radius),
              int(center[1] + np.sin(angle) * radius))
        cv2.line(outer, p1, p2, 1.0, 6, cv2.LINE_AA)
    for k in range(3):
        angle = 2 * np.pi * k / 3
        dot = (int(center[0] + np.cos(angle) * radius),
               int(center[1] + np.sin(angle) * radius))
        cv2.circle(outer, dot, int(radius * 0.09), 1.0, -1, cv2.LINE_AA)

    # --- Lớp giữa: các cung đứt đoạn ---
    mid_radius = int(radius * 0.64)
    span = 360.0 / arcs
    for k in range(arcs):
        start = k * span
        cv2.ellipse(middle, center, (mid_radius, mid_radius), 0,
                    start, start + span * 0.58, 1.0, 6, cv2.LINE_AA)

    # --- Lớp trong: ngôi sao 5 cánh + vòng tròn bao quanh ---
    star_radius = radius * 0.45
    points = []
    for k in range(5):
        # Nối các đỉnh cách nhau 2 bước -> nét liền thành ngôi sao 5 cánh
        angle = (k * 2) * (2 * np.pi / 5) - np.pi / 2
        points.append((int(center[0] + np.cos(angle) * star_radius),
                       int(center[1] + np.sin(angle) * star_radius)))
    cv2.polylines(inner, [np.array(points, dtype=np.int32)], True, 1.0, 5, cv2.LINE_AA)
    cv2.circle(inner, center, int(star_radius * 1.18), 0.8, 3, cv2.LINE_AA)

    return outer, middle, inner


class MagicCircles:
    """
    Vẽ vòng ma pháp ở các đầu ngón tay.

    Tham số:
        radius_ratio : bán kính vòng nhỏ ở đầu ngón, so với kích thước bàn tay
        big_ratio    : bán kính vòng lớn khi xoè đủ 4 ngón (trừ ngón trỏ)
        link_dots    : số khối hạt trôi trên chùm nối 2 tay
        link_speed   : tốc độ trôi của các khối hạt đó
        merge_time   : thời gian (giây) để gộp các vòng nhỏ thành 1 vòng lớn, và ngược lại
        min_big_time : vòng lớn phải hiện ít nhất bao nhiêu giây rồi mới được
                       phép tách trở lại thành các vòng nhỏ
        exit_delay   : điều kiện "thiếu ngón" phải kéo dài liên tục bấy nhiêu
                       giây thì mới thực sự tách vòng (chống rung nhận diện)
        color        : màu vòng (BGR) - mặc định đỏ
        spin_speed   : tốc độ quay cơ bản (radian mỗi frame)
        ticks / arcs : số vạch chia trên vòng ngoài / số cung ở lớp giữa
        glow         : độ sáng tổng thể
        blur         : độ toả sáng quanh nét (0 = nét sắc, không toả)
    """

    def __init__(self, radius_ratio=0.25, color=(40, 40, 255), spin_speed=0.15,
                 ticks=12, arcs=5, glow=1.25, blur=1.6,
                 big_ratio=0.85, merge_time=0.1, min_big_time=1.0,
                 exit_delay=0.35, link_dots=7, link_speed=0.35):
        self.radius_ratio = radius_ratio
        self.big_ratio = big_ratio
        self.merge_time = merge_time
        self.min_big_time = min_big_time
        self.exit_delay = exit_delay
        self.link_dots = link_dots
        self.link_speed = link_speed
        self.color = np.array(color, dtype=np.float32)
        self.spin_speed = spin_speed
        self.glow = glow
        self.blur = blur
        self.enabled = True

        self._templates = _build_templates(ticks, arcs)
        self._time = 0.0
        self._layer = None
        self._dirty = None
        self._merge = {}      # tay -> mức độ đã gộp thành vòng lớn (0..1), chỉ để chạy hoạt hình
        self._state = {}      # tay -> trạng thái NHỎ/LỚN của máy trạng thái
        self._last_time = None

    # ------------------------------------------------------------------ public

    def update_merge(self, hands_info):
        """
        Cập nhật trạng thái vòng của từng bàn tay bằng 1 MÁY TRẠNG THÁI.

        Mỗi bàn tay chỉ có 2 trạng thái: NHỎ (các vòng ở đầu ngón) và LỚN (1
        vòng giữa lòng bàn tay). Biến `merge` (0..1) chỉ là biến chạy hoạt hình
        đi theo trạng thái, KHÔNG phải thứ quyết định trạng thái.

        Vì sao phải làm vậy: MediaPipe luôn rung nhẹ ở ngón áp út và ngón út khi
        bàn tay hơi nghiêng. Nếu để trạng thái bám thẳng vào kết quả nhận diện
        từng frame, mỗi frame đọc hụt 1 ngón sẽ kéo `merge` tụt xuống rồi lại
        dâng lên - vòng lớn co bóp liên tục, đúng hiện tượng "nhấp nháy".

        Luật chuyển trạng thái (bất đối xứng - vào dễ, ra khó):
            NHỎ -> LỚN : ngay khi đủ 4 ngón xoè
            LỚN -> NHỎ : phải thoả CẢ HAI điều kiện
                         (1) đã ở trạng thái LỚN ít nhất `min_big_time` giây
                         (2) điều kiện "thiếu ngón" kéo dài LIÊN TỤC ít nhất
                             `exit_delay` giây - chỉ cần 1 frame đọc lại đủ ngón
                             là đồng hồ này reset về 0

        Tách riêng khỏi draw() để chương trình chính đọc được mức gộp TRƯỚC khi
        vẽ hạt - hệ hạt cần biết tay nào đang mở vòng lớn để trải hạt ra khắp
        bàn tay.
        """
        now = time.perf_counter()
        # dt bị chặn trên để lần chạy đầu (hoặc lúc máy khựng) không nhảy cóc
        dt = 0.016 if self._last_time is None else min(now - self._last_time, 0.1)
        self._last_time = now
        step = dt / max(self.merge_time, 1e-3)

        seen = set()
        for hand in hands_info:
            handedness = hand["handedness"]
            seen.add(handedness)
            fingers_up = hand.get("fingers_up", [1] * 5)
            # Gộp khi 4 ngón (trừ ngón cái) cùng xoè; ngón cái không tính
            enough_fingers = all(fingers_up[i] for i in MERGE_FINGERS)

            state = self._state.setdefault(
                handedness, {"big": False, "since": now, "bad_since": None})

            if not state["big"]:
                if enough_fingers:
                    state["big"] = True
                    state["since"] = now
                    state["bad_since"] = None
            else:
                if enough_fingers:
                    # Đọc lại đủ ngón -> xoá đồng hồ đếm "thiếu ngón"
                    state["bad_since"] = None
                else:
                    if state["bad_since"] is None:
                        state["bad_since"] = now
                    held_enough = (now - state["since"]) >= self.min_big_time
                    missing_enough = (now - state["bad_since"]) >= self.exit_delay
                    if held_enough and missing_enough:
                        state["big"] = False
                        state["since"] = now
                        state["bad_since"] = None

            # `merge` chỉ chạy tuyến tính về đích do trạng thái quy định.
            # Phải KẸP đúng vào đích: viết kiểu "merge += step nếu chưa tới,
            # ngược lại -= step" sẽ làm merge nhảy qua nhảy lại quanh 1.0 mãi
            # (tới đích rồi thì nhánh else lại kéo nó xuống) - chính là nguyên
            # nhân vòng lớn phồng xẹp liên tục.
            target = 1.0 if state["big"] else 0.0
            merge = self._merge.get(handedness, 0.0)
            if merge < target:
                merge = min(target, merge + step)
            elif merge > target:
                merge = max(target, merge - step)
            self._merge[handedness] = float(merge)

        for handedness in list(self._merge):
            if handedness not in seen:
                self._merge.pop(handedness, None)
                self._state.pop(handedness, None)
        return self._merge

    def merge_amount(self, handedness):
        """Mức độ đã gộp (0..1) của 1 bàn tay - 1.0 nghĩa là đang mở vòng lớn."""
        return self._merge.get(handedness, 0.0)

    def draw(self, frame, hands_info):
        """
        Vẽ vòng ma pháp của tất cả đầu ngón đang giơ lên `frame` (cộng ánh sáng).
        Trả về True nếu có vẽ vòng nào.
        """
        if not self.enabled:
            return False

        height, width = frame.shape[:2]
        if self._layer is None or self._layer.shape != (height, width):
            self._layer = np.zeros((height, width), dtype=np.float32)

        self._time += self.spin_speed
        self._dirty = None          # hình chữ nhật bao quanh tất cả các vòng
        big_circles = []            # (tâm, bán kính, mức gộp) của các vòng lớn

        for hand in hands_info:
            landmarks = hand["landmarks"]
            fingers_up = hand.get("fingers_up", [1] * 5)
            handedness = hand["handedness"]

            wrist = np.array([landmarks[WRIST_ID].x * width,
                              landmarks[WRIST_ID].y * height], dtype=np.float32)
            palm = np.array([landmarks[MIDDLE_MCP_ID].x * width,
                             landmarks[MIDDLE_MCP_ID].y * height], dtype=np.float32)
            hand_size = float(np.linalg.norm(palm - wrist))
            radius = self.radius_ratio * hand_size
            if radius < 5:
                continue

            # Mức độ "đã gộp" do update_merge() tính sẵn theo thời gian thật
            merge = self._merge.get(handedness, 0.0)

            # Tâm lòng bàn tay = trung bình cổ tay + 4 gốc ngón
            center_ids = (WRIST_ID, 5, MIDDLE_MCP_ID, 13, 17)
            palm_center = np.mean(
                [[landmarks[i].x * width, landmarks[i].y * height] for i in center_ids],
                axis=0).astype(np.float32)

            # --- 5 vòng nhỏ: trôi về tâm bàn tay và mờ dần khi đang gộp ---
            if merge < 0.97:
                for finger_index, tip_id in CIRCLE_FINGERS:
                    if not fingers_up[finger_index]:
                        continue        # ngón đang gập -> không vẽ vòng
                    tip = np.array([landmarks[tip_id].x * width,
                                    landmarks[tip_id].y * height], dtype=np.float32)
                    position = tip + (palm_center - tip) * merge
                    # Lệch pha theo từng ngón để các vòng không quay trùng nhịp
                    self._stamp(tuple(position), radius * (1.0 - 0.35 * merge),
                                phase=finger_index * 0.9, intensity=1.0 - merge)

            # --- Vòng lớn giữa lòng bàn tay: lớn dần và rõ dần, quay chậm hơn ---
            if merge > 0.03:
                big_radius = radius + (self.big_ratio * hand_size - radius) * merge
                self._stamp(tuple(palm_center), big_radius, phase=0.0,
                            intensity=merge, spin_scale=0.55, pulse=0.0)
                big_circles.append((palm_center, big_radius, merge))

        # Cả 2 tay cùng mở vòng lớn -> nối 2 tâm bằng 1 chùm hạt đỏ
        if len(big_circles) >= 2:
            (c1, r1, m1), (c2, r2, m2) = big_circles[0], big_circles[1]
            strength = min(m1, m2)
            if strength > 0.55:
                self._draw_link(c1, c2, min(r1, r2), strength)

        if self._dirty is None:
            return False

        # Chỉ làm mờ + tô màu + cộng trong vùng có vòng, không đụng cả khung hình
        pad = int(self.blur * 4) + 2
        x0 = max(self._dirty[0] - pad, 0)
        y0 = max(self._dirty[1] - pad, 0)
        x1 = min(self._dirty[2] + pad, width)
        y1 = min(self._dirty[3] + pad, height)

        patch = self._layer[y0:y1, x0:x1]
        if self.blur > 0:
            # Quầng sáng mờ + cộng lại nét gốc để lõi vẫn đặc
            patch = cv2.GaussianBlur(patch, (0, 0), self.blur) * 0.85 + patch

        glow = (patch * self.glow)[:, :, None] * self.color[None, None, :]
        region = frame[y0:y1, x0:x1].astype(np.float32) + glow
        frame[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)

        # Dọn sạch vùng vừa dùng để frame sau vẽ lại từ đầu
        self._layer[y0:y1, x0:x1] = 0.0
        return True

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    # --------------------------------------------------------------- internals

    def _draw_link(self, center_a, center_b, radius, strength):
        """
        Vẽ chùm sáng nối 2 tâm vòng lớn: 1 dải sáng chạy thẳng giữa 2 tâm, cộng
        thêm vài khối hạt to trôi dọc theo nó.

        Các khối hạt được rải cách đều rồi cùng trôi theo thời gian (vị trí lấy
        phần lẻ của `t`), nên nhìn như dòng năng lượng chảy từ tay này sang tay
        kia. Khối ở giữa đường to hơn khối ở 2 đầu cho chùm trông phình ở giữa.
        """
        p1 = np.asarray(center_a, dtype=np.float32)
        p2 = np.asarray(center_b, dtype=np.float32)
        length = float(np.linalg.norm(p2 - p1))
        if length < 10:
            return

        layer = self._layer
        core_width = max(2, int(radius * 0.22 * strength))

        # Dải sáng nối 2 tâm (2 lớp: lõi mảnh sáng, vỏ dày mờ hơn)
        cv2.line(layer, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)),
                 0.45 * strength, core_width * 3, cv2.LINE_AA)
        cv2.line(layer, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)),
                 0.95 * strength, core_width, cv2.LINE_AA)

        # Các khối hạt to trôi dọc chùm
        for k in range(self.link_dots):
            t = (self._time * self.link_speed + k / self.link_dots) % 1.0
            position = p1 + (p2 - p1) * t
            # Phình ở giữa, thu nhỏ về 2 đầu
            bulge = 0.45 + 0.55 * np.sin(np.pi * t)
            dot_radius = max(2, int(radius * 0.30 * bulge * strength))
            cv2.circle(layer, tuple(np.round(position).astype(int)), dot_radius,
                       0.9 * strength, -1, cv2.LINE_AA)

        pad = int(radius * 0.4) + 4
        self._expand_dirty(int(min(p1[0], p2[0])) - pad, int(min(p1[1], p2[1])) - pad,
                           int(max(p1[0], p2[0])) + pad, int(max(p1[1], p2[1])) + pad)

    def _expand_dirty(self, x0, y0, x1, y1):
        """Mở rộng hình chữ nhật cần xử lý (làm mờ + cộng màu) cho khớp vùng vừa vẽ."""
        height, width = self._layer.shape
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, width), min(y1, height)
        if x1 <= x0 or y1 <= y0:
            return
        if self._dirty is None:
            self._dirty = [x0, y0, x1, y1]
        else:
            self._dirty = [min(self._dirty[0], x0), min(self._dirty[1], y0),
                           max(self._dirty[2], x1), max(self._dirty[3], y1)]

    def _stamp(self, center, radius, phase=0.0, intensity=1.0, spin_scale=1.0,
               pulse=0.05):
        """
        Xoay + thu nhỏ 3 ảnh mẫu rồi dán vào 1 vị trí trên khung hình.

        `pulse` là biên độ "nhịp thở" (bán kính phồng xẹp nhè nhẹ). Vòng lớn
        truyền pulse=0 vì ở cỡ lớn, cùng biên độ đó nhìn thành nhấp nháy to nhỏ
        liên tục chứ không còn là nhịp thở nữa.
        """
        if intensity <= 0.02:
            return
        t = self._time * spin_scale + phase
        if pulse:
            radius = radius * (1.0 + pulse * np.sin(t * 2.2))

        patch = int(radius * 2.3) | 1            # cạnh vùng dán (số lẻ)
        if patch < 9:
            return

        cx, cy = center
        x0, y0 = int(round(cx - patch / 2)), int(round(cy - patch / 2))
        x1, y1 = x0 + patch, y0 + patch

        height, width = self._layer.shape
        # Cắt bớt phần lọt ra ngoài khung hình
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x1, width), min(y1, height)
        if sx1 <= sx0 or sy1 <= sy0:
            return

        scale = (radius * 2.0) / TEMPLATE_SIZE
        # 3 lớp quay khác tốc độ, lớp giữa quay ngược chiều
        angles = (np.degrees(t), -np.degrees(t * 1.8), np.degrees(t * 2.6))

        stamp = np.zeros((patch, patch), dtype=np.float32)
        for template, angle in zip(self._templates, angles):
            matrix = cv2.getRotationMatrix2D((TEMPLATE_SIZE / 2, TEMPLATE_SIZE / 2),
                                             angle, scale)
            # Dời tâm ảnh mẫu về tâm vùng dán
            matrix[0, 2] += patch / 2 - TEMPLATE_SIZE / 2
            matrix[1, 2] += patch / 2 - TEMPLATE_SIZE / 2
            warped = cv2.warpAffine(template, matrix, (patch, patch),
                                    flags=cv2.INTER_LINEAR)
            # Phải GỘP bằng phép lấy max: nếu cho warpAffine ghi thẳng vào cùng
            # 1 ảnh thì lớp sau ghi đè phần nền đen lên lớp trước, chỉ còn lớp
            # cuối cùng hiện ra.
            np.maximum(stamp, warped, out=stamp)

        if intensity < 0.999:
            stamp *= intensity

        region = stamp[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
        np.maximum(self._layer[sy0:sy1, sx0:sx1], region,
                   out=self._layer[sy0:sy1, sx0:sx1])

        self._expand_dirty(sx0, sy0, sx1, sy1)
