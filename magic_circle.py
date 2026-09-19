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

VÙNG ÂM BẢN: vòng lớn hiện ra thì vùng không gian quanh nó bị ĐẢO MÀU (âm bản):

    - 1 bàn tay mở vòng lớn  -> 1 vành âm bản bao quanh bàn tay đó, bán kính
      `single_ratio` lần bán kính vòng lớn
    - cả 2 tay cùng mở vòng  -> vùng âm bản kéo dài thành 1 dải nối 2 bàn tay

Riêng chỗ vòng ma pháp nằm thì được CHỪA RA, không đảo màu (xem giải thích
trong _invert_zone). Mép vùng được làm mềm nên loang dần chứ không lộ viền cắt.
Khi 2 tay cùng mở vòng, hạt đỏ còn toả ra dọc đoạn nối 2 tâm (phần hạt do
particles.py lo).

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
        invert_ratio : bề rộng vùng đảo màu khi CẢ 2 tay mở vòng, so với bán kính vòng
        single_ratio : bán kính vùng đảo màu khi CHỈ 1 tay mở vòng, so với bán kính vòng
        invert_keep  : bán kính chỗ chừa ra quanh mỗi vòng ma pháp (không đảo màu)
        invert_feather: độ mềm của mép vùng đảo màu (pixel)
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
                 exit_delay=0.35, invert_ratio=2.8, single_ratio=3.0,
                 invert_keep=1.02, invert_feather=10.0):
        self.radius_ratio = radius_ratio
        self.big_ratio = big_ratio
        self.merge_time = merge_time
        self.min_big_time = min_big_time
        self.exit_delay = exit_delay
        self.invert_ratio = invert_ratio
        self.single_ratio = single_ratio
        self.invert_keep = invert_keep
        self.invert_feather = invert_feather
        self.link = None      # (tâm A, tâm B, độ đậm) của chùm nối ở frame vừa vẽ
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

    def big_circles(self, hands_info, frame_shape):
        """
        Danh sách (tâm, bán kính, mức gộp) của các vòng lớn đang mở.

        Dùng chung cho cả 2 lượt vẽ: lượt đảo màu nền (draw_invert) và lượt vẽ
        vòng (draw).
        """
        height, width = frame_shape[:2]
        result = []
        for hand in hands_info:
            merge = self._merge.get(hand["handedness"], 0.0)
            if merge <= 0.03:
                continue
            landmarks = hand["landmarks"]
            wrist = np.array([landmarks[WRIST_ID].x * width,
                              landmarks[WRIST_ID].y * height], dtype=np.float32)
            palm = np.array([landmarks[MIDDLE_MCP_ID].x * width,
                             landmarks[MIDDLE_MCP_ID].y * height], dtype=np.float32)
            hand_size = float(np.linalg.norm(palm - wrist))
            radius = self.radius_ratio * hand_size
            if radius < 5:
                continue
            center_ids = (WRIST_ID, 5, MIDDLE_MCP_ID, 13, 17)
            center = np.mean([[landmarks[i].x * width, landmarks[i].y * height]
                              for i in center_ids], axis=0).astype(np.float32)
            big_radius = radius + (self.big_ratio * hand_size - radius) * merge
            result.append((center, big_radius, merge))
        return result

    def draw_invert(self, frame, hands_info):
        """
        Lượt 1: đảo màu (âm bản) vùng quanh các vòng lớn.

        Tách riêng khỏi draw() và gọi TRƯỚC khi vẽ hạt, để hạt và vòng ma pháp
        giữ nguyên màu của chúng - chỉ có hình nền thật (bàn tay, khung cảnh)
        mới bị đảo màu. Nếu đảo màu sau khi vẽ hạt thì vệt lửa vàng cam sẽ hoá
        xanh lơ trong vùng đó.
        """
        self.link = None
        if not self.enabled:
            return

        circles = self.big_circles(hands_info, frame.shape)
        if len(circles) >= 2:
            (c1, r1, m1), (c2, r2, m2) = circles[0], circles[1]
            strength = min(m1, m2)
            if strength > 0.55:
                self._invert_zone(frame, [(c1, r1), (c2, r2)], strength)
                # Báo đoạn nối 2 tâm ra ngoài để hệ hạt bắn hạt đỏ toả dọc nó
                self.link = (c1, c2, strength)
        elif len(circles) == 1:
            center, radius, merge = circles[0]
            if merge > 0.55:
                self._invert_zone(frame, [(center, radius)], merge)

    def merge_amount(self, handedness):
        """Mức độ đã gộp (0..1) của 1 bàn tay - 1.0 nghĩa là đang mở vòng lớn."""
        return self._merge.get(handedness, 0.0)

    def draw(self, frame, hands_info):
        """
        Lượt 2: vẽ vòng ma pháp của tất cả đầu ngón đang giơ lên `frame` (cộng
        ánh sáng). Gọi SAU draw_invert() và sau khi vẽ hạt.
        Trả về True nếu có vẽ vòng nào.
        """
        if not self.enabled:
            return False

        height, width = frame.shape[:2]
        if self._layer is None or self._layer.shape != (height, width):
            self._layer = np.zeros((height, width), dtype=np.float32)

        self._time += self.spin_speed
        self._dirty = None          # hình chữ nhật bao quanh tất cả các vòng

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
        # Cộng ánh sáng bằng cv2.add trên uint8 (tự chặn trần 255) - nhanh hơn
        # hẳn so với đổi cả vùng ảnh sang float rồi clip trong numpy
        glow8 = np.clip(glow, 0, 255).astype(np.uint8)
        roi = frame[y0:y1, x0:x1]
        cv2.add(roi, glow8, dst=roi)

        # Dọn sạch vùng vừa dùng để frame sau vẽ lại từ đầu
        self._layer[y0:y1, x0:x1] = 0.0
        return True

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    # --------------------------------------------------------------- internals

    def _invert_zone(self, frame, circles, strength):
        """
        Đảo màu (âm bản) vùng không gian quanh vòng lớn.

        `circles` là danh sách (tâm, bán kính) của các vòng lớn đang mở:
            - 1 phần tử  -> vùng đảo màu là hình TRÒN bán kính `single_ratio`
              lần bán kính vòng, bao quanh bàn tay đó
            - 2 phần tử  -> vùng đảo màu là hình BẦU DỤC kéo dài nối 2 bàn tay,
              bề ngang `invert_ratio` lần bán kính vòng

        Chỗ bản thân vòng ma pháp nằm thì được CHỪA RA (`invert_keep`), vì 2 lý
        do: giữ đúng màu đỏ của vòng như yêu cầu, và vì cảnh webcam thường tối
        nên đảo màu xong vùng đó sáng trắng - vòng lại vẽ theo kiểu CỘNG ánh
        sáng, cộng lên nền đã sáng thì cháy trắng, mất hẳn màu.

        Mặt nạ được làm mềm ở mép (`invert_feather`) rồi pha giữa ảnh gốc và
        ảnh âm bản:  kết quả = gốc * (1 - m) + (255 - gốc) * m

        Phép pha dùng hàm số nguyên của OpenCV (multiply/add trên ảnh uint8)
        chứ không đổi cả vùng ảnh sang float trong numpy: vùng này khá lớn,
        làm kiểu numpy tốn ~14 ms/frame, dùng OpenCV chỉ còn ~1,3 ms mà sai
        lệch màu trung bình chưa tới 0,2/255.
        """
        if not circles:
            return

        height, width = frame.shape[:2]
        points = [np.asarray(c, dtype=np.float32) for c, _ in circles]
        radii = [r for _, r in circles]

        if len(circles) >= 2:
            reach = max(radii) * self.invert_ratio
        else:
            reach = radii[0] * self.single_ratio

        pad = int(reach + self.invert_feather * 2) + 4
        x0 = int(max(min(p[0] for p in points) - pad, 0))
        y0 = int(max(min(p[1] for p in points) - pad, 0))
        x1 = int(min(max(p[0] for p in points) + pad, width))
        y1 = int(min(max(p[1] for p in points) + pad, height))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return

        # Mặt nạ dựng ở NỬA độ phân giải rồi mới phóng to: làm mờ là phần tốn
        # nhất, tính ở nửa độ phân giải rẻ hơn ~4 lần mà mép vốn đã mềm nên
        # phóng to lại không thấy khác.
        scale = 0.5
        small_w = max(8, int((x1 - x0) * scale))
        small_h = max(8, int((y1 - y0) * scale))
        mask = np.zeros((small_h, small_w), dtype=np.float32)
        offset = np.array([x0, y0], dtype=np.float32)

        if len(circles) >= 2:
            p1 = (points[0] - offset) * scale
            p2 = (points[1] - offset) * scale
            delta = p2 - p1
            length = float(np.linalg.norm(delta))
            center = (p1 + p2) / 2.0
            half_width = max(radii[0], radii[1]) * self.invert_ratio * scale
            axes = (int(length / 2 + half_width), int(half_width))
            angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
            cv2.ellipse(mask, (int(center[0]), int(center[1])), axes, angle,
                        0, 360, 1.0, -1, cv2.LINE_AA)
        else:
            center = (points[0] - offset) * scale
            cv2.circle(mask, (int(center[0]), int(center[1])),
                       int(radii[0] * self.single_ratio * scale), 1.0, -1, cv2.LINE_AA)

        # Chừa chỗ của từng vòng ma pháp ra (vẽ đè giá trị 0)
        for point, radius in zip(points, radii):
            local = (point - offset) * scale
            cv2.circle(mask, (int(local[0]), int(local[1])),
                       max(int(radius * self.invert_keep * scale), 1), 0.0, -1, cv2.LINE_AA)

        if self.invert_feather > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), self.invert_feather * scale)

        mask8 = np.clip(mask * strength * 255.0, 0, 255).astype(np.uint8)
        mask8 = cv2.resize(mask8, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        mask3 = cv2.cvtColor(mask8, cv2.COLOR_GRAY2BGR)

        roi = frame[y0:y1, x0:x1]
        inverted = cv2.bitwise_not(roi)                 # 255 - ảnh gốc
        frame[y0:y1, x0:x1] = cv2.add(
            cv2.multiply(roi, cv2.bitwise_not(mask3), scale=1 / 255.0),
            cv2.multiply(inverted, mask3, scale=1 / 255.0))

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
