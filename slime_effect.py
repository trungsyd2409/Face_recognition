"""
slime_effect.py
Hiệu ứng "slime" (chất nhờn / gel) bám giữa ngón cái và ngón trỏ của từng bàn tay.

Ý tưởng: dùng kỹ thuật METABALL. Mỗi đầu ngón là 1 "quả cầu chất lỏng"; tại mỗi
điểm ảnh ta tính 1 giá trị trường (field):

    F(p) = tổng của  r_i^2 / khoảng_cách(p, tâm_i)^2

Chỗ nào F > 1 thì coi là nằm TRONG khối slime. Nhờ cách cộng dồn này, khi 2 quả
cầu lại gần nhau thì bề mặt của chúng tự động "dính" liền thành 1 khối - đúng
kiểu chất nhờn, không cần vẽ tay đường nối.

Thêm vào đó:
- Một chuỗi quả cầu phụ nằm dọc đoạn nối 2 đầu ngón tạo thành SỢI slime. Càng
  kéo 2 ngón ra xa, bán kính các quả cầu ở giữa càng nhỏ -> sợi thắt eo lại.
  Riêng chuỗi này gộp bằng phép LẤY MAX (không cộng dồn), nếu không hàng chục
  quả cầu chồng nhau sẽ cộng lại thành 1 cục to thay vì 1 sợi mảnh.
- Sợi võng xuống theo trọng lực (sag) và rung nhẹ theo thời gian cho "mềm". Độ
  võng có QUÁN TÍNH: nó chạy theo tay chậm hơn 1 nhịp, nên khi bạn di chuyển
  tay nhanh thì slime bị trễ lại và đung đưa như chất lỏng nặng.
- Trong lúc đang chảy, slime thỉnh thoảng NHỎ GIỌT từ điểm thấp nhất của sợi và
  từ mặt dưới 2 khối gel ở đầu ngón.
- Kéo quá xa -> sợi ĐỨT, sinh ra vài GIỌT slime rơi xuống theo trọng lực.
- Bề mặt có khúc xạ nhẹ (ảnh nền bị bẻ cong theo độ dốc của trường F), có viền
  sáng và đốm sáng phản chiếu (specular) cho giống gel bóng.

Dùng trong main_webcam.py: tạo 1 đối tượng SlimeEffect rồi gọi
`slime.update_and_draw(frame, hands_info)` mỗi frame.
"""

import time

import cv2
import numpy as np

# Chỉ số landmark cần dùng (theo chuẩn MediaPipe Hands)
WRIST_ID = 0
MIDDLE_MCP_ID = 9      # gốc ngón giữa - dùng đo "kích thước" bàn tay
THUMB_TIP_ID = 4
INDEX_TIP_ID = 8

GEL_COLOR_BGR = (250, 250, 252)   # màu slime mặc định (trắng sữa)


def _dist(point_a, point_b):
    return float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))


class SlimeEffect:
    """
    Quản lý trạng thái (sợi đang dính hay đã đứt, các giọt đang rơi) và vẽ hiệu
    ứng slime lên frame.

    Tham số chính:
        gel_color     : màu slime (BGR)
        render_scale  : tính trường F ở độ phân giải thu nhỏ cho nhanh (0.5 = 1/2)
        break_ratio   : kéo xa quá (break_ratio x kích thước bàn tay) thì sợi đứt
        blob_ratio    : bán kính quả cầu ở đầu ngón, theo kích thước bàn tay
        bridge_blobs  : số quả cầu phụ rải dọc sợi slime
        sag_ratio     : độ võng xuống của sợi (càng lớn càng chảy nhiều)
        sag_follow    : sợi bắt kịp tay nhanh hay chậm (0-1, càng nhỏ càng ì)
        drip_chance   : xác suất mỗi frame nhỏ 1 giọt khi slime đang chảy
        gravity       : gia tốc rơi của các giọt slime (pixel/frame^2)
    """

    def __init__(self, gel_color=GEL_COLOR_BGR, render_scale=0.5,
                 break_ratio=1.7, blob_ratio=0.15, bridge_blobs=12,
                 sag_ratio=0.55, sag_follow=0.15, drip_chance=0.045,
                 gravity=1.1):
        self.gel_color = np.array(gel_color, dtype=np.float32)
        self.render_scale = render_scale
        self.break_ratio = break_ratio
        self.blob_ratio = blob_ratio
        self.bridge_blobs = bridge_blobs
        self.sag_ratio = sag_ratio
        self.sag_follow = sag_follow
        self.drip_chance = drip_chance
        self.gravity = gravity

        self._sag_state = {}   # "Left"/"Right" -> độ võng hiện tại (có quán tính)
        self._connected = {}   # "Left"/"Right" -> sợi slime còn dính hay không
        self._droplets = []    # các giọt đang rơi: dict(x, y, vx, vy, r)
        self._start_time = time.time()

    # ---------------------------------------------------------------- public

    def update_and_draw(self, frame, hands_info):
        """
        Cập nhật trạng thái + vẽ slime cho tất cả bàn tay trong `hands_info`
        (định dạng trả về của utils.detect_hands). Vẽ trực tiếp lên `frame`.
        """
        height, width = frame.shape[:2]
        now = time.time() - self._start_time

        blobs = []          # (x, y, r) - các quả cầu gộp bằng phép CỘNG
        chains = []         # list các chuỗi quả cầu của sợi, gộp bằng phép MAX
        highlights = []     # (x, y, r) - các quả cầu chính, dùng để chấm sáng

        for hand in hands_info:
            hand_blobs, hand_chain, hand_highlights = self._build_hand_blobs(
                hand, width, height, now)
            blobs.extend(hand_blobs)
            if hand_chain:
                chains.append(hand_chain)
            highlights.extend(hand_highlights)

        # Bàn tay biến mất khỏi khung hình -> coi như sợi của tay đó đã đứt
        seen = {hand["handedness"] for hand in hands_info}
        for handedness in list(self._connected):
            if handedness not in seen:
                self._connected.pop(handedness, None)
                self._sag_state.pop(handedness, None)

        self._update_droplets(width, height)
        for drop in self._droplets:
            blobs.append((drop["x"], drop["y"], drop["r"]))
            highlights.append((drop["x"], drop["y"], drop["r"]))

        if blobs:
            self._render_blobs(frame, blobs, chains, highlights)

    def reset(self):
        """Xoá hết trạng thái (dùng khi bật/tắt chế độ slime)."""
        self._connected.clear()
        self._sag_state.clear()
        self._droplets.clear()

    # --------------------------------------------------------------- private

    def _build_hand_blobs(self, hand, width, height, now):
        """
        Tính các quả cầu metaball cho 1 bàn tay. Trả về:
            (blobs_cong, chuoi_soi, blobs_danh_sang)
        - blobs_cong: 2 đầu ngón, gộp vào trường bằng phép CỘNG (để chúng dính
          vào nhau và dính với sợi một cách mượt mà)
        - chuoi_soi: các quả cầu dọc sợi slime, gộp bằng phép LẤY MAX
        """
        landmarks = hand["landmarks"]
        handedness = hand["handedness"]

        def to_px(landmark):
            return (landmark.x * width, landmark.y * height)

        wrist = to_px(landmarks[WRIST_ID])
        middle_mcp = to_px(landmarks[MIDDLE_MCP_ID])
        thumb_tip = to_px(landmarks[THUMB_TIP_ID])
        index_tip = to_px(landmarks[INDEX_TIP_ID])

        hand_size = _dist(wrist, middle_mcp)
        if hand_size < 10:          # tay quá nhỏ/ở quá xa -> bỏ qua
            return [], [], []

        tip_radius = self.blob_ratio * hand_size
        gap = _dist(thumb_tip, index_tip)
        break_length = self.break_ratio * hand_size

        # Trạng thái dính/đứt, có "trễ" (hysteresis): đứt khi kéo quá xa, chỉ
        # dính lại khi 2 ngón chụm lại đủ gần - tránh chớp tắt liên tục.
        connected = self._connected.get(handedness, True)
        if connected and gap > break_length:
            connected = False
            self._spawn_droplets(thumb_tip, index_tip, tip_radius)
        elif not connected and gap < break_length * 0.45:
            connected = True
        self._connected[handedness] = connected

        blobs = [(thumb_tip[0], thumb_tip[1], tip_radius),
                 (index_tip[0], index_tip[1], tip_radius)]
        highlights = list(blobs)
        chain = []

        # Chỉ dựng sợi khi 2 ngón đã tách ra đủ xa; lúc 2 đầu ngón còn chồng lên
        # nhau thì bản thân 2 quả cầu đã là 1 khối rồi, không cần sợi.
        if connected and gap > tip_radius:
            stretch = min(gap / break_length, 1.0)
            # Sợi càng bị kéo dãn thì phần giữa càng mảnh
            waist = max(0.14, 0.75 * (1.0 - stretch) ** 0.7)

            # Độ võng xuống theo trọng lực. Giá trị này đi theo tay có QUÁN
            # TÍNH (lọc thông thấp): kéo tay ra nhanh thì slime chảy xuống từ
            # từ chứ không nhảy ngay tới hình dạng mới.
            target_sag = self.sag_ratio * gap * (0.45 + 0.55 * stretch)
            sag = self._sag_state.get(handedness, target_sag)
            sag += (target_sag - sag) * self.sag_follow
            self._sag_state[handedness] = sag

            wobble = 0.04 * hand_size * np.sin(now * 6.0)

            # Đặt các quả cầu đủ dày để sợi liền mạch, kể cả khi đã rất mảnh
            steps = int(np.clip(gap / max(2.0, 0.25 * tip_radius * waist),
                                self.bridge_blobs, 60))
            lowest = None
            for i in range(1, steps):
                t = i / steps
                # curve: 0 ở 2 đầu, 1 ở giữa. Mũ 0.65 làm đáy sợi bè ra, trông
                # nặng và "chảy" hơn so với một đường cong tròn đều.
                curve = (1.0 - (2.0 * t - 1.0) ** 2) ** 0.65
                x = thumb_tip[0] + (index_tip[0] - thumb_tip[0]) * t
                y = thumb_tip[1] + (index_tip[1] - thumb_tip[1]) * t
                y += (sag + wobble) * curve
                # Phần đáy dày hơn 1 chút: chất lỏng dồn xuống chỗ thấp nhất
                radius = tip_radius * (waist * (1.0 + 0.6 * curve)
                                       + (1.0 - waist) * (1.0 - curve) ** 1.5)
                chain.append((x, y, radius))
                if lowest is None or y > lowest[1]:
                    lowest = (x, y, radius)

            # Thỉnh thoảng nhỏ 1 giọt từ điểm thấp nhất của sợi
            if lowest is not None and np.random.random() < self.drip_chance * (0.4 + stretch):
                self._add_droplet(lowest[0], lowest[1] + lowest[2],
                                  max(3.0, lowest[2] * 1.4), spread=1.0)

        # Gel ở đầu ngón cũng đọng lại rồi nhỏ giọt xuống
        if np.random.random() < self.drip_chance * 0.5:
            drip_from = thumb_tip if np.random.random() < 0.5 else index_tip
            self._add_droplet(drip_from[0], drip_from[1] + tip_radius,
                              tip_radius * 0.45, spread=0.6)

        return blobs, chain, highlights

    def _add_droplet(self, x, y, radius, spread=1.0):
        """Thêm 1 giọt slime rơi tự do tại (x, y)."""
        self._droplets.append({
            "x": float(x) + np.random.uniform(-spread, spread),
            "y": float(y),
            "vx": np.random.uniform(-0.6, 0.6) * spread,
            "vy": np.random.uniform(0.0, 1.0),
            "r": float(radius),
        })
        self._droplets = self._droplets[-16:]   # giới hạn số giọt cho nhẹ máy

    def _spawn_droplets(self, thumb_tip, index_tip, tip_radius):
        """Sợi vừa đứt -> sinh vài giọt slime rơi xuống ở khoảng giữa 2 ngón."""
        mid_x = (thumb_tip[0] + index_tip[0]) / 2.0
        mid_y = (thumb_tip[1] + index_tip[1]) / 2.0
        for _ in range(np.random.randint(2, 4)):
            self._add_droplet(
                mid_x + np.random.uniform(-tip_radius, tip_radius),
                mid_y + np.random.uniform(-tip_radius, tip_radius),
                tip_radius * np.random.uniform(0.35, 0.65),
                spread=1.5,
            )

    def _update_droplets(self, width, height):
        """Cập nhật vị trí các giọt đang rơi, bỏ giọt đã ra khỏi khung hình."""
        alive = []
        for drop in self._droplets:
            drop["vy"] += self.gravity
            drop["x"] += drop["vx"]
            drop["y"] += drop["vy"]
            drop["r"] *= 0.995
            if drop["y"] - drop["r"] < height and drop["r"] > 1.5 and -50 < drop["x"] < width + 50:
                alive.append(drop)
        self._droplets = alive

    def _render_blobs(self, frame, blobs, chains, highlights):
        """Dựng trường metaball từ các quả cầu rồi tô vùng slime lên `frame`."""
        height, width = frame.shape[:2]
        all_blobs = list(blobs) + [b for chain in chains for b in chain]
        max_radius = max(b[2] for b in all_blobs)

        # Ở xa, F ~ (tổng r^2) / d^2, nên bề mặt F = 1 chắc chắn nằm trong bán
        # kính sqrt(tổng r^2) - lấy dư 1 chút làm lề cho vùng tính toán.
        reach = np.sqrt(sum(b[2] ** 2 for b in blobs) +
                        max((b[2] ** 2 for b in all_blobs), default=1.0))
        pad = 1.5 * reach + 4.0

        x1 = int(max(min(b[0] for b in all_blobs) - pad, 0))
        y1 = int(max(min(b[1] for b in all_blobs) - pad, 0))
        x2 = int(min(max(b[0] for b in all_blobs) + pad, width))
        y2 = int(min(max(b[1] for b in all_blobs) + pad, height))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return

        roi = frame[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        # Tính trường F ở độ phân giải thu nhỏ cho nhẹ, rồi phóng to lại
        small_w = max(8, int(roi_w * self.render_scale))
        small_h = max(8, int(roi_h * self.render_scale))
        sx = roi_w / small_w
        sy = roi_h / small_h

        grid_x = (np.arange(small_w, dtype=np.float32) + 0.5) * sx + x1
        grid_y = (np.arange(small_h, dtype=np.float32) + 0.5) * sy + y1
        gx, gy = np.meshgrid(grid_x, grid_y)

        field = np.zeros((small_h, small_w), dtype=np.float32)
        for bx, by, br in blobs:
            d2 = (gx - bx) ** 2 + (gy - by) ** 2
            field += (br * br) / (d2 + 1.0)

        # Sợi slime: lấy MAX trên cả chuỗi rồi mới cộng vào trường chung, để
        # nhiều quả cầu chồng nhau không cộng dồn thành 1 cục phình to.
        for chain in chains:
            chain_field = np.zeros((small_h, small_w), dtype=np.float32)
            for bx, by, br in chain:
                d2 = (gx - bx) ** 2 + (gy - by) ** 2
                np.maximum(chain_field, (br * br) / (d2 + 1.0), out=chain_field)
            field += chain_field

        field = cv2.resize(field, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
        field = cv2.GaussianBlur(field, (0, 0), 1.2)

        mask = field > 1.0
        if not mask.any():
            return

        # Khúc xạ: bẻ cong ảnh nền theo độ dốc của trường F (giống nhìn qua gel)
        grad_x = cv2.Sobel(field, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(field, cv2.CV_32F, 0, 1, ksize=5)
        refract = 5.0
        base_x, base_y = np.meshgrid(
            np.arange(roi_w, dtype=np.float32), np.arange(roi_h, dtype=np.float32))
        map_x = np.clip(base_x - np.clip(grad_x, -3, 3) * refract, 0, roi_w - 1)
        map_y = np.clip(base_y - np.clip(grad_y, -3, 3) * refract, 0, roi_h - 1)
        warped = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT).astype(np.float32)

        # Càng vào sâu trong khối slime thì màu càng đậm và đục hơn
        depth = np.clip((field - 1.0) / 1.2, 0.0, 1.0)[:, :, None]
        alpha = 0.35 + 0.35 * depth
        gel = warped * (1.0 - alpha) + self.gel_color * alpha

        # Viền ngoài sáng hơn (ánh sáng viền của bề mặt cong)
        rim = np.clip(1.0 - np.abs(field - 1.08) / 0.18, 0.0, 1.0)[:, :, None]
        gel += rim * 70.0

        # Đốm sáng phản chiếu trên các quả cầu chính
        spec = np.zeros((roi_h, roi_w), dtype=np.float32)
        for hx, hy, hr in highlights:
            cx = int(hx - x1 - 0.32 * hr)
            cy = int(hy - y1 - 0.38 * hr)
            if 0 <= cx < roi_w and 0 <= cy < roi_h:
                cv2.circle(spec, (cx, cy), max(2, int(0.22 * hr)), 1.0, -1)
        if spec.any():
            spec = cv2.GaussianBlur(spec, (0, 0), max(2.0, 0.08 * max_radius))
            gel += (spec / max(spec.max(), 1e-6) * 150.0)[:, :, None]

        roi[mask] = np.clip(gel, 0, 255).astype(np.uint8)[mask]
