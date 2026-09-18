"""
particles.py
Hệ hạt (particle system) bắn ra từ đầu ngón tay: vệt lửa, tuyết, khói độc...

Cách hoạt động:

1. SINH HẠT - mỗi frame, tại mỗi đầu ngón ĐANG GIƠ sinh ra vài hạt mới (ngón
   gập lại thì không bắn hạt). RIÊNG NGÓN CÁI không bao giờ bắn hạt - nó được
   để trống hẳn. Khi 1 bàn tay mở "vòng ma pháp lớn" (trường
   `palm_glow` của bàn tay đó chạy từ 0 lên 1), hạt chuyển từ bắn ở đầu ngón
   sang RẢI ĐỀU khắp bàn tay với độ sáng thấp, làm cả bàn tay ửng sáng mà không
   lấn át màu của vòng ma pháp. Vận tốc ban
   đầu của hạt = vận tốc của chính đầu ngón đó (tính bằng hiệu vị trí so với
   frame trước) cộng thêm chút ngẫu nhiên. Vung tay càng nhanh thì hạt sinh ra
   càng nhiều và bắn càng mạnh - đúng như cảm giác vẩy lửa.

2. VẬT LÝ - mỗi frame cập nhật toàn bộ hạt bằng vài phép toán numpy trên MẢNG
   (không dùng vòng lặp Python), nên hàng nghìn hạt vẫn chạy thời gian thực:
        vận tốc += trọng lực (mặc định ÂM = bốc lên như lửa)
        vận tốc *= hệ số cản (drag)
        vị trí  += vận tốc
        tuổi thọ -= 1
   Hạt hết tuổi thọ hoặc bay ra ngoài khung hình thì bị loại.

3. HÚT VÀO KHI CHỤM NGÓN - nếu 1 bàn tay đang chụm ngón cái + trỏ, các hạt bị
   kéo về điểm chụm theo lực kiểu hấp dẫn (tỉ lệ nghịch với bình phương khoảng
   cách), kèm 1 thành phần vuông góc nên chúng xoáy tròn quanh điểm chụm thay
   vì rơi thẳng vào.

4. VẼ - không vẽ từng hạt bằng cv2.circle (chậm và trông cứng). Thay vào đó cộng
   dồn màu của các hạt vào 1 ảnh đệm độ phân giải thấp bằng np.add.at, rồi làm
   mờ (Gaussian blur) 2 lần với 2 bán kính khác nhau: bán kính nhỏ cho lõi sáng,
   bán kính lớn cho quầng sáng. Ảnh đệm còn được giữ lại một phần qua các frame
   (nhân với hệ số < 1) nên hạt kéo theo VỆT SÁNG mờ dần phía sau.

   Ảnh đệm được CỘNG vào khung hình (additive blending) - đúng cách ánh sáng
   thật hoạt động, nên vệt lửa trông phát sáng chứ không như dán hình lên.
"""

import cv2
import numpy as np

# Đầu ngón bắn hạt: trỏ, giữa, áp út, út - KHÔNG có ngón cái.
# Mỗi phần tử là (chỉ số ngón trong fingers_up, chỉ số landmark của đầu ngón).
EMITTER_FINGERS = [(1, 8), (2, 12), (3, 16), (4, 20)]

THUMB_TIP_ID = 4
INDEX_TIP_ID = 8

# Màu riêng cho lớp hạt toả ra từ chùm nối 2 tay (BGR) - cùng tông đỏ với vòng
LINK_COLOR = (40, 40, 255)

# Landmark thuộc ngón cái (trừ điểm gốc 1, vốn nằm sát lòng bàn tay) - loại
# khỏi lớp hạt rải quanh bàn tay để ngón cái hoàn toàn "sạch"
THUMB_LANDMARKS = (2, 3, 4)

# Bảng màu: danh sách các mốc màu (BGR) đi từ lúc hạt vừa sinh ra (nóng nhất,
# phần tử đầu) tới lúc sắp tắt (nguội nhất, phần tử cuối).
COLOR_THEMES = {
    "lua":     [(210, 252, 255), (60, 200, 255), (10, 95, 255), (0, 15, 90)],
    "bang":    [(255, 250, 240), (255, 210, 120), (230, 130, 40), (90, 40, 10)],
    "doc":     [(200, 255, 220), (120, 255, 150), (40, 200, 60), (10, 70, 20)],
    "tim":     [(255, 240, 255), (255, 150, 220), (200, 60, 150), (80, 10, 60)],
}
THEME_NAMES = list(COLOR_THEMES)


def _theme_lut(theme_name, steps=64):
    """
    Dựng bảng tra màu (lookup table) bằng cách nội suy tuyến tính giữa các mốc
    màu của 1 bảng màu. Tra bảng nhanh hơn nhiều so với nội suy lại cho từng hạt.
    """
    stops = np.array(COLOR_THEMES[theme_name], dtype=np.float32)
    positions = np.linspace(0.0, 1.0, len(stops))
    query = np.linspace(0.0, 1.0, steps)
    lut = np.stack([np.interp(query, positions, stops[:, channel])
                    for channel in range(3)], axis=1)
    return lut.astype(np.float32)


class ParticleSystem:
    """
    Hệ hạt bắn ra từ các đầu ngón tay.

    Tham số:
        max_particles : số hạt tối đa (chặn để không tụt FPS)
        spawn_per_tip : số hạt sinh mỗi frame ở mỗi đầu ngón khi tay đứng yên
        speed_spawn   : vung tay nhanh thì sinh thêm tối đa bao nhiêu hạt nữa
        gravity       : gia tốc theo trục dọc (pixel/frame^2). Giá trị ÂM nghĩa
                        là kéo hạt bay LÊN (trục y của ảnh hướng xuống dưới),
                        cho hạt bốc lên như lửa/khói thay vì rơi xuống
        speed_scale   : hệ số nhân vận tốc ban đầu của hạt
        drag          : hệ số cản không khí mỗi frame
        life_range    : tuổi thọ hạt (số frame)
        pull_strength : lực hút khi chụm ngón (pixel/frame^2 ở sát tâm)
        trail         : phần vệt sáng giữ lại qua mỗi frame (0 = không có vệt)
        brightness    : độ chói tổng thể của hạt (giảm xuống nếu thấy loá)
        palm_glow_rate: số hạt rải quanh bàn tay mỗi frame khi vòng lớn mở
        palm_glow_dim : độ sáng của lớp hạt rải đó (để không át màu vòng)
        render_scale  : độ phân giải ảnh đệm so với khung hình
    """

    def __init__(self, max_particles=2500, spawn_per_tip=2, speed_spawn=8,
                 gravity=-0.45, drag=0.97, life_range=(18, 42), speed_scale=1.5,
                 pull_strength=110.0, trail=0.68, render_scale=0.5,
                 theme="lua", brightness=0.75,
                 palm_glow_rate=14, palm_glow_dim=0.42):
        self.max_particles = max_particles
        self.spawn_per_tip = spawn_per_tip
        self.speed_spawn = speed_spawn
        self.gravity = gravity
        self.speed_scale = speed_scale
        self.drag = drag
        self.life_range = life_range
        self.pull_strength = pull_strength
        self.trail = trail
        self.render_scale = render_scale
        self.brightness = brightness
        self.palm_glow_rate = palm_glow_rate
        self.palm_glow_dim = palm_glow_dim

        self.theme = theme
        self._lut = _theme_lut(theme)
        self.gravity_on = True
        self.pulling = False     # có bàn tay nào đang chụm ngón hút hạt không

        # Mỗi thuộc tính của hạt là 1 MẢNG numpy, không phải list các object -
        # nhờ vậy cập nhật toàn bộ hạt chỉ bằng vài phép tính trên mảng.
        self.pos = np.zeros((0, 2), dtype=np.float32)
        self.vel = np.zeros((0, 2), dtype=np.float32)
        self.life = np.zeros(0, dtype=np.float32)
        self.life_max = np.zeros(0, dtype=np.float32)
        self.bright = np.zeros(0, dtype=np.float32)
        # Hệ số trọng lực riêng của từng hạt: hạt bắn ở đầu ngón chịu trọng lực
        # đầy đủ (1.0), còn hạt rải quanh bàn tay gần như không (≈0.1) để chúng
        # bám vào bàn tay thay vì bay vọt đi mất
        self.gscale = np.zeros(0, dtype=np.float32)
        # Hạt nào thuộc chùm nối 2 tay: tô màu đỏ cố định thay vì lấy màu theo
        # bảng màu của hệ hạt
        self.is_link = np.zeros(0, dtype=bool)

        self._prev_tips = {}      # (tay, ngón) -> vị trí frame trước
        self._glow = None         # ảnh đệm phát sáng, giữ lại giữa các frame

    # ------------------------------------------------------------------ public

    def update(self, hands_info, width, height):
        """Sinh hạt mới từ các đầu ngón + cập nhật vật lý cho toàn bộ hạt."""
        self._spawn_from_hands(hands_info, width, height)
        self._apply_pull(hands_info, width, height)
        self._step_physics(width, height)

    def draw(self, frame, extra_glow=None):
        """
        Vẽ toàn bộ hạt lên frame bằng cách cộng thêm ánh sáng (additive).

        `extra_glow` (tuỳ chọn) là ảnh đệm phát sáng nhỏ của hiệu ứng khác, cùng
        cỡ với ảnh đệm của hệ hạt - gộp vào đây để chỉ phải phóng to 1 lần.
        """
        height, width = frame.shape[:2]
        scale = self.render_scale
        small_h, small_w = int(height * scale), int(width * scale)

        if self._glow is None or self._glow.shape[:2] != (small_h, small_w):
            self._glow = np.zeros((small_h, small_w, 3), dtype=np.float32)

        # Vệt sáng: giữ lại 1 phần ánh sáng của frame trước rồi mới cộng hạt mới
        self._glow *= self.trail

        if len(self.pos):
            x = np.clip((self.pos[:, 0] * scale).astype(np.int32), 0, small_w - 1)
            y = np.clip((self.pos[:, 1] * scale).astype(np.int32), 0, small_h - 1)

            # Màu theo "tuổi": hạt mới sinh nóng sáng, hạt sắp tắt nguội và tối
            ratio = np.clip(1.0 - self.life / np.maximum(self.life_max, 1), 0, 0.999)
            colors = self._lut[(ratio * (len(self._lut) - 1)).astype(np.int32)]
            if self.is_link.any():
                # Hạt toả ra từ chùm nối 2 tay giữ nguyên tông đỏ, không đổi
                # theo bảng màu của hệ hạt
                colors = colors.copy()
                colors[self.is_link] = LINK_COLOR
            intensity = ((self.life / np.maximum(self.life_max, 1))[:, None]
                         * self.bright[:, None] * self.brightness)

            np.add.at(self._glow, (y, x), colors * intensity)

        # Chặn trần độ sáng: khi hàng nghìn hạt chồng lên nhau (lúc gom hạt
        # thành quả cầu) tổng cộng dồn sẽ vọt lên rất cao và cháy trắng xoá,
        # mất hết màu. Cắt trần giữ được màu mà vẫn rất sáng.
        np.minimum(self._glow, 105.0, out=self._glow)

        # Lõi sáng (mờ ít) + quầng sáng (mờ nhiều) -> ra chất lửa phát quang
        core = cv2.GaussianBlur(self._glow, (0, 0), 1.1)
        halo = cv2.GaussianBlur(self._glow, (0, 0), 5.0)
        glow = core * 1.0 + halo * 0.6

        if extra_glow is not None and extra_glow.shape == glow.shape:
            glow = glow + extra_glow

        # Chuyển sang uint8 NGAY ở độ phân giải nhỏ rồi mới phóng to và cộng
        # bằng cv2.add (tự chặn trần 255). Làm phép cộng ở dạng float trên ảnh
        # gốc tốn hơn nhiều lần mà kết quả không khác, vì phần vượt 255 đằng
        # nào cũng bị cắt.
        glow8 = np.clip(glow, 0, 255).astype(np.uint8)
        glow8 = cv2.resize(glow8, (width, height), interpolation=cv2.INTER_LINEAR)
        cv2.add(frame, glow8, dst=frame)

    def set_theme(self, name):
        self.theme = name
        self._lut = _theme_lut(name)

    def next_theme(self):
        index = (THEME_NAMES.index(self.theme) + 1) % len(THEME_NAMES)
        self.set_theme(THEME_NAMES[index])
        return self.theme

    def clear(self):
        self.pos = np.zeros((0, 2), dtype=np.float32)
        self.vel = np.zeros((0, 2), dtype=np.float32)
        self.life = np.zeros(0, dtype=np.float32)
        self.life_max = np.zeros(0, dtype=np.float32)
        self.bright = np.zeros(0, dtype=np.float32)
        self.gscale = np.zeros(0, dtype=np.float32)
        self.is_link = np.zeros(0, dtype=bool)
        self._prev_tips.clear()
        if self._glow is not None:
            self._glow[:] = 0

    @property
    def count(self):
        return len(self.pos)

    # --------------------------------------------------------------- internals

    def _spawn_from_hands(self, hands_info, width, height):
        """Sinh hạt tại từng đầu ngón, mạnh yếu tuỳ tốc độ vung tay."""
        new_pos, new_vel, new_life, new_bright, new_gscale = [], [], [], [], []
        seen = set()

        for hand in hands_info:
            handedness = hand["handedness"]
            fingers_up = hand.get("fingers_up", [1] * 5)
            # Mức độ bàn tay đang mở vòng ma pháp lớn (0..1), do chương trình
            # chính gán vào trước khi gọi update()
            palm_glow = float(hand.get("palm_glow", 0.0))

            if palm_glow > 0.02:
                spawned = self._spawn_palm_glow(hand, width, height, palm_glow)
                if spawned:
                    new_pos.append(spawned[0]); new_vel.append(spawned[1])
                    new_life.append(spawned[2]); new_bright.append(spawned[3])
                    new_gscale.append(np.full(len(spawned[0]), 0.10, dtype=np.float32))

            for finger_index, tip_id in EMITTER_FINGERS:
                if not fingers_up[finger_index]:
                    # Ngón đang gập: không bắn hạt, và quên luôn vị trí cũ để
                    # lúc giơ lại không bị tính ra 1 vận tốc giả rất lớn
                    self._prev_tips.pop((handedness, tip_id), None)
                    continue

                key = (handedness, tip_id)
                seen.add(key)

                lm = hand["landmarks"][tip_id]
                position = np.array([lm.x * width, lm.y * height], dtype=np.float32)

                previous = self._prev_tips.get(key)
                self._prev_tips[key] = position
                if previous is None:
                    continue

                # Vận tốc của đầu ngón = quãng đường đi được trong 1 frame
                tip_velocity = position - previous
                speed = float(np.linalg.norm(tip_velocity))

                # Vung càng nhanh, sinh càng nhiều hạt. Bàn tay đang mở vòng
                # lớn thì bớt bắn ở đầu ngón, nhường chỗ cho lớp hạt rải đều.
                count = self.spawn_per_tip + int(min(speed / 6.0, 1.0) * self.speed_spawn)
                count = int(round(count * (1.0 - 0.75 * palm_glow)))
                if count <= 0:
                    continue

                spread = (1.2 + speed * 0.18) * self.speed_scale
                offsets = np.random.normal(0.0, 3.0, size=(count, 2)).astype(np.float32)
                jitter = np.random.normal(0.0, spread, size=(count, 2)).astype(np.float32)

                new_pos.append(position + offsets)
                # Hạt mang theo phần lớn vận tốc của ngón tay -> bắn theo hướng vung
                new_vel.append(tip_velocity * (0.55 * self.speed_scale) + jitter)
                new_life.append(np.random.uniform(self.life_range[0], self.life_range[1], count))
                new_bright.append(np.random.uniform(0.55, 1.0, count))
                new_gscale.append(np.ones(count, dtype=np.float32))

        for key in list(self._prev_tips):
            if key not in seen:
                self._prev_tips.pop(key, None)

        if not new_pos:
            return

        self._append(np.concatenate(new_pos),
                     np.concatenate(new_vel),
                     np.concatenate(new_life),
                     np.concatenate(new_bright),
                     np.concatenate(new_gscale),
                     np.zeros(len(np.concatenate(new_life)), dtype=bool))

    def emit_link(self, point_a, point_b, amount=1.0, rate=16, speed=3.2):
        """
        Bắn hạt TOẢ RA hai bên từ chùm sáng nối 2 bàn tay.

        Mỗi frame chọn ngẫu nhiên vài điểm trên đoạn thẳng nối 2 tâm, rồi bắn
        hạt theo phương VUÔNG GÓC với đoạn đó (ngẫu nhiên sang trái hoặc phải),
        nên hạt trông như bị chùm năng lượng hất ra hai bên.

        Nhóm hạt này được đánh dấu `is_link` để lúc vẽ dùng màu đỏ cố định, thay
        vì màu theo bảng màu (lửa/băng/độc/tím) của các hạt thường.
        """
        p1 = np.asarray(point_a, dtype=np.float32)
        p2 = np.asarray(point_b, dtype=np.float32)
        direction = p2 - p1
        length = float(np.linalg.norm(direction))
        count = int(round(rate * amount))
        if length < 10 or count <= 0:
            return

        direction = direction / length
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)

        t = np.random.random(count).astype(np.float32)[:, None]
        positions = p1 + (p2 - p1) * t
        positions += normal * np.random.normal(0.0, 3.0, (count, 1)).astype(np.float32)

        side = np.where(np.random.random((count, 1)) < 0.5, -1.0, 1.0).astype(np.float32)
        strength = np.random.uniform(0.4, 1.0, (count, 1)).astype(np.float32) * speed
        velocities = (normal * side * strength
                      + direction * np.random.normal(0.0, 0.8, (count, 1)).astype(np.float32))

        life = np.random.uniform(self.life_range[0] * 0.7,
                                 self.life_range[1] * 0.9, count).astype(np.float32)
        bright = np.random.uniform(0.7, 1.2, count).astype(np.float32)

        self._append(positions, velocities, life, bright,
                     np.full(count, 0.25, dtype=np.float32),
                     np.ones(count, dtype=bool))

    def _append(self, pos, vel, life, bright, gscale, is_link):
        """Nối thêm 1 nhóm hạt mới vào các mảng, cắt bớt nếu vượt quá giới hạn."""
        self.pos = np.concatenate([self.pos, pos.astype(np.float32)])
        self.vel = np.concatenate([self.vel, vel.astype(np.float32)])
        self.life = np.concatenate([self.life, life.astype(np.float32)])
        self.life_max = np.concatenate([self.life_max, life.astype(np.float32)])
        self.bright = np.concatenate([self.bright, bright.astype(np.float32)])
        self.gscale = np.concatenate([self.gscale, gscale.astype(np.float32)])
        self.is_link = np.concatenate([self.is_link, is_link])

        if len(self.pos) > self.max_particles:
            keep = slice(len(self.pos) - self.max_particles, None)
            self.pos, self.vel = self.pos[keep], self.vel[keep]
            self.life, self.life_max = self.life[keep], self.life_max[keep]
            self.bright, self.gscale = self.bright[keep], self.gscale[keep]
            self.is_link = self.is_link[keep]

    def _spawn_palm_glow(self, hand, width, height, amount):
        """
        Rải hạt đều khắp bàn tay (dùng khi vòng ma pháp lớn đang mở).

        Cách rải: bốc ngẫu nhiên 1 landmark của bàn tay (bỏ các landmark thuộc
        ngón cái) rồi lệch đi 1 chút - như vậy hạt bám đúng hình bàn tay (cả lòng lẫn các ngón) thay vì
        rải trong 1 hình chữ nhật bao quanh, vốn sẽ lòi ra ngoài viền tay.

        Hạt ở đây bay chậm, gần như không chịu trọng lực (nếu không chúng sẽ bốc
        lên và rời khỏi bàn tay), sống lâu hơn và mờ hơn hẳn hạt bắn ở đầu ngón,
        để cả bàn tay chỉ ửng sáng chứ không át mất màu của vòng ma pháp.
        """
        landmarks = hand["landmarks"]
        count = int(round(self.palm_glow_rate * amount))
        if count <= 0:
            return None

        points = np.array([[lm.x * width, lm.y * height]
                           for i, lm in enumerate(landmarks)
                           if i not in THUMB_LANDMARKS], dtype=np.float32)
        hand_size = float(np.linalg.norm(
            np.array([landmarks[9].x * width, landmarks[9].y * height]) -
            np.array([landmarks[0].x * width, landmarks[0].y * height])))
        if hand_size < 5:
            return None

        picks = points[np.random.randint(0, len(points), count)]
        jitter = np.random.normal(0.0, hand_size * 0.16, size=(count, 2)).astype(np.float32)
        velocity = np.random.normal(0.0, 0.5, size=(count, 2)).astype(np.float32)
        life = np.random.uniform(self.life_range[0] * 0.8,
                                 self.life_range[1] * 1.1, count).astype(np.float32)
        bright = (np.random.uniform(0.35, 0.75, count).astype(np.float32)
                  * self.palm_glow_dim * amount)
        return picks + jitter, velocity, life, bright

    def _apply_pull(self, hands_info, width, height):
        """Chụm ngón cái + trỏ -> hút các hạt về điểm chụm và cho xoáy quanh nó."""
        self.pulling = False
        if not len(self.pos):
            return

        for hand in hands_info:
            landmarks = hand["landmarks"]
            thumb = landmarks[THUMB_TIP_ID]
            index = landmarks[INDEX_TIP_ID]
            wrist = landmarks[0]
            middle_mcp = landmarks[9]

            hand_size = np.hypot(wrist.x - middle_mcp.x, wrist.y - middle_mcp.y)
            pinch_gap = np.hypot(thumb.x - index.x, thumb.y - index.y)
            if hand_size <= 0 or pinch_gap / hand_size > 0.4:
                continue        # tay này không chụm ngón

            center = np.array([(thumb.x + index.x) / 2 * width,
                               (thumb.y + index.y) / 2 * height], dtype=np.float32)

            delta = center - self.pos
            dist = np.sqrt(np.maximum((delta ** 2).sum(axis=1), 1.0))
            # Chặn khoảng cách tối thiểu, nếu không hạt sát tâm sẽ nhận lực vô
            # hạn và bị bắn văng đi thay vì bị gom lại
            dist_clamped = np.maximum(dist, 22.0)
            direction = delta / dist[:, None]

            # Lực hút giảm theo 1/r (dịu hơn 1/r^2), và có trần
            accel = np.minimum(self.pull_strength / dist_clamped, 6.0)
            # Thành phần vuông góc -> hạt xoáy tròn quanh điểm chụm
            tangent = np.stack([-direction[:, 1], direction[:, 0]], axis=1)

            self.vel += direction * accel[:, None] + tangent * (accel * 0.45)[:, None]

            # Hạt ở gần tâm bị hãm lại (nếu không chúng chỉ lướt qua rồi bay đi)
            # và được giữ sống lâu hơn, tạo thành 1 quả cầu hạt xoáy trong tay
            near = dist < 95.0
            self.vel[near] *= 0.88
            self.life[near] += 0.7
            self.pulling = True

    def _step_physics(self, width, height):
        """Cập nhật vận tốc, vị trí, tuổi thọ; loại các hạt đã tắt hoặc bay ra ngoài."""
        if not len(self.pos):
            return

        if self.gravity_on:
            # Đang gom hạt thì giảm trọng lực, nếu không quả cầu hạt bay mất
            self.vel[:, 1] += (self.gravity * (0.15 if self.pulling else 1.0)) * self.gscale
        self.vel *= self.drag
        self.pos += self.vel
        self.life -= 1.0

        alive = ((self.life > 0) &
                 (self.pos[:, 0] > -40) & (self.pos[:, 0] < width + 40) &
                 (self.pos[:, 1] > -40) & (self.pos[:, 1] < height + 60))
        if not alive.all():
            self.pos, self.vel = self.pos[alive], self.vel[alive]
            self.life, self.life_max = self.life[alive], self.life_max[alive]
            self.bright, self.gscale = self.bright[alive], self.gscale[alive]
            self.is_link = self.is_link[alive]
