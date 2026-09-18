"""
hand_control.py
Chuyển cử chỉ bàn tay (kết quả của utils.detect_hands) thành các phép biến đổi
cho model 3D: xoay, phóng to/thu nhỏ, di chuyển, đổi model.

Bảng điều khiển:

    Cử chỉ                              Tác dụng
    ---------------------------------   --------------------------------------
    Chụm ngón cái + trỏ rồi kéo tay     Xoay model (kéo ngang = xoay quanh trục
                                        đứng, kéo dọc = lật lên/xuống)
    2 tay, đưa xa nhau / lại gần nhau   Phóng to / thu nhỏ
    Nắm tay rồi di chuyển               Kéo model đi trong khung hình
    Giơ N ngón (1 tay, giữ yên 1 nhịp)  Đổi sang model thứ N trong danh sách

Nguyên tắc chung: mọi cử chỉ đều điều khiển theo KIỂU TƯƠNG ĐỐI (so với frame
trước) chứ không gán giá trị tuyệt đối - nhờ vậy model không bị "nhảy" mỗi khi
MediaPipe nhận diện lệch một chút.

Giá trị cuối cùng còn được làm mượt (lọc thông thấp) trước khi đưa cho renderer,
nên chuyển động trông trôi chảy chứ không giật theo từng frame.
"""

import numpy as np

from utils import is_pinching

PALM_ID = 9          # gốc ngón giữa - dùng làm "tâm bàn tay"
INDEX_TIP_ID = 8


def _palm_center(landmarks, width, height):
    lm = landmarks[PALM_ID]
    return np.array([lm.x * width, lm.y * height], dtype=np.float32)


def _index_tip(landmarks, width, height):
    lm = landmarks[INDEX_TIP_ID]
    return np.array([lm.x * width, lm.y * height], dtype=np.float32)


class ModelController:
    """
    Giữ trạng thái hiển thị của model (góc xoay, cỡ, vị trí, model nào đang
    chọn) và cập nhật trạng thái đó theo cử chỉ tay mỗi frame.

    Tham số:
        rotate_speed  : kéo 1 pixel thì xoay bao nhiêu radian
        zoom_limits   : giới hạn nhỏ nhất / lớn nhất của hệ số phóng
        smooth        : độ mượt (0-1), càng nhỏ càng mượt nhưng càng "trễ" tay
        switch_frames : giơ N ngón phải giữ yên bao nhiêu frame mới đổi model
        idle_spin     : tốc độ tự xoay khi không có tay nào điều khiển
    """

    def __init__(self, rotate_speed=0.012, zoom_limits=(0.25, 4.0),
                 smooth=0.35, switch_frames=12, idle_spin=0.006):
        self.rotate_speed = rotate_speed
        self.zoom_limits = zoom_limits
        self.smooth = smooth
        self.switch_frames = switch_frames
        self.idle_spin = idle_spin

        # Giá trị đích (cử chỉ tác động vào đây)
        self.target_yaw = 0.6
        self.target_pitch = 0.3
        self.target_scale = 1.0
        self.target_offset = np.zeros(2, dtype=np.float32)

        # Giá trị đang hiển thị (chạy đuổi theo giá trị đích cho mượt)
        self.yaw = self.target_yaw
        self.pitch = self.target_pitch
        self.scale = self.target_scale
        self.offset = self.target_offset.copy()

        self.model_index = 0
        self.auto_spin = True
        self.action = "cho cu chi"     # mô tả cử chỉ đang nhận, để hiện lên màn hình

        self._prev_pinch = {}          # tay -> vị trí đầu ngón trỏ frame trước
        self._prev_fist = {}           # tay -> tâm bàn tay frame trước
        self._prev_two_hand_dist = None
        self._finger_count = 0
        self._finger_count_frames = 0

    # ------------------------------------------------------------------ input

    def update(self, hands_info, width, height, model_count=1):
        """Đọc cử chỉ của frame hiện tại và cập nhật trạng thái model."""
        rotating = self._handle_rotate(hands_info, width, height)
        moving = self._handle_move(hands_info, width, height)
        zooming = self._handle_zoom(hands_info, width, height, rotating or moving)
        switched = self._handle_model_switch(hands_info, model_count,
                                             rotating or moving or zooming)

        if rotating:
            self.action = "xoay"
        elif moving:
            self.action = "di chuyen"
        elif zooming:
            self.action = "phong to / thu nho"
        elif switched:
            self.action = f"doi model #{self.model_index + 1}"
        else:
            self.action = "cho cu chi"
            if self.auto_spin and not hands_info:
                self.target_yaw += self.idle_spin

        self._smooth_towards_target()
        return self.action

    def _handle_rotate(self, hands_info, width, height):
        """Chụm ngón cái + trỏ rồi kéo tay -> xoay model."""
        active = False
        seen = set()

        for hand in hands_info:
            handedness = hand["handedness"]
            seen.add(handedness)
            if not is_pinching(hand["landmarks"]):
                continue

            position = _index_tip(hand["landmarks"], width, height)
            previous = self._prev_pinch.get(handedness)
            if previous is not None:
                delta = position - previous
                self.target_yaw += float(delta[0]) * self.rotate_speed
                self.target_pitch += float(delta[1]) * self.rotate_speed
                # Chặn không cho lật quá đỉnh đầu / quá đáy (tránh model lộn ngược)
                self.target_pitch = float(np.clip(self.target_pitch, -1.4, 1.4))
                active = True
            self._prev_pinch[handedness] = position

        # Tay nào thôi chụm (hoặc biến mất) thì xoá mốc cũ
        for handedness in list(self._prev_pinch):
            if handedness not in seen or not any(
                h["handedness"] == handedness and is_pinching(h["landmarks"])
                for h in hands_info
            ):
                self._prev_pinch.pop(handedness, None)

        return active

    def _handle_move(self, hands_info, width, height):
        """Nắm tay rồi di chuyển -> kéo model đi theo."""
        active = False
        fists = set()

        for hand in hands_info:
            handedness = hand["handedness"]
            if sum(hand["fingers_up"]) > 0:      # không phải nắm tay
                continue
            fists.add(handedness)

            position = _palm_center(hand["landmarks"], width, height)
            previous = self._prev_fist.get(handedness)
            if previous is not None:
                self.target_offset += position - previous
                active = True
            self._prev_fist[handedness] = position

        for handedness in list(self._prev_fist):
            if handedness not in fists:
                self._prev_fist.pop(handedness, None)

        return active

    def _handle_zoom(self, hands_info, width, height, busy):
        """
        2 tay đưa xa nhau / lại gần nhau -> phóng to / thu nhỏ.
        Tỉ lệ thay đổi được tính theo TỈ SỐ khoảng cách so với frame trước, nên
        kéo tay ở gần hay xa camera đều cho cảm giác như nhau.
        """
        if busy or len(hands_info) < 2:
            self._prev_two_hand_dist = None
            return False

        a = _palm_center(hands_info[0]["landmarks"], width, height)
        b = _palm_center(hands_info[1]["landmarks"], width, height)
        distance = float(np.linalg.norm(a - b))
        if distance < 1e-3:
            return False

        active = False
        if self._prev_two_hand_dist:
            ratio = distance / self._prev_two_hand_dist
            ratio = float(np.clip(ratio, 0.85, 1.18))   # chặn nhảy cóc khi nhận diện lỗi
            if abs(ratio - 1.0) > 0.004:
                self.target_scale = float(np.clip(self.target_scale * ratio,
                                                  *self.zoom_limits))
                active = True
        self._prev_two_hand_dist = distance
        return active

    def _handle_model_switch(self, hands_info, model_count, busy):
        """Giơ N ngón bằng 1 tay và giữ yên -> chuyển sang model thứ N."""
        if busy or model_count <= 1 or len(hands_info) != 1:
            self._finger_count_frames = 0
            return False

        count = sum(hands_info[0]["fingers_up"])
        if count < 1 or count > 5:
            self._finger_count_frames = 0
            return False

        if count == self._finger_count:
            self._finger_count_frames += 1
        else:
            self._finger_count = count
            self._finger_count_frames = 1

        if self._finger_count_frames == self.switch_frames:
            new_index = (count - 1) % model_count
            if new_index != self.model_index:
                self.model_index = new_index
                self.reset_view(keep_model=True)
                return True
        return False

    def _smooth_towards_target(self):
        """Cho giá trị hiển thị chạy đuổi theo giá trị đích (lọc thông thấp)."""
        k = self.smooth
        self.yaw += (self.target_yaw - self.yaw) * k
        self.pitch += (self.target_pitch - self.pitch) * k
        self.scale += (self.target_scale - self.scale) * k
        self.offset += (self.target_offset - self.offset) * k

    # ----------------------------------------------------------------- khác

    def reset_view(self, keep_model=False):
        """Đưa góc nhìn về mặc định (phím 'r')."""
        self.target_yaw, self.target_pitch = 0.6, 0.3
        self.target_scale = 1.0
        self.target_offset = np.zeros(2, dtype=np.float32)
        self.yaw, self.pitch = self.target_yaw, self.target_pitch
        self.scale = self.target_scale
        self.offset = self.target_offset.copy()
        self._prev_pinch.clear()
        self._prev_fist.clear()
        self._prev_two_hand_dist = None
        if not keep_model:
            self.model_index = 0

    def next_model(self, model_count):
        if model_count > 0:
            self.model_index = (self.model_index + 1) % model_count
            self.reset_view(keep_model=True)
