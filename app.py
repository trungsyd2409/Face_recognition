"""
app.py
Backend web (Flask) cho bản chạy trên trình duyệt của main_webcam.py.

Kiến trúc: trình duyệt tự lấy webcam (getUserMedia), chụp từng frame gửi lên
đây qua HTTP POST (ảnh JPEG). Backend chạy ĐÚNG pipeline xử lý như
main_webcam.py (nhận diện khuôn mặt + cảm xúc, nhận diện tay + cử chỉ, tứ giác
2 tay + hiệu ứng xoay vòng khi chụm ngón), vẽ overlay lên frame, rồi trả ảnh
JPEG kết quả về cho trình duyệt hiển thị.

Vì đây là 1 backend dùng chung, trạng thái xoay vòng hiệu ứng / cache cảm xúc
được lưu theo từng `session_id` (trình duyệt tự sinh 1 lần khi mở trang, gửi
kèm mỗi request) - lưu tạm trong bộ nhớ (mất khi restart server). Cách này đủ
dùng cho demo/portfolio 1-vài người dùng cùng lúc, KHÔNG phù hợp production
quy mô lớn (cần Redis/DB nếu muốn scale nhiều instance).

Chạy thử ở máy (không cần Docker):
    pip install -r requirements-web.txt
    python app.py
Sau đó mở trình duyệt tại: http://localhost:8000
"""

import threading
import time
import uuid

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

from utils import (
    detect_faces, detect_emotion,
    detect_hands, draw_hand_landmarks,
    get_two_hand_quad_points, draw_quad_outline,
    is_pinching, apply_quad_color_effect, COLOR_EFFECT_CYCLE,
)

app = Flask(__name__, static_folder="static", static_url_path="")

# Chỉ chạy lại DeepFace (rất chậm so với phần còn lại) mỗi N request cho MỖI
# session, các request còn lại dùng lại kết quả gần nhất - giống cơ chế
# RECOGNIZE_EVERY_N_FRAMES trong main_webcam.py.
RECOGNIZE_EVERY_N_REQUESTS = 5

# Sau chừng này giây không thấy request nào, coi như session đã đóng tab và dọn
# trạng thái để không rò rỉ bộ nhớ theo thời gian.
SESSION_MAX_AGE_SECONDS = 600

_sessions = {}

# Các model dùng chung (YuNet, DeepFace, MediaPipe HandLandmarker) trong utils.py
# không đảm bảo an toàn khi gọi đồng thời từ nhiều thread - lock này đảm bảo
# tại 1 thời điểm chỉ có 1 request thực sự chạy qua pipeline xử lý ảnh, dù
# gunicorn có thể dùng nhiều thread để nhận request cùng lúc.
_processing_lock = threading.Lock()

EMOTION_COLORS = {
    "happy": (0, 200, 0),
    "sad": (255, 120, 0),
    "angry": (0, 0, 255),
    "surprise": (0, 220, 255),
    "fear": (200, 0, 200),
    "disgust": (0, 128, 100),
    "neutral": (180, 180, 180),
    "Unknown": (255, 255, 255),
}

EMOTION_LABELS_VI = {
    "happy": "Vui",
    "sad": "Buon",
    "angry": "Gian",
    "surprise": "Ngac nhien",
    "fear": "So hai",
    "disgust": "Ghe tom",
    "neutral": "Binh thuong",
    "Unknown": "Khong xac dinh",
}

HAND_TEXT_COLOR = (255, 255, 255)


def _get_session_state(session_id):
    if session_id not in _sessions:
        _sessions[session_id] = {
            "request_count": 0,
            "last_emotion_results": {},
            "current_effect_index": 0,
            "was_pinching": {"Left": False, "Right": False},
        }
    state = _sessions[session_id]
    state["last_seen"] = time.time()
    return state


def _cleanup_old_sessions():
    now = time.time()
    stale_ids = [
        sid for sid, s in _sessions.items()
        if now - s.get("last_seen", now) > SESSION_MAX_AGE_SECONDS
    ]
    for sid in stale_ids:
        del _sessions[sid]


def _process_frame(frame, state):
    """Chạy toàn bộ pipeline xử lý lên `frame` (sửa trực tiếp), dùng `state`
    (dict) để lưu trạng thái xuyên suốt các lần gọi của CÙNG 1 session."""
    state["request_count"] += 1

    # ---- Khuôn mặt + cảm xúc ----
    faces = detect_faces(frame)
    for (x, y, w, h) in faces:
        pos_key = (x // 50, y // 50)

        if state["request_count"] % RECOGNIZE_EVERY_N_REQUESTS == 0:
            face_img = frame[y:y + h, x:x + w]
            emotion, confidence = detect_emotion(face_img)
            state["last_emotion_results"][pos_key] = (emotion, confidence)
        else:
            emotion, confidence = state["last_emotion_results"].get(pos_key, ("Detecting...", 0.0))

        color = EMOTION_COLORS.get(emotion, (255, 255, 255))
        label = EMOTION_LABELS_VI.get(emotion, emotion)
        if emotion in EMOTION_COLORS and emotion != "Unknown":
            text = f"{label} ({confidence:.0f}%)"
        else:
            text = label

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ---- Bàn tay + cử chỉ (mode "image" - xử lý từng request độc lập) ----
    hands_info = detect_hands(frame, running_mode="image")
    frame_h, frame_w = frame.shape[:2]

    left_hand = next((h for h in hands_info if h["handedness"] == "Left"), None)
    right_hand = next((h for h in hands_info if h["handedness"] == "Right"), None)

    # Cử chỉ "chụm ngón" -> xoay vòng hiệu ứng (cạnh lên, không đổi liên tục)
    for hand in hands_info:
        handedness = hand["handedness"]
        pinching_now = is_pinching(hand["landmarks"])
        if pinching_now and not state["was_pinching"].get(handedness, False):
            state["current_effect_index"] = (state["current_effect_index"] + 1) % len(COLOR_EFFECT_CYCLE)
        state["was_pinching"][handedness] = pinching_now

    if left_hand and right_hand:
        quad_points = get_two_hand_quad_points(
            left_hand["landmarks"], right_hand["landmarks"], frame_w, frame_h,
        )
        current_effect = COLOR_EFFECT_CYCLE[state["current_effect_index"]]
        apply_quad_color_effect(frame, quad_points, current_effect)
        draw_quad_outline(frame, quad_points)

        quad_top_point = min(quad_points, key=lambda p: p[1])
        cv2.putText(
            frame, f"Hieu ung: {current_effect}",
            (quad_top_point[0], max(quad_top_point[1] - 15, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

    for hand in hands_info:
        draw_hand_landmarks(frame, hand["landmarks"])
        wrist = hand["landmarks"][0]
        text_x = int(wrist.x * frame_w)
        text_y = max(int(wrist.y * frame_h) - 20, 20)
        cv2.putText(
            frame, f'{hand["handedness"]}: {hand["gesture"]}', (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, HAND_TEXT_COLOR, 2,
        )

    return frame


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/session", methods=["POST"])
def create_session():
    """Trình duyệt gọi 1 lần khi mở trang để lấy 1 session_id mới."""
    session_id = uuid.uuid4().hex
    _get_session_state(session_id)
    return jsonify({"session_id": session_id})


@app.route("/api/process", methods=["POST"])
def process_frame_endpoint():
    _cleanup_old_sessions()

    session_id = request.form.get("session_id")
    if not session_id:
        return jsonify({"error": "missing session_id"}), 400

    file = request.files.get("frame")
    if file is None:
        return jsonify({"error": "missing frame"}), 400

    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "invalid image"}), 400

    state = _get_session_state(session_id)

    try:
        with _processing_lock:
            frame = _process_frame(frame, state)
    except FileNotFoundError as e:
        # Model chưa được tải (thiếu bước download_model.py lúc build)
        return jsonify({"error": str(e)}), 500

    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return jsonify({"error": "encode failed"}), 500

    return app.response_class(buffer.tobytes(), mimetype="image/jpeg")


@app.route("/healthz")
def healthz():
    """Endpoint kiểm tra sức khoẻ server, Render dùng để biết service còn sống."""
    return jsonify({"status": "ok", "active_sessions": len(_sessions)})


if __name__ == "__main__":
    # Chạy thử ở máy. Khi deploy thật (Render), Dockerfile dùng gunicorn thay
    # vì Flask dev server không phù hợp chạy production.
    app.run(host="0.0.0.0", port=8000, debug=False)
