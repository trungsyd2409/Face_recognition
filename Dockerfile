# Dockerfile - dùng để deploy backend web (app.py) lên Render (hoặc bất kỳ
# host nào hỗ trợ Docker). KHÔNG dùng để chạy bản desktop (main_webcam.py) -
# bản đó cần webcam + màn hình thật trên máy, không chạy trong container được.

FROM python:3.11-slim

# libgl1 + libglib2.0-0: vài hàm của OpenCV (kể cả bản headless) vẫn cần các
# thư viện hệ thống này lúc runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY utils.py app.py download_model.py ./
COPY static/ ./static/

# Tải sẵn model (YuNet + HandLandmarker) ngay lúc build image, để container
# chạy lên là dùng được luôn - tránh request đầu tiên bị chậm/timeout vì phải
# tải model giữa lúc đang phục vụ người dùng.
RUN python download_model.py

ENV PORT=8000
EXPOSE 8000

# gunicorn thay cho Flask dev server (không phù hợp chạy production).
# --workers 1: mỗi worker load riêng model DeepFace/MediaPipe (tốn RAM), free
#   tier server thường ít RAM nên chỉ chạy 1 worker; nếu cần phục vụ nhiều
#   người dùng đồng thời hơn, tăng RAM instance trước rồi mới tăng --workers.
# --threads 4: vẫn nhận nhiều kết nối cùng lúc, nhưng phần xử lý ảnh thực sự
#   được khoá tuần tự bằng threading.Lock() trong app.py (model dùng chung
#   không đảm bảo thread-safe).
# --timeout 120: xử lý 1 frame bằng DeepFace có thể mất vài giây trên CPU yếu.
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 app:app
