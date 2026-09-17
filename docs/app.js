// app.js
// Ban web (chay hoan toan trong trinh duyet, khong can server Python) cua
// tinh nang nhan dien ban tay + cu chi + tu giac 2 tay + hieu ung mau vung
// ben trong tu giac - port lai tu utils.py/main_webcam.py (ban Python) sang
// JavaScript, dung MediaPipe Tasks Vision (chay qua WASM ngay trong trinh
// duyet, khong gui frame webcam len server nao ca).
//
// Vi chay hoan toan client-side, ban nay co the deploy len GitHub Pages
// (host file tinh mien phi) ma khong bi gioi han RAM/CPU nhu server free
// tier, va khong bi giat do do tre mang (khac voi kien truc server Python
// gui/nhan frame qua HTTP).

import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const startBtn = document.getElementById("startBtn");
const statusEl = document.getElementById("status");
const leftGestureEl = document.getElementById("leftGesture");
const rightGestureEl = document.getElementById("rightGesture");
const effectNameEl = document.getElementById("effectName");

// ==================== Hang so landmark (giong utils.py) ====================
const THUMB_TIP_ID = 4;
const INDEX_TIP_ID = 8;
const MIDDLE_TIP_ID = 12;
const RING_TIP_ID = 16;
const PINKY_TIP_ID = 20;
const FINGER_TIP_IDS = [4, 8, 12, 16, 20];

// 13 hieu ung xoay vong khi pinch, DUNG THU TU nhu COLOR_EFFECT_CYCLE trong
// utils.py de giu nhat quan trai nghiem giua 2 ban.
const COLOR_EFFECT_CYCLE = [
  "invert", "r0", "g0", "b0",
  "pixelate", "noise", "swirl", "wave",
  "blur", "edge", "heatmap", "grayscale", "sepia",
];

let currentEffectIndex = 0;
const wasPinching = { Left: false, Right: false };

// ==================== Khoi tao HandLandmarker (MediaPipe Tasks Vision) ====================
let handLandmarker = null;

async function initHandLandmarker() {
  statusEl.textContent = "Dang tai model nhan dien ban tay...";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });
  statusEl.textContent = "Da tai xong model.";
}

// ==================== Cu chi (giong _classify_gesture trong utils.py) ====================
function classifyGesture(landmarks, handedness) {
  const fingersUp = [];

  // Ngon cai: so sanh toa do x giua dau ngon va khop IP, huong phu thuoc tay
  // trai/phai. Luu y: MediaPipe web tra ve handedness theo goc nhin CAMERA
  // (da tu dong bu mirroring cho webcam selfie), giong hanh vi ban Python.
  const thumbTip = landmarks[FINGER_TIP_IDS[0]];
  const thumbIp = landmarks[FINGER_TIP_IDS[0] - 1];
  if (handedness === "Right") {
    fingersUp.push(thumbTip.x < thumbIp.x ? 1 : 0);
  } else {
    fingersUp.push(thumbTip.x > thumbIp.x ? 1 : 0);
  }

  // 4 ngon con lai: dang gio neu dau ngon (tip) nam cao hon (y nho hon) khop
  // giua (pip).
  for (const tipId of FINGER_TIP_IDS.slice(1)) {
    const tip = landmarks[tipId];
    const pip = landmarks[tipId - 2];
    fingersUp.push(tip.y < pip.y ? 1 : 0);
  }

  const totalUp = fingersUp.reduce((a, b) => a + b, 0);

  if (totalUp === 0) return "Nam tay";
  if (totalUp === 5) return "Xoe tay";
  if (fingersUp.join(",") === "1,0,0,0,0") return "Thumbs up";
  return `${totalUp} ngon tay`;
}

// ==================== Pinch detection (giong is_pinching) ====================
function landmarkDistance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function isPinching(landmarks, ratioThreshold = 0.4) {
  const thumbTip = landmarks[THUMB_TIP_ID];
  const indexTip = landmarks[INDEX_TIP_ID];
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];

  const pinchDist = landmarkDistance(thumbTip, indexTip);
  const refDist = landmarkDistance(wrist, middleMcp);
  if (refDist === 0) return false;

  return pinchDist / refDist < ratioThreshold;
}

// ==================== Tu giac 2 tay (giong get_two_hand_quad_points) ====================
function landmarkToPixel(landmark, width, height) {
  return [landmark.x * width, landmark.y * height];
}

function getTwoHandQuadPoints(leftLandmarks, rightLandmarks, width, height,
                               tipIdA = INDEX_TIP_ID, tipIdB = THUMB_TIP_ID) {
  const leftA = landmarkToPixel(leftLandmarks[tipIdA], width, height);
  const leftB = landmarkToPixel(leftLandmarks[tipIdB], width, height);
  const rightB = landmarkToPixel(rightLandmarks[tipIdB], width, height);
  const rightA = landmarkToPixel(rightLandmarks[tipIdA], width, height);
  return [leftA, leftB, rightB, rightA];
}

function drawQuadOutline(context, quadPoints, color = "rgb(255,255,255)", lineWidth = 2) {
  context.save();
  context.strokeStyle = color;
  context.lineWidth = lineWidth;
  context.beginPath();
  context.moveTo(quadPoints[0][0], quadPoints[0][1]);
  for (let i = 1; i < quadPoints.length; i++) {
    context.lineTo(quadPoints[i][0], quadPoints[i][1]);
  }
  context.closePath();
  context.stroke();
  context.restore();
}

function drawQuadVertices(context, quadPoints, color = "rgb(0,200,255)", radius = 8) {
  context.save();
  context.fillStyle = color;
  for (const [x, y] of quadPoints) {
    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    context.fill();
  }
  context.restore();
}

// ==================== Hieu ung mau vung tu giac (giong apply_quad_color_effect) ====================

// Kiem tra 1 diem co nam trong da giac (point-in-polygon, thuat toan ray
// casting) - dung de tao mask gioi han hieu ung CHI trong vung tu giac.
function pointInPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    const intersect =
      yi > y !== yj > y &&
      x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-9) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function boundingRect(quadPoints, width, height) {
  const xs = quadPoints.map((p) => p[0]);
  const ys = quadPoints.map((p) => p[1]);
  let x0 = Math.max(0, Math.floor(Math.min(...xs)));
  let y0 = Math.max(0, Math.floor(Math.min(...ys)));
  let x1 = Math.min(width, Math.ceil(Math.max(...xs)));
  let y1 = Math.min(height, Math.ceil(Math.max(...ys)));
  return [x0, y0, Math.max(0, x1 - x0), Math.max(0, y1 - y0)];
}

// Ap 1 ham bien doi pixel (transformFn) len vung ROI (bounding box cua tu
// giac), roi chi "dan" lai nhung pixel nam THUC SU ben trong tu giac (dung
// mask point-in-polygon) - dam bao hieu ung khong lan ra ngoai tu giac, kha
// vien tuong tu _apply_roi_effect trong utils.py.
function applyRoiEffect(context, quadPoints, canvasWidth, canvasHeight, transformFn) {
  const [rx, ry, rw, rh] = boundingRect(quadPoints, canvasWidth, canvasHeight);
  if (rw <= 0 || rh <= 0) return;

  const original = context.getImageData(rx, ry, rw, rh);
  const transformed = transformFn(original, rw, rh);

  // Mask: pixel nao thuc su nam trong tu giac (toa do tuyet doi tren canvas).
  const outData = original.data; // ghi de truc tiep len ban goc, chi thay doi pixel trong mask
  for (let py = 0; py < rh; py++) {
    for (let px = 0; px < rw; px++) {
      const absX = rx + px;
      const absY = ry + py;
      if (pointInPolygon(absX, absY, quadPoints)) {
        const idx = (py * rw + px) * 4;
        outData[idx] = transformed.data[idx];
        outData[idx + 1] = transformed.data[idx + 1];
        outData[idx + 2] = transformed.data[idx + 2];
        outData[idx + 3] = 255;
      }
    }
  }
  context.putImageData(original, rx, ry);
}

function clamp255(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

function effectInvert(imgData) {
  const d = new Uint8ClampedArray(imgData.data);
  for (let i = 0; i < d.length; i += 4) {
    d[i] = 255 - d[i];
    d[i + 1] = 255 - d[i + 1];
    d[i + 2] = 255 - d[i + 2];
  }
  return { data: d };
}

function effectZeroChannel(imgData, channelIndex) {
  const d = new Uint8ClampedArray(imgData.data);
  for (let i = 0; i < d.length; i += 4) {
    d[i + channelIndex] = 0;
  }
  return { data: d };
}

function effectPixelate(imgData, w, h, blockSize = 14) {
  const src = imgData.data;
  const d = new Uint8ClampedArray(src.length);
  for (let by = 0; by < h; by += blockSize) {
    for (let bx = 0; bx < w; bx += blockSize) {
      let r = 0, g = 0, b = 0, count = 0;
      const bw = Math.min(blockSize, w - bx);
      const bh = Math.min(blockSize, h - by);
      for (let y = by; y < by + bh; y++) {
        for (let x = bx; x < bx + bw; x++) {
          const idx = (y * w + x) * 4;
          r += src[idx]; g += src[idx + 1]; b += src[idx + 2];
          count++;
        }
      }
      r = Math.round(r / count); g = Math.round(g / count); b = Math.round(b / count);
      for (let y = by; y < by + bh; y++) {
        for (let x = bx; x < bx + bw; x++) {
          const idx = (y * w + x) * 4;
          d[idx] = r; d[idx + 1] = g; d[idx + 2] = b; d[idx + 3] = 255;
        }
      }
    }
  }
  return { data: d };
}

function effectNoise(imgData, w, h, amount = 45) {
  const src = imgData.data;
  const d = new Uint8ClampedArray(src.length);
  for (let i = 0; i < src.length; i += 4) {
    const n = (Math.random() * 2 - 1) * amount;
    d[i] = clamp255(src[i] + n);
    d[i + 1] = clamp255(src[i + 1] + n);
    d[i + 2] = clamp255(src[i + 2] + n);
    d[i + 3] = 255;
  }
  return { data: d };
}

function effectSwirl(imgData, w, h, strength = 3.0) {
  const src = imgData.data;
  const d = new Uint8ClampedArray(src.length);
  const cx = w / 2, cy = h / 2;
  const maxRadius = Math.min(w, h) / 2;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = x - cx, dy = y - cy;
      const radius = Math.hypot(dx, dy);
      let srcX = x, srcY = y;
      if (radius < maxRadius) {
        const percent = (maxRadius - radius) / maxRadius;
        const theta = strength * percent * percent;
        const cosT = Math.cos(theta), sinT = Math.sin(theta);
        srcX = cx + dx * cosT - dy * sinT;
        srcY = cy + dx * sinT + dy * cosT;
      }
      const sx = Math.min(w - 1, Math.max(0, Math.round(srcX)));
      const sy = Math.min(h - 1, Math.max(0, Math.round(srcY)));
      const srcIdx = (sy * w + sx) * 4;
      const dstIdx = (y * w + x) * 4;
      d[dstIdx] = src[srcIdx]; d[dstIdx + 1] = src[srcIdx + 1];
      d[dstIdx + 2] = src[srcIdx + 2]; d[dstIdx + 3] = 255;
    }
  }
  return { data: d };
}

function effectWave(imgData, w, h, amplitude = 8, wavelength = 25) {
  const src = imgData.data;
  const d = new Uint8ClampedArray(src.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const srcX = x + amplitude * Math.sin((2 * Math.PI * y) / wavelength);
      const srcY = y + amplitude * Math.sin((2 * Math.PI * x) / wavelength);
      const sx = Math.min(w - 1, Math.max(0, Math.round(srcX)));
      const sy = Math.min(h - 1, Math.max(0, Math.round(srcY)));
      const srcIdx = (sy * w + sx) * 4;
      const dstIdx = (y * w + x) * 4;
      d[dstIdx] = src[srcIdx]; d[dstIdx + 1] = src[srcIdx + 1];
      d[dstIdx + 2] = src[srcIdx + 2]; d[dstIdx + 3] = 255;
    }
  }
  return { data: d };
}

// Blur dung canvas filter (nhanh + chat luong tot hon tu code convolution
// thu cong) tren 1 offscreen canvas rieng cho vung ROI.
function effectBlur(imgData, w, h, blurPx = 10) {
  const off = document.createElement("canvas");
  off.width = w; off.height = h;
  const octx = off.getContext("2d");
  octx.putImageData(imgData, 0, 0);

  const off2 = document.createElement("canvas");
  off2.width = w; off2.height = h;
  const octx2 = off2.getContext("2d");
  octx2.filter = `blur(${blurPx}px)`;
  octx2.drawImage(off, 0, 0);

  return octx2.getImageData(0, 0, w, h);
}

function effectEdge(imgData, w, h) {
  const src = imgData.data;
  const gray = new Float32Array(w * h);
  for (let i = 0, p = 0; i < src.length; i += 4, p++) {
    gray[p] = 0.299 * src[i] + 0.587 * src[i + 1] + 0.114 * src[i + 2];
  }
  const d = new Uint8ClampedArray(src.length);
  const gx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const gy = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let sx = 0, sy = 0, k = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const v = gray[(y + ky) * w + (x + kx)];
          sx += v * gx[k]; sy += v * gy[k]; k++;
        }
      }
      const mag = clamp255(Math.hypot(sx, sy));
      const idx = (y * w + x) * 4;
      d[idx] = mag; d[idx + 1] = mag; d[idx + 2] = mag; d[idx + 3] = 255;
    }
  }
  return { data: d };
}

// Bang mau xap xi kieu "jet" colormap (xanh duong -> xanh la -> vang -> do)
function heatColor(t) {
  // t trong [0, 1]
  const r = clamp255(Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3)))));
  const g = clamp255(Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2)))));
  const b = clamp255(Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1)))));
  return [r, g, b];
}

function effectHeatmap(imgData) {
  const src = imgData.data;
  const d = new Uint8ClampedArray(src.length);
  for (let i = 0; i < src.length; i += 4) {
    const gray = (0.299 * src[i] + 0.587 * src[i + 1] + 0.114 * src[i + 2]) / 255;
    const [r, g, b] = heatColor(gray);
    d[i] = r; d[i + 1] = g; d[i + 2] = b; d[i + 3] = 255;
  }
  return { data: d };
}

function effectGrayscale(imgData) {
  const src = imgData.data;
  const d = new Uint8ClampedArray(src.length);
  for (let i = 0; i < src.length; i += 4) {
    const gray = clamp255(0.299 * src[i] + 0.587 * src[i + 1] + 0.114 * src[i + 2]);
    d[i] = gray; d[i + 1] = gray; d[i + 2] = gray; d[i + 3] = 255;
  }
  return { data: d };
}

function effectSepia(imgData) {
  const src = imgData.data;
  const d = new Uint8ClampedArray(src.length);
  for (let i = 0; i < src.length; i += 4) {
    const r = src[i], g = src[i + 1], b = src[i + 2];
    d[i] = clamp255(0.393 * r + 0.769 * g + 0.189 * b);
    d[i + 1] = clamp255(0.349 * r + 0.686 * g + 0.168 * b);
    d[i + 2] = clamp255(0.272 * r + 0.534 * g + 0.131 * b);
    d[i + 3] = 255;
  }
  return { data: d };
}

const ROI_EFFECT_FUNCTIONS = {
  invert: (img) => effectInvert(img),
  r0: (img) => effectZeroChannel(img, 0),
  g0: (img) => effectZeroChannel(img, 1),
  b0: (img) => effectZeroChannel(img, 2),
  pixelate: (img, w, h) => effectPixelate(img, w, h),
  noise: (img, w, h) => effectNoise(img, w, h),
  swirl: (img, w, h) => effectSwirl(img, w, h),
  wave: (img, w, h) => effectWave(img, w, h),
  blur: (img, w, h) => effectBlur(img, w, h),
  edge: (img, w, h) => effectEdge(img, w, h),
  heatmap: (img) => effectHeatmap(img),
  grayscale: (img) => effectGrayscale(img),
  sepia: (img) => effectSepia(img),
};

function applyQuadColorEffect(context, quadPoints, canvasWidth, canvasHeight, effect) {
  const fn = ROI_EFFECT_FUNCTIONS[effect];
  if (!fn) return;
  applyRoiEffect(context, quadPoints, canvasWidth, canvasHeight, (img, w, h) => fn(img, w, h));
}

// ==================== Vong lap chinh ====================
let lastVideoTime = -1;

function renderLoop() {
  if (video.readyState >= 2 && handLandmarker) {
    const nowMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
      lastVideoTime = video.currentTime;
      const results = handLandmarker.detectForVideo(video, nowMs);
      processResults(results);
    }
  }
  requestAnimationFrame(renderLoop);
}

function processResults(results) {
  const w = canvas.width;
  const h = canvas.height;

  ctx.save();
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(video, 0, 0, w, h);

  const hands = [];
  if (results.landmarks && results.landmarks.length > 0) {
    for (let i = 0; i < results.landmarks.length; i++) {
      const handedness = results.handednesses[i][0].categoryName; // "Left" | "Right"
      const landmarks = results.landmarks[i];
      const gesture = classifyGesture(landmarks, handedness);
      hands.push({ handedness, landmarks, gesture });
    }
  }

  const leftHand = hands.find((h) => h.handedness === "Left");
  const rightHand = hands.find((h) => h.handedness === "Right");

  // Cu chi "chum ngon" (pinch) -> xoay vong hieu ung, chi o canh len (vua
  // cham, khong doi lien tuc trong luc giu chum) - giong main_webcam.py.
  for (const hand of hands) {
    const pinchingNow = isPinching(hand.landmarks);
    if (pinchingNow && !wasPinching[hand.handedness]) {
      currentEffectIndex = (currentEffectIndex + 1) % COLOR_EFFECT_CYCLE.length;
    }
    wasPinching[hand.handedness] = pinchingNow;
  }

  if (leftHand && rightHand) {
    const quadPoints = getTwoHandQuadPoints(leftHand.landmarks, rightHand.landmarks, w, h);
    const currentEffect = COLOR_EFFECT_CYCLE[currentEffectIndex];
    applyQuadColorEffect(ctx, quadPoints, w, h, currentEffect);
    drawQuadOutline(ctx, quadPoints);
    drawQuadVertices(ctx, quadPoints);
    effectNameEl.textContent = currentEffect;
  } else {
    effectNameEl.textContent = "-";
  }

  leftGestureEl.textContent = leftHand ? leftHand.gesture : "-";
  rightGestureEl.textContent = rightHand ? rightHand.gesture : "-";

  ctx.restore();
}

// ==================== Khoi dong webcam ====================
async function startWebcam() {
  startBtn.disabled = true;
  statusEl.textContent = "Dang xin quyen truy cap webcam...";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 960, height: 720 },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    if (!handLandmarker) {
      await initHandLandmarker();
    }

    statusEl.textContent = "Dang chay - dua 2 tay vao khung hinh de thu tu giac.";
    requestAnimationFrame(renderLoop);
  } catch (err) {
    statusEl.textContent = `Loi: ${err.message}`;
    startBtn.disabled = false;
  }
}

startBtn.addEventListener("click", startWebcam);
