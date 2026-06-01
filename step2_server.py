"""
BƯỚC 2 — FASTAPI SERVER + WEB DEMO REALTIME
Chạy: uvicorn step2_server:app --host 0.0.0.0 --port 8000 --reload
Mở:   http://localhost:8000
"""

import os, json, asyncio, time, base64
from collections import deque
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Cấu hình ──────────────────────────────────────────────────────────────
MODEL_PATH    = r"C:\Users\MYPC\Downloads\Dataset\Models\best_bilstm.keras"
LABEL_MAP_PATH= r"C:\Users\MYPC\Downloads\Dataset\Models\label_map.json"
N_FRAMES      = 30
N_FEATURES    = 225
CONF_THRESHOLD= 0.75   # độ tin cậy tối thiểu để nhận nhãn

# ── Load model & label map ─────────────────────────────────────────────────
print("⏳ Đang tải model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_MAP_PATH, encoding="utf-8") as f:
    id_to_label = {int(k): v for k, v in json.load(f).items()}
print(f"✅ Model sẵn sàng — {len(id_to_label)} nhãn")

# ── MediaPipe ──────────────────────────────────────────────────────────────
mp_holistic   = mp.solutions.holistic
mp_drawing    = mp.solutions.drawing_utils
mp_draw_styles= mp.solutions.drawing_styles

holistic_instance = mp_holistic.Holistic(
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ── Shoulder normalization (giống train) ───────────────────────────────────
def normalize_frame(frame_features: np.ndarray) -> np.ndarray:
    kps = frame_features.reshape(-1, 3)
    ls, rs = kps[11], kps[12]
    center = (ls + rs) / 2
    scale  = np.linalg.norm(rs - ls)
    if scale < 1e-6: scale = 1.0
    return ((kps - center) / scale).flatten().astype(np.float32)

# ── State per connection ───────────────────────────────────────────────────
class SignState:
    def __init__(self):
        self.buffer    : deque = deque(maxlen=N_FRAMES)   # sliding window keypoints
        self.velocity  : deque = deque(maxlen=10)          # hand velocity
        self.gloss_seq : list  = []                        # chuỗi nhãn đã nhận diện
        self.last_gloss: str   = ""
        self.last_conf : float = 0.0
        self.fps_times : deque = deque(maxlen=30)

    def hand_energy(self, kps: np.ndarray) -> float:
        """Tổng chuyển động cổ tay trái (63) và phải (189) → dùng detect ký hiệu"""
        lw = kps[63:66]; rw = kps[189:192]
        return float(np.linalg.norm(lw) + np.linalg.norm(rw))

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(title="VSL Realtime Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Web UI ─────────────────────────────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>VSL Realtime Demo</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0f1117; color: #e2e8f0; font-family: 'Segoe UI', sans-serif;
         display: flex; flex-direction: column; align-items: center; min-height: 100vh; }
  header { width: 100%; background: #1a1d2e; padding: 14px 24px;
           display: flex; align-items: center; gap: 12px; border-bottom: 1px solid #2d3250; }
  header h1 { font-size: 1.2rem; font-weight: 700; letter-spacing: .5px; }
  .badge { font-size: .7rem; background: #4c9be8; padding: 3px 10px;
           border-radius: 99px; font-weight: 600; }
  .main { display: flex; gap: 20px; padding: 20px; width: 100%; max-width: 1280px; flex-wrap: wrap; }

  /* Video panel */
  .video-panel { position: relative; flex: 1 1 640px; }
  #canvas { width: 100%; border-radius: 12px; background: #000; display: block; }
  .overlay-badges { position: absolute; top: 10px; left: 10px; display: flex; gap: 8px; }
  .state-badge { padding: 4px 12px; border-radius: 99px; font-size: .75rem; font-weight: 700; }
  .state-REST      { background: #2d3250; color: #a0aec0; }
  .state-SIGNING   { background: #276749; color: #9ae6b4; }
  .state-CONFIRMED { background: #1a365d; color: #90cdf4; }
  .fps-badge { position: absolute; top: 10px; right: 10px;
               background: rgba(0,0,0,.6); padding: 3px 8px;
               border-radius: 6px; font-size: .75rem; }

  /* Side panel */
  .side { flex: 0 0 320px; display: flex; flex-direction: column; gap: 16px; }

  .card { background: #1a1d2e; border-radius: 12px; padding: 16px; border: 1px solid #2d3250; }
  .card h3 { font-size: .75rem; text-transform: uppercase; letter-spacing: 1px;
             color: #718096; margin-bottom: 10px; }

  /* Current prediction */
  #pred-label { font-size: 2.4rem; font-weight: 800; color: #90cdf4; letter-spacing: 1px;
                min-height: 3rem; transition: all .2s; }
  #pred-conf  { font-size: .85rem; color: #a0aec0; margin-top: 4px; }
  .conf-bar   { height: 6px; background: #2d3250; border-radius: 99px; margin-top: 8px; }
  #conf-fill  { height: 100%; background: #4c9be8; border-radius: 99px;
                transition: width .3s; width: 0%; }

  /* Energy bar */
  .energy-track { height: 8px; background: #2d3250; border-radius: 99px; margin-top: 6px; }
  #energy-fill  { height: 100%; background: #68d391; border-radius: 99px;
                  transition: width .1s; width: 0%; }

  /* Gloss sequence */
  #gloss-box { display: flex; flex-wrap: wrap; gap: 6px; min-height: 40px; }
  .gloss-chip { background: #2d3250; border-radius: 6px; padding: 4px 10px;
                font-size: .8rem; font-weight: 600; }
  .gloss-chip.new { background: #2c5282; color: #bee3f8; }

  /* Sentence */
  #sentence-box { font-size: 1rem; line-height: 1.6; color: #e2e8f0; min-height: 2rem; }

  /* Controls */
  .btn { width: 100%; padding: 10px; border: none; border-radius: 8px;
         font-size: .9rem; font-weight: 600; cursor: pointer; transition: opacity .2s; }
  .btn:hover { opacity: .85; }
  .btn-start  { background: #276749; color: #fff; }
  .btn-stop   { background: #742a2a; color: #fff; }
  .btn-clear  { background: #2d3250; color: #a0aec0; }

  /* Latency */
  .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .metric { background: #0f1117; border-radius: 8px; padding: 10px; text-align: center; }
  .metric .val { font-size: 1.3rem; font-weight: 700; color: #90cdf4; }
  .metric .lbl { font-size: .7rem; color: #718096; margin-top: 2px; }
</style>
</head>
<body>
<header>
  <span>🤟</span>
  <h1>VSL Realtime Recognition</h1>
  <span class="badge">BiLSTM + MediaPipe</span>
</header>

<div class="main">
  <!-- Video -->
  <div class="video-panel">
    <canvas id="canvas" width="640" height="480"></canvas>
    <div class="overlay-badges">
      <span id="state-badge" class="state-badge state-REST">REST</span>
    </div>
    <span class="fps-badge" id="fps-display">0 FPS</span>
  </div>

  <!-- Side -->
  <div class="side">
    <!-- Prediction -->
    <div class="card">
      <h3>Nhận diện hiện tại</h3>
      <div id="pred-label">—</div>
      <div id="pred-conf">Confidence: —</div>
      <div class="conf-bar"><div id="conf-fill"></div></div>
    </div>

    <!-- Energy -->
    <div class="card">
      <h3>Năng lượng chuyển động tay</h3>
      <div class="energy-track"><div id="energy-fill"></div></div>
    </div>

    <!-- Gloss -->
    <div class="card">
      <h3>Chuỗi gloss</h3>
      <div id="gloss-box"><span style="color:#4a5568;font-size:.8rem">Chưa có...</span></div>
    </div>

    <!-- Sentence (LLM — bước 3) -->
    <div class="card">
      <h3>Câu tiếng Việt (LLM)</h3>
      <div id="sentence-box" style="color:#4a5568">Chờ LLM tích hợp...</div>
    </div>

    <!-- Controls -->
    <div class="card">
      <h3>Điều khiển</h3>
      <div style="display:flex;flex-direction:column;gap:8px">
        <button class="btn btn-start" id="btn-start" onclick="startCam()">▶ Bắt đầu</button>
        <button class="btn btn-stop"  id="btn-stop"  onclick="stopCam()" disabled>■ Dừng</button>
        <button class="btn btn-clear" onclick="clearGloss()">✕ Xoá chuỗi gloss</button>
      </div>
    </div>

    <!-- Metrics -->
    <div class="card">
      <h3>Thống kê</h3>
      <div class="metrics">
        <div class="metric"><div class="val" id="m-latency">—</div><div class="lbl">Latency (ms)</div></div>
        <div class="metric"><div class="val" id="m-total">0</div><div class="lbl">Ký hiệu nhận</div></div>
        <div class="metric"><div class="val" id="m-frames">0</div><div class="lbl">Frames xử lý</div></div>
        <div class="metric"><div class="val" id="m-conf-avg">—</div><div class="lbl">Conf trung bình</div></div>
      </div>
    </div>
  </div>
</div>

<script>
let ws, stream, animId;
let frameCount = 0, totalSigns = 0, confSum = 0, confCnt = 0;
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');
let video;

async function startCam() {
  stream = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}});
  video  = document.createElement('video');
  video.srcObject = stream; video.play();
  document.getElementById('btn-start').disabled = true;
  document.getElementById('btn-stop').disabled  = false;

  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage  = handleMsg;
  ws.onopen     = () => loop();
}

function stopCam() {
  cancelAnimationFrame(animId);
  if (stream) stream.getTracks().forEach(t=>t.stop());
  if (ws) ws.close();
  document.getElementById('btn-start').disabled = false;
  document.getElementById('btn-stop').disabled  = true;
}

function clearGloss() {
  document.getElementById('gloss-box').innerHTML =
    '<span style="color:#4a5568;font-size:.8rem">Chưa có...</span>';
  if (ws && ws.readyState===1) ws.send(JSON.stringify({action:'clear_gloss'}));
}

let lastFrameTime = 0;
function loop() {
  animId = requestAnimationFrame(loop);
  const now = performance.now();
  if (now - lastFrameTime < 66) return; // ~15 FPS gửi lên server
  lastFrameTime = now;

  if (!video || video.readyState < 2) return;
  ctx.drawImage(video, 0, 0, 640, 480);
  canvas.toBlob(blob => {
    if (ws && ws.readyState===1) {
      blob.arrayBuffer().then(buf => ws.send(buf));
      frameCount++;
      document.getElementById('m-frames').textContent = frameCount;
    }
  }, 'image/jpeg', 0.7);
}

function handleMsg(evt) {
  const data = JSON.parse(evt.data);

  // Annotated frame
  if (data.frame) {
    const img = new Image();
    img.onload = () => ctx.drawImage(img, 0, 0, 640, 480);
    img.src = 'data:image/jpeg;base64,' + data.frame;
  }

  // State badge
  const badge = document.getElementById('state-badge');
  badge.textContent = data.state || 'REST';
  badge.className   = 'state-badge state-' + (data.state || 'REST');

  // FPS
  if (data.fps) document.getElementById('fps-display').textContent = data.fps.toFixed(1)+' FPS';

  // Energy
  if (data.energy !== undefined) {
    const pct = Math.min(data.energy * 200, 100);
    document.getElementById('energy-fill').style.width = pct + '%';
  }

  // Prediction
  if (data.label) {
    document.getElementById('pred-label').textContent = data.label;
    document.getElementById('pred-conf').textContent  = `Confidence: ${(data.conf*100).toFixed(1)}%`;
    document.getElementById('conf-fill').style.width  = (data.conf*100)+'%';
    document.getElementById('m-latency').textContent  = data.latency_ms || '—';
    if (data.is_new) {
      totalSigns++;
      document.getElementById('m-total').textContent = totalSigns;
      confSum += data.conf; confCnt++;
      document.getElementById('m-conf-avg').textContent =
        (confSum/confCnt*100).toFixed(1)+'%';
    }
  }

  // Gloss sequence
  if (data.gloss_seq && data.gloss_seq.length > 0) {
    const box = document.getElementById('gloss-box');
    box.innerHTML = data.gloss_seq.map((g,i) =>
      `<span class="gloss-chip${i===data.gloss_seq.length-1?' new':''}">${g}</span>`
    ).join('');
  }

  // Sentence from LLM
  if (data.sentence) {
    document.getElementById('sentence-box').textContent = data.sentence;
    document.getElementById('sentence-box').style.color = '#e2e8f0';
  }
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML

# ── WebSocket inference loop ───────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state = SignState()
    prev_kp: Optional[np.ndarray] = None
    signing_frames = 0
    SIGNING_THRESHOLD = 0.04   # ngưỡng energy bắt đầu ký hiệu
    CONFIRM_FRAMES    = 6      # số frame im lặng để confirm
    silence_frames    = 0
    current_state     = "REST"

    try:
        while True:
            t0 = time.perf_counter()
            data = await ws.receive()

            # Lệnh điều khiển từ client
            if "text" in data:
                cmd = json.loads(data["text"])
                if cmd.get("action") == "clear_gloss":
                    state.gloss_seq = []
                continue

            # Frame JPEG
            jpg = np.frombuffer(data["bytes"], np.uint8)
            frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame = cv2.resize(frame, (640, 480))
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic_instance.process(rgb)

            # Vẽ landmarks
            frame.flags.writeable = True
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(frame, res.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw_styles.get_default_pose_landmarks_style())
            if res.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, res.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS)
            if res.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, res.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS)

            # Extract keypoints
            pose = np.zeros((33,3), np.float32)
            lh   = np.zeros((21,3), np.float32)
            rh   = np.zeros((21,3), np.float32)
            if res.pose_landmarks:
                for i,lm in enumerate(res.pose_landmarks.landmark): pose[i]=[lm.x,lm.y,lm.z]
            if res.left_hand_landmarks:
                for i,lm in enumerate(res.left_hand_landmarks.landmark): lh[i]=[lm.x,lm.y,lm.z]
            if res.right_hand_landmarks:
                for i,lm in enumerate(res.right_hand_landmarks.landmark): rh[i]=[lm.x,lm.y,lm.z]

            raw_kp = np.concatenate([pose,lh,rh]).flatten()
            norm_kp= normalize_frame(raw_kp)
            state.buffer.append(norm_kp)

            # Tính hand energy (velocity)
            energy = 0.0
            if prev_kp is not None:
                energy = float(np.linalg.norm(norm_kp[99:225] - prev_kp[99:225]))
            prev_kp = norm_kp.copy()
            state.velocity.append(energy)
            avg_energy = float(np.mean(state.velocity))

            # State machine: REST → SIGNING → CONFIRMED
            label, conf, is_new = state.last_gloss, state.last_conf, False

            if avg_energy > SIGNING_THRESHOLD:
                signing_frames += 1; silence_frames = 0
                current_state = "SIGNING"
            else:
                silence_frames += 1
                if current_state == "SIGNING" and silence_frames >= CONFIRM_FRAMES:
                    current_state = "CONFIRMED"
                    # Inference
                    if len(state.buffer) >= N_FRAMES:
                        seq = np.array(list(state.buffer)[-N_FRAMES:], np.float32)
                        seq = seq[np.newaxis, ...]          # (1,30,225)
                        probs = model.predict(seq, verbose=0)[0]
                        pred_id   = int(np.argmax(probs))
                        pred_conf = float(probs[pred_id])
                        if pred_conf >= CONF_THRESHOLD:
                            label = id_to_label[pred_id]
                            conf  = pred_conf
                            if label != state.last_gloss:
                                state.gloss_seq.append(label)
                                state.last_gloss = label
                                state.last_conf  = conf
                                is_new = True
                    signing_frames = 0
                elif silence_frames > CONFIRM_FRAMES * 2:
                    current_state = "REST"

            # FPS
            state.fps_times.append(time.perf_counter())
            fps = 0.0
            if len(state.fps_times) >= 2:
                fps = (len(state.fps_times)-1) / (state.fps_times[-1]-state.fps_times[0]+1e-9)

            latency_ms = round((time.perf_counter()-t0)*1000, 1)

            # Encode annotated frame
            _, jpg_out = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64  = base64.b64encode(jpg_out.tobytes()).decode()

            await ws.send_text(json.dumps({
                "frame"     : frame_b64,
                "state"     : current_state,
                "label"     : label,
                "conf"      : round(conf, 4),
                "is_new"    : is_new,
                "gloss_seq" : state.gloss_seq[-10:],
                "energy"    : round(avg_energy, 4),
                "fps"       : round(fps, 1),
                "latency_ms": latency_ms,
            }))

    except WebSocketDisconnect:
        pass
    finally:
        print("📴 Client ngắt kết nối")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("step2_server:app", host="0.0.0.0", port=8000, reload=False)