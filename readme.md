# 🤟 Hệ thống Nhận diện Ngôn ngữ Ký hiệu Việt Nam (VSL)
### End-to-end Pipeline: Thu thập dữ liệu → Huấn luyện → Inference → LLM

---

## Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE TỔNG QUÁT                       │
│                                                                  │
│  [Camera]                                                 │
│       │                                                          │
│       ▼                                                          │
│  [MediaPipe Holistic]  ──→  225 keypoints/frame                 │
│       │                     (33 Pose + 21 LH + 21 RH × 3 trục) │
│       ▼                                                          │
│  [Shoulder Normalization + Interpolation → 30 frames]           │
│       │                                                          │
│       ▼                                                          │
│  [BiLSTM Model]  ──→  Gloss (nhãn ký hiệu đơn lẻ)             │
│       │                                                          │
│       ▼                                                          │
│  [Gloss Sequence Accumulator]                                    │
│       │                                                          │
│       ▼                                                          │
│  [LLM (vit5 / GPT-4o)]  ──→  Câu tiếng Việt tự nhiên          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## GIAI ĐOẠN 1 — Thu thập & Xây dựng Dataset

### Mô tả
- Thu thập video ký hiệu từ **3 người thực hiện**
- **100 từ/ký hiệu** trong ngôn ngữ ký hiệu Việt Nam
- Mỗi video là 1 lần thực hiện 1 ký hiệu
- Gán nhãn thủ công qua file `label_gold.csv`

### Cấu trúc dataset
```
Dataset/
├── Videos/          # Video gốc (.mp4)
│   ├── XIN_CHAO_p1_001.mp4
│   └── ...
├── Labels/
│   └── label_gold.csv   # cột: VIDEO, LABEL
└── Models/          # Output sau train
```

### Thông số
| Thông số | Giá trị |
|----------|---------|
| Số nhãn | 100 ký hiệu |
| Số người quay | 3 |
| Định dạng | MP4, full quality |
| Gán nhãn | Thủ công (label_gold.csv) |

---

## GIAI ĐOẠN 2 — Trích xuất đặc trưng (Feature Extraction)

### Công cụ: MediaPipe Holistic

```
Frame gốc (full resolution, không nén)
    │
    ▼
MediaPipe Holistic (model_complexity=2)
    ├── Pose landmarks      → 33 điểm × 3 trục = 99 features
    ├── Left Hand landmarks → 21 điểm × 3 trục = 63 features
    └── Right Hand landmarks→ 21 điểm × 3 trục = 63 features
                                                  ──────────
                                            Tổng: 225 features / frame
```

> **Lý do bỏ Face Mesh:** Face landmarks (468 điểm) không đóng góp
> đáng kể cho phân loại VSL nhưng tăng chiều đặc trưng lên ~1600,
> gây overfitting trên dataset nhỏ.

### Chuẩn hóa Shoulder Normalization
```python
center = (vai_trái + vai_phải) / 2
scale  = ||vai_phải - vai_trái||
keypoints_norm = (keypoints - center) / scale
```
- Loại bỏ ảnh hưởng của khoảng cách người ký đến camera
- Bất biến với vị trí và tầm vóc người dùng

### Nội suy thời gian
- Mỗi video có độ dài khác nhau → nội suy tuyến tính về đúng **30 frames**
- 30 frames ≈ 2 giây @ 15 FPS = đủ bao phủ 1 ký hiệu hoàn chỉnh

---

## GIAI ĐOẠN 3 — Tăng cường dữ liệu (Data Augmentation)

Mỗi video gốc sinh ra **30 bản augment** bằng các phép biến đổi:

| Kỹ thuật | Mô tả | Xác suất |
|----------|-------|----------|
| Gaussian Noise | Thêm nhiễu vật lý nhỏ (σ=0.012) | 60% |
| Scale | Co giãn tay ±12% | 50% |
| Time Shift | Dịch thời gian ±3 frames | 50% |
| Time Warp | Co/giãn tốc độ ký hiệu ±15% | 40% |
| Mirror | Hoán đổi tay trái ↔ phải + flip trục X | 50% |

**Kết quả:** Dataset từ N video gốc → **N × 31 mẫu** sau augmentation

---

## GIAI ĐOẠN 4 — Huấn luyện mô hình (Training)

### Kiến trúc: Bidirectional LSTM

```
Input: (30 frames, 225 features)
    │
    ▼
BiLSTM(128)  return_sequences=True   ← nắm bắt ngữ cảnh 2 chiều
    │
Dropout(0.3)
    │
BiLSTM(64)   return_sequences=False  ← tổng hợp toàn chuỗi
    │
Dropout(0.3)
    │
Dense(256, relu)
    │
Dropout(0.4)
    │
Dense(128, relu)
    │
Dense(100, softmax)  ← 100 nhãn ký hiệu
```

### Cấu hình huấn luyện
| Tham số | Giá trị |
|---------|---------|
| Optimizer | Adam (lr=1e-3) |
| Loss | Sparse Categorical Crossentropy |
| Batch size | 32 |
| Epochs tối đa | 150 |
| Early stopping | patience=20 (monitor val_accuracy) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=7) |
| Train/Val split | 80/20, stratified |

### Output
- `best_bilstm.keras` — model tốt nhất theo val_accuracy
- `label_map.json` — ánh xạ id → nhãn
- `training_log.csv` — lịch sử loss/accuracy từng epoch

---

## GIAI ĐOẠN 5 — Đánh giá (Evaluation)

### A. Đánh giá model chính
- Top-1 Accuracy, Top-5 Accuracy
- Confusion Matrix (toàn bộ 100 lớp)
- Per-class Precision / Recall / F1

### B. Ablation Study — 5 cấu hình

| Cấu hình | Features | Kiến trúc |
|----------|----------|-----------|
| Pose only | 99 | BiLSTM |
| Hand only | 126 | BiLSTM |
| Pose+Hand | 225 | GRU (baseline) |
| Pose+Hand | 225 | Transformer |
| **Pose+Hand** | **225** | **BiLSTM (đề xuất)** |

> Ablation study chứng minh lý do chọn BiLSTM + Pose+Hand
> là tốt nhất → điền vào **Table II** bài báo

### C. Test trên người mới (Cross-person)
- Người thứ 4 chưa xuất hiện trong tập train
- Đo accuracy cross-person → số liệu generalization

---

## GIAI ĐOẠN 6 — Inference Realtime (Server)

### Kiến trúc server: FastAPI + WebSocket

```
Browser (Webcam)
    │  JPEG frame ~15 FPS qua WebSocket
    ▼
FastAPI Server
    ├── MediaPipe Holistic → 225 keypoints
    ├── Shoulder normalization
    ├── Sliding window buffer (30 frames)
    ├── State machine:
    │     REST → [energy > threshold] → SIGNING
    │     SIGNING → [silence ≥ 6 frames] → CONFIRMED → inference
    │     CONFIRMED → REST
    ├── BiLSTM.predict() → (nhãn, confidence)
    └── Broadcast JSON về browser
```

### State machine detect ký hiệu
```
Trạng thái   Điều kiện chuyển
─────────────────────────────────────────
REST       → SIGNING    : hand energy > 0.04
SIGNING    → CONFIRMED  : im lặng ≥ 6 frames
CONFIRMED  → REST       : sau khi inference xong
```

### Web Dashboard
- Live stream có vẽ landmarks
- State badge (REST / SIGNING / CONFIRMED)
- Confidence bar + Energy bar
- Gloss sequence ticker
- Metrics: FPS, latency, số ký hiệu nhận

---

## GIAI ĐOẠN 7 — LLM Sinh câu (NLG)

### Pipeline

```
Gloss sequence: ["XIN", "CHÀO", "TÊN", "TÔI", "LÀ", "NAM"]
    │
    ▼
[Bước 7A] Sinh dataset cặp (gloss → câu) bằng GPT-4o
    │        100 từ × tổ hợp 2-5 từ → ~500-1000 cặp
    │
    ▼
[Bước 7B] Finetune VietAI/vit5-base trên domain VSL
    │        So sánh vit5 gốc vs vit5 finetuned → BLEU score
    │
    ▼
[Bước 7C] RAG (Retrieval-Augmented Generation)
    │        Embed câu tham chiếu bằng PhoBERT + FAISS index
    │        Retrieve top-3 câu tương tự → đưa vào prompt
    │
    ▼
Output: "Xin chào, tên tôi là Nam."
```

### Đánh giá LLM
| Metric | Mô tả |
|--------|-------|
| BLEU-1 | Unigram precision |
| BLEU-2 | Bigram precision |
| BLEU-4 | Standard MT metric |
| Latency | ms/câu |

> So sánh 3 hệ thống: Prompt only / Finetuned vit5 / Finetuned + RAG
> → **Table III** bài báo

---

## Tổng hợp số liệu cần cho bài báo

| Bảng | Nội dung | File |
|------|----------|------|
| Table I | Thống kê dataset (100 từ, 3 người, N mẫu) | thống kê thủ công |
| Table II | Ablation study — 5 cấu hình | `ablation_results.csv` |
| Table III | BLEU score — 3 hệ thống LLM | `bleu_scores.csv` |
| Table IV | Latency end-to-end (MediaPipe + BiLSTM + LLM) | đo từ dashboard |
| Figure 1 | Kiến trúc hệ thống tổng quan | vẽ tay hoặc draw.io |
| Figure 2 | Confusion matrix 100 lớp | `confusion_matrix.png` |
| Figure 3 | Ablation chart | `ablation_chart.png` |
| Figure 4 | Training curves (loss/accuracy) | từ `training_log.csv` |

---

## Checklist trước khi nộp KSE 2025

### Kỹ thuật
- [ ] Test accuracy trên người thứ 4 (cross-person evaluation)
- [ ] Ablation study 5 cấu hình chạy xong
- [ ] Dataset gloss → câu sinh xong (~500 cặp)
- [ ] Finetune vit5 + đo BLEU
- [ ] RAG với FAISS tích hợp xong
- [ ] Latency end-to-end đo thực tế

### Bài viết
- [ ] Abstract (150-200 từ)
- [ ] Introduction + Related Work
- [ ] Methodology (pipeline + kiến trúc)
- [ ] Experiments & Results (điền số liệu)
- [ ] Conclusion
- [ ] Proofread tiếng Anh

---

## Stack công nghệ

| Thành phần | Công nghệ |
|-----------|-----------|
| Feature extraction | MediaPipe Holistic |
| Deep learning | TensorFlow / Keras |
| Model | Bidirectional LSTM |
| API server | FastAPI + WebSocket |
| Frontend | HTML / JavaScript |
| LLM | VietAI/vit5-base (finetune) + GPT-4o |
| RAG | FAISS + PhoBERT embeddings |
| Evaluation | scikit-learn, sacrebleu |
| Language | Python 3.11 |