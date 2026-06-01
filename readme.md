Dưới đây là tệp tin **`README.md`** hoàn chỉnh, chi tiết và có chiều sâu học thuật nhất, tích hợp toàn bộ các thuật toán lai được định lượng hóa chi tiết bằng toán học để bạn sử dụng cho kho lưu trữ mã nguồn của mình:

---

# 🤟 Hệ thống Nhận diện và Dịch Ngôn ngữ Ký hiệu Việt Nam (VSL) Thời Gian Thực
### Hệ thống Lai hai giai đoạn (Two-Stage Pipeline): Trích xuất đa phương thức 345 đặc trưng → Huấn luyện BiLSTM → Bộ lọc chuỗi liên tục Hybrid (CVPR 2024 & ESWA 2024) → NLG Translator (vit5 + RAG)

---

## 🌟 1. Tổng quan Kiến trúc Hệ thống

Hệ thống được thiết kế theo mô hình **Hệ thống Lai 2 giai đoạn (Two-Stage Hybrid System)** nhằm giải quyết hai thách thức lớn nhất của bài toán nhận diện ngôn ngữ ký hiệu liên tục (CSLR) là: **Sự thiếu hụt tập dữ liệu câu dài song song** và **Lỗi ranh giới từ do nhiễu chuyển tiếp (co-articulation)** giữa các cử chỉ.

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE TỔNG QUÁT                       │
│                                                                  │
│  [Camera / Stream Video liên tục]                               │
│       │                                                          │
│       ▼                                                          │
│  [MediaPipe Holistic]  ──→  345 keypoints/frame                 │
│       │                     (33 Pose + 21 LH + 21 RH + 40 Lips) │
│       ▼                                                          │
│  [Shoulder Normalization + Interpolation → 30 frames]           │
│       │                                                          │
│       ▼                                                          │
│  [BiLSTM Predictor (345)] ──→ Xác suất Softmax của các lớp       │
│       │                                                          │
│       ▼                                                          │
│  [Bộ lọc lai Hybrid Post-Processing (CVPR'24 & ESWA'24)]        │
│       ├── 1. Hàng đợi bỏ phiếu đa số (Voting Bag B=7)            │
│       ├── 2. Khóa khoảng trống sụt giảm (Gap Suppression)        │
│       └── 3. Phát hiện đỉnh cực đại (Peak-Valley Detection)      │
│       │                                                          │
│       ▼                                                          │
│  [Chuỗi ký hiệu nén sạch (Gloss Sequence)]                      │
│       │                                                          │
│       ▼                                                          │
│  [LLM (vit5 / GPT-4o)]  ──→  Câu tiếng Việt tự nhiên          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 2. Chi tiết các Giai đoạn Xử lý trong Pipeline

### Giai đoạn 2.1: Quản lý Dữ liệu trực tiếp (Folder-based Dataset)
Để loại bỏ sự phụ thuộc vào các tệp chỉ mục CSV thủ công dễ sai lệch, hệ thống sử dụng cấu trúc thư mục dữ liệu trực tiếp:
*   Mỗi thư mục con nằm trong thư mục gốc `videos_update` biểu thị cho chính **Tên của Nhãn (Label)**.
*   Bên trong thư mục con chứa các video gốc (`W...mp4`) và các video bổ sung góc quay nghiêng trái (`_L`), nghiêng phải (`_R`) của nhiều người ký khác nhau.
*   Mô hình tự động quét tên thư mục để sinh ra bảng ánh xạ nhãn (`label_map.json`) khi bắt đầu huấn luyện.

---

### Giai đoạn 2.2: Trích xuất đặc trưng đa phương thức (345 Đặc trưng)
Để phân biệt các cử chỉ có hướng đi của tay giống nhau nhưng khác biệt ở biểu cảm khuôn mặt hoặc khẩu hình miệng, hệ thống trích xuất **345 tọa độ không gian** bằng bộ công cụ **MediaPipe Holistic (model_complexity=2)**:
*   **Pose (Thân mình):** 33 điểm mốc $\times$ 3 trục $(x, y, z)$ = **99 đặc trưng** (lấy thông tin khung vai và hướng di chuyển cánh tay).
*   **Left Hand (Tay trái):** 21 điểm $\times$ 3 trục = **63 đặc trưng**.
*   **Right Hand (Tay phải):** 21 điểm $\times$ 3 trục = **63 đặc trưng**.
*   **Lips Mesh (Khẩu hình miệng):** 40 điểm $\times$ 3 trục = **120 đặc trưng** (trích xuất theo chỉ mục vành môi của Face Mesh).

> **Giải pháp tối ưu hóa Face Mesh:** Không lấy toàn bộ 468 điểm mốc trên mặt (gây quá tải tính toán và overfitting), hệ thống chỉ lọc ra đúng **40 điểm mốc môi (Lips Mesh)** dựa theo chỉ mục cố định để bảo toàn thông tin khẩu hình miệng với dung lượng nhẹ nhất.

---

### Giai đoạn 2.3: Chuẩn hóa khoảng cách vai & Nội suy thời gian
*   **Shoulder Normalization (Chuẩn hóa vai):** Để mô hình không bị ảnh hưởng bởi việc người dùng đứng xa hay gần camera, tọa độ của tất cả các khớp được dịch chuyển gốc về tâm vai (trung điểm vai trái và vai phải) và chia cho tỷ lệ độ dài xương vai.
    $$\mathbf{C} = \frac{\mathbf{P}_{left\_shoulder} + \mathbf{P}_{right\_shoulder}}{2}$$
    $$S = \|\mathbf{P}_{right\_shoulder} - \mathbf{P}_{left\_shoulder}\|_2$$
    $$\mathbf{P}'_{i} = \frac{\mathbf{P}_{i} - \mathbf{C}}{S}$$
    *(Trong đó $\mathbf{C}$ là tâm vai, $S$ là tỷ lệ vai, và $\mathbf{P}'_{i}$ là tọa độ khớp sau chuẩn hóa).*

*   **Linear Interpolation (Nội suy tuyến tính):** Các video có thời lượng khác nhau được nội suy tuyến tính 1 chiều (`scipy.interpolate.interp1d`) đưa về độ dài thống nhất **`N_FRAMES = 30`** (khoảng 1 giây ở tốc độ 30 FPS).

---

### Giai đoạn 2.4: Tăng cường dữ liệu không gian - thời gian (Augmentation)
Mỗi mẫu dữ liệu gốc được biến đổi ngẫu nhiên thành 30 mẫu khác nhau để chống Overfitting:
*   *Co giãn tốc độ (Time Warp):* Co giãn thời gian cử chỉ ngẫu nhiên $\pm 15\%$ bằng cách nội suy lại số khung hình.
*   *Dịch thời gian (Time Shift):* Dịch chuyển thời điểm thực hiện cử chỉ tới/lui $\pm 3$ khung hình và đắp biên (`edge padding`).
*   *Lật gương bàn tay (Mirror):* Hoán đổi đặc trưng tay trái và tay phải, đồng thời nhân đối xứng trục X của Pose và Lips Mesh (`x = -x`) để mô phỏng cử chỉ bằng tay thuận khác nhau.

---

### Giai đoạn 2.5: Huấn luyện BiLSTM & Bộ nhớ đệm (Smart Caching)
Mô hình chính được xây dựng bằng mạng hồi quy hai chiều Bidirectional LSTM gồm 2 tầng chồng lên nhau:
*   **BiLSTM tầng 1 (128 units):** Trả về chuỗi đầu ra (`return_sequences=True`) để học ngữ cảnh chuyển động tiến - lùi thời gian sâu của cử chỉ.
*   **BiLSTM tầng 2 (64 units):** Chỉ trả về vectơ trạng thái cuối cùng (`return_sequences=False`) để tổng hợp toàn bộ chuyển động của chuỗi.
*   **Mạng nơ-ron Dense kết nối đầy đủ:** Đi qua 2 tầng Dense (256, 128) kích hoạt ReLU kèm tỷ lệ Dropout cao (0.3 - 0.4) chống Overfitting, trước khi đưa ra phân phối xác suất qua tầng Softmax.

*   **Cơ chế tự lưu bộ nhớ đệm (Caching):** Quá trình chạy MediaPipe Holistic trên CPU rất tốn thời gian. Hệ thống tích hợp cơ chế tự động ghi nhận dữ liệu đã trích xuất ra các file nhị phân **`X_cached.npy`** và **`y_cached.npy`** trong thư mục `Models/`. Ở các phiên huấn luyện tiếp theo, mô hình sẽ nạp trực tiếp file cache này chỉ mất **1 giây**, loại bỏ hoàn toàn thời gian trích xuất lặp lại.

---

## 🧠 3. Chi tiết thuật toán Bộ lọc hậu xử lý lai (Hybrid Post-Processing Engine)

Đây là đóng góp khoa học cốt lõi của hệ thống, giúp chạy nhận diện thời gian thực trên webcam hoặc video chuỗi dài cực kỳ chính xác mà không bị nháy nhãn. Gọi chuỗi xác suất Softmax nhận được từ các cửa sổ trượt liên tiếp là $\mathbf{P} = \{P_1, P_2, \dots, P_T\}$, trong đó tại mỗi thời điểm cửa sổ $t$, mô hình trả về một vectơ xác suất của $C$ lớp từ:
$$P_t = [p_{t,1}, p_{t,2}, \dots, p_{t,C}] \quad \text{với} \quad \sum_{c=1}^{C} p_{t,c} = 1.0$$

---

### Lớp 1: Bộ lọc ổn định hóa bằng hàng đợi bỏ phiếu đa số (Majority Voting Bag - CVPR 2024)

Khi cửa sổ trượt di chuyển với bước nhảy $S = 5$ khung hình, sự chồng lấn thông tin giữa các cửa sổ kề nhau là rất lớn. Tại các vùng chuyển tiếp của tay, mô hình dễ bị phân vân giữa các nhãn tương đồng, dẫn đến hiện tượng nháy nhãn liên tục (flickering).

1.  **Xác định nhãn thô của cửa sổ ($L_t$):**
    Tại cửa sổ $t$, nhãn dự đoán thô $L_t$ được xác định dựa trên điểm số tin cậy tối đa:
    $$L_t = \begin{cases} \text{argmax}_{c} (p_{t,c}) & \text{nếu} \quad \max(P_t) \ge \text{CONF\_THRESHOLD} \quad \text{và} \quad E_t \ge \text{ENERGY\_THRESHOLD} \\ \text{None} (\emptyset) & \text{ngược lại} \end{cases}$$
    *(Trong đó $E_t$ là năng lượng chuyển động tay của cửa sổ $t$, giúp lọc bỏ vùng tĩnh).*

2.  **Cơ chế Bỏ phiếu đa số (Voting):**
    Hệ thống duy trì một hàng đợi (Queue Bag) chứa kết quả của $B = 7$ cửa sổ gần nhất: $\mathbf{B}_t = [L_{t-B+1}, \dots, L_t]$. Nhãn đồng thuận $V_t$ của cửa sổ $t$ được quyết định bởi số phiếu quá bán:
    $$V_t = \begin{cases} l^* & \text{nếu} \quad \text{Count}(l^*, \mathbf{B}_t) \ge \text{MIN\_VOTES} \quad \text{với} \quad l^* = \text{argmax}_{l \neq \emptyset} \text{Count}(l, \mathbf{B}_t) \\ \text{None} & \text{ngược lại} \end{cases}$$
    *(Với cấu hình $B = 7$, `MIN_VOTES` được đặt bằng $4$. Nếu không có nhãn nào đạt tối thiểu 4 phiếu, hệ thống trả về nhãn trống `None`).*

---

### Lớp 2: Bộ khóa khoảng trống sụt giảm xác suất (Temporal Gap Suppression - ESWA 2024)

Người ký không chuyên thường có xu hướng ký ngập ngừng hoặc không đều tay. Điều này khiến đồ thị xác suất của từ đang thực hiện bị sụt giảm tạm thời dưới ngưỡng tin cậy trong vài khung hình (trả về nhãn `None`), trước khi tăng trở lại trên ngưỡng. 

Để giải quyết, hệ thống thiết lập một **ngưỡng khóa khoảng tĩnh** `MAX_GAP_WINDOWS = 3` cửa sổ (tương đương khoảng 15 khung hình hay 0.5 giây):

*   **Cơ chế chờ (Pending State):** Khi chuỗi nhãn đang nhận diện từ $A$ đột ngột chuyển sang `None`, hệ thống chưa đóng từ ngay lập tức mà đưa phân đoạn vào trạng thái chờ và bắt đầu đếm số cửa sổ tĩnh: $G_{counter}$.
*   **Kích hoạt Khóa khoảng trống (Gap Lock):**
    *   Nếu nhãn $A$ xuất hiện trở lại trước khi $G_{counter} > \text{MAX\_GAP\_WINDOWS}$, hệ thống sẽ **khóa khoảng trống**, coi toàn bộ các cửa sổ tĩnh ở giữa là một phần chuyển động chuyển tiếp trong tiến trình ký từ $A$ và gộp chúng lại vào cùng một phân đoạn.
    *   Nếu $G_{counter} > \text{MAX\_GAP\_WINDOWS}$, hệ thống xác nhận người ký đã hoàn tất từ và chính thức đóng phân đoạn của từ $A$.
    *   Nếu một từ mới $B \neq A$ xuất hiện trong thời gian chờ, hệ thống sẽ **lập tức đóng từ $A$** và mở phân đoạn mới cho $B$ để bảo toàn ranh giới cứng giữa hai từ khác nhau.

---

### Lớp 3: Bộ phân tách ranh giới Đỉnh - Thung lũng (Peak-Valley Boundary Segmentation - ESWA 2024)

Lớp lọc thứ ba đảm nhận nhiệm vụ định vị chính xác tâm cử chỉ và giải quyết bài toán: **Làm sao phân biệt giữa một cử chỉ ký rất chậm (độ dài kéo dài) với việc người dùng thực sự muốn ký lặp lại từ đó 2 lần kề nhau?**

#### 1. Định vị đỉnh cử chỉ (Peak Detection)
Đối với một phân đoạn cử chỉ $A$ kéo dài từ cửa sổ thứ $start$ đến cửa sổ thứ $end$, hệ thống dò tìm **Đỉnh cực đại địa phương (Local Maximum)**:
$$t_{peak} = \text{argmax}_{i \in [start, end]} (p_{i, A})$$
Thời điểm khung hình tại trung tâm của cửa sổ $t_{peak}$ được đánh mốc là **Tâm cử chỉ (Center of Gesture)** - nơi cử chỉ đạt độ rõ nét và chuẩn xác cao nhất. Giá trị $p_{t_{peak}, A}$ được ghi nhận là độ tin cậy tối đa của phân đoạn.

#### 2. Phân tách thung lũng (Valley Split)
Nếu người dùng thực hiện lặp lại từ $A$ hai lần liên tiếp (ví dụ ký từ "bạn bạn" để nhấn mạnh), tay của họ sẽ di chuyển về trạng thái trung gian giữa hai lần ký. Lúc này, đường cong xác suất của nhãn $A$ sẽ xuất hiện hai đỉnh (Peak) rõ rệt và bị ngăn cách bởi một điểm sụt giảm sâu ở giữa - gọi là **Thung lũng xác suất (Valley)**.

Hệ thống quét chuỗi xác suất của phân đoạn $[p_{start, A}, \dots, p_{end, A}]$ để tìm cửa sổ thung lũng $k$ thỏa mãn điều kiện cực tiểu địa phương:
$$p_{k, A} < p_{k-1, A} \quad \text{và} \quad p_{k, A} < p_{k+1, A}$$

Tại điểm thung lũng $k$ này, hệ thống tính toán độ sâu sụt giảm xác suất so với đỉnh cực đại:
$$\Delta = p_{t_{peak}, A} - p_{k, A}$$

*   Nếu $\Delta > \text{VALLEY\_THRESHOLD}$ (mặc định đặt là `0.30` theo thực nghiệm của bài báo ESWA), hệ thống xác nhận có hành vi lặp cử chỉ chủ ý và **tiến hành cắt đôi phân đoạn** tại vị trí $k$ thành hai phân đoạn từ $A$ độc lập.
*   Nếu $\Delta \le \text{VALLEY\_THRESHOLD}$, hệ thống coi đây chỉ là sự dao động xác suất nhỏ khi ký chậm và giữ nguyên làm một phân đoạn duy nhất.

---

## 📝 5. Giải thuật chuyển ngữ nâng cao (LLM & RAG Translation)

Pha sinh câu tự nhiên (NLG) chuyển dịch chuỗi nhãn viết hoa (Gloss) thành câu tiếng Việt hoàn chỉnh thông qua giải thuật:

```
[Chuỗi Gloss sạch] ──► [FAISS Index] ──► [Truy vấn top-3 câu tương tự] ──► [ vit5-base Finetuned + RAG ] ──► [Câu dịch]
```

*   **Finetuned VietAI/vit5-base:** Mô hình học máy dịch thuật dạng Sequence-to-Sequence được tinh chỉnh trực tiếp trên tập dữ liệu câu ghép VSL để tối ưu hóa khả năng sắp xếp lại trật tự từ (đặc thù ngữ pháp ngôn ngữ ký hiệu thường bị đảo ngược so với văn bản nói).
*   **Hạ tầng RAG (Retrieval-Augmented Generation):**
    *   Sử dụng PhoBERT để nhúng toàn bộ câu tiếng Việt chuẩn trong cơ sở dữ liệu thành các vectơ đặc trưng và lưu trữ vào chỉ mục của thư viện **FAISS** (Facebook AI Similarity Search).
    *   Khi có chuỗi Gloss đầu ra, FAISS sẽ truy vấn không gian vector để lấy ra 3 cặp câu tương đồng nhất đưa vào Prompt làm ngữ cảnh mồi (In-context learning), ép mô hình vit5 dịch chuẩn xác ngữ pháp Tiếng Việt và ngăn ngừa lỗi dịch sai nghĩa.

---

