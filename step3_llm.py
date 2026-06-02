import os
import sys
import io
import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Đồng bộ hiển thị tiếng Việt trên Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── Cấu hình đường dẫn ────────────────────────────────────────────────────────
BASE_DIR      = r"C:\Users\MYPC\Downloads\Dataset"
CORPUS_PATH   = os.path.join(BASE_DIR, "Labels", "vsl_corpus.json")
MODEL_DIR     = os.path.join(BASE_DIR, "Models")
EVAL_DIR      = os.path.join(BASE_DIR, "Evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

VIT5_RAG_MODEL_PATH = os.path.join(MODEL_DIR, "vit5_rag_best")

# Thư viện toàn cục để tránh load lại mô hình nhiều lần
_tokenizer = None
_model = None

# ══════════════════════════════════════════════════════════════════════════
# PHẦN A — NẠP VÀ PHÂN CHIA TẬP DỮ LIỆU CÂU SONG SONG (CORPUS)
# ══════════════════════════════════════════════════════════════════════════
# Dữ liệu dự phòng nếu chưa có file JSON
FALLBACK_CORPUS = [
    {"gloss": "XIN CHÀO BẠN", "text": "Xin chào bạn."},
    {"gloss": "TÔI TÊN LÀ NAM", "text": "Tôi tên là Nam."},
    {"gloss": "BẠN KHOẺ KHÔNG", "text": "Bạn có khoẻ không?"},
    {"gloss": "CẢM ƠN BẠN GIÚP ĐỠ", "text": "Cảm ơn bạn đã giúp đỡ tôi."},
    {"gloss": "TÔI MUỐN ĂN CƠM", "text": "Tôi muốn ăn cơm."},
    {"gloss": "GIA ĐÌNH BỐ MẸ TÔI", "text": "Đây là gia đình bố mẹ của tôi."},
    {"gloss": "BẠN ĐI ĐÂU ĐẤT", "text": "Bạn đang đi đâu thế?"},
    {"gloss": "XIN LỖI TÔI NHẦM KHÔNG HIỂU", "text": "Xin lỗi, tôi bị nhầm nên không hiểu."},
    {"gloss": "TÔI YÊU BẠN BỐ MẸ ÔNG BÀ BÌNH AN", "text": "Tôi yêu bạn và chúc bố mẹ, ông bà luôn bình an."},
    {"gloss": "CHÀO BUỔI SÁNG ÔNG BÀ SỨC KHỎE BÌNH AN", "text": "Con xin chào buổi sáng ông bà, chúc ông bà luôn sức khỏe và bình an."}
]

if os.path.exists(CORPUS_PATH):
    print(f"📂 [CORPUS] Đang nạp dữ liệu huấn luyện: {CORPUS_PATH}", flush=True)
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        VSL_CORPUS = json.load(f)
    print(f"✅ Đã nạp thành công {len(VSL_CORPUS)} cặp câu song song.", flush=True)
else:
    print(f"⚠ Không tìm thấy {CORPUS_PATH}. Sử dụng FALLBACK_CORPUS.", flush=True)
    VSL_CORPUS = FALLBACK_CORPUS

# PHÂN CHIA TẬP TRONG/NGOÀI ĐỂ ĐÁNH GIÁ CHUẨN KHOA HỌC (80% Train / 20% Test)
train_corpus, test_corpus = train_test_split(VSL_CORPUS, test_size=0.2, random_state=42)
print(f"📊 Phân chia tập huấn luyện: {len(train_corpus)} câu | Tập kiểm thử độc lập: {len(test_corpus)} câu", flush=True)

# ══════════════════════════════════════════════════════════════════════════
# PHẦN B — ĐỘNG CƠ TRUY VẤN RAG (TF-IDF + COSINE RETRIEVAL)
# ══════════════════════════════════════════════════════════════════════════
class SignLanguageRAG:
    def __init__(self, corpus: List[Dict[str, str]]):
        self.corpus = corpus
        self.gloss_list = [item["gloss"] for item in corpus]
        self.vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.gloss_list)

    def retrieve(self, query_gloss: str, top_k: int = 2, skip_exact_match: bool = False) -> List[Dict[str, str]]:
        """Truy vấn lấy các cặp câu Gloss tương đồng từ tập Corpus"""
        query_vector = self.vectorizer.transform([query_gloss])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            is_exact = self.gloss_list[idx].strip().upper() == query_gloss.strip().upper()
            # Bỏ chính nó khi chạy huấn luyện/đánh giá chéo, nhưng giữ lại khi chạy thực tế (Inference)
            if skip_exact_match and is_exact:
                continue
            results.append(self.corpus[idx])
            if len(results) == top_k:
                break
        return results

    def build_prompt(self, query_gloss: str, top_k: int = 2, skip_exact_match: bool = False) -> str:
        refs = self.retrieve(query_gloss, top_k=top_k, skip_exact_match=skip_exact_match)
        prompt_parts = []
        for i, ref in enumerate(refs):
            prompt_parts.append(f"Ref {i+1}: {ref['gloss']} -> {ref['text']}")
        prompt_parts.append(f"Input: {query_gloss} ->")
        return " | ".join(prompt_parts)

# Động cơ RAG chỉ được phép tìm kiếm dữ liệu tham khảo trong tập Train_Corpus
rag_engine = SignLanguageRAG(train_corpus)

# ══════════════════════════════════════════════════════════════════════════
# PHẦN C — LAZY LOADING MODEL (NẠP MỘT LẦN DUY NHẤT)
# ══════════════════════════════════════════════════════════════════════════
def get_model_and_tokenizer(use_gpu: bool = True):
    """Nạp mô hình vào RAM/VRAM một lần duy nhất khi khởi chạy"""
    global _tokenizer, _model
    import torch
    
    if _model is None or _tokenizer is None:
        load_path = VIT5_RAG_MODEL_PATH if os.path.exists(VIT5_RAG_MODEL_PATH) else "VietAI/vit5-base"
        print(f"⏳ [LAZY LOAD] Đang nạp mô hình dịch thuật từ: {load_path}...", flush=True)
        
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _tokenizer = AutoTokenizer.from_pretrained(load_path)
        _model = AutoModelForSeq2SeqLM.from_pretrained(load_path)
        
        device = "cuda" if (torch.cuda.is_available() and use_gpu) else "cpu"
        _model.to(device)
        print(f"✅ [LAZY LOAD] Nạp thành công mô hình lên thiết bị: {device.upper()}", flush=True)
        
    return _model, _tokenizer

# ══════════════════════════════════════════════════════════════════════════
# PHẦN D — QUY TRÌNH HUẤN LUYỆN (FINETUNING TRÊN GPU/CPU)
# ══════════════════════════════════════════════════════════════════════════
def train_vit5_rag():
    """Finetuning mô hình VietAI/vit5-base cấu trúc Prompt RAG"""
    print("\n⚡ [HUẤN LUYỆN RAG] Khởi động quá trình Finetuning VietAI/vit5-base...")
    
    try:
        from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, 
                                  Seq2SeqTrainer, Seq2SeqTrainingArguments, 
                                  DataCollatorForSeq2Seq)
        from datasets import Dataset
        import torch
    except ImportError:
        print("❌ Thiếu thư viện transformers hoặc datasets.")
        return

    # 1. Chỉ dùng dữ liệu Train_Corpus để huấn luyện
    train_data = []
    for item in train_corpus:
        # Bắt buộc skip_exact_match=True khi train để mô hình không học vẹt
        prompt = rag_engine.build_prompt(item["gloss"], top_k=2, skip_exact_match=True)
        train_data.append({
            "input_text": prompt,
            "target_text": item["text"]
        })
    
    df_train = pd.DataFrame(train_data)
    dataset = Dataset.from_pandas(df_train)

    # 2. Tải mô hình nền
    model_name = "VietAI/vit5-base"
    print(f"   Đang nạp mô hình nền: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"   Môi trường huấn luyện thiết bị: {device.upper()}")

    # 3. Tiền xử lý dữ liệu
    def preprocess_function(examples):
        model_inputs = tokenizer(examples["input_text"], max_length=256, truncation=True)
        labels = tokenizer(text_target=examples["target_text"], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 4. Thiết lập tham số (Giảm xuống 30 epochs để tránh overfitting)
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "vit5_rag_checkpoints"),
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=30,  # Tối ưu hóa 30 Epochs
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("🚀 Bắt đầu quá trình Finetuning...")
    trainer.train()

    # Lưu mô hình đã tối ưu hóa
    model.save_pretrained(VIT5_RAG_MODEL_PATH)
    tokenizer.save_pretrained(VIT5_RAG_MODEL_PATH)
    print(f"✅ HOÀN TẤT HUẤN LUYỆN! Đã lưu mô hình tại: {VIT5_RAG_MODEL_PATH}")

# ══════════════════════════════════════════════════════════════════════════
# PHẦN E — DỊCH THUẬT THỜI GIAN THỰC (INFERENCE)
# ══════════════════════════════════════════════════════════════════════════
def translate_gloss_to_text(gloss_list: List[str], use_gpu: bool = True, skip_exact_match: bool = False) -> Tuple[str, float]:
    """
    Dịch chuỗi cử chỉ sang câu văn Tiếng Việt bằng mô hình Finetuned Vit5 + RAG (Lazy-Loaded)
    """
    t0 = time.perf_counter()
    query_gloss = " ".join(gloss_list).strip().upper()
    
    # 1. Truy vấn RAG dựng Prompt gợi ý ngữ cảnh
    prompt = rag_engine.build_prompt(query_gloss, top_k=2, skip_exact_match=skip_exact_match)

    # 2. Khởi chạy sinh câu bằng mô hình Vit5 đã lưu trữ trên RAM/VRAM
    try:
        import torch
        model, tokenizer = get_model_and_tokenizer(use_gpu=use_gpu)
        
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
            
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        latency = time.perf_counter() - t0
        return translated_text, latency
        
    except Exception as e:
        fallback_text = query_gloss.capitalize() + "."
        latency = time.perf_counter() - t0
        return f"[Dự phòng] {fallback_text} (Chi tiết lỗi: {e})", latency

# ══════════════════════════════════════════════════════════════════════════
# PHẦN F — ĐÁNH GIÁ CHẤT LƯỢNG DỊCH THUẬT TRÊN TẬP TEST ĐỘC LẬP
# ══════════════════════════════════════════════════════════════════════════
def run_translation_evaluation():
    """Đo lường điểm BLEU trên tập TEST độc lập chưa từng xuất hiện lúc huấn luyện"""
    print("\n🔬 KHỞI CHẠY ĐÁNH GIÁ CHẤT LƯỢNG DỊCH THUẬT TRÊN TẬP TEST NGOÀI...")
    
    hypotheses, references = [], []
    latencies = []

    # Tiến hành chạy thử nghiệm dịch thuật trên tập test_corpus (20% dữ liệu)
    for item in test_corpus:
        gloss_list = item["gloss"].split()
        ref_text   = item["text"]
        
        # Bắt buộc đặt skip_exact_match=True khi đánh giá để kiểm thử thực lực dịch thuật của mô hình
        pred_text, latency = translate_gloss_to_text(gloss_list, use_gpu=True, skip_exact_match=True)
        
        hypotheses.append(pred_text)
        references.append(ref_text)
        latencies.append(latency)
        
        print(f"  [Gloss VSL]: {item['gloss']}")
        print(f"   -> Gốc chuẩn: {ref_text}")
        print(f"   -> Bản dịch : {pred_text} ({latency*1000:.0f}ms)\n")

    # Tính toán chỉ số BLEU-1, BLEU-2, BLEU-4
    bleu_results = {}
    try:
        import sacrebleu
        for n in [1, 2, 4]:
            score = sacrebleu.corpus_bleu(hypotheses, [references], max_ngram_order=n).score
            bleu_results[f"BLEU-{n}"] = round(score, 2)
    except ImportError:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        sf = SmoothingFunction().method1
        hyps_tok = [h.split() for h in hypotheses]
        refs_tok = [[r.split()] for r in references]
        for n in [1, 2, 4]:
            weights = tuple([1/n]*n + [0]*(4-n))
            score = corpus_bleu(refs_tok, hyps_tok, weights=weights, smoothing_function=sf)
            bleu_results[f"BLEU-{n}"] = round(score * 100, 2)

    avg_latency_ms = np.mean(latencies) * 1000
    print("📊 BẢNG ĐIỂM CHẤT LƯỢNG DỊCH THUẬT LAI (TABLE III - CHUẨN NGOÀI):")
    for k, v in bleu_results.items():
        print(f"   - {k:10s}: {v}%")
    print(f"   - Độ trễ trung bình: {avg_latency_ms:.1f} ms")

    # Ghi báo cáo ra file CSV
    df_eval = pd.DataFrame([bleu_results])
    df_eval["Avg_Latency_ms"] = round(avg_latency_ms, 1)
    df_eval.to_csv(os.path.join(EVAL_DIR, "bleu_scores.csv"), index=False)
    print(f"✅ Đã lưu kết quả đánh giá thực thực tại: {os.path.join(EVAL_DIR, 'bleu_scores.csv')}")

# ── ENTRY POINT ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="eval", choices=["train", "eval", "test_single"])
    args = parser.parse_args()

    if args.mode == "train":
        train_vit5_rag()
    elif args.mode == "eval":
        run_translation_evaluation()
    elif args.mode == "test_single":
        test_gloss = ["XIN", "CHÀO", "BẠN", "GIÚP", "ĐỠ"]
        ans, lat = translate_gloss_to_text(test_gloss)
        print(f"\n[Dịch thử mẫu] {test_gloss} -> {ans} ({lat*1000:.1f}ms)")