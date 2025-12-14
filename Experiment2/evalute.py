import json
import torch
import re
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, T5ForConditionalGeneration
import evaluate
from src.config import Config

# --- CẤU HÌNH ---
MODEL_PATH = "./results/final_model"
TEST_FILE = "./data/test.json"
OUTPUT_FILE = "question_rouge_results.json"
# ----------------

def parse_questions_only(text_output):
    """
    Hàm này chỉ trích xuất phần nội dung câu hỏi từ output của model.
    Input: "question: Biển nào giáp Cali? answer: TBD [SEP] question: Sacramento ở đâu? answer: ..."
    Output: "Biển nào giáp Cali? Sacramento ở đâu?"
    """
    questions = []
    # Tách các cặp bằng [SEP]
    pairs = text_output.split(Config.PAIR_SEP.strip())
    
    for pair in pairs:
        # Tìm vị trí tag 'question:' và 'answer:'
        q_start = pair.find(Config.Q_TAG.strip())
        a_start = pair.find(Config.A_TAG.strip())
        
        if q_start != -1 and a_start != -1:
            # Cắt lấy phần text ở giữa question và answer
            # + len(...) để bỏ qua chữ "question: "
            q_text = pair[q_start + len(Config.Q_TAG.strip()): a_start].strip()
            if q_text:
                questions.append(q_text)
                
    # Nối các câu hỏi lại thành 1 chuỗi để tính ROUGE
    return " ".join(questions)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị: {device}")

    # 1. Load Metrics & Model
    print("Đang tải ROUGE và Model...")
    rouge = evaluate.load("rouge")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    # 2. Load và Gom nhóm dữ liệu Test (Group by Context)
    print("Đang xử lý dữ liệu test...")
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Dictionary: Key = Context, Value = List các câu hỏi thật (Ground Truth)
    context_map = defaultdict(list)
    
    for item in raw_data:
        ctx = item.get('context', "").strip()
        quest = item.get('question', "").strip()
        
        # Chỉ lấy nếu có context và question (bỏ qua answers)
        if ctx and quest:
            context_map[ctx].append(quest)

    print(f"Tổng số đoạn văn (Context) cần đánh giá: {len(context_map)}")

    # 3. Chạy đánh giá
    predictions = [] # List các chuỗi câu hỏi model sinh ra
    references = []  # List các chuỗi câu hỏi thật
    
    details = [] # Lưu chi tiết để kiểm tra

    print("Bắt đầu sinh câu hỏi và so sánh...")
    for context, real_questions_list in tqdm(context_map.items()):
        
        # --- A. Model sinh text ---
        input_text = Config.QA_PREFIX + context
        inputs = tokenizer(
            input_text, 
            max_length=Config.MAX_SOURCE_LENGTH, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=Config.MAX_TARGET_LENGTH,
                num_beams=4,
                early_stopping=True
            )
        
        # Output thô (chứa cả Q và A)
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # --- B. Trích xuất chỉ câu hỏi (Prediction) ---
        pred_questions_str = parse_questions_only(raw_output)
        
        # --- C. Chuẩn bị câu hỏi thật (Reference) ---
        # Nối tất cả câu hỏi thật lại thành 1 chuỗi
        ref_questions_str = " ".join(real_questions_list)
        
        # Lưu lại để tính điểm
        predictions.append(pred_questions_str)
        references.append(ref_questions_str)
        
        details.append({
            "context_snippet": context[:100] + "...",
            "generated_questions": pred_questions_str,
            "real_questions": ref_questions_str
        })

    # 4. Tính ROUGE
    print("\nĐang tính toán điểm ROUGE cho Question...")
    results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

    print("\n" + "="*40)
    print("KẾT QUẢ ROUGE (CHỈ TÍNH CÂU HỎI)")
    print("="*40)
    print(f"ROUGE-1: {results['rouge1']:.4f}")
    print(f"ROUGE-2: {results['rouge2']:.4f}")
    print(f"ROUGE-L: {results['rougeL']:.4f}")
    print("="*40)

    # 5. Lưu kết quả
    output_data = {
        "metrics": results,
        "details": details
    }
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"Đã lưu kết quả chi tiết vào: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()