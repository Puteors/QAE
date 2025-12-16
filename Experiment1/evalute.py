import json
import torch
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM # Dùng thư viện PEFT cho Qwen LoRA
import evaluate
from src.config import Config

# --- CẤU HÌNH ---
# Đường dẫn tới thư mục chứa adapter đã train (vd: ./results_qwen/final_model)
MODEL_PATH = "./results_qwen/checkpoint-100" 
TEST_FILE = "./data/test.json"
OUTPUT_FILE = "question_rouge_results_qwen.json"
# ----------------

def parse_questions_only(text_output):
    """
    Trích xuất câu hỏi từ chuỗi output raw.
    Input: "question: Q1? answer: A1 [SEP] question: Q2? answer: A2"
    Output: "Q1? Q2?"
    """
    questions = []
    # Tách các cặp bằng separator
    sep = Config.PAIR_SEP.strip()
    pairs = text_output.split(sep)
    
    q_tag = Config.Q_TAG.strip() # "question:"
    a_tag = Config.A_TAG.strip() # "answer:"
    
    for pair in pairs:
        pair = pair.strip()
        # Tìm vị trí tag question và answer
        # Lưu ý: Qwen có thể sinh thêm khoảng trắng, nên dùng find linh hoạt
        q_start = pair.find(q_tag)
        a_start = pair.find(a_tag)
        
        if q_start != -1 and a_start != -1 and q_start < a_start:
            # Cắt lấy phần text ở giữa question và answer
            q_text = pair[q_start + len(q_tag): a_start].strip()
            if q_text:
                questions.append(q_text)
                
    # Nối các câu hỏi lại thành 1 chuỗi để tính ROUGE
    return " ".join(questions)

def main():
    # 1. Load Metrics & Model
    print("Đang tải ROUGE...")
    rouge = evaluate.load("rouge")

    print(f"Đang tải Model Qwen từ: {MODEL_PATH}")
    
    # Load Tokenizer (từ base model name trong config để đảm bảo đúng)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model (Qwen + LoRA)
    # device_map="auto" sẽ tự động dùng GPU nếu có
    model = AutoPeftModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    model.eval()

    # 2. Load và Gom nhóm dữ liệu Test
    print(f"Đang đọc dữ liệu từ: {TEST_FILE}")
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Dictionary: Key = Context, Value = List các câu hỏi thật
    context_map = defaultdict(list)
    for item in raw_data:
        ctx = item.get('context', "").strip()
        quest = item.get('question', "").strip()
        if ctx and quest:
            context_map[ctx].append(quest)

    print(f"Tổng số đoạn văn (Context) cần đánh giá: {len(context_map)}")

    # 3. Chạy đánh giá
    predictions = []
    references = []
    details = []

    print("Bắt đầu sinh câu hỏi...")
    
    # Duyệt qua từng context
    for context, real_questions_list in tqdm(context_map.items()):
        
        # --- A. Tạo Prompt (Phải giống hệt lúc train trong dataset.py) ---
        input_text = f"{Config.QA_PREFIX}{context}\n\nResponse: "
        
        inputs = tokenizer(
            input_text, 
            return_tensors="pt",
            truncation=True,
            max_length=Config.MAX_SOURCE_LENGTH
        ).to(model.device)

        # --- B. Generate ---
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=Config.MAX_TARGET_LENGTH, # Dùng max_new_tokens cho CausalLM
                num_beams=1,           # Qwen thường chạy tốt với greedy search (beams=1)
                do_sample=False,       # Để đánh giá ổn định (deterministic), nên tắt sample
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # --- C. Xử lý Output (Quan trọng cho Qwen) ---
        # outputs chứa cả [Input IDs + Generated IDs]. Cần cắt bỏ Input.
        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_len:]
        
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # --- D. Trích xuất & So sánh ---
        pred_questions_str = parse_questions_only(raw_output)
        ref_questions_str = " ".join(real_questions_list)
        
        predictions.append(pred_questions_str)
        references.append(ref_questions_str)
        
        details.append({
            "context_snippet": context[:100] + "...",
            "generated_raw": raw_output,
            "extracted_questions": pred_questions_str,
            "ground_truth": ref_questions_str
        })

    # 4. Tính ROUGE
    print("\nĐang tính toán điểm ROUGE...")
    # Kiểm tra nếu predictions rỗng (model chưa học được gì) để tránh lỗi
    if not any(predictions):
        print("Cảnh báo: Model không sinh ra được câu hỏi nào hợp lệ!")
        results = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    else:
        results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

    print("\n" + "="*40)
    print("KẾT QUẢ ROUGE (Qwen - Questions Only)")
    print("="*40)
    print(f"ROUGE-1: {results['rouge1'] * 100:.2f}")
    print(f"ROUGE-2: {results['rouge2'] * 100:.2f}")
    print(f"ROUGE-L: {results['rougeL'] * 100:.2f}")
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