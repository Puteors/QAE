import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5GemmaForConditionalGeneration
from src.config import Config

class QAGenerator:
    def __init__(self, model_path):
        # Thiết lập thiết bị (GPU nếu có)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Đang chạy trên thiết bị: {self.device}")

        print(f"Đang tải Tokenizer: {Config.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        
        print(f"Đang tải Model từ: {model_path}")
        self.model = T5GemmaForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def parse_output(self, text_output):
        """
        Chuyển đổi chuỗi text raw thành list dict JSON
        Input: "question: Q1 answer: A1 [SEP] question: Q2 answer: A2"
        """
        qa_list = []
        # Tách các cặp bằng token phân cách
        pairs = text_output.split(Config.PAIR_SEP.strip())
        
        for pair in pairs:
            try:
                # Tìm vị trí tag question và answer
                q_idx = pair.find(Config.Q_TAG.strip())
                a_idx = pair.find(Config.A_TAG.strip())
                
                if a_idx != -1 and q_idx != -1:
                    # Cắt chuỗi lấy nội dung
                    q_text = pair[q_idx + len(Config.Q_TAG.strip()): a_idx].strip()
                    a_text = pair[a_idx + len(Config.A_TAG.strip()):].strip()
                    
                    if q_text and a_text:
                        qa_list.append({
                            "question": q_text,
                            "answers": a_text
                        })
            except Exception:
                continue
        return qa_list

    def generate(self, context):
        input_text = Config.QA_PREFIX + context
        
        # Tokenize và đưa vào GPU
        inputs = self.tokenizer(
            input_text, 
            max_length=Config.MAX_SOURCE_LENGTH, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # Sinh văn bản
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=Config.MAX_TARGET_LENGTH,
                num_beams=4, # Beam search để câu văn mượt hơn
                early_stopping=True
            )
        
        # Decode kết quả
        decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.parse_output(decoded_text)

def main():
    # --- CẤU HÌNH ---
    input_file = "./data/test.json"       # File dữ liệu đầu vào
    model_path = "./results/final_model"  # Đường dẫn model đã train
    output_file = "predictions.json"      # Tên file kết quả sẽ lưu
    # ----------------

    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file {input_file}")
        return

    # 1. Khởi tạo Generator
    generator = QAGenerator(model_path)

    # 2. Đọc dữ liệu
    print(f"Đang đọc dữ liệu từ {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 3. Lọc Unique Context
    # Vì file test.json lặp lại context nhiều lần cho các câu hỏi khác nhau.
    # Ta chỉ cần chạy model 1 lần cho mỗi context.
    unique_data_map = {}
    for item in raw_data:
        ctx = item['context']
        if ctx not in unique_data_map:
            unique_data_map[ctx] = {
                "id": item['id'],       # Lưu ID đại diện
                "title": item.get('title', "")
            }
            
    print(f"Tổng số dòng trong file gốc: {len(raw_data)}")
    print(f"Số đoạn văn (context) cần xử lý: {len(unique_data_map)}")

    # 4. Chạy dự đoán (Inference)
    results = []
    
    # Dùng tqdm để hiển thị thanh tiến trình
    for context, info in tqdm(unique_data_map.items(), desc="Đang sinh câu hỏi"):
        
        # Gọi model để sinh Q&A
        generated_qa_list = generator.generate(context)
        
        # Lưu kết quả vào list
        results.append({
            "id": info['id'],
            "title": info['title'],
            "context": context,
            "generated_qa": generated_qa_list
        })

    # 5. Lưu ra file JSON
    print(f"Đang lưu kết quả vào {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print("Hoàn tất!")

if __name__ == "__main__":
    main()