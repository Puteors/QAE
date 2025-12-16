import torch
import json
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from src.config import Config

class QAGenerator:
    def __init__(self, model_path):
        print(f"Loading model from: {model_path}")
        
        # 1. Load Tokenizer
        # Nên load từ base model name trong Config để đảm bảo đúng vocabulary
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Load Model + LoRA Adapter
        # AutoPeftModelForCausalLM tự động load base model và merge adapter
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",      # Tự động dùng GPU
            torch_dtype="auto",     # Tự động chọn float16/bfloat16
            trust_remote_code=True
        )
        self.model.eval() # Chuyển sang chế độ inference

    def parse_output(self, text_output):
        """
        Chuyển đổi chuỗi text raw thành list json.
        Xử lý trường hợp text có thể bị lỗi format nhẹ.
        """
        qa_list = []
        
        # 1. Tách các cặp bằng [SEP]
        # Xóa khoảng trắng thừa ở separator để split chuẩn hơn
        sep = Config.PAIR_SEP.strip()
        pairs = text_output.split(sep)
        
        # Lấy tag chuẩn từ config và strip
        q_tag = Config.Q_TAG.strip() # "question:"
        a_tag = Config.A_TAG.strip() # "answer:"
        
        for pair in pairs:
            pair = pair.strip()
            if not pair: continue
            
            # Logic tách Question và Answer
            # Tìm vị trí của "answer:"
            a_idx = pair.find(a_tag)
            
            # Nếu tìm thấy tag answer
            if a_idx != -1:
                # Phần question nằm trước a_idx
                # Cần xóa tag "question:" ở đầu nếu có
                q_part = pair[:a_idx].strip()
                if q_part.lower().startswith(q_tag.lower()):
                    q_text = q_part[len(q_tag):].strip()
                else:
                    # Trường hợp model quên sinh tag question
                    q_text = q_part.strip()

                # Phần answer nằm sau a_idx + độ dài tag
                a_text = pair[a_idx + len(a_tag):].strip()
                
                if q_text and a_text:
                    qa_list.append({
                        "question": q_text,
                        "answers": a_text
                    })
                
        return qa_list

    def generate(self, context):
        # 1. Tạo Prompt (BẮT BUỘC GIỐNG LÚC TRAIN)
        # Trong dataset.py: prompt_text = f"{Config.QA_PREFIX}{item['context']}\n\nResponse: "
        input_text = f"{Config.QA_PREFIX}{context}\n\nResponse: "
        
        # 2. Tokenize & đưa lên GPU
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt",
            truncation=True,
            max_length=Config.MAX_SOURCE_LENGTH
        ).to(self.model.device)
        
        # 3. Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=Config.MAX_TARGET_LENGTH, # Chỉ định số từ mới tối đa được sinh ra
                num_beams=1,           # Qwen chat tốt với greedy search (hoặc tăng lên 4 nếu muốn)
                do_sample=True,        # Sampling giúp câu văn tự nhiên hơn
                temperature=0.7,       # Độ sáng tạo
                top_p=0.9,
                repetition_penalty=1.1, # Tránh lặp từ
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 4. Cắt bỏ phần Input (Chỉ lấy phần mới sinh ra)
        # outputs chứ cả [Input IDs + Generated IDs]
        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_len:]
        
        # 5. Decode
        decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # print("--- Raw Output ---")
        # print(decoded_text)
        # print("------------------")
        
        # 6. Parse thành JSON
        return self.parse_output(decoded_text)

if __name__ == "__main__":
    # Đường dẫn thư mục chứa model đã train (nơi có file adapter_config.json)
    model_path = "./results_qwen/final_model" 
    
    # Kiểm tra xem đường dẫn có tồn tại không
    import os
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy model tại {model_path}")
    else:
        try:
            generator = QAGenerator(model_path)
            
            sample_context = """Phạm Văn Đồng (1 tháng 3 năm 1906 – 29 tháng 4 năm 2000) là Thủ tướng đầu tiên của nước Cộng hòa Xã hội chủ nghĩa Việt Nam từ năm 1976. Ông có tên gọi thân mật là Tô."""
            
            print(f"\nContext: {sample_context}\n")
            print("Đang tạo câu hỏi...")
            
            result_json = generator.generate(sample_context)
            
            # In kết quả đẹp
            print(json.dumps(result_json, ensure_ascii=False, indent=4))
            
        except Exception as e:
            print(f"Có lỗi xảy ra: {e}")