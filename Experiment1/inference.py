import re
from transformers import AutoTokenizer, T5ForConditionalGeneration
from src.config import Config
from peft import AutoPeftModelForCausalLM

class QAGenerator:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.model = AutoPeftModelForCausalLM.from_pretrained(model_path)
        self.model.eval()

    def parse_output(self, text_output):
        """
        Chuyển đổi chuỗi text raw thành list json mong muốn
        Input: "question: Ai là Tô? answer: PVĐ [SEP] question: Năm nào? answer: 1987"
        Output: [{'question': 'Ai là Tô?', 'answers': 'PVĐ'}, ...]
        """
        qa_list = []
        
        # 1. Tách các cặp bằng [SEP]
        pairs = text_output.split(Config.PAIR_SEP.strip())
        
        for pair in pairs:
            # 2. Dùng Regex hoặc find để tách question và answer
            # Pattern tìm: question: (nội dung) answer: (nội dung)
            try:
                # Tìm vị trí của tag answer
                a_idx = pair.find(Config.A_TAG.strip())
                q_idx = pair.find(Config.Q_TAG.strip())
                
                if a_idx != -1 and q_idx != -1:
                    # Cắt chuỗi
                    q_text = pair[q_idx + len(Config.Q_TAG.strip()): a_idx].strip()
                    a_text = pair[a_idx + len(Config.A_TAG.strip()):].strip()
                    
                    qa_list.append({
                        "question": q_text,
                        "answers": a_text
                    })
            except Exception as e:
                continue
                
        return qa_list

    def generate(self, context):
        input_text = Config.QA_PREFIX + context
        inputs = self.tokenizer(
            input_text, 
            max_length=Config.MAX_SOURCE_LENGTH, 
            truncation=True, 
            return_tensors="pt"
        )
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=Config.MAX_TARGET_LENGTH,
            num_beams=4,
            early_stopping=True
        )
        
        decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Gọi hàm parse để trả về đúng format JSON list
        return self.parse_output(decoded_text)

if __name__ == "__main__":
    # Đường dẫn model sau khi train xong
    model_path = "./results/final_model" 
    
    generator = QAGenerator(model_path)
    
    sample_context = """Phạm Văn Đồng (1 tháng 3 năm 1906 – 29 tháng 4 năm 2000) là Thủ tướng đầu tiên của nước Cộng hòa Xã hội chủ nghĩa Việt Nam từ năm 1976. Ông có tên gọi thân mật là Tô."""
    
    # Kết quả trả về sẽ là List JSON
    result_json = generator.generate(sample_context)
    
    # In ra để kiểm tra
    import json
    print(json.dumps(result_json, ensure_ascii=False, indent=4))