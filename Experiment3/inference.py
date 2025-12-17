# src/inference.py
import re
import os
import torch
from transformers import AutoTokenizer, T5GemmaForConditionalGeneration
from peft import PeftModel, PeftConfig
from src.config import Config


class QAGenerator:
    def __init__(self, model_path, merge_weights=False):
        """
        Khởi tạo QA Generator.
        
        Args:
            model_path: Đường dẫn tới model (LoRA adapter hoặc full model)
            merge_weights: Nếu True, merge LoRA weights vào base model (nhanh hơn khi inference)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = self._load_model(model_path, merge_weights)
        self.model.eval()
        print(f"✅ Model loaded on: {self.device}")

    def _load_model(self, model_path, merge_weights):
        """Load model - tự động detect PEFT hoặc full model"""
        
        # Kiểm tra xem có phải PEFT model không
        is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        
        if is_peft:
            print("🔧 Đang load PEFT/LoRA model...")
            
            # Load PEFT config để lấy base model name
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_name = peft_config.base_model_name_or_path
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load base model
            base_model = T5GemmaForConditionalGeneration.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Load LoRA adapter
            model = PeftModel.from_pretrained(base_model, model_path)
            
            # Merge weights nếu cần (inference nhanh hơn)
            if merge_weights:
                print("🔀 Đang merge LoRA weights...")
                model = model.merge_and_unload()
                print("✅ Đã merge weights thành công!")
            
        else:
            print("📦 Đang load Full model...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load full model
            model = T5GemmaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        return model, tokenizer

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
            pair = pair.strip()
            if not pair:
                continue
                
            # 2. Dùng Regex hoặc find để tách question và answer
            try:
                # Tìm vị trí của tag answer
                a_idx = pair.find(Config.A_TAG.strip())
                q_idx = pair.find(Config.Q_TAG.strip())
                
                if a_idx != -1 and q_idx != -1:
                    # Cắt chuỗi
                    q_text = pair[q_idx + len(Config.Q_TAG.strip()): a_idx].strip()
                    a_text = pair[a_idx + len(Config.A_TAG.strip()):].strip()
                    
                    if q_text and a_text:  # Chỉ thêm nếu cả 2 không rỗng
                        qa_list.append({
                            "question": q_text,
                            "answers": a_text
                        })
            except Exception as e:
                continue
                
        return qa_list

    def generate(self, context, num_beams=4, max_length=None, num_return_sequences=1):
        """
        Generate câu hỏi và câu trả lời từ context.
        
        Args:
            context: Đoạn văn bản đầu vào
            num_beams: Số beams cho beam search
            max_length: Độ dài tối đa output
            num_return_sequences: Số lượng kết quả trả về
            
        Returns:
            List các cặp Q&A dạng dict
        """
        if max_length is None:
            max_length = Config.MAX_TARGET_LENGTH
            
        input_text = Config.QA_PREFIX + context
        
        inputs = self.tokenizer(
            input_text, 
            max_length=Config.MAX_SOURCE_LENGTH, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                early_stopping=True,
                do_sample=False,
            )
        
        # Decode và parse
        if num_return_sequences == 1:
            decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self.parse_output(decoded_text)
        else:
            # Trả về nhiều kết quả
            results = []
            for output in outputs:
                decoded_text = self.tokenizer.decode(output, skip_special_tokens=True)
                results.append(self.parse_output(decoded_text))
            return results

    def generate_with_sampling(self, context, temperature=0.7, top_p=0.9, top_k=50):
        """
        Generate với sampling để tạo đa dạng hơn.
        """
        input_text = Config.QA_PREFIX + context
        
        inputs = self.tokenizer(
            input_text, 
            max_length=Config.MAX_SOURCE_LENGTH, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=Config.MAX_TARGET_LENGTH,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        
        decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.parse_output(decoded_text)

    def batch_generate(self, contexts, batch_size=8, num_beams=4):
        """
        Generate cho nhiều contexts cùng lúc.
        """
        all_results = []
        
        for i in range(0, len(contexts), batch_size):
            batch_contexts = contexts[i:i + batch_size]
            batch_inputs = [Config.QA_PREFIX + ctx for ctx in batch_contexts]
            
            inputs = self.tokenizer(
                batch_inputs,
                max_length=Config.MAX_SOURCE_LENGTH,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=Config.MAX_TARGET_LENGTH,
                    num_beams=num_beams,
                    early_stopping=True,
                )
            
            for output in outputs:
                decoded_text = self.tokenizer.decode(output, skip_special_tokens=True)
                all_results.append(self.parse_output(decoded_text))
        
        return all_results


def main():
    import json
    
    # ============== CẤU HÌNH ==============
    # Chọn 1 trong 2 đường dẫn:
    
    # Option 1: Load LoRA adapter
    model_path = "./results/lora_adapter"
    
    # Option 2: Load full model (nếu không dùng PEFT)
    # model_path = "./results/final_model"
    
    # Option 3: Load checkpoint cụ thể
    # model_path = "./results/checkpoint-500"
    
    # ============== KHỞI TẠO ==============
    generator = QAGenerator(
        model_path=model_path,
        merge_weights=True  # True để inference nhanh hơn
    )
    
    # ============== TEST ĐƠN ==============
    sample_context = """
    Phạm Văn Đồng (1 tháng 3 năm 1906 – 29 tháng 4 năm 2000) là Thủ tướng đầu tiên 
    của nước Cộng hòa Xã hội chủ nghĩa Việt Nam từ năm 1976. Ông có tên gọi thân mật là Tô.
    """
    
    print("=" * 50)
    print("📝 Context:")
    print(sample_context.strip())
    print("=" * 50)
    
    # Generate Q&A
    result_json = generator.generate(sample_context)
    
    print("\n🎯 Generated Q&A:")
    print(json.dumps(result_json, ensure_ascii=False, indent=4))
    
    # ============== TEST BATCH ==============
    print("\n" + "=" * 50)
    print("📚 Batch Generation Test:")
    
    contexts = [
        "Python là ngôn ngữ lập trình được tạo bởi Guido van Rossum vào năm 1991.",
        "Hà Nội là thủ đô của Việt Nam với hơn 8 triệu dân.",
    ]
    
    batch_results = generator.batch_generate(contexts)
    for i, (ctx, res) in enumerate(zip(contexts, batch_results)):
        print(f"\n--- Context {i+1} ---")
        print(f"Input: {ctx[:50]}...")
        print(f"Output: {json.dumps(res, ensure_ascii=False)}")


if __name__ == "__main__":
    main()