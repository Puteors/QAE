import json
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from src.config import Config

class QAGenDataset(Dataset):
    def __init__(self, json_path, tokenizer):
        self.tokenizer = tokenizer
        # Gọi hàm load dữ liệu
        self.data = self.load_and_group_data(json_path)
        
    def load_and_group_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 1. Gom nhóm các câu hỏi cùng context
        grouped = defaultdict(list)
        for item in raw_data:
            context = item.get('context', '')
            q = item.get('question', '')
            
            # Logic lấy answer giữ nguyên của bạn
            a = ""
            if not item.get('is_impossible', False) and item.get('answers') and item['answers'].get('text'):
                a = item['answers']['text'][0]
            elif item.get('is_impossible', False) and item.get('plausible_answers'):
                a = item['plausible_answers']['text'][0]
            
            if a:
                grouped[context].append((q, a))
        
        # 2. Tạo format training
        dataset = []
        for context, qa_list in grouped.items():
            # Tạo chuỗi target: "question: A answer: B [SEP] question: C answer: D"
            pair_strings = []
            for q, a in qa_list:
                pair_str = f"{Config.Q_TAG}{q}{Config.A_TAG}{a}"
                pair_strings.append(pair_str)
            
            target_text = Config.PAIR_SEP.join(pair_strings)
            
            dataset.append({
                "context": context,
                "target": target_text
            })
        
        print(f"Đã load {len(dataset)} mẫu dữ liệu từ {path}")
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Chuẩn bị Text (Giống ý bạn nhưng thêm format cho Qwen)
        # Input: "generate_qa: Nội dung bài đọc..."
        # Qwen Instruct cần format rõ ràng hơn chút, ví dụ thêm xuống dòng
        prompt_text = f"{Config.QA_PREFIX}{item['context']}\n\nResponse: "
        target_text = item['target'] # Chuỗi các câu hỏi/đáp án
        
        # 2. NỐI CHUỖI (BẮT BUỘC VỚI QWEN)
        # Full text = Prompt + Target + EOS Token
        full_text = prompt_text + target_text + self.tokenizer.eos_token
        
        # 3. Tokenize toàn bộ
        encoded = self.tokenizer(
            full_text,
            max_length=Config.MAX_SOURCE_LENGTH, # Qwen chỉ dùng 1 max length chung
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoded.input_ids[0]
        attention_mask = encoded.attention_mask[0]
        
        # 4. Tạo Labels (Masking phần Prompt)
        labels = input_ids.clone()
        
        # Tính độ dài của prompt để mask đi (không tính loss phần prompt)
        # Tokenize riêng prompt để đếm
        prompt_ids = self.tokenizer(
            prompt_text, 
            truncation=True, 
            max_length=Config.MAX_SOURCE_LENGTH,
            add_special_tokens=False # Cẩn thận với BOS token
        ).input_ids
        
        prompt_len = len(prompt_ids)
        
        # Nếu full text bị cắt ngắn (truncation), đảm bảo không lỗi index
        mask_len = min(prompt_len, len(labels))
        
        # Gán -100 để PyTorch bỏ qua khi tính Loss
        labels[:mask_len] = -100
        
        # Nếu padding="max_length", cần mask cả phần padding (token=0 hoặc pad_token_id)
        if self.tokenizer.pad_token_id is not None:
            labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,          
            "attention_mask": attention_mask,
            "labels": labels                 
        }