# src/dataset.py
import json
from torch.utils.data import Dataset
from collections import defaultdict
from src.config import Config

class QAGenDataset(Dataset):
    def __init__(self, json_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = self.load_and_group_data(json_path)
        
    def load_and_group_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        grouped = defaultdict(list)
        for item in raw_data:
            # 1. Bỏ qua các câu hỏi "gài bẫy" (không có câu trả lời thật)
            # Dòng này giúp loại bỏ mẫu dữ liệu số 4 (is_impossible=True)
            if item.get('is_impossible', False):
                continue

            # 2. Xử lý sạch văn bản (Cleaning)
            context = item['context'].replace('–', '-').replace('—', '-')
            question = item['question'].replace('–', '-').replace('—', '-')
            
            # 3. Lấy câu trả lời AN TOÀN (Sửa lỗi crash tại đây)
            answer_text = ""
            answers = item.get('answers') # Lấy object answers ra
            
            # Kiểm tra kỹ: 
            # - answers phải khác None (để tránh lỗi NoneType)
            # - answers phải có key 'text'
            # - list answers['text'] phải có phần tử (len > 0)
            if answers and answers.get('text') and len(answers['text']) > 0:
                raw_ans = answers['text'][0]
                answer_text = raw_ans.replace('–', '-').replace('—', '-')
            
            # Chỉ thêm vào nếu tìm được câu trả lời hợp lệ
            if answer_text:
                grouped[context].append((question, answer_text))
        
        # Tạo dataset định dạng chuỗi
        dataset = []
        for context, qa_list in grouped.items():
            pair_strings = []
            for q, a in qa_list:
                pair_str = f"{Config.Q_TAG}{q}{Config.A_TAG}{a}"
                pair_strings.append(pair_str)
            
            target_text = Config.PAIR_SEP.join(pair_strings)
            
            dataset.append({
                "context": context,
                "target": target_text
            })
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = Config.QA_PREFIX + item['context']
        target_text = item['target']

        # Tokenize Input
        # Lưu ý: Không dùng return_tensors="pt" ở đây để tránh warning DataCollator
        inputs = self.tokenizer(
            input_text,
            max_length=Config.MAX_SOURCE_LENGTH,
            padding="max_length",
            truncation=True,
        )

        # Tokenize Output (Target)
        targets = self.tokenizer(
            target_text,
            max_length=Config.MAX_TARGET_LENGTH,
            padding="max_length",
            truncation=True,
        )

        return {
            "input_ids": inputs.input_ids,          # Trả về list int
            "attention_mask": inputs.attention_mask,# Trả về list int
            "labels": targets.input_ids             # Trả về list int
        }