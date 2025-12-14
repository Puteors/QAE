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
        
        # 1. Gom nhóm các câu hỏi cùng context
        grouped = defaultdict(list)
        for item in raw_data:
            context = item['context']
            q = item['question']
            
            # Lấy câu trả lời (ưu tiên text thật, hoặc plausible nếu impossible=True)
            a = ""
            if not item['is_impossible'] and item['answers']['text']:
                a = item['answers']['text'][0]
            elif item['is_impossible'] and item.get('plausible_answers'):
                a = item['plausible_answers']['text'][0]
            
            if a: # Chỉ lấy nếu có câu trả lời
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
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = Config.QA_PREFIX + item['context']
        target_text = item['target']

        # Tokenize Input
        # BỎ 'return_tensors="pt"' đi
        inputs = self.tokenizer(
            input_text,
            max_length=Config.MAX_SOURCE_LENGTH,
            padding="max_length",
            truncation=True,
        )

        # Tokenize Output
        # BỎ 'return_tensors="pt"' đi
        targets = self.tokenizer(
            target_text,
            max_length=Config.MAX_TARGET_LENGTH,
            padding="max_length",
            truncation=True,
        )

        return {
            "input_ids": inputs.input_ids,          # Đây là List[int]
            "attention_mask": inputs.attention_mask,# Đây là List[int]
            "labels": targets.input_ids             # Đây là List[int]
        }