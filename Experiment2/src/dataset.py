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
            if item.get('is_impossible', False):
                continue

            # 2. Xử lý sạch văn bản (Cleaning)
            # Thay thế các dấu gạch ngang lạ bằng dấu trừ bình thường
            context = item['context'].replace('–', '-').replace('—', '-')
            question = item['question'].replace('–', '-').replace('—', '-')
            
            # Lấy câu trả lời
            answer_text = ""
            if item['answers']['text']:
                answer_text = item['answers']['text'][0]
                answer_text = answer_text.replace('–', '-').replace('—', '-')
            
            if answer_text:
                grouped[context].append((question, answer_text))
        
        # Tạo dataset
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

        # Tokenize (trả về list int, không dùng pt tensor ở đây để tránh warning)
        inputs = self.tokenizer(
            input_text,
            max_length=Config.MAX_SOURCE_LENGTH,
            padding="max_length",
            truncation=True,
        )

        targets = self.tokenizer(
            target_text,
            max_length=Config.MAX_TARGET_LENGTH,
            padding="max_length",
            truncation=True,
        )

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": targets.input_ids
        }