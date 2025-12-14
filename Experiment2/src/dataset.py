import json
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from src.config import Config

class QAGenDataset(Dataset):
    def __init__(self, json_path, tokenizer, split="train"):
        self.tokenizer = tokenizer
        self.data = self.load_and_group_data(json_path)
        
    def load_and_group_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Gom nhóm theo context
        grouped_data = defaultdict(list)
        for item in raw_data:
            context = item['context']
            question = item['question']
            
            # Xử lý lấy câu trả lời
            answer_text = ""
            if not item['is_impossible']:
                if item['answers']['text']:
                    answer_text = item['answers']['text'][0]
            elif item.get('plausible_answers'):
                answer_text = item['plausible_answers']['text'][0]
            
            # Chỉ thêm nếu có câu trả lời
            if answer_text:
                # Format: "hỏi: ... đáp: ..."
                pair = f"hỏi: {question} đáp: {answer_text}"
                grouped_data[context].append(pair)
        
        # Chuyển về list các mẫu training
        dataset = []
        for context, qa_pairs in grouped_data.items():
            # Nối các cặp Q&A bằng dấu gạch đứng hoặc token đặc biệt
            target_text = Config.SEP_TOKEN.join(qa_pairs)
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
        inputs = self.tokenizer(
            input_text,
            max_length=Config.MAX_SOURCE_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize Output (Target)
        targets = self.tokenizer(
            target_text,
            max_length=Config.MAX_TARGET_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": targets.input_ids.squeeze()
        }