import json
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        self.data = []
        for item in raw_data:
            # Lấy câu trả lời đầu tiên làm chuẩn
            if item['answers']['text']:
                self.data.append({
                    'context': item['context'],
                    'question': item['question'],
                    'answer_text': item['answers']['text'][0],
                    'answer_start': item['answers']['answer_start'][0]
                })

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class AEDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, max_len=512):
        super().__init__(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        start_char = item['answer_start']
        end_char = start_char + len(item['answer_text'])

        # Tokenize context có trả về offset mapping để biết token nào ứng với ký tự nào
        tokenized_inputs = self.tokenizer(
            context,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        input_ids = tokenized_inputs["input_ids"].squeeze()
        attention_mask = tokenized_inputs["attention_mask"].squeeze()
        offset_mapping = tokenized_inputs["offset_mapping"].squeeze().tolist()
        
        # Tạo nhãn: 0 (O), 1 (B - Bắt đầu), 2 (I - Bên trong)
        labels = [0] * len(input_ids)
        
        for i, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0: continue # Skip special tokens like CLS/SEP
            
            # Kiểm tra xem token này có nằm trọn trong vùng answer không
            if start >= start_char and end <= end_char:
                if start == start_char:
                    labels[i] = 1 # B-Answer
                else:
                    labels[i] = 2 # I-Answer
                    
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long)
        }

class QGDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, max_len_input=512, max_len_target=64):
        super().__init__(data_path)
        self.tokenizer = tokenizer
        self.max_len_input = max_len_input
        self.max_len_target = max_len_target

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        ans_text = item['answer_text']
        ans_start = item['answer_start']
        
        # Đánh dấu câu trả lời bằng thẻ <hl>
        # Cần cẩn thận cắt chuỗi chính xác
        prefix = context[:ans_start]
        suffix = context[ans_start + len(ans_text):]
        input_text = f"{prefix}<hl> {ans_text} <hl>{suffix}"
        
        target_text = item['question']

        inputs = self.tokenizer(
            input_text, 
            max_length=self.max_len_input, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        targets = self.tokenizer(
            target_text, 
            max_length=self.max_len_target, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )

        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": targets.input_ids.squeeze() # labels cho model tính loss
        }