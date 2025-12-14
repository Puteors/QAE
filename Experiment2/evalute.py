import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    T5ForConditionalGeneration, 
    DataCollatorForSeq2Seq
)
import evaluate
from src.config import Config
from src.dataset import QAGenDataset

def main():
    # 1. Cấu hình
    model_path = "./results/final_model"
    test_data_path = "./data/test.json"
    output_file = "test_results_rouge.json"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang sử dụng thiết bị: {device}")

    # 2. Load Model & Tokenizer
    print("Đang tải model...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # 3. Load Metrics
    rouge = evaluate.load("rouge")

    # 4. Load Dataset
    print("Đang tải dữ liệu test...")
    test_dataset = QAGenDataset(test_data_path, tokenizer)
    
    # Sử dụng DataCollator để tự động pad batch
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, # Hoặc giảm xuống nếu VRAM đầy (vd: 4 hoặc 2)
        shuffle=False, 
        collate_fn=data_collator
    )

    print("Bắt đầu sinh câu hỏi và đánh giá...")
    
    all_preds = []
    all_labels = []

    # 5. Vòng lặp dự đoán (Thủ công để tránh lỗi NoneType)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Chuyển dữ liệu sang GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Sinh văn bản (Generate)
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=Config.MAX_TARGET_LENGTH,
                num_beams=4,
                early_stopping=True
            )

            # Decode kết quả sinh ra
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Decode nhãn thật (Labels)
            # Thay -100 bằng pad_id để decode không lỗi
            labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_preds.extend(decoded_preds)
            all_labels.extend(decoded_labels)

    # 6. Tính ROUGE Score
    print("\nĐang tính điểm ROUGE...")
    result = rouge.compute(predictions=all_preds, references=all_labels, use_stemmer=True)

    # 7. In kết quả
    print("\n" + "="*30)
    print("KẾT QUẢ ĐÁNH GIÁ (ROUGE)")
    print("="*30)
    print(f"ROUGE-1: {result['rouge1']:.4f}")
    print(f"ROUGE-2: {result['rouge2']:.4f}")
    print(f"ROUGE-L: {result['rougeL']:.4f}")
    print("="*30)

    # 8. Lưu file JSON chi tiết
    final_output = {
        "metrics": result,
        "samples": []
    }

    # Lưu 100 mẫu đầu tiên vào file để kiểm tra (lưu hết sẽ rất nặng nếu data lớn)
    for p, l in zip(all_preds, all_labels):
        final_output["samples"].append({
            "prediction": p,
            "ground_truth": l
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    
    print(f"Đã lưu kết quả vào: {output_file}")

if __name__ == "__main__":
    main()