import json
import numpy as np
import os
from functools import partial
from transformers import (
    AutoTokenizer, 
    T5ForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from src.config import Config
from src.dataset import QAGenDataset
from src.metrics import compute_metrics

def main():
    # 1. Cấu hình đường dẫn
    model_path = "./results/final_model"  # Đường dẫn model đã train xong
    test_data_path = "./data/test.json"   # File test
    output_file = "test_results_rouge.json" # File kết quả đầu ra

    print(f"Đang tải model từ: {model_path}")
    
    # 2. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    # 3. Chuẩn bị Dataset (Tái sử dụng class QAGenDataset)
    print("Đang xử lý dữ liệu test...")
    test_dataset = QAGenDataset(test_data_path, tokenizer)
    
    # 4. Cấu hình tham số đánh giá (Inference Arguments)
    eval_args = Seq2SeqTrainingArguments(
        output_dir="./eval_temp",
        per_device_eval_batch_size=Config.BATCH_SIZE,
        predict_with_generate=True,  # BẮT BUỘC: Để model sinh text rồi mới tính ROUGE
        generation_max_length=Config.MAX_TARGET_LENGTH, # Độ dài tối đa khi sinh
        fp16=True, # Nếu có GPU
        remove_unused_columns=False
    )

    # 5. Data Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 6. Khởi tạo Trainer (Chỉ để dùng hàm predict)
    # Dùng partial để truyền tokenizer vào hàm compute_metrics có sẵn của bạn
    compute_metrics_func = partial(compute_metrics, tokenizer=tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        data_collator=data_collator,
        processing_class=tokenizer, # Thay thế tokenizer=tokenizer để tránh warning
        compute_metrics=compute_metrics_func
    )

    # 7. Chạy dự đoán trên tập Test
    print("Đang chạy đánh giá (Inference)...")
    predict_results = trainer.predict(test_dataset)

    # 8. Lấy chỉ số ROUGE từ kết quả trả về
    metrics = predict_results.metrics
    
    # Xóa các key không cần thiết (như tốc độ chạy, bộ nhớ...)
    metrics = {k: v for k, v in metrics.items() if "rouge" in k}
    
    print("\n" + "="*30)
    print("KẾT QUẢ ROUGE TRÊN TẬP TEST")
    print("="*30)
    print(json.dumps(metrics, indent=4))
    print("="*30)

    # 9. Decode kết quả để lưu lại (Prediction vs Ground Truth)
    print("Đang giải mã kết quả để lưu file...")
    
    # Predictions (Model sinh ra)
    decoded_preds = tokenizer.batch_decode(
        predict_results.predictions, 
        skip_special_tokens=True
    )
    
    # Labels (Đáp án đúng)
    # Thay -100 bằng pad_token_id để decode không lỗi
    labels = np.where(
        predict_results.label_ids != -100, 
        predict_results.label_ids, 
        tokenizer.pad_token_id
    )
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 10. Lưu tất cả vào file JSON
    final_output = {
        "metrics": metrics,
        "samples": []
    }

    # Ghép từng cặp để dễ so sánh
    for pred, label in zip(decoded_preds, decoded_labels):
        final_output["samples"].append({
            "prediction": pred,
            "ground_truth": label
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)

    print(f"\nĐã lưu kết quả chi tiết vào: {output_file}")

if __name__ == "__main__":
    main()