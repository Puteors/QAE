import evaluate
import numpy as np
from src.config import Config

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    
    # --- SỬA LỖI Ở ĐÂY ---
    # 1. Xử lý PREDICTIONS: Thay thế -100 bằng pad_token_id TRƯỚC khi decode
    # Nếu không làm bước này, batch_decode sẽ báo lỗi OverflowError
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # 2. Xử lý LABELS: Thay thế -100 bằng pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # (Tùy chọn) Làm sạch khoảng trắng thừa để tính điểm chính xác hơn
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # Tính ROUGE
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
    }