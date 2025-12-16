import evaluate
import numpy as np
from src.config import Config

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    
    # Decode token thành text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Thay thế -100 trong labels (do HuggingFace pad labels bằng -100)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Tính ROUGE
    # rougeLsum thường tốt cho task tóm tắt/tạo sinh
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
    }