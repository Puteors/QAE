import torch
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
from src.dataset import AEDataset
from src.metrics import AEMetrics

def train_ae_model(train_path, val_path, test_path, output_dir="./outputs/ae_model"):
    model_checkpoint = "microsoft/mdeberta-v3-base"
    
    print(f"--- Loading Tokenizer: {model_checkpoint} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # Cấu hình nhãn
    id2label = {0: "O", 1: "B-ANS", 2: "I-ANS"}
    label2id = {"O": 0, "B-ANS": 1, "I-ANS": 2}
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint, 
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

    # Load 3 bộ dữ liệu
    print("--- Loading Datasets ---")
    train_dataset = AEDataset(train_path, tokenizer)
    val_dataset = AEDataset(val_path, tokenizer)
    test_dataset = AEDataset(test_path, tokenizer)
    
    # Metrics
    ae_metrics = AEMetrics()
    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",          # Kiểm tra trên tập Validation theo bước
        eval_steps=500,                 # 500 bước check 1 lần
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,             # Lưu 2 model tốt nhất
        load_best_model_at_end=True,    # Load model tốt nhất dựa trên Validation Loss
        metric_for_best_model="loss",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,       # Dùng Validation để tinh chỉnh
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=ae_metrics.compute
    )

    print("--- Bắt đầu Training AE ---")
    trainer.train()
    
    print("--- Lưu model tốt nhất ---")
    model.save_pretrained(f"{output_dir}/best")
    tokenizer.save_pretrained(f"{output_dir}/best")

    # Đánh giá trên tập TEST (Độc lập)
    print("--- Đánh giá trên tập TEST ---")
    test_results = trainer.evaluate(test_dataset)
    
    # Ghi log kết quả Test ra file
    with open(f"{output_dir}/test_results.json", "w") as f:
        json.dump(test_results, f, indent=4)
    
    print("Kết quả Test AE:", test_results)