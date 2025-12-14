import torch
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from src.dataset import QGDataset
from src.metrics import QGMetrics

def train_qg_model(train_path, val_path, test_path, output_dir="./outputs/qg_model"):
    model_checkpoint = "vinai/bartpho-word-base"
    
    print(f"--- Loading Tokenizer: {model_checkpoint} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<hl>']})
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model.resize_token_embeddings(len(tokenizer))

    print("--- Loading Datasets ---")
    train_dataset = QGDataset(train_path, tokenizer)
    val_dataset = QGDataset(val_path, tokenizer)
    test_dataset = QGDataset(test_path, tokenizer)
    
    qg_metrics = QGMetrics(tokenizer) 
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",          # Kiểm tra trên Validation
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,    # Load model validation loss thấp nhất
        metric_for_best_model="loss",   

        learning_rate=3e-5,
        per_device_train_batch_size=2,      # GIẢM TỪ 4 XUỐNG 2
        per_device_eval_batch_size=2,       # GIẢM TỪ 4 XUỐNG 2
        gradient_accumulation_steps=2,      # Tích lũy 2 bước (2x2 = 4 hiệu quả)

        num_train_epochs=10,
        weight_decay=0.01,
        predict_with_generate=True,     # Để tính BLEU/BERTScore
        fp16=torch.cuda.is_available(),
        logging_dir=f"{output_dir}/logs",
        logging_steps=100
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,       # Validation Set
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=qg_metrics.compute
    )

    print("--- Bắt đầu Training QG ---")
    trainer.train()
    
    print("--- Lưu model tốt nhất ---")
    model.save_pretrained(f"{output_dir}/best")
    tokenizer.save_pretrained(f"{output_dir}/best")

    # Đánh giá trên tập TEST
    print("--- Đánh giá trên tập TEST ---")
    # predict() sẽ chạy generate và trả về metrics + predictions
    test_results = trainer.predict(test_dataset)
    
    print("Kết quả Test QG (Metrics):", test_results.metrics)
    
    # Lưu kết quả test
    with open(f"{output_dir}/test_results.json", "w") as f:
        json.dump(test_results.metrics, f, indent=4)