import sys
import os
from functools import partial
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, # Dùng cho T5, Flan-T5
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, TaskType, get_peft_model

# Import module nội bộ
from src.config import Config
from src.dataset import QAGenDataset
from src.metrics import compute_metrics

def main():
    print(f"Loading configuration for model: {Config.MODEL_NAME}")

    # 1. Load Tokenizer & Model
    # trust_remote_code=True thường cần thiết cho các model mới như Qwen
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, trust_remote_code=True)
    
    # LƯU Ý: AutoModelForSeq2SeqLM dành cho Encoder-Decoder (T5). 
    # Nếu dùng Qwen (Decoder-only), code này có thể báo lỗi kiến trúc.
    model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME, trust_remote_code=True)

    # Cấu hình LoRA thống nhất từ Config
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, 
        inference_mode=False, 
        r=Config.LORA_R, 
        lora_alpha=Config.LORA_ALPHA, 
        lora_dropout=Config.LORA_DROPOUT
    )
    
    model = get_peft_model(model, peft_config)
    print("Trainable parameters:")
    model.print_trainable_parameters()

    # 2. Prepare Datasets
    # Sử dụng đường dẫn từ Config
    if os.path.exists(Config.TRAIN_FILE):
        train_dataset = QAGenDataset(Config.TRAIN_FILE, tokenizer)
    else:
        raise FileNotFoundError(f"Không tìm thấy file train tại: {Config.TRAIN_FILE}")

    if os.path.exists(Config.VAL_FILE):
        val_dataset = QAGenDataset(Config.VAL_FILE, tokenizer)
    else:
        print(f"Cảnh báo: Không tìm thấy file validation tại {Config.VAL_FILE}. Việc đánh giá sẽ bị bỏ qua hoặc lỗi.")
        val_dataset = None

    # 3. Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model
    )

    # 4. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        
        # --- CẤU HÌNH STEPS (Từ Config) ---
        eval_strategy="steps",
        eval_steps=Config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=Config.EVAL_STEPS,
        
        # --- CẤU HÌNH BEST MODEL (Từ Config) ---
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        save_total_limit=Config.SAVE_TOTAL_LIMIT,
        
        # --- HYPERPARAMETERS (Từ Config) ---
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        weight_decay=Config.WEIGHT_DECAY,
        num_train_epochs=Config.EPOCHS,
        
        # --- KHÁC ---
        predict_with_generate=True,
        fp16=True, # Đảm bảo GPU hỗ trợ FP16
        logging_dir=os.path.join(Config.OUTPUT_DIR, 'logs'),
        logging_steps=Config.LOGGING_STEPS,
        report_to="none" # Tắt wandb nếu không cần
    )

    # 5. Initialize Trainer
    compute_metrics_func = partial(compute_metrics, tokenizer=tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer, # Phiên bản mới dùng processing_class thay cho tokenizer
        data_collator=data_collator,
        compute_metrics=compute_metrics_func
    )

    # 6. Train
    print("Bắt đầu huấn luyện...")
    trainer.train()

    # 7. Save final model
    save_path = os.path.join(Config.OUTPUT_DIR, "final_model")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path) # Nên lưu cả tokenizer
    print(f"Đã lưu mô hình và tokenizer tại: {save_path}")

if __name__ == "__main__":
    main()