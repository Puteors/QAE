import sys
import os
import torch
from functools import partial
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,   # <--- THAY ĐỔI: Dùng CausalLM cho Qwen
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, TaskType, get_peft_model

# Import nội bộ
from src.config import Config
from src.dataset import QAGenDataset
from src.metrics import compute_metrics

def main():
    print(f"Loading Qwen Model: {Config.MODEL_NAME}")

    # 1. Load Tokenizer
    # trust_remote_code=True cần thiết cho các dòng Qwen cũ hoặc custom
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, trust_remote_code=True)
    
    # CẤU HÌNH TOKENIZER CHO QWEN (Rất quan trọng)
    # Qwen không có pad_token mặc định, ta dùng eos_token làm pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Khi training CausalLM, padding side không quá quan trọng, nhưng để 'right' an toàn hơn với collator này
    tokenizer.padding_side = "right" 

    # 2. Load Model (Decoder-only)
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        trust_remote_code=True,
        dtype="auto",       # Tự động chọn float16 hoặc bfloat16 tùy GPU
        device_map="auto"         # Tự động chia model vào GPU
    )
    
    # Bật gradient checkpointing để tiết kiệm VRAM (giảm 50% VRAM)
    model.gradient_checkpointing_enable() 
    model.enable_input_require_grads()    # Cần thiết cho LoRA + Checkpointing

    # 3. LoRA Configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,   # <--- THAY ĐỔI: Task là CAUSAL_LM
        inference_mode=False, 
        r=Config.LORA_R, 
        lora_alpha=Config.LORA_ALPHA, 
        lora_dropout=Config.LORA_DROPOUT,
        target_modules=Config.LORA_TARGET_MODULES # <--- THAY ĐỔI: Target vào các lớp của Qwen
    )
    
    model = get_peft_model(model, peft_config)
    print("\nTham số huấn luyện (Trainable Parameters):")
    model.print_trainable_parameters()

    # 4. Prepare Datasets
    # Lưu ý: Class QAGenDataset của bạn cần đảm bảo trả về input_ids và labels
    if os.path.exists(Config.TRAIN_FILE):
        train_dataset = QAGenDataset(Config.TRAIN_FILE, tokenizer)
        val_dataset = QAGenDataset(Config.VAL_FILE, tokenizer)
    else:
        raise FileNotFoundError(f"File dữ liệu không tồn tại: {Config.TRAIN_FILE}")

    # 5. Data Collator
    # Dùng DataCollatorForSeq2Seq vẫn ổn vì nó giúp padding batch động
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model,
        padding=True,
        pad_to_multiple_of=8
    )

    # 6. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        
        # Steps configuration
        eval_strategy="steps",
        eval_steps=Config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=Config.EVAL_STEPS,
        save_total_limit=Config.SAVE_TOTAL_LIMIT,
        
        # Hyperparameters
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.EPOCHS,
        weight_decay=Config.WEIGHT_DECAY,
        
        # Optimization
        fp16=True,                # Bắt buộc True cho GPU để chạy nhanh
        gradient_checkpointing=True, # Tiết kiệm bộ nhớ
        
        # Generation configuration (cho metrics)
        predict_with_generate=True,
        generation_max_length=Config.MAX_TARGET_LENGTH,
        
        logging_dir='./logs',
        logging_steps=Config.LOGGING_STEPS,
        report_to="none"
    )

    # 7. Trainer
    # Dùng partial để truyền tokenizer vào metrics function
    compute_metrics_func = partial(compute_metrics, tokenizer=tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer, 
        data_collator=data_collator,
        compute_metrics=compute_metrics_func
    )

    # Fix lỗi cache warning của Qwen khi dùng gradient checkpointing
    model.config.use_cache = False 

    # 8. Train
    print("\nBắt đầu huấn luyện Qwen...")
    trainer.train()

    # 9. Save
    print("Đang lưu model...")
    save_path = os.path.join(Config.OUTPUT_DIR, "final_model")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Hoàn tất! Model lưu tại: {save_path}")

if __name__ == "__main__":
    main()