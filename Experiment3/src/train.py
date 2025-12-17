# train.py
import sys
import os
import torch
from transformers import (
    AutoTokenizer, 
    T5GemmaForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)
from src.config import Config
from src.dataset import QAGenDataset
from src.metrics import compute_metrics
from functools import partial


def print_trainable_parameters(model):
    """In số lượng tham số có thể huấn luyện"""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"📊 Trainable params: {trainable_params:,} || "
        f"All params: {all_params:,} || "
        f"Trainable%: {100 * trainable_params / all_params:.2f}%"
    )


def load_model_and_tokenizer():
    """Load tokenizer và model với PEFT config"""
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # --- [FIX 1] Xử lý PAD TOKEN (Bắt buộc để tránh lỗi indices) ---
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("ℹ️ Đã gán pad_token = eos_token")
    
    # Cập nhật padding side (T5 thường là right hoặc mặc định)
    tokenizer.padding_side = "right" 
    # -----------------------------------------------------------
    
    # 2. Cấu hình Quantization
    bnb_config = None
    if Config.USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=Config.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("✅ Đang sử dụng 4-bit Quantization (QLoRA)")
    
    # --- [FIX 2] Xử lý Device Map ---
    # Nếu dùng 4-bit: Buộc phải dùng device_map, nhưng ta ép về GPU 0 để tránh lỗi split
    # Nếu KHÔNG dùng 4-bit: Bỏ device_map để Trainer tự quản lý (ổn định nhất)
    
    device_map_config = None
    if Config.USE_4BIT:
        device_map_config = {"": 0} # Ép toàn bộ model vào GPU 0
    else:
        device_map_config = None    # Để None, Trainer sẽ tự move model vào cuda:0
        
    print(f"⚙️ Device Map config: {device_map_config}")

    # 3. Load Model
    model = T5GemmaForConditionalGeneration.from_pretrained(
        Config.MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.float16, # Luôn dùng fp16 cho nhẹ
        device_map=device_map_config 
    )

    # Đồng bộ config model với tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # 4. Áp dụng PEFT/LoRA (nếu bật)
    if Config.USE_PEFT:
        print("=" * 50)
        print("🔧 Đang cấu hình PEFT/LoRA...")
        
        # Chuẩn bị model cho k-bit training
        if Config.USE_4BIT:
            model = prepare_model_for_kbit_training(model)
        
        # Cấu hình LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=Config.LORA_DROPOUT,
            target_modules=Config.LORA_TARGET_MODULES,
            bias="none",
            inference_mode=False,
        )
        
        # Áp dụng LoRA
        model = get_peft_model(model, lora_config)
        
        # [FIX 3] Đảm bảo model in/out đúng kiểu dữ liệu khi dùng Trainer
        # Giúp tránh lỗi tensor mismatch
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32) # Cast layer đang train về float32 để ổn định
                
        print_trainable_parameters(model)
        print("=" * 50)
    
    return model, tokenizer


def main():
    # 1. Load Model & Tokenizer
    print("📥 Đang tải model và tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    # 2. Prepare Datasets
    print("📂 Đang tải datasets...")
    train_dataset = QAGenDataset("./data/train.json", tokenizer)
    val_dataset = QAGenDataset("./data/validation.json", tokenizer)
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    # 3. Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model
    )

    # 4. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        
        # --- EVALUATION & SAVING ---
        eval_strategy="steps",
        eval_steps=Config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=Config.EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        save_total_limit=Config.SAVE_TOTAL_LIMIT,
        
        # --- TRAINING PARAMS ---
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        weight_decay=0.01,
        num_train_epochs=Config.EPOCHS,
        warmup_ratio=0.1,              # Warmup 10% số steps
        
        # --- GENERATION ---
        predict_with_generate=True,
        generation_max_length=Config.MAX_TARGET_LENGTH,
        
        # --- OPTIMIZATION ---
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        # --- LOGGING ---
        logging_dir='./logs',
        logging_steps=50,
        report_to="tensorboard",
    )

    # 5. Initialize Trainer
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

    # 6. Train
    print("\n🚀 Bắt đầu huấn luyện...")
    trainer.train()

    # 7. Save Model
    if Config.USE_PEFT:
        # Lưu LoRA adapter
        lora_save_path = os.path.join(Config.OUTPUT_DIR, "lora_adapter")
        model.save_pretrained(lora_save_path)
        tokenizer.save_pretrained(lora_save_path)
        print(f"✅ Đã lưu LoRA adapter tại: {lora_save_path}")
    else:
        # Lưu full model
        final_path = os.path.join(Config.OUTPUT_DIR, "final_model")
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"✅ Đã lưu model tại: {final_path}")


if __name__ == "__main__":
    main()