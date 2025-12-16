import sys
import os
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
from functools import partial
from peft import LoraConfig, TaskType, get_peft_model

def main():
    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(Config.MODEL_NAME)
    peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 2. Prepare Datasets
    # Lưu ý: Cần chỉnh đường dẫn đúng tới file json của bạn
    train_dataset = QAGenDataset("./data/train.json", tokenizer)
    val_dataset = QAGenDataset("./data/validation.json", tokenizer)
    # test_dataset = QAGenDataset("./data/test.json", tokenizer)

    # 3. Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model
    )

    # 4. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        
        # --- CẤU HÌNH STEPS ---
        eval_strategy="steps",    # Đánh giá theo steps thay vì epoch
        eval_steps=Config.EVAL_STEPS,   # Số bước mỗi lần đánh giá (vd: 100)
        
        save_strategy="steps",          # Lưu model theo steps (phải khớp với eval)
        save_steps=Config.EVAL_STEPS,   # Số bước mỗi lần lưu
        
        # --- CẤU HÌNH BEST MODEL ---
        load_best_model_at_end=True,    # Load lại model ngon nhất khi train xong
        metric_for_best_model="rougeL", # Chọn model có điểm ROUGE-L cao nhất (thay vì Loss thấp nhất)
        greater_is_better=True,         # ROUGE càng cao càng tốt
        save_total_limit=Config.SAVE_TOTAL_LIMIT, # Chỉ giữ 2 model tốt nhất
        
        # --- CÁC THAM SỐ KHÁC ---
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        weight_decay=0.01,
        num_train_epochs=Config.EPOCHS,
        predict_with_generate=True,     # BẮT BUỘC để tính được ROUGE
        fp16=True,                      # Dùng GPU thì để True
        logging_dir='./logs',
        logging_steps=50,               # In log loss mỗi 50 bước
    )

    # 5. Initialize Trainer
    # Dùng partial để truyền tokenizer vào hàm compute_metrics
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
    print("Bắt đầu huấn luyện...")
    trainer.train()

    # 7. Save final model
    trainer.save_model(os.path.join(Config.OUTPUT_DIR, "final_model"))
    print("Đã lưu mô hình!")

if __name__ == "__main__":
    main()