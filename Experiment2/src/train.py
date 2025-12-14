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

def main():
    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(Config.MODEL_NAME)

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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=Config.EPOCHS,
        predict_with_generate=True, # Bắt buộc True để tính ROUGE khi eval
        fp16=True, # Bật nếu dùng GPU
        logging_dir='./logs',
        logging_steps=100,
        load_best_model_at_end=True,
    )

    # 5. Initialize Trainer
    # Dùng partial để truyền tokenizer vào hàm compute_metrics
    compute_metrics_func = partial(compute_metrics, tokenizer=tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
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