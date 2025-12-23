# src/train.py
import argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    Trainer,
    Seq2SeqTrainer,
)

from peft import LoraConfig, get_peft_model, TaskType
from src.config import QAConfig, AEConfig
from src.dataset import load_json, qa_gen_example, ae_tokenize_and_align


def apply_lora(model, task_type: TaskType):
    """
    Apply LoRA (PEFT) to model.

    target_modules phụ thuộc kiến trúc model; nếu gặp lỗi,
    bạn cần đổi list này theo layer names thực tế của model.
    """
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=task_type,
        target_modules=["query_proj", "key_proj", "value_proj", "dense"],
    )
    return get_peft_model(model, lora)


def train_bartpho(train_path, valid_path, cfg: QAConfig):
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

    if cfg.use_peft:
        # Seq2Seq => SEQ_2_SEQ_LM
        model = apply_lora(model, TaskType.SEQ_2_SEQ_LM)

    train_data = [qa_gen_example(x) for x in load_json(train_path)]
    valid_data = [qa_gen_example(x) for x in load_json(valid_path)]

    ds_train = Dataset.from_list(train_data)
    ds_valid = Dataset.from_list(valid_data)

    def preprocess(batch):
        x = tok(batch["source"], max_length=cfg.max_source_len, truncation=True)
        y = tok(text_target=batch["target"], max_length=cfg.max_target_len, truncation=True)
        x["labels"] = y["input_ids"]
        return x

    ds_train = ds_train.map(preprocess, batched=True, remove_columns=ds_train.column_names)
    ds_valid = ds_valid.map(preprocess, batched=True, remove_columns=ds_valid.column_names)

    collator = DataCollatorForSeq2Seq(tok, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,

        # ✅ đúng tên tham số
        evaluation_strategy="steps",
        eval_steps=1000,

        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        predict_with_generate=True,
        fp16=True,
        logging_steps=50,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        data_collator=collator,
        tokenizer=tok,
        # Nếu muốn tính metric khi train QA, có thể thêm compute_metrics ở đây.
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)


def train_mdeberta_ae(train_path, valid_path, cfg: AEConfig):
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(cfg.model_name)

    if cfg.use_peft:
        # Extractive QA => QUESTION_ANSWERING
        model = apply_lora(model, TaskType.QUESTION_ANSWERING)

    train_examples = load_json(train_path)
    valid_examples = load_json(valid_path)

    train_feats = ae_tokenize_and_align(tok, train_examples, cfg.max_len, cfg.doc_stride)
    valid_feats = ae_tokenize_and_align(tok, valid_examples, cfg.max_len, cfg.doc_stride)

    ds_train = Dataset.from_list(train_feats)
    ds_valid = Dataset.from_list(valid_feats)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,

        # ✅ Giữ eval_loss khi train AE (mỗi n step)
        evaluation_strategy="steps",
        eval_steps=1000,

        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        fp16=True,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        tokenizer=tok,
        # ✅ Không compute_metrics => chỉ eval_loss
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["qa", "ae"], required=True)
    ap.add_argument("--train", default="data/train.json")
    ap.add_argument("--valid", default="data/validation.json")
    args = ap.parse_args()

    if args.task == "qa":
        train_bartpho(args.train, args.valid, QAConfig())
    else:
        train_mdeberta_ae(args.train, args.valid, AEConfig())


if __name__ == "__main__":
    main()
