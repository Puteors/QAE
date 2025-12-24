# src/train.py
import argparse
import os
import json
import shutil
from datetime import datetime
from typing import List

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
    TrainerCallback,
)

from peft import LoraConfig, get_peft_model, TaskType
from src.config import QAConfig, AEConfig
from src.dataset import load_json, qa_gen_example, ae_tokenize_and_align
from src.metrics import compute_rough_and_bertscore_from_eval_pred


# -------------------------
# Callback: save best-K checkpoints by eval_loss
# -------------------------
class SaveBestKByEvalLossCallback(TrainerCallback):
    """
    Lưu TOP-K checkpoint có eval_loss thấp nhất vào <output_dir>/best_checkpoints.
    """

    def __init__(self, best_dir: str, k: int = 3, tokenizer=None):
        self.best_dir = best_dir
        self.k = k
        self.tokenizer = tokenizer
        self.best: List[tuple[float, str]] = []  # (loss, path)
        os.makedirs(self.best_dir, exist_ok=True)

    def _prune(self):
        self.best.sort(key=lambda x: x[0])
        while len(self.best) > self.k:
            loss, path = self.best.pop()  # remove worst
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return control

        loss = metrics.get("eval_loss", None)
        if loss is None:
            return control

        trainer = kwargs.get("trainer", None)
        if trainer is None:
            return control

        loss = float(loss)
        step = int(state.global_step)
        save_path = os.path.join(self.best_dir, f"step-{step}_loss-{loss:.4f}")

        should_save = (len(self.best) < self.k) or (loss < max(x[0] for x in self.best))
        if should_save:
            trainer.save_model(save_path)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(save_path)

            self.best.append((loss, save_path))
            self._prune()

        return control


# -------------------------
# Callback: log metrics to file (JSONL)
# -------------------------
class MetricsFileLoggerCallback(TrainerCallback):
    """
    Ghi metrics trong quá trình train vào file JSONL.
    Mỗi lần evaluate sẽ append 1 dòng JSON.

    File: <output_dir>/metrics_log.jsonl
    """

    def __init__(self, output_dir: str, filename: str = "metrics_log.jsonl"):
        self.output_dir = output_dir
        self.filepath = os.path.join(output_dir, filename)
        os.makedirs(self.output_dir, exist_ok=True)

        # đảm bảo file tồn tại
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", encoding="utf-8") as f:
                f.write("")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return control

        record = {
            "time": datetime.now().isoformat(),
            "global_step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
        }
        for k, v in metrics.items():
            try:
                record[k] = float(v)
            except Exception:
                record[k] = v

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return control


# -------------------------
# LoRA helpers
# -------------------------
def _find_target_modules(model, candidates: List[List[str]]) -> List[str]:
    """
    Trả về list target_modules phù hợp với model (dò theo tên submodule).
    Ưu tiên option mà tất cả modules đều tồn tại; nếu không thì lấy những cái match được.
    """
    names = set(n for n, _ in model.named_modules())

    def exists_any(suffix: str) -> bool:
        return any(n == suffix or n.endswith("." + suffix) for n in names)

    for option in candidates:
        if all(exists_any(m) for m in option):
            return option

    for option in candidates:
        matched = [m for m in option if exists_any(m)]
        if matched:
            return matched

    return []


def apply_lora(model, task_type: TaskType):
    """
    Apply LoRA với target_modules phù hợp theo kiến trúc.
    """
    candidates = [
        ["q_proj", "k_proj", "v_proj", "out_proj"],        # BART/mBART
        ["q_proj", "v_proj"],                              # fallback
        ["query", "key", "value"],                         # BERT-like
        ["query_proj", "key_proj", "value_proj", "dense"], # một số kiến trúc khác
    ]

    target_modules = _find_target_modules(model, candidates)
    if not target_modules:
        raise ValueError(
            "Không tìm thấy target_modules phù hợp để gắn LoRA. "
            "Hãy kiểm tra tên modules trong model.named_modules()."
        )

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=task_type,
        target_modules=target_modules,
    )
    print(f"[LoRA] task_type={task_type} target_modules={target_modules}")
    return get_peft_model(model, lora)


# -------------------------
# Train: QA (BartPho seq2seq)
# -------------------------
def train_bartpho(train_path, valid_path, cfg: QAConfig):
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

    if cfg.use_peft:
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

        # theo yêu cầu giữ eval_strategy
        eval_strategy="steps",
        eval_steps=100,

        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        predict_with_generate=True,
        fp16=True,
        logging_steps=50,
        report_to="none",
    )

    best_cb = SaveBestKByEvalLossCallback(
        best_dir=os.path.join(cfg.output_dir, "best_checkpoints"),
        k=3,
        tokenizer=tok,
    )
    log_cb = MetricsFileLoggerCallback(output_dir=cfg.output_dir)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        data_collator=collator,
        tokenizer=tok,
        compute_metrics=lambda p: compute_rough_and_bertscore_from_eval_pred(
            p,
            tokenizer=tok,
            bert_lang="vi",
            bert_model_type="xlm-roberta-base",  # hoặc "vinai/phobert-base"
        ),
        callbacks=[best_cb, log_cb],
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)


# -------------------------
# Train: AE (mDeBERTa extractive QA)
# -------------------------
def train_mdeberta_ae(train_path, valid_path, cfg: AEConfig):
    # Nếu bạn đã cài protobuf thì dùng use_fast=True, nếu chưa thì dùng False
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)
    model = AutoModelForQuestionAnswering.from_pretrained(cfg.model_name)

    if cfg.use_peft:
        # ✅ Fix 2: nếu TaskType không có QUESTION_ANSWERING thì fallback TOKEN_CLS
        qa_task = getattr(TaskType, "QUESTION_ANSWERING", TaskType.TOKEN_CLS)
        model = apply_lora(model, qa_task)

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

        # theo yêu cầu giữ eval_strategy
        eval_strategy="steps",
        eval_steps=100,

        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        fp16=True,
        logging_steps=50,
        report_to="none",
    )

    best_cb = SaveBestKByEvalLossCallback(
        best_dir=os.path.join(cfg.output_dir, "best_checkpoints"),
        k=3,
        tokenizer=tok,
    )
    log_cb = MetricsFileLoggerCallback(output_dir=cfg.output_dir)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        tokenizer=tok,
        callbacks=[best_cb, log_cb],
        # AE: chỉ eval_loss
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
