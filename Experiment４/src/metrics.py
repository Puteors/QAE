# src/metrics.py
from __future__ import annotations

from typing import Dict, List, Tuple
import re

import evaluate
import numpy as np


# -------------------------
# Text utils
# -------------------------
def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


# -------------------------
# ROUGE (cached) -> dùng làm rough1/rough2/roughL
# -------------------------
_ROUGE = None


def get_rouge():
    global _ROUGE
    if _ROUGE is None:
        _ROUGE = evaluate.load("rouge")
    return _ROUGE


def rouge_as_rough_scores(
    preds: List[str],
    refs: List[str],
    use_stemmer: bool = False,
) -> Dict[str, float]:
    """
    Map ROUGE -> rough metrics:
      - rough1 = rouge1
      - rough2 = rouge2
      - roughL = rougeL
    """
    if len(preds) == 0:
        return {"rough1": 0.0, "rough2": 0.0, "roughL": 0.0}

    rouge = get_rouge()
    result = rouge.compute(
        predictions=preds,
        references=refs,
        use_stemmer=use_stemmer,  # tiếng Việt thường không cần stemmer
    )
    return {
        "rough1": float(result["rouge1"]),
        "rough2": float(result["rouge2"]),
        "roughL": float(result["rougeL"]),
    }


# -------------------------
# BERTScore (cached)
# -------------------------
_BERTSCORE = None


def get_bertscore():
    global _BERTSCORE
    if _BERTSCORE is None:
        _BERTSCORE = evaluate.load("bertscore")
    return _BERTSCORE


def bertscore_scores(
    preds: List[str],
    refs: List[str],
    lang: str = "vi",
    model_type: str = "xlm-roberta-base",  # hoặc "vinai/phobert-base"
    rescale_with_baseline: bool = False,
) -> Dict[str, float]:
    if len(preds) == 0:
        return {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0}

    bs = get_bertscore()
    out = bs.compute(
        predictions=preds,
        references=refs,
        lang=lang,
        model_type=model_type,
        rescale_with_baseline=rescale_with_baseline,
    )
    return {
        "bertscore_p": float(sum(out["precision"]) / len(out["precision"])),
        "bertscore_r": float(sum(out["recall"]) / len(out["recall"])),
        "bertscore_f1": float(sum(out["f1"]) / len(out["f1"])),
    }


# -------------------------
# Decode helpers (Trainer eval_pred)
# -------------------------
def replace_ignore_index(
    labels: np.ndarray,
    pad_token_id: int,
    ignore_index: int = -100,
) -> np.ndarray:
    return np.where(labels != ignore_index, labels, pad_token_id)


def decode_batch(tokenizer, token_ids: np.ndarray) -> List[str]:
    return tokenizer.batch_decode(token_ids, skip_special_tokens=True)


def decode_preds_and_labels(
    eval_pred,
    tokenizer,
    ignore_index: int = -100,
) -> Tuple[List[str], List[str]]:
    predictions, labels = eval_pred
    decoded_preds = decode_batch(tokenizer, predictions)

    labels = replace_ignore_index(labels, tokenizer.pad_token_id, ignore_index=ignore_index)
    decoded_labels = decode_batch(tokenizer, labels)

    # normalize nhẹ cho ổn định
    decoded_preds = [normalize_text(x) for x in decoded_preds]
    decoded_labels = [normalize_text(x) for x in decoded_labels]
    return decoded_preds, decoded_labels


def filter_nonempty_refs(
    preds: List[str],
    refs: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Lọc các cặp có ref (gold) rỗng để tránh làm bẩn metric.
    """
    kept_p, kept_r = [], []
    for p, r in zip(preds, refs):
        if normalize_text(r) != "":
            kept_p.append(p)
            kept_r.append(r)
    return kept_p, kept_r


# -------------------------
# Public metrics APIs
# -------------------------
def compute_rough_and_bertscore_from_text(
    preds: List[str],
    refs: List[str],
    bert_lang: str = "vi",
    bert_model_type: str = "xlm-roberta-base",
) -> Dict[str, float]:
    """
    Dùng cho script evaluate (đọc preds.json: pred/gold).
    """
    preds_f, refs_f = filter_nonempty_refs(preds, refs)

    metrics: Dict[str, float] = {}
    metrics.update(rouge_as_rough_scores(preds_f, refs_f))
    metrics.update(bertscore_scores(preds_f, refs_f, lang=bert_lang, model_type=bert_model_type))
    metrics["n_eval"] = float(len(refs_f))
    return metrics


def compute_rough_and_bertscore_from_eval_pred(
    eval_pred,
    tokenizer,
    bert_lang: str = "vi",
    bert_model_type: str = "xlm-roberta-base",
) -> Dict[str, float]:
    """
    Dùng cho Trainer.compute_metrics (seq2seq).
    """
    preds, refs = decode_preds_and_labels(eval_pred, tokenizer)
    return compute_rough_and_bertscore_from_text(
        preds,
        refs,
        bert_lang=bert_lang,
        bert_model_type=bert_model_type,
    )
