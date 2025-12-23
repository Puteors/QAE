# src/metrics.py
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import re

import evaluate
import numpy as np


# -------------------------
# Utils: normalize / tokenize
# -------------------------
def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def whitespace_tokens(s: str) -> List[str]:
    s = normalize_text(s)
    return s.split() if s else []


# -------------------------
# Rough metrics: EM, token-F1
# -------------------------
def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    p = whitespace_tokens(pred)
    g = whitespace_tokens(gold)

    if len(p) == 0 and len(g) == 0:
        return 1.0
    if len(p) == 0 or len(g) == 0:
        return 0.0

    # multiset overlap
    counts = {}
    for t in p:
        counts[t] = counts.get(t, 0) + 1

    hit = 0
    for t in g:
        if counts.get(t, 0) > 0:
            hit += 1
            counts[t] -= 1

    if hit == 0:
        return 0.0

    prec = hit / len(p)
    rec = hit / len(g)
    return 2 * prec * rec / (prec + rec)


def rough_scores(preds: List[str], refs: List[str]) -> Dict[str, float]:
    if len(preds) == 0:
        return {"em": 0.0, "token_f1": 0.0}

    em = sum(exact_match(p, r) for p, r in zip(preds, refs)) / len(preds)
    f1 = sum(token_f1(p, r) for p, r in zip(preds, refs)) / len(preds)
    return {"em": float(em), "token_f1": float(f1)}


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
    model_type: str = "xlm-roberta-base",
    rescale_with_baseline: bool = False,
) -> Dict[str, float]:
    """
    Trả về trung bình P/R/F1.
    Lưu ý: cần refs (gold). Nếu refs rỗng -> không tính được.
    """
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
# (Optional) Decode helpers for Trainer eval_pred
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
# Public: compute all metrics for evaluate
# -------------------------
def compute_rough_and_bertscore_from_text(
    preds: List[str],
    refs: List[str],
    bert_lang: str = "vi",
    bert_model_type: str = "xlm-roberta-base",
) -> Dict[str, float]:
    """
    Dùng cho evaluate script (đọc preds.json: pred/gold).
    """
    preds_f, refs_f = filter_nonempty_refs(preds, refs)
    metrics = {}
    metrics.update(rough_scores(preds_f, refs_f))
    metrics.update(bertscore_scores(preds_f, refs_f, lang=bert_lang, model_type=bert_model_type))
    metrics["n_eval"] = float(len(preds_f))
    return metrics


def compute_rough_and_bertscore_from_eval_pred(
    eval_pred,
    tokenizer,
    bert_lang: str = "vi",
    bert_model_type: str = "xlm-roberta-base",
) -> Dict[str, float]:
    """
    Dùng nếu bạn muốn nhét vào Trainer.compute_metrics (seq2seq).
    """
    preds, refs = decode_preds_and_labels(eval_pred, tokenizer)
    return compute_rough_and_bertscore_from_text(
        preds,
        refs,
        bert_lang=bert_lang,
        bert_model_type=bert_model_type,
    )
