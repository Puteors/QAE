# src/inference.py
import argparse
import json
from typing import Any, Dict, List, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
)

def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

@torch.inference_mode()
def predict_bartpho(
    model_dir: str,
    items: List[Dict[str, Any]],
    max_source_len: int = 512,
    max_new_tokens: int = 64,
    num_beams: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[Dict[str, Any]]:
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()

    outputs: List[Dict[str, Any]] = []
    for ex in items:
        q = ex["question"].strip()
        c = ex["context"].strip()
        source = f"Câu hỏi: {q}\nNgữ cảnh: {c}"

        enc = tok(
            source,
            max_length=max_source_len,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        gen_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
        pred = tok.decode(gen_ids[0], skip_special_tokens=True).strip()

        outputs.append({
            "id": ex.get("id"),
            "uit_id": ex.get("uit_id"),
            "question": ex.get("question"),
            "pred": pred,
            "gold": (ex.get("answers", {}) or {}).get("text", [""])[0] if ex.get("answers") else "",
        })
    return outputs

def _best_span_from_logits(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    max_answer_len: int = 30,
) -> Tuple[int, int, float]:
    """
    Return (best_start, best_end, score)
    """
    s = start_logits.squeeze(0)
    e = end_logits.squeeze(0)

    best_score = -1e9
    best_i, best_j = 0, 0

    # brute force within max_answer_len (OK for single example)
    s_cpu = s.detach().cpu()
    e_cpu = e.detach().cpu()
    for i in range(len(s_cpu)):
        j_max = min(len(e_cpu) - 1, i + max_answer_len)
        for j in range(i, j_max + 1):
            score = float(s_cpu[i] + e_cpu[j])
            if score > best_score:
                best_score = score
                best_i, best_j = i, j
    return best_i, best_j, best_score

@torch.inference_mode()
def predict_mdeberta_ae(
    model_dir: str,
    items: List[Dict[str, Any]],
    max_len: int = 384,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_answer_len: int = 40,
) -> List[Dict[str, Any]]:
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir).to(device)
    model.eval()

    outputs: List[Dict[str, Any]] = []
    for ex in items:
        question = ex["question"].strip()
        context = ex["context"]

        enc = tok(
            question,
            context,
            truncation="only_second",
            max_length=max_len,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)
        start_i, end_i, _ = _best_span_from_logits(out.start_logits, out.end_logits, max_answer_len=max_answer_len)

        # map token span -> char span using offsets
        start_char, end_char = offsets[start_i][0], offsets[end_i][1]
        if end_char <= start_char:
            pred = ""
        else:
            pred = context[start_char:end_char].strip()

        outputs.append({
            "id": ex.get("id"),
            "uit_id": ex.get("uit_id"),
            "question": ex.get("question"),
            "pred": pred,
            "gold": (ex.get("answers", {}) or {}).get("text", [""])[0] if ex.get("answers") else "",
        })
    return outputs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["qa", "ae"], required=True, help="qa=BARTpho gen, ae=mDeBERTa extractive")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input_json", default="data/test.json")
    ap.add_argument("--output_json", default="outputs/preds.json")

    # gen params
    ap.add_argument("--max_source_len", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--num_beams", type=int, default=4)

    # ae params
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--max_answer_len", type=int, default=40)

    args = ap.parse_args()

    items = load_json(args.input_json)
    if args.task == "qa":
        preds = predict_bartpho(
            model_dir=args.model_dir,
            items=items,
            max_source_len=args.max_source_len,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
    else:
        preds = predict_mdeberta_ae(
            model_dir=args.model_dir,
            items=items,
            max_len=args.max_len,
            max_answer_len=args.max_answer_len,
        )

    save_json(args.output_json, preds)
    print(f"Saved {len(preds)} predictions to: {args.output_json}")

if __name__ == "__main__":
    main()
