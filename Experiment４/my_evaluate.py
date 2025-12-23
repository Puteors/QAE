"""
my_eraavaluate.py

Đánh giá "rough" (EM + token-F1) và BERTScore cho 2 mô hình:
- Generative QA: vinai/bartpho-syllable (Seq2Seq)  -> dùng inference.predict_bartpho
- Extractive QA: microsoft/mdeberta-v3-base (QA)   -> dùng inference.predict_mdeberta_ae

Yêu cầu: input_json phải có GOLD (answers.text[0]) thì mới tính được metric.
Ngôn ngữ: tiếng Việt (lang='vi' cho BERTScore).
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List, Tuple

import evaluate

from inference import predict_bartpho, predict_mdeberta_ae


# -------------------------
# I/O
# -------------------------
def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_gold(ex: Dict[str, Any]) -> str:
    ans = ex.get("answers") or {}
    texts = ans.get("text") or []
    if isinstance(texts, list) and len(texts) > 0 and texts[0] is not None:
        return str(texts[0]).strip()
    return ""


# -------------------------
# Rough metrics (EM + token F1)
# -------------------------
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    # tối giản, phù hợp tiếng Việt: lower + gộp khoảng trắng
    s = (s or "").strip().lower()
    s = _WS.sub(" ", s)
    return s


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    p = normalize_text(pred).split() if pred else []
    g = normalize_text(gold).split() if gold else []

    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    # multiset overlap
    counts: Dict[str, int] = {}
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


def rough_scores(pairs: List[Tuple[str, str]]) -> Dict[str, float]:
    if not pairs:
        return {"em": 0.0, "token_f1": 0.0}

    em = sum(exact_match(p, g) for p, g in pairs) / len(pairs)
    f1 = sum(token_f1(p, g) for p, g in pairs) / len(pairs)
    return {"em": float(em), "token_f1": float(f1)}


# -------------------------
# BERTScore (cached)
# -------------------------
_BS = None


def get_bertscore():
    global _BS
    if _BS is None:
        _BS = evaluate.load("bertscore")
    return _BS


def bertscore_scores(
    preds: List[str],
    refs: List[str],
    model_type: str,
    lang: str = "vi",
    rescale_with_baseline: bool = False,
) -> Dict[str, float]:
    """
    model_type: tên backbone để compute BERTScore (vd: 'vinai/phobert-base', 'xlm-roberta-base', ...)
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
# Evaluation flow
# -------------------------
def run_inference(
    task: str,
    model_dir: str,
    items: List[Dict[str, Any]],
    max_source_len: int = 512,
    max_new_tokens: int = 64,
    num_beams: int = 4,
    max_len: int = 384,
    max_answer_len: int = 40,
) -> List[Dict[str, Any]]:
    if task == "qa":
        return predict_bartpho(
            model_dir=model_dir,
            items=items,
            max_source_len=max_source_len,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
    return predict_mdeberta_ae(
        model_dir=model_dir,
        items=items,
        max_len=max_len,
        max_answer_len=max_answer_len,
    )


def evaluate_outputs(
    outputs: List[Dict[str, Any]],
    bertscore_model: str,
) -> Dict[str, float]:
    # lọc gold rỗng để metric không bị bẩn
    pairs = [(x.get("pred", ""), x.get("gold", "")) for x in outputs if normalize_text(x.get("gold", "")) != ""]
    preds = [p for p, _ in pairs]
    refs = [g for _, g in pairs]

    metrics: Dict[str, float] = {}
    metrics["n_total"] = float(len(outputs))
    metrics["n_eval"] = float(len(pairs))  # số mẫu có gold

    metrics.update(rough_scores(pairs))
    metrics.update(bertscore_scores(preds, refs, model_type=bertscore_model, lang="vi"))

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["qa", "ae"], required=True, help="qa=BARTpho gen, ae=mDeBERTa extractive")

    # model dirs (đã train hoặc pretrained đã fine-tune và save_pretrained)
    ap.add_argument("--model_dir", required=True)

    # data
    ap.add_argument("--input_json", default="data/validation.json", help="Nên dùng validation có answers để chấm điểm")
    ap.add_argument("--save_preds", default="", help="Nếu set, sẽ lưu predictions ra JSON path này")

    # BERTScore settings (Vietnamese)
    ap.add_argument("--bertscore_model", default="vinai/phobert-base", help="VD: vinai/phobert-base hoặc xlm-roberta-base")

    # gen params (qa)
    ap.add_argument("--max_source_len", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--num_beams", type=int, default=4)

    # ae params
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--max_answer_len", type=int, default=40)

    # sampling
    ap.add_argument("--n", type=int, default=0, help="0=all, else take first n samples")

    args = ap.parse_args()

    items = load_json(args.input_json)
    if args.n and args.n > 0:
        items = items[: args.n]

    # đảm bảo outputs có gold (để chấm điểm)
    # inference.py đã làm safe_gold, nhưng ta cảnh báo nếu tất cả gold rỗng.
    outputs = run_inference(
        task=args.task,
        model_dir=args.model_dir,
        items=items,
        max_source_len=args.max_source_len,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        max_len=args.max_len,
        max_answer_len=args.max_answer_len,
    )

    if args.save_preds:
        save_json(args.save_preds, outputs)
        print(f"Saved {len(outputs)} predictions to: {args.save_preds}")

    n_gold = sum(1 for x in outputs if safe_gold(x).strip() != "" or (x.get("gold","") or "").strip() != "")
    if n_gold == 0:
        print("⚠️ Không thấy GOLD trong input/output. Không thể tính rough/BERTScore trên test không nhãn.")
        print("   Hãy dùng data/validation.json (có answers) hoặc một test set có answers.")
        return

    metrics = evaluate_outputs(outputs, bertscore_model=args.bertscore_model)

    print("\n=== Evaluation (Vietnamese) ===")
    print(f"task: {args.task}")
    print(f"model_dir: {args.model_dir}")
    print(f"input_json: {args.input_json}")
    print(f"bertscore_model: {args.bertscore_model}")
    print(f"n_total: {int(metrics['n_total'])} | n_eval (non-empty gold): {int(metrics['n_eval'])}")
    print(f"EM: {metrics['em']:.4f}")
    print(f"Token-F1: {metrics['token_f1']:.4f}")
    print(f"BERTScore P/R/F1: {metrics['bertscore_p']:.4f} / {metrics['bertscore_r']:.4f} / {metrics['bertscore_f1']:.4f}")

    # show a few examples
    print("\nExamples (first 5):")
    for x in outputs[:5]:
        print("-" * 60)
        print("Q:", x.get("question"))
        print("P:", x.get("pred"))
        print("G:", x.get("gold"))


if __name__ == "__main__":
    main()
