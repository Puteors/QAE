# src/test_preprocess.py
import argparse
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer

from src.dataset import load_json, qa_gen_example, ae_tokenize_and_align


def _safe_get_gold(ex: Dict[str, Any]) -> str:
    ans = ex.get("answers", {}) or {}
    txt = ans.get("text", [])
    return (txt[0] if txt else "") or ""


def check_bartpho_preprocess(
    examples: List[Dict[str, Any]],
    model_name: str,
    max_source_len: int = 512,
    max_target_len: int = 64,
    k: int = 5,
) -> None:
    tok = AutoTokenizer.from_pretrained(model_name)
    print("\n" + "=" * 80)
    print(f"[QA GEN] Tokenizer: {model_name}")

    show = 0
    for ex in examples:
        eg = qa_gen_example(ex)
        src, tgt = eg["source"], eg["target"]

        enc = tok(src, truncation=True, max_length=max_source_len)
        dec = tok(text_target=tgt, truncation=True, max_length=max_target_len)

        if show < k:
            print("-" * 80)
            print("ID:", ex.get("id"))
            print("Q :", ex.get("question"))
            print("Gold target:", tgt)
            print(f"source_len={len(enc['input_ids'])} target_len={len(dec['input_ids'])}")
            print("source_preview:", src[:200].replace("\n", " ") + ("..." if len(src) > 200 else ""))
        show += 1

    print(f"Checked {min(len(examples), k)} samples (printed).")


def _extract_span_text_from_feature(
    tokenizer,
    feature: Dict[str, Any],
    context: str,
) -> str:
    """
    feature: output của ae_tokenize_and_align() chứa input_ids + start/end_positions.
    Lưu ý: feature đang là tokenized theo question+context, offsets không giữ lại trong feature.
    Vì vậy ta chỉ decode span token rồi strip; đây chỉ để sanity check.
    (Decode token span có thể khác context substring một chút do tokenizer.)
    """
    s = int(feature["start_positions"])
    e = int(feature["end_positions"])
    if e < s:
        return ""
    ids = feature["input_ids"][s : e + 1]
    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
    return text


def check_ae_preprocess(
    examples: List[Dict[str, Any]],
    model_name: str,
    max_len: int = 384,
    doc_stride: int = 128,
    k: int = 10,
) -> None:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    print("\n" + "=" * 80)
    print(f"[AE/EXTRACTIVE] Tokenizer: {model_name}")
    print(f"max_len={max_len}, doc_stride={doc_stride}")

    # Tạo features (có thể nhiều feature/1 example do overflow)
    feats = ae_tokenize_and_align(tok, examples, max_len=max_len, doc_stride=doc_stride)
    print(f"Examples: {len(examples)} -> Features: {len(feats)}")

    # Sanity check theo example gốc: kiểm tra answer_start slice khớp gold
    # và kiểm tra token span decode khớp (rough) gold trên feature đầu tiên match id (nếu có)
    # (ae_tokenize_and_align hiện không giữ ex_id trong feature; nên ta check rough theo thứ tự)
    printed = 0
    ok_slice = 0
    total_has_ans = 0

    for idx, ex in enumerate(examples):
        if printed >= k:
            break

        gold = _safe_get_gold(ex)
        if ex.get("is_impossible", False) or not gold:
            continue

        total_has_ans += 1
        context = ex["context"]
        start_char = ex["answers"]["answer_start"][0]
        end_char = start_char + len(gold)
        slice_gold = context[start_char:end_char]

        slice_match = (slice_gold.strip() == gold.strip())
        ok_slice += int(slice_match)

        # feature tương ứng (rough theo index; nếu dữ liệu ít overflow thì đúng)
        # nếu overflow nhiều, đoạn này chỉ để tham khảo.
        feat = feats[min(idx, len(feats) - 1)]
        pred_tok_span = _extract_span_text_from_feature(tok, feat, context)

        print("-" * 80)
        print("ID:", ex.get("id"))
        print("Q :", ex.get("question"))
        print("Gold:", gold)
        print("Context slice by answer_start:", slice_gold)
        print("Slice match gold:", slice_match)
        print("Token-span decode (rough):", pred_tok_span)
        print("start_positions/end_positions:", feat["start_positions"], feat["end_positions"])
        printed += 1

    if total_has_ans > 0:
        print(f"\nSanity: context[answer_start:...] matches gold = {ok_slice}/{total_has_ans}")
    else:
        print("\nNo answerable examples found to check.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", default="data/train.json")
    ap.add_argument("--k", type=int, default=5, help="number of samples to print per task")

    # QA gen
    ap.add_argument("--qa_model", default="vinai/bartpho-syllable")
    ap.add_argument("--max_source_len", type=int, default=512)
    ap.add_argument("--max_target_len", type=int, default=64)

    # AE
    ap.add_argument("--ae_model", default="microsoft/mdeberta-v3-base")
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--doc_stride", type=int, default=128)

    ap.add_argument("--only", choices=["qa", "ae", "both"], default="both")
    args = ap.parse_args()

    examples = load_json(args.input_json)

    if args.only in ("qa", "both"):
        check_bartpho_preprocess(
            examples,
            model_name=args.qa_model,
            max_source_len=args.max_source_len,
            max_target_len=args.max_target_len,
            k=args.k,
        )

    if args.only in ("ae", "both"):
        check_ae_preprocess(
            examples,
            model_name=args.ae_model,
            max_len=args.max_len,
            doc_stride=args.doc_stride,
            k=max(args.k, 5),
        )


if __name__ == "__main__":
    main()
