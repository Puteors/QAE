# src/dataset.py
import json
from typing import Dict, Any, List, Tuple


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# QA generative (BartPho)
# -------------------------
def qa_gen_example(ex: Dict[str, Any]) -> Dict[str, str]:
    """
    seq2seq: source -> target
    Input mong đợi:
      - question
      - context
      - answers: {"text":[...]} (SQuAD style)
    """
    q = (ex.get("question") or "").strip()
    c = (ex.get("context") or "").strip()

    answers = ex.get("answers", {}) or {}
    ans = ""
    if isinstance(answers, dict):
        texts = answers.get("text", [])
        if texts:
            ans = (texts[0] or "").strip()

    return {"source": f"Câu hỏi: {q}\nNgữ cảnh: {c}", "target": ans}


# -------------------------
# Helper: char span -> token span
# -------------------------
def _char_to_token_span(
    offsets: List[Tuple[int, int]],
    start_char: int,
    end_char: int
) -> Tuple[int, int]:
    """
    offsets: list[(start,end)] theo token index.
    start_char/end_char: char-index trên context gốc.

    return (token_start, token_end) inclusive.
    Nếu không tìm thấy -> (-1,-1)
    """
    token_start = token_end = None

    # tìm token_start: token có offset bao phủ start_char
    for i, (s, e) in enumerate(offsets):
        if s <= start_char < e:
            token_start = i
            break

    # tìm token_end: token có offset bao phủ end_char-1
    # (một số implement dùng end_char nằm trong [s,e], mình giữ logic như bạn)
    for i, (s, e) in enumerate(offsets):
        if s < end_char <= e:
            token_end = i
            break

    if token_start is None or token_end is None:
        return -1, -1
    return token_start, token_end


# -------------------------
# Extractive QA tokenize + align (AE)
# -------------------------
def ae_tokenize_and_align(
    tokenizer,
    examples: List[Dict[str, Any]],
    max_len: int = 384,
    doc_stride: int = 128
) -> List[Dict[str, Any]]:
    """
    Trả về features dạng list[dict] cho AutoModelForQuestionAnswering.

    Output ONLY gồm:
      - input_ids
      - attention_mask
      - start_positions
      - end_positions
      - token_type_ids (nếu tokenizer có)

    ✅ KHÔNG BAO GIỜ tạo key "labels"
    ✅ Cần tokenizer FAST (use_fast=True) vì dùng return_offsets_mapping
    """
    features: List[Dict[str, Any]] = []

    for ex in examples:
        question = (ex.get("question") or "").strip()
        context = ex.get("context") or ""

        answers = ex.get("answers", {}) or {}
        is_impossible = bool(ex.get("is_impossible", False))

        tok = tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=max_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        for i in range(len(tok["input_ids"])):
            input_ids = tok["input_ids"][i]
            attention_mask = tok["attention_mask"][i]
            offsets = tok["offset_mapping"][i]
            sequence_ids = tok.sequence_ids(i)  # None / 0(question) / 1(context)

            # cls index (thường là token 0)
            cls_index = 0
            if tokenizer.cls_token_id is not None and tokenizer.cls_token_id in input_ids:
                cls_index = input_ids.index(tokenizer.cls_token_id)

            # chỉ giữ offsets cho context, còn lại set (0,0)
            # (đúng như code bạn)
            context_offsets: List[Tuple[int, int]] = []
            for (o, sid) in zip(offsets, sequence_ids):
                context_offsets.append(o if sid == 1 else (0, 0))

            # default: impossible -> answer tại CLS
            start_pos = end_pos = cls_index

            if (not is_impossible) and isinstance(answers, dict):
                texts = answers.get("text", [])
                starts = answers.get("answer_start", [])

                if texts and starts:
                    ans_text = (texts[0] or "")
                    start_char = int(starts[0])
                    end_char = start_char + len(ans_text)

                    start_tok, end_tok = _char_to_token_span(context_offsets, start_char, end_char)

                    # Nếu không tìm thấy trong span (do truncation), dùng CLS
                    if start_tok != -1 and end_tok != -1:
                        start_pos, end_pos = int(start_tok), int(end_tok)

            feat = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "start_positions": int(start_pos),
                "end_positions": int(end_pos),
            }

            # nếu tokenizer có token_type_ids (BERT-style)
            if "token_type_ids" in tok:
                feat["token_type_ids"] = tok["token_type_ids"][i]

            # ✅ không đưa offset_mapping / overflow mapping ra ngoài dataset
            # để tránh Trainer hiểu nhầm hoặc pass vào model
            features.append(feat)

    return features
