# src/dataset.py
import json
from typing import Dict, Any, List, Tuple

def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def qa_gen_example(ex: Dict[str, Any]) -> Dict[str, str]:
    # seq2seq: source -> target
    q = ex["question"].strip()
    c = ex["context"].strip()
    ans = ex["answers"]["text"][0].strip() if ex.get("answers") and ex["answers"]["text"] else ""
    return {"source": f"Câu hỏi: {q}\nNgữ cảnh: {c}", "target": ans}

def _char_to_token_span(offsets: List[Tuple[int,int]], start_char: int, end_char: int) -> Tuple[int,int]:
    # tìm token_start: token có offset bao phủ start_char
    token_start = token_end = None
    for i, (s,e) in enumerate(offsets):
        if s <= start_char < e:
            token_start = i
            break
    for i, (s,e) in enumerate(offsets):
        if s < end_char <= e:
            token_end = i
            break
    if token_start is None or token_end is None:
        return -1, -1
    return token_start, token_end

def ae_tokenize_and_align(tokenizer, examples: List[Dict[str, Any]], max_len=384, doc_stride=128):
    """
    Trả về features dạng list[dict] cho AutoModelForQuestionAnswering.
    """
    features = []
    for ex in examples:
        question = ex["question"].strip()
        context = ex["context"]

        answers = ex.get("answers", {})
        is_impossible = bool(ex.get("is_impossible", False))

        # HuggingFace QA pattern: tokenize(question, context) + overflow
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
            sequence_ids = tok.sequence_ids(i)  # None / 0 (question) / 1 (context)
            cls_index = input_ids.index(tokenizer.cls_token_id) if tokenizer.cls_token_id in input_ids else 0

            # chỉ giữ offsets cho context, còn lại set (0,0)
            context_offsets = []
            for (o, sid) in zip(offsets, sequence_ids):
                context_offsets.append(o if sid == 1 else (0, 0))

            if is_impossible or not answers or not answers.get("text"):
                start_pos = end_pos = cls_index
            else:
                ans_text = answers["text"][0]
                start_char = answers["answer_start"][0]
                end_char = start_char + len(ans_text)

                start_tok, end_tok = _char_to_token_span(context_offsets, start_char, end_char)
                if start_tok == -1 or end_tok == -1:
                    start_pos = end_pos = cls_index
                else:
                    start_pos, end_pos = start_tok, end_tok

            features.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "start_positions": start_pos,
                "end_positions": end_pos,
            })
    return features
