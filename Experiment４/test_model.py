# test_model.py
import argparse
import json
from typing import Any, Dict, List

from inference import predict_bartpho, predict_mdeberta_ae


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["qa", "ae"], required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input_json", default="data/test.json")
    ap.add_argument("--output_json", default="outputs/test_preds.json")
    ap.add_argument("--n_show", type=int, default=10)

    # gen params (qa)
    ap.add_argument("--max_source_len", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--num_beams", type=int, default=4)

    # ae params
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--max_answer_len", type=int, default=40)

    args = ap.parse_args()

    items = load_json(args.input_json)
    print(f"Loaded {len(items)} samples from {args.input_json}")

    if args.task == "qa":
        outputs = predict_bartpho(
            model_dir=args.model_dir,
            items=items,
            max_source_len=args.max_source_len,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
    else:
        outputs = predict_mdeberta_ae(
            model_dir=args.model_dir,
            items=items,
            max_len=args.max_len,
            max_answer_len=args.max_answer_len,
        )

    # với test.json answers=null => gold sẽ là ""
    # ta chỉ cần lưu pred
    save_json(args.output_json, outputs)
    print(f"Saved predictions to {args.output_json}")

    print("\nPreview:")
    for x in outputs[: args.n_show]:
        print("-" * 60)
        print("id:", x.get("id"))
        print("Q :", x.get("question"))
        print("P :", x.get("pred"))

if __name__ == "__main__":
    main()
