# src/inference_test.py
import argparse
import json
from typing import Any, Dict, List

from tqdm import tqdm

from inference import predict_bartpho, predict_mdeberta_ae
from my_evaluate import bertscore


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["qa", "ae"], required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input_json", default="data/validation.json")
    ap.add_argument("--n", type=int, default=0, help="0=all, else take first n samples")
    args = ap.parse_args()

    items = load_json(args.input_json)
    if args.n and args.n > 0:
        items = items[: args.n]

    print(f"Running inference on {len(items)} samples...")

    # 👉 tqdm bao ngoài vì inference trả list
    with tqdm(total=len(items), desc="Inference", unit="sample") as pbar:
        if args.task == "qa":
            outputs = predict_bartpho(args.model_dir, items)
        else:
            outputs = predict_mdeberta_ae(args.model_dir, items)
        pbar.update(len(items))

    # lọc gold rỗng (tránh làm bẩn BERTScore)
    pairs = [(x["pred"], x["gold"]) for x in outputs if (x["gold"] or "").strip() != ""]
    if len(pairs) == 0:
        print("⚠️ No non-empty gold answers for evaluation.")
        return

    preds = [p for p, g in pairs]
    refs  = [g for p, g in pairs]

    scores = bertscore(preds, refs)
    print("BERTScore:", scores)

    # show a few examples
    print("\nExamples:")
    for x in outputs[:5]:
        print("-" * 60)
        print("Q:", x["question"])
        print("P:", x["pred"])
        print("G:", x["gold"])


if __name__ == "__main__":
    main()
