# src/inference_test.py
import argparse
import json
from typing import Any, Dict, List

from inference import predict_bartpho, predict_mdeberta_ae
from src.evaluate import bertscore  # dùng file evaluate.py mình đưa trước đó

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

    if args.task == "qa":
        outputs = predict_bartpho(args.model_dir, items)
    else:
        outputs = predict_mdeberta_ae(args.model_dir, items)

    preds = [x["pred"] for x in outputs]
    refs = [x["gold"] for x in outputs]

    scores = bertscore(preds, refs)
    print("BERTScore:", scores)

    # show a few examples
    for x in outputs[:5]:
        print("-" * 60)
        print("Q:", x["question"])
        print("P:", x["pred"])
        print("G:", x["gold"])

if __name__ == "__main__":
    main()
