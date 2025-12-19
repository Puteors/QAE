# my_evaluate.py
import evaluate

def bertscore(preds, refs, model_type="xlm-roberta-base"):
    bs = evaluate.load("bertscore")
    out = bs.compute(
        predictions=preds,
        references=refs,
        lang="vi",
        model_type=model_type,
        rescale_with_baseline=True,
    )
    return {
        "P": float(sum(out["precision"]) / len(out["precision"])),
        "R": float(sum(out["recall"]) / len(out["recall"])),
        "F1": float(sum(out["f1"]) / len(out["f1"])),
    }
