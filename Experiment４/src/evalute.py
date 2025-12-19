# src/evaluate.py
from typing import List
import evaluate
from src.config import EvalConfig

def bertscore(preds: List[str], refs: List[str], cfg=EvalConfig()):
    bs = evaluate.load("bertscore")
    out = bs.compute(
        predictions=preds,
        references=refs,
        lang="vi",
        model_type=cfg.bertscore_model,
        rescale_with_baseline=True,
    )
    return {
        "P": float(sum(out["precision"]) / len(out["precision"])),
        "R": float(sum(out["recall"]) / len(out["recall"])),
        "F1": float(sum(out["f1"]) / len(out["f1"])),
    }
