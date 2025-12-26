# src/config.py
from dataclasses import dataclass

@dataclass
class QAConfig:
    model_name: str = "vinai/bartpho-syllable"   # hoặc bartpho-word
    max_source_len: int = 512
    max_target_len: int = 64
    lr: float = 2e-5
    batch_size: int = 4
    epochs: int = 2
    output_dir: str = "outputs/bartpho_qa"
    use_peft: bool = True

@dataclass
class AEConfig:
    model_name: str = "microsoft/mdeberta-v3-base"
    max_len: int = 384
    doc_stride: int = 128
    lr: float = 2e-5
    batch_size: int = 8
    epochs: int = 2
    output_dir: str = "outputs/mdeberta_ae"
    use_peft: bool = False

@dataclass
class EvalConfig:
    bertscore_model: str = "vinai/phobert-base"  # Vietnamese-friendly; cũng có thể dùng xlm-roberta-large
