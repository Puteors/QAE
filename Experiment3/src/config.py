# src/config.py
class Config:
    # ============== MODEL ==============
    MODEL_NAME = "google/t5gemma-2b-2b-prefixlm-it"
    
    # ============== TOKENIZER ==============
    MAX_SOURCE_LENGTH = 1024
    MAX_TARGET_LENGTH = 512
    
    # ============== TRAINING ==============
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-4      # Tăng LR khi dùng LoRA (từ 2e-5 lên 2e-4)
    EPOCHS = 10
    OUTPUT_DIR = "./results"
    
    # ============== EVALUATION & SAVING ==============
    EVAL_STEPS = 100
    SAVE_TOTAL_LIMIT = 2
    
    # ============== DATA FORMAT ==============
    PAIR_SEP = " [SEP] "
    Q_TAG = "question: "
    A_TAG = " answer: "
    QA_PREFIX = "generate_qa: "
    
    # ============== PEFT/LoRA CONFIG ==============
    USE_PEFT = True                    # Bật/tắt PEFT
    LORA_R = 16                        # Rank (8, 16, 32, 64)
    LORA_ALPHA = 32                    # Alpha scaling (thường = 2 * R)
    LORA_DROPOUT = 0.1                 # Dropout rate
    LORA_TARGET_MODULES = [            # Target modules cho Gemma
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj",
        "gate_proj", 
        "up_proj", 
        "down_proj"
    ]
    
    # ============== QUANTIZATION (Optional) ==============
    USE_4BIT = False                   # Bật QLoRA để tiết kiệm thêm VRAM
    BNB_4BIT_QUANT_TYPE = "nf4"
    BNB_4BIT_COMPUTE_DTYPE = "float16"