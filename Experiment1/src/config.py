class Config:
    # --- MODEL ---
    # Hãy đảm bảo tên model chính xác trên HuggingFace
    # Ví dụ: "Qwen/Qwen2.5-1.5B-Instruct" hoặc "Qwen/Qwen2.5-3B"
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" 
    
    # --- DATA PATHS ---
    TRAIN_FILE = "./data/train.json"
    VAL_FILE = "./data/validation.json"
    
    # --- HYPERPARAMETERS ---
    # Qwen có context window lớn, nhưng để train nhanh demo thì để thấp
    MAX_SOURCE_LENGTH = 512 
    MAX_TARGET_LENGTH = 512
    BATCH_SIZE = 2             # Giảm batch size vì Qwen tốn VRAM hơn T5
    LEARNING_RATE = 2e-4       # Qwen thường dùng LR cao hơn T5 một chút với LoRA
    EPOCHS = 1
    WEIGHT_DECAY = 0.01
    
    # --- STEPS & SAVING ---
    OUTPUT_DIR = "./results_qwen"
    EVAL_STEPS = 50
    SAVE_TOTAL_LIMIT = 2
    LOGGING_STEPS = 10
    
    # --- PEFT / LORA CONFIG CHO QWEN ---
    LORA_R = 16                # Tăng R lên chút cho model lớn
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    # Các modules quan trọng của Qwen cần được áp dụng LoRA
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # --- FORMATTING ---
    # Qwen (Instruct) thường dùng Chat Template, nhưng ở đây ta dùng format đơn giản:
    # Input: Generate QA based on: {context}
    # Response: Question: {q} Answer: {a}
    PROMPT_TEMPLATE = "Generate QA based on the following text:\n{context}\n\nResponse:"