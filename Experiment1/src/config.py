class Config:
    # --- MODEL & TOKENIZER ---
    MODEL_NAME = "Qwen/Qwen3-4B" 
    
    # --- DATA PATHS ---
    TRAIN_FILE = "./data/train.json"
    VAL_FILE = "./data/validation.json"
    TEST_FILE = "./data/test.json"
    
    # --- HYPERPARAMETERS ---
    MAX_SOURCE_LENGTH = 1024
    MAX_TARGET_LENGTH = 512
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5
    EPOCHS = 1
    WEIGHT_DECAY = 0.01
    
    # --- STEPS & SAVING ---
    OUTPUT_DIR = "./results"
    EVAL_STEPS = 100          
    SAVE_TOTAL_LIMIT = 2      
    LOGGING_STEPS = 50
    
    # --- PEFT / LORA CONFIG ---
    LORA_R = 8
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    
    # --- FORMATTING (Prompt Engineering) ---
    PAIR_SEP = " [SEP] "
    Q_TAG = "question: "
    A_TAG = " answer: "
    QA_PREFIX = "generate_qa: "