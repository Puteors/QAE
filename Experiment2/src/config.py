class Config:
    MODEL_NAME = "VietAI/vit5-base"
    MAX_SOURCE_LENGTH = 1024
    MAX_TARGET_LENGTH = 512
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5
    EPOCHS = 10               # Tăng Epoch lên vì ta sẽ dừng sớm (early stopping) hoặc lấy best step
    OUTPUT_DIR = "./results"
    
    # Cấu hình Steps
    EVAL_STEPS = 100          # Đánh giá và lưu sau mỗi 100 bước
    SAVE_TOTAL_LIMIT = 2      # Chỉ giữ lại 2 checkpoint tốt nhất để tiết kiệm ổ cứng
    
    # Token định dạng (giữ nguyên như cũ)
    PAIR_SEP = " [SEP] "
    Q_TAG = "question: "
    A_TAG = " answer: "
    QA_PREFIX = "generate_qa: "