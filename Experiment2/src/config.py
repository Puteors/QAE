class Config:
    MODEL_NAME = "VietAI/vit5-base"
    MAX_SOURCE_LENGTH = 1024
    MAX_TARGET_LENGTH = 512
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5
    EPOCHS = 5
    OUTPUT_DIR = "./results"
    
    # Định nghĩa các token đánh dấu để model học và sau này mình dùng để tách chuỗi
    PAIR_SEP = " [SEP] "     # Ngăn cách giữa các cặp Q&A
    Q_TAG = "question: "     # Đánh dấu bắt đầu câu hỏi
    A_TAG = " answer: "      # Đánh dấu bắt đầu câu trả lời (lưu ý khoảng trắng)
    QA_PREFIX = "generate_qa: " 