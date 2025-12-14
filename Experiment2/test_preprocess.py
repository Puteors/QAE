import sys
import os
from transformers import AutoTokenizer
from src.config import Config
from src.dataset import QAGenDataset

def check_data():
    # 1. Load Tokenizer
    print(f"Đang tải tokenizer: {Config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # 2. Load Dataset
    data_path = "./data/train.json"
    if not os.path.exists(data_path):
        print(f"Lỗi: Không tìm thấy file tại {data_path}")
        return

    print("Đang xử lý dữ liệu qua QAGenDataset...")
    dataset = QAGenDataset(data_path, tokenizer)
    
    print(f"\n=== TỔNG QUAN ===")
    print(f"Tổng số mẫu (Unique Contexts): {len(dataset)}")
    
    # 3. Soi chi tiết 3 mẫu đầu tiên
    print("\n=== KIỂM TRA CHI TIẾT 3 MẪU ĐẦU TIÊN ===")
    
    # Lặp qua 3 phần tử đầu
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        
        input_ids = sample['input_ids']
        labels = sample['labels']
        
        # Decode ngược lại thành text để kiểm tra
        # skip_special_tokens=False để nhìn thấy cả padding và </s>
        decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
        decoded_label = tokenizer.decode(labels, skip_special_tokens=False)
        
        print(f"\n--- Mẫu số {i+1} ---")
        
        # Kiểm tra độ dài
        print(f"Độ dài Input IDs: {len(input_ids)} (Max: {Config.MAX_SOURCE_LENGTH})")
        print(f"Độ dài Label IDs: {len(labels)} (Max: {Config.MAX_TARGET_LENGTH})")
        
        # In nội dung
        print(f"\n[INPUT RAW]:\n{decoded_input[:300]} ... (đã cắt bớt)")
        print(f"\n[TARGET RAW] (Đây là cái model sẽ học):\n{decoded_label}")
        
        print("-" * 50)

if __name__ == "__main__":
    check_data()