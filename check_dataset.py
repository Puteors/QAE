import torch
from transformers import AutoTokenizer
from src.dataset import AEDataset

# Cấu hình đường dẫn
DATA_PATH = "./data/train.json"
MODEL_NAME = "microsoft/mdeberta-v3-base"

def check_label_alignment():
    print(f"--- Đang tải Tokenizer: {MODEL_NAME} ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print(f"--- Đang tải dữ liệu từ: {DATA_PATH} ---")
    try:
        # Load dataset sử dụng class đã viết trong src/dataset.py
        dataset = AEDataset(DATA_PATH, tokenizer)
    except Exception as e:
        print(f"Lỗi khi load dataset: {e}")
        return

    print(f"\nTổng số mẫu dữ liệu hợp lệ: {len(dataset)}")
    print("Kiểm tra 5 mẫu đầu tiên:\n")

    # Mapping ID sang tên nhãn để dễ đọc
    id2label = {0: "O", 1: "B-ANS", 2: "I-ANS"}

    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        
        # Lấy thông tin gốc (Text) từ biến nội bộ của dataset
        raw_item = dataset.data[i]
        original_answer = raw_item['answer_text']
        context_snippet = raw_item['context'][:60] + "..."

        # Lấy thông tin đã mã hóa (Tensor)
        input_ids = sample['input_ids']
        labels = sample['labels']
        
        # Convert IDs ngược lại thành Tokens (text)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        print(f"{'='*20} MẪU {i+1} {'='*20}")
        print(f"Context: {context_snippet}")
        print(f"Answer Gốc: '{original_answer}'")
        
        # Tìm xem model "nhìn thấy" câu trả lời là gì dựa trên nhãn
        seen_tokens = []
        seen_labels = []
        decoded_ids = []
        
        for idx, (token, label_id) in enumerate(zip(tokens, labels)):
            label_id = label_id.item() # Convert tensor to int
            if label_id != 0: # Nếu không phải là O (Outside)
                label_str = id2label[label_id]
                seen_tokens.append(f"{token}")
                seen_labels.append(label_str)
                decoded_ids.append(input_ids[idx])

        # Reconstruct lại chuỗi từ các token được gán nhãn
        reconstructed_answer = tokenizer.decode(decoded_ids).strip()

        print(f"Tokens được gán nhãn: {seen_tokens}")
        print(f"Các nhãn tương ứng : {seen_labels}")
        print(f"Answer Tái tạo      : '{reconstructed_answer}'")
        
        # So sánh sơ bộ
        # Xóa khoảng trắng để so sánh cho dễ (do tokenizer hay thêm khoảng trắng)
        if original_answer.replace(" ", "") in reconstructed_answer.replace(" ", ""):
             print(">> ĐÁNH GIÁ: ✅ KHỚP (OK)")
        else:
             print(">> ĐÁNH GIÁ: ⚠️ KHÔNG KHỚP (Cần kiểm tra lại logic gán nhãn)")
        print("\n")

if __name__ == "__main__":
    check_label_alignment()