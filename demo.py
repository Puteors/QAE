import os
from src.inference import QGPipeline

# 1. Đường dẫn đến model tốt nhất đã lưu sau khi train
AE_MODEL_PATH = "./outputs/ae_model/best"
QG_MODEL_PATH = "./outputs/qg_model/best"

def main():
    # Kiểm tra xem đã train xong chưa
    if not os.path.exists(AE_MODEL_PATH) or not os.path.exists(QG_MODEL_PATH):
        print("LỖI: Chưa tìm thấy model đã huấn luyện.")
        print("Vui lòng chạy 'python run_training.py' trước!")
        return

    print("--- Đang tải mô hình (Loading Models)... ---")
    pipeline = QGPipeline(AE_MODEL_PATH, QG_MODEL_PATH)
    print("--- Tải xong! Sẵn sàng demo. ---")

    # 2. Đoạn văn mẫu (Bạn có thể thay đổi tùy ý)
    paragraph = """
    Hồ Chí Minh (19 tháng 5 năm 1890 – 2 tháng 9 năm 1969) là một nhà cách mạng và chính khách người Việt Nam. 
    Ông là người sáng lập Đảng Cộng sản Việt Nam, từng là Chủ tịch nước Việt Nam Dân chủ Cộng hòa từ năm 1945 đến năm 1969. 
    Tại Đại hội lần thứ II của Đảng (1951), ông được bầu làm Chủ tịch Ban Chấp hành Trung ương Đảng Lao động Việt Nam.
    """

    print(f"\nĐoạn văn đầu vào:\n{paragraph.strip()}\n")
    print("-" * 50)

    # 3. Thực hiện dự đoán
    results = pipeline.predict(paragraph)

    # 4. In kết quả
    if not results:
        print("Không tìm thấy câu hỏi/câu trả lời nào phù hợp.")
    else:
        for i, item in enumerate(results, 1):
            print(f"Cặp {i}:")
            print(f"  ➢ Câu trả lời (AE): {item['answer']}")
            print(f"  ➢ Câu hỏi (QG)    : {item['question']}")
            print("-" * 50)

if __name__ == "__main__":
    main()