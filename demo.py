from src.inference import QGPipeline

# Đường dẫn đến model đã train xong
ae_path = "./outputs/ae_model/best"
qg_path = "./outputs/qg_model/best"

pipeline = QGPipeline(ae_path, qg_path)

paragraph = "Phạm Văn Đồng là Thủ tướng đầu tiên của nước Cộng hòa Xã hội chủ nghĩa Việt Nam từ năm 1976."
results = pipeline.predict(paragraph)

print("Đoạn văn:", paragraph)
for item in results:
    print(f"- Answer: {item['answer']}")
    print(f"- Question: {item['question']}")