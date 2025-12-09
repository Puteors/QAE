import os
from src.train_ae import train_ae_model
from src.train_qg import train_qg_model

# Cấu hình đường dẫn dữ liệu
TRAIN_DATA = "./data/train.json"
VAL_DATA = "./data/validation.json"
TEST_DATA = "./data/test.json"

def main():
    # Kiểm tra file tồn tại
    for path in [TRAIN_DATA, VAL_DATA, TEST_DATA]:
        if not os.path.exists(path):
            print(f"LỖI: Không tìm thấy file {path}")
            return

    # 1. Huấn luyện Answer Extraction
    print("\n" + "="*40)
    print(" BƯỚC 1: HUẤN LUYỆN AE MODEL (mDeBERTa)")
    print("="*40)
    train_ae_model(TRAIN_DATA, VAL_DATA, TEST_DATA)
    
    # 2. Huấn luyện Question Generation
    print("\n" + "="*40)
    print(" BƯỚC 2: HUẤN LUYỆN QG MODEL (BARTpho)")
    print("="*40)
    train_qg_model(TRAIN_DATA, VAL_DATA, TEST_DATA)

    print("\n" + "="*40)
    print(" HOÀN TẤT TOÀN BỘ QUÁ TRÌNH")
    print("="*40)

if __name__ == "__main__":
    main()