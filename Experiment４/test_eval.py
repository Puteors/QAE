from my_evaluate import bertscore

preds = ["Lâm Bá Kiệt", "Thủ tướng"]
refs  = ["Lâm Bá Kiệt", "Thủ tướng Chính phủ"]

print(bertscore(preds, refs))
