import numpy as np
import evaluate

class AEMetrics:
    def __init__(self):
        """
        Metrics cho bài toán Answer Extraction (Token Classification).
        Giữ nguyên logic cũ: Seqeval (Precision, Recall, F1).
        """
        print("Loading AE metrics: seqeval...")
        self.metric = evaluate.load("seqeval")
        self.id2label = {0: "O", 1: "B-ANS", 2: "I-ANS"}

    def compute(self, eval_preds):
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

class QGMetrics:
    def __init__(self, tokenizer):
        """
        Metrics cho bài toán Question Generation.
        Sử dụng: ROUGE và BERTScore.
        """
        self.tokenizer = tokenizer
        
        print("Loading QG metrics: ROUGE & BERTScore...")
        # Load ROUGE
        self.rouge_metric = evaluate.load("rouge")
        # Load BERTScore
        self.bertscore_metric = evaluate.load("bertscore")

    def compute(self, eval_preds):
        preds, labels = eval_preds
        
        if isinstance(preds, tuple):
            preds = preds[0]
            
        # 1. Giải mã (Decode)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Xử lý nhãn (-100 -> pad_id)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 2. Làm sạch văn bản
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # 3. Tính ROUGE
        # use_stemmer=False vì stemmer tiếng Anh không hiệu quả với tiếng Việt
        result_rouge = self.rouge_metric.compute(
            predictions=decoded_preds, 
            references=decoded_labels,
            use_stemmer=False 
        )
        
        # 4. Tính BERTScore (Tiếng Việt)
        result_bert = self.bertscore_metric.compute(
            predictions=decoded_preds, 
            references=decoded_labels, 
            lang="vi"
        )
        
        # 5. Trả về kết quả tổng hợp
        return {
            "rouge1": result_rouge["rouge1"],
            "rouge2": result_rouge["rouge2"],
            "rougeL": result_rouge["rougeL"],
            "bertscore_f1": np.mean(result_bert["f1"])
        }