import numpy as np
import evaluate

class AEMetrics:
    def __init__(self):
        """
        Metrics cho bài toán Answer Extraction (Token Classification).
        Sử dụng 'seqeval' để tính Precision, Recall, F1 cho các thực thể (Entity).
        """
        print("Loading AE metrics: seqeval...")
        self.metric = evaluate.load("seqeval")
        # Định nghĩa map từ ID sang Label (khớp với dataset: 0->O, 1->B, 2->I)
        self.id2label = {0: "O", 1: "B-ANS", 2: "I-ANS"}

    def compute(self, eval_preds):
        predictions, labels = eval_preds
        
        # Predictions đang là logits -> lấy argmax để ra id class
        predictions = np.argmax(predictions, axis=2)

        # Loại bỏ các label đặc biệt (-100) và convert ID sang Text Label
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Tính toán kết quả
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
        Metrics bao gồm: BLEU (sacrebleu) và BERTScore.
        """
        self.tokenizer = tokenizer
        print("Loading QG metrics: SacreBLEU & BERTScore...")
        self.bleu_metric = evaluate.load("sacrebleu")
        self.bertscore_metric = evaluate.load("bertscore")

    def compute(self, eval_preds):
        preds, labels = eval_preds
        
        if isinstance(preds, tuple):
            preds = preds[0]
            
        # Giải mã predictions
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Xử lý nhãn (-100 -> pad_id)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Làm sạch văn bản
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Tính BLEU
        # SacreBLEU yêu cầu references dạng list of list
        result_bleu = self.bleu_metric.compute(
            predictions=decoded_preds, 
            references=[[l] for l in decoded_labels]
        )
        
        # Tính BERTScore (Tiếng Việt)
        result_bert = self.bertscore_metric.compute(
            predictions=decoded_preds, 
            references=decoded_labels, 
            lang="vi"
        )
        
        return {
            "bleu": result_bleu["score"],
            "bertscore_f1": np.mean(result_bert["f1"])
        }