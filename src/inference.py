import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSeq2SeqLM

class QGPipeline:
    def __init__(self, ae_model_path, qg_model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading models on {self.device}...")
        
        # Load AE
        self.ae_tokenizer = AutoTokenizer.from_pretrained(ae_model_path)
        self.ae_model = AutoModelForTokenClassification.from_pretrained(ae_model_path).to(self.device)
        
        # Load QG
        self.qg_tokenizer = AutoTokenizer.from_pretrained(qg_model_path)
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg_model_path).to(self.device)

    def extract_answers(self, text):
        inputs = self.ae_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.ae_model(**inputs).logits
        
        predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
        tokens = self.ae_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        answers = []
        current_ans = []
        
        # Logic giải mã BIO tags đơn giản
        for token, label in zip(tokens, predictions):
            if label == 1: # B-Answer
                if current_ans:
                    answers.append(self.ae_tokenizer.convert_tokens_to_string(current_ans).replace(" ", " ").strip())
                    current_ans = []
                current_ans.append(token)
            elif label == 2: # I-Answer
                if current_ans: current_ans.append(token)
            else: # O
                if current_ans:
                    answers.append(self.ae_tokenizer.convert_tokens_to_string(current_ans).replace(" ", " ").strip())
                    current_ans = []
                    
        # Lọc kết quả trùng và quá ngắn
        return list(set([ans for ans in answers if len(ans) > 2]))

    def generate_questions(self, context, answers):
        qa_pairs = []
        for ans in answers:
            # Tạo input đánh dấu highlight
            # Lưu ý: chỉ replace lần xuất hiện đầu tiên để đơn giản hóa
            input_text = context.replace(ans, f"<hl> {ans} <hl>", 1)
            
            inputs = self.qg_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.qg_model.generate(
                    inputs["input_ids"],
                    max_length=64,
                    num_beams=5, # Beam search để câu hỏi mượt hơn
                    early_stopping=True
                )
            
            question = self.qg_tokenizer.decode(outputs[0], skip_special_tokens=True)
            qa_pairs.append({"answer": ans, "question": question})
            
        return qa_pairs

    def predict(self, paragraph):
        answers = self.extract_answers(paragraph)
        results = self.generate_questions(paragraph, answers)
        return results