import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .data_processing import text_preprocessor, load_config

class SentimentPredictor:
    def __init__(self, config_path="config/params.yml"):
        config = load_config(config_path)
        model_path = config['model']['save_path']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("SentimentPredictor initialized and model loaded.")

    def predict(self, text):
        """Predicts sentiment for a single piece of text."""
        # Preprocess the input text
        processed_text = text_preprocessor(text)
        
        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64 # Should match training config
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        predicted_class_id = logits.argmax().item()
        sentiment = "HAPPY" if predicted_class_id == 0 else "SAD"
        
        return {
            "text": text,
            "sentiment": sentiment,
            "label_id": predicted_class_id
        }