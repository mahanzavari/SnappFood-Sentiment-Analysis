import torch
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data_processing import load_config, load_and_clean_data, text_preprocessor

def evaluate_final_model():
    print("--- Loading Configuration and Data for Final Evaluation ---")
    config = load_config()
    df = load_and_clean_data(config)
    
    tqdm.pandas(desc="Preprocessing Test Comments")
    df['comment'] = df['comment'].progress_apply(text_preprocessor)

    print("\n--- Re-creating the Held-Out Test Set ---")
    X = df['comment']
    y = df['label_id']

    # We must use the exact same splits as in train.py to get the correct test set
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=config['train']['test_size'], random_state=42, stratify=y
    )
    print(f"Evaluating final model on {len(X_test)} test samples.")

    print("\n--- Loading Final Production Model ---")
    model_path = config['model']['save_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run src/train.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    test_encodings = tokenizer(X_test.tolist(), padding="max_length", truncation=True, max_length=config['train']['max_length'], return_tensors="pt")
    test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(y_test.values))
    test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'])

    print("\n--- Running Inference on Test Set ---")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            all_preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n\n--- Final Model Performance Report ---")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['HAPPY', 'SAD']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['HAPPY', 'SAD'])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix for Final Model on Test Set")
    plt.show()

if __name__ == "__main__":
    evaluate_final_model()