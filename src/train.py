import pandas as pd
import torch 
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from tqdm import tqdm
import os

from data_processing import load_config, load_and_clean_data, text_preprocessor

def train_model():
    """Main function to orchestrate the model training process"""
    config = load_config()
    df = load_and_clean_data(config)

    df['comment'] = df['comment'].apply(text_preprocessor)

    X = df['comment']
    y = df['label_id']
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(config['train']['test_size'] + config['train']['val_size']), random_state=42, stratify=y
    )
    val_size_adjusted = config['train']['val_size'] / (config['train']['test_size'] + config['train']['val_size'])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1-val_size_adjusted), random_state=42, stratify=y_temp
    )

    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_name'])

    def tokenize_data(texts, max_len):
        return tokenizer(texts.tolist(), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    
    train_encodings = tokenize_data(X_train, config['train']['max_length'])
    val_encodings = tokenize_data(X_val, config['train']['max_length'])

    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(y_train.values))
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(y_val.values))

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'])

    model = AutoModelForSequenceClassification.from_pretrained(config['model']['base_name'], num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config['train']['learning_rate'])

    # Training Loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(config['train']['epochs']):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            os.makedirs(config['model']['save_path'], exist_ok=True)
            model.save_pretrained(config['model']['save_path'])
            tokenizer.save_pretrained(config['model']['save_path'])
            print(f"Model saved to {config['model']['save_path']}")
        else:
            patience_counter += 1
            if patience_counter >= config['train']['patience']:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    train_model()