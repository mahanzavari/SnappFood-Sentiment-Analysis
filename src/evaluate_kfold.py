# src/evaluate_kfold.py

import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from data_processing import load_config, load_and_clean_data, text_preprocessor

tqdm.pandas(desc="Preprocessing Comments")

def train_and_evaluate_fold(config, train_dataset, val_dataset, fold_num):
    """Trains a model on one fold and evaluates it."""
    print(f"\n--- Starting Fold {fold_num} ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(config['model']['base_name'], num_labels=2)
    model.to(device)

    best_params = {'learning_rate': 2e-05, 'weight_decay': 0.01, 'batch_size': 32} 
    
    optimizer = AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'])

    epochs = config['train']['epochs']
    num_training_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
    )

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
    
    # Evaluation loop
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro')

def run_kfold_evaluation():
    config = load_config()
    df = load_and_clean_data(config)
    df['comment'] = df['comment'].progress_apply(text_preprocessor)

    X = df['comment'].values
    y = df['label_id'].values
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_name'])
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accuracies, fold_f1_scores = [], []

    for fold_num, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_encodings = tokenizer(X_train.tolist(), padding="max_length", truncation=True, max_length=config['train']['max_length'], return_tensors="pt")
        val_encodings = tokenizer(X_val.tolist(), padding="max_length", truncation=True, max_length=config['train']['max_length'], return_tensors="pt")

        train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(y_train))
        val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(y_val))

        acc, f1 = train_and_evaluate_fold(config, train_dataset, val_dataset, fold_num)
        fold_accuracies.append(acc)
        fold_f1_scores.append(f1)
        print(f"Fold {fold_num} -> Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")

    print("\n--- K-Fold Cross-Validation Complete ---")
    print(f"Average Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"Average Macro F1-Score: {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}")

if __name__ == "__main__":
    run_kfold_evaluation()