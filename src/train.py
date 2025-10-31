# src/train.py

import torch
import optuna
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

from data_processing import load_config, load_and_clean_data, text_preprocessor

# Initialize tqdm for pandas (must be done once)
tqdm.pandas(desc="Preprocessing Comments")

def objective(trial, config, train_dataset, val_dataset):
    """
    The objective function for Optuna to optimize.
    A 'trial' represents a single run with a specific set of hyperparameters.
    """
    # 1. Suggest Hyperparameters for this trial
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # 2. Setup model and optimizer for this trial
    model = AutoModelForSequenceClassification.from_pretrained(config['model']['base_name'], num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 3. Setup learning rate scheduler
    epochs = config['train']['epochs']
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps) # 10% warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # 4. Training and Validation Loop
    for epoch in range(epochs):
        model.train()
        for batch in train_loader: # Removed tqdm here for cleaner logs during search
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Validation at the end of each epoch
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        trial.report(avg_val_loss, epoch)

        # Pruning: Stop unpromising trials early
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return avg_val_loss

def train_final_model(config, best_params, X_train_full, y_train_full, tokenizer):
    """
    Trains the final model on the entire training dataset using the best hyperparameters.
    """
    print("\n" + "="*50)
    print(" STEP 2: TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("="*50)
    print(f"Using hyperparameters: {best_params}")

    # 1. Prepare full training dataset
    train_encodings = tokenizer(X_train_full.tolist(), padding="max_length", truncation=True, max_length=config['train']['max_length'], return_tensors="pt")
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(y_train_full.values))
    
    # Use the best batch size found by Optuna
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    
    # 2. Initialize new model, optimizer, and scheduler
    model = AutoModelForSequenceClassification.from_pretrained(config['model']['base_name'], num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    
    epochs = config['train']['epochs']
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # 3. Final Training Loop (no validation, as we use all data for training)
    model.train()
    for epoch in range(epochs):
        for batch in tqdm(train_loader, desc=f"Final Training Epoch {epoch+1}/{epochs}"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

    # 4. Save the final model
    save_path = config['model']['save_path']
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nFinal production model saved to: {save_path}")

def run_training_pipeline():
    """Main function to orchestrate the entire training pipeline."""
    config = load_config()
    df = load_and_clean_data(config)
    df['comment'] = df['comment'].progress_apply(text_preprocessor)

    # Data Splitting s
    # We create three sets:
    # 1. train_for_search: To train models during HPO
    # 2. val_for_search: To evaluate models during HPO
    # 3. test_set: Held-out set, never touched during training or HPO
    # 4. train_full: The combination of (1) and (2) for final model training
    X = df['comment']
    y = df['label_id']
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=config['train']['test_size'], random_state=42, stratify=y
    )
    X_train_search, X_val_search, y_train_search, y_val_search = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
    )

    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_name'])
    
    train_search_encodings = tokenizer(X_train_search.tolist(), padding="max_length", truncation=True, max_length=config['train']['max_length'], return_tensors="pt")
    val_search_encodings = tokenizer(X_val_search.tolist(), padding="max_length", truncation=True, max_length=config['train']['max_length'], return_tensors="pt")

    train_search_dataset = TensorDataset(train_search_encodings['input_ids'], train_search_encodings['attention_mask'], torch.tensor(y_train_search.values))
    val_search_dataset = TensorDataset(val_search_encodings['input_ids'], val_search_encodings['attention_mask'], torch.tensor(y_val_search.values))

    # HYPERPARAMETER SEARCH 
    print("="*50)
    print(" STEP 1: STARTING HYPERPARAMETER SEARCH WITH OPTUNA")
    print("="*50)
    study = optuna.create_study(direction="minimize", study_name="sentiment_analysis_optimization", pruner=optuna.pruners.MedianPruner())
    study.optimize(
        lambda trial: objective(trial, config, train_search_dataset, val_search_dataset), 
        n_trials=20
    )
    
    print("\n--- Hyperparameter Search Complete ---")
    print(f"Best validation loss: {study.best_trial.value:.4f}")
    print("Best hyperparameters found: ", study.best_params)

    # FINAL MODEL TRAINING
    # Now, we re-train on the full training data (X_train_full) using the best params.
    best_hyperparameters = study.best_params
    train_final_model(config, best_hyperparameters, X_train_full, y_train_full, tokenizer)


if __name__ == "__main__":
    run_training_pipeline()
"""
for production I want the model to be run on CPU as well, what method should I use? 
quantization? 
knowledge distilation? 
or other methods?"""