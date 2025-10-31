# src/train.py

import torch
import optuna
import os
import yaml
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

from data_processing import load_config, load_and_clean_data, text_preprocessor

tqdm.pandas(desc="Preprocessing Comments")

def objective(trial, config, train_dataset, val_dataset):
    """The objective function for Optuna to optimize."""
    hyperparams = {}
    for param_config in config['hpo']['parameters']:
        name = param_config['name']
        param_type = param_config['type']
        param_args = param_config['params']
        if param_type == 'categorical':
            hyperparams[name] = trial.suggest_categorical(name, **param_args)
        elif param_type == 'float':
            hyperparams[name] = trial.suggest_float(name, **param_args)
    
    model = AutoModelForSequenceClassification.from_pretrained(config['model']['base_name'], num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=hyperparams['learning_rate'], weight_decay=hyperparams['weight_decay'])
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], num_workers=0)
    
    epochs = config['train']['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * epochs * len(train_loader)), num_training_steps=epochs * len(train_loader))
    
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch+1}", leave=False):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return avg_val_loss

def train_final_model(config, best_params, X_train_full, y_train_full, tokenizer):
    """Trains the final model on the entire training dataset."""
    print("\n" + "="*50)
    print(" TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("="*50)
    print(f"Using hyperparameters: {best_params}")

    train_encodings = tokenizer(X_train_full.tolist(), padding="max_length", truncation=True, max_length=config['train']['max_length'], return_tensors="pt")
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(y_train_full.values))
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, num_workers=0)
    
    model = AutoModelForSequenceClassification.from_pretrained(config['model']['base_name'], num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    
    epochs = config['train']['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * epochs * len(train_loader)), num_training_steps=epochs * len(train_loader))

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
    
    save_path = config['model']['save_path']
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nFinal production model saved to: {save_path}")

if __name__ == "__main__":
    config = load_config()
    
    print("\nLoading and Cleaning Data...")
    df = load_and_clean_data(config)
    df['comment'] = df['comment'].progress_apply(text_preprocessor)
    print("Data processing complete.")

    X = df['comment']
    y = df['label_id']
    X_train_full, _, y_train_full, _ = train_test_split(
        X, y, test_size=config['train']['test_size'], random_state=42, stratify=y
    )
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_name'])
    
    best_hyperparameters = {}

    if config['hpo']['enabled']:
        print("\nMODE: Hyperparameter Search ENABLED.")
        print("="*50)
        print(" STEP 1: STARTING HYPERPARAMETER SEARCH WITH OPTUNA")
        print("="*50)

        # Create a smaller validation set just for the search
        X_train_search, X_val_search, y_train_search, y_val_search = train_test_split(
            X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
        )
        train_search_encodings = tokenizer(X_train_search.tolist(), padding="max_length", truncation=True, max_length=config['train']['max_length'], return_tensors="pt")
        val_search_encodings = tokenizer(X_val_search.tolist(), padding="max_length", truncation=True, max_length=config['train']['max_length'], return_tensors="pt")
        train_search_dataset = TensorDataset(train_search_encodings['input_ids'], train_search_encodings['attention_mask'], torch.tensor(y_train_search.values))
        val_search_dataset = TensorDataset(val_search_encodings['input_ids'], val_search_encodings['attention_mask'], torch.tensor(y_val_search.values))

        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        study.optimize(
            lambda trial: objective(trial, config, train_search_dataset, val_search_dataset), 
            n_trials=config['hpo']['n_trials']
        )
        
        print("\n--- Hyperparameter Search Complete ---")
        best_hyperparameters = study.best_params
        print("Best hyperparameters found: ", best_hyperparameters)

        # Save the results for future use
        best_params_path = "config/best_params.yml"
        with open(best_params_path, 'w') as f:
            yaml.dump(best_hyperparameters, f)
        print(f"Best hyperparameters saved to {best_params_path}")

    else:
        print("\nMODE: Hyperparameter Search DISABLED.")
        print("Skipping Optuna search and using fixed parameters from config file.")
        best_hyperparameters = config['fixed_hyperparameters']

    train_final_model(config, best_hyperparameters, X_train_full, y_train_full, tokenizer)