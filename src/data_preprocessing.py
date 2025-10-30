import pandas as pd
import yaml
import re
from hazm import Normalizer, WordTokenizer
import warnings 

# warnings.filterwarnings("ignore")
normalizer = Normalizer()

def load_config(config_path ='config/params.yml'):
    """Loads the parameters from the configuratiom file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def load_and_clean_data(config):
    """Loads raw data and applies all cleaning steps from the notebook."""
    df = pd.read_csv(config["data"]["raw_path"], on_bad_lines='skip', delimiter=',')

    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['comment'], keep='first', inplace=True)

    df['label'] = df['label'].str.upper().str.strip()
    df = df[~((df['label'] == 'HAPPY') & (df['label_id'] != 0))]
    df = df[~((df['label'] == 'SAD') & (df['label_id'] != 1))]
    df['label_id'] = df['label_id'].astype(int)

    # Filter non-Persian and long comments
    df = df[~df['comment'].str.contains(r'[a-zA-Z]', na=False)]
    df['word_count'] = df['comment'].apply(lambda x: len(str(x).split()))
    max_words = df['word_count'].quantile(config['preprocess']['max_words_quantile'])
    df = df[df['word_count'] <= max_words]

    # Reset index and select relevant columns
    df = df[['comment', 'label_id']].reset_index(drop=True)
    print(f"Data loaded and cleaned. Shape: {df.shape}")
    return df

def text_preprocessor(text):
    """
    Simple preprocessor for BERT.
    Just normalizes and cleans unwanted characters. No stopword removal or stemming.
    """
    # normalizer = Normalizer() # Huge bottleneck
    text = normalizer.normalize(text)
    # Remove anything that's not Persian characters, numbers, or whitespace
    text = re.sub(r'[^\w\s\u0600-\u06FF]', '', text)
    return text