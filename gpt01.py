# gpt01.py
# Script d'enrichissement complet du dataset 1m pour 5 paires
# Inclut rotation de clés Google Custom Search, backoff, HMM, embeddings BERT et PCA

import os
import sys
import logging
import pandas as pd
import numpy as np
import requests
import time
import warnings
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from hmmlearn import hmm
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning

# === CONFIGURATION ===
# Symboles et plage de dates
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT']
start_date = '2024-01-01'
end_date   = '2025-02-02'
# Pattern CSV d'entrée dans le répertoire courant
input_pattern = '{symbol}_1m.csv'
# Répertoire et fichier de sortie
dataset_dir = 'dataset'
output_file = os.path.join(dataset_dir, 'crypto_dataset_complet.parquet')

# Environnement BERT
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BERT_MODEL = 'ElKulako/cryptobert'

# Liste de credentials (clé API + engine id) pour Google Custom Search
API_CREDENTIALS = [
    {
        'key': os.getenv('SEARCH_API_KEY1', 'AIzaSyAnWLOQ1NemZEi1YMOIk2NIjVyOgJ66XfQ'),
        'cx':  os.getenv('SEARCH_ENGINE_ID1', 'b0f490be8cf904edc')
    },
    {
        'key': os.getenv('SEARCH_API_KEY2', 'AIzaSyBxT1xuZf_3CsnEnQs__HZJX-S_Ak6NS0E'),
        'cx':  os.getenv('SEARCH_ENGINE_ID2', 'f054bdc505bee4cfa')
    },
    {
        'key': os.getenv('SEARCH_API_KEY3', 'AIzaSyAnWLOQ1NemZEi1YMOIk2NIjVyOgJ66XfQ'),
        'cx':  os.getenv('SEARCH_ENGINE_ID3', '61ae17808394d4ffa')
    }
]
enable_search = True

# Logging setup
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)
logger = setup_logger()

# === FONCTIONS UTILITAIRES ===

def get_news_snippets(query: str, cache: dict) -> str:
    """
    Récupère les snippets quotidiens en utilisant la rotation de credentials
    et backoff exponentiel en cas de code 429.
    """
    if not enable_search:
        return ''
    if query in cache:
        return cache[query]
    for cred in API_CREDENTIALS:
        backoff = 1
        for attempt in range(3):
            try:
                resp = requests.get(
                    'https://www.googleapis.com/customsearch/v1',
                    params={'key': cred['key'], 'cx': cred['cx'], 'q': query},
                    timeout=10
                )
            except Exception as e:
                logger.warning(f"SearchAPI network error: {e}")
                break
            if resp.status_code == 200:
                items = resp.json().get('items', [])
                text = ' '.join(item.get('snippet','') for item in items[:5])
                cache[query] = text
                return text
            elif resp.status_code == 429:
                logger.warning(f"Key {cred['key']} rate limited, backoff {backoff}s")
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                logger.warning(f"Key {cred['key']} error {resp.status_code}")
                break
    return cache.get(query, '')


def get_bert_embeddings(texts: list, tokenizer, model, device, batch_size: int = 32) -> np.ndarray:
    """
    Génère les embeddings CLS pour une liste de textes via HuggingFace.
    Affiche une barre de progression.
    """
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc='BERT embeddings'):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=64
            )
            ids = encoded['input_ids'].to(device)
            mask = encoded['attention_mask'].to(device)
            outputs = model(input_ids=ids, attention_mask=mask)
            cls_emb = outputs.last_hidden_state[:,0,:].cpu().numpy()
            embeddings.append(cls_emb)
    return np.vstack(embeddings)

# === TRAITEMENT PRINCIPAL ===
if __name__ == '__main__':
    all_dfs = []
    for symbol in symbols:
        csv_file = input_pattern.format(symbol=symbol)
        if not os.path.isfile(csv_file):
            logger.warning(f"File not found: {csv_file}, skipping.")
            continue
        logger.info(f"Processing {symbol}")
        # Lecture CSV et filtrage par date
        df = pd.read_csv(csv_file, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = df.loc[start_date:end_date]
        if df.empty:
            logger.warning(f"No data for {symbol} in range {start_date} to {end_date}.")
            continue
        # Ajout des colonnes symbol et date
        df['symbol'] = symbol
        df['date'] = df.index.date
        # Normalisation OHLCV
        scaler = MinMaxScaler()
        df[['open','high','low','close','volume']] = scaler.fit_transform(
            df[['open','high','low','close','volume']]
        )
        # Détection des régimes via HMM
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        X = df[['close','volume']].values
        hmm_model = hmm.GaussianHMM(n_components=3, covariance_type='diag', n_iter=500, random_state=42)
        hmm_model.fit(X)
        df['market_regime'] = hmm_model.predict(X)
        # Récupération des snippets d'actualité (quotidien)
        cache = {}
        unique_dates = df['date'].unique()
        snippets_map = {date: get_news_snippets(f"{symbol} crypto market news {date}", cache)
                        for date in unique_dates}
        df['news_snippets'] = df['date'].map(snippets_map)
        # Génération des embeddings BERT
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        bert_model = AutoModel.from_pretrained(BERT_MODEL).to(DEVICE)
        emb = get_bert_embeddings(df['news_snippets'].tolist(), tokenizer, bert_model, DEVICE)
        bert_cols = [f'bert_{i}' for i in range(emb.shape[1])]
        df[bert_cols] = emb
        # Calcul des Market Condition Proxies (PCA)
        pca = PCA(n_components=3, random_state=42)
        df[['mcp_1','mcp_2','mcp_3']] = pca.fit_transform(
            df[['close','volume','market_regime']]
        )
        all_dfs.append(df)
    if not all_dfs:
        logger.error('No datasets generated, exiting.')
        sys.exit(1)
    full_df = pd.concat(all_dfs)
    os.makedirs(dataset_dir, exist_ok=True)
    try:
        full_df.to_parquet(output_file, compression='snappy')
        logger.info(f"Full dataset saved: {output_file}")
    except Exception as e:
        logger.error(f"Parquet save error: {e}, falling back to CSV.")
        csv_fallback = output_file.replace('.parquet', '.csv')
        full_df.to_csv(csv_fallback, index=True)
        logger.info(f"CSV fallback saved: {csv_fallback}")
