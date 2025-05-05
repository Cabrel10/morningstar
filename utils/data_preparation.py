import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import yfinance as yf
from transformers import AutoTokenizer, TFAutoModel


class CryptoBERTEmbedder:
    """Génère des embeddings crypto avec CryptoBERT (ElKulako/cryptobert)"""

    def __init__(self, model_name="ElKulako/cryptobert", use_cache=True):
        self.cache_dir = Path("data/llm_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.use_cache = use_cache
        self.model_name = model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = TFAutoModel.from_pretrained(self.model_name, from_pt=True)
            self.online = True
            logging.info(f"CryptoBERT (ElKulako/cryptobert) chargé avec succès")
        except Exception as e:
            logging.warning(f"Impossible de charger CryptoBERT: {str(e)} - Mode hors ligne activé")
            self.online = False

    def embed_text(self, text: str) -> np.ndarray:
        """Génère un embedding pour un texte financier"""
        # Vérifier si l'embedding est déjà en cache
        if self.use_cache:
            cache_path = self.cache_dir / f"{hash(text)}.npy"
            if cache_path.exists():
                try:
                    return np.load(cache_path)
                except Exception as e:
                    logging.warning(f"Erreur lors du chargement du cache: {str(e)}")
        
        if not self.online:
            # Retourner un vecteur aléatoire si le modèle n'est pas disponible
            logging.warning("Mode hors ligne: génération d'un embedding aléatoire")
            random_embedding = np.random.normal(0, 0.1, 768)
            return random_embedding / np.linalg.norm(random_embedding)
        
        # Générer l'embedding
        inputs = self.tokenizer(text, return_tensors="tf", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        embedding = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy().flatten()
        
        # Sauvegarder l'embedding en cache
        if self.use_cache:
            try:
                cache_path = self.cache_dir / f"{hash(text)}.npy"
                np.save(cache_path, embedding)
            except Exception as e:
                logging.warning(f"Erreur lors de la sauvegarde du cache: {str(e)}")
        
        return embedding


class NewsProcessor:
    """Classe pour collecter et traiter les actualités financières"""

    def __init__(self, api_key: Optional[str] = None, use_cryptobert: bool = True):
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        self.base_url = "https://newsapi.org/v2/everything"
        self.embedder = CryptoBERTEmbedder() if use_cryptobert else None

    def fetch_news(self, query: str, days: int = 7) -> pd.DataFrame:
        """Récupère les actualités financières avec embeddings"""
        if not self.api_key:
            raise ValueError("API key manquante pour NewsAPI")

        params = {
            "q": query,
            "from": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
            "to": datetime.now().strftime("%Y-%m-%d"),
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": self.api_key,
            "pageSize": 100,
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            articles = response.json().get("articles", [])

            data = []
            for article in articles:
                news_data = {
                    "timestamp": pd.to_datetime(article["publishedAt"]),
                    "title": article["title"],
                    "content": article["content"][:500] if article["content"] else "",
                }

                if self.embedder and news_data["content"]:
                    news_data["embedding"] = self.embedder.embed_text(news_data["content"])

                data.append(news_data)

            return pd.DataFrame(data)

        except Exception as e:
            logging.error(f"Erreur NewsAPI: {str(e)}")
            return pd.DataFrame()


logger = logging.getLogger(__name__)


def load_raw_data(input_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load market data from either:
    - CSV file if input is a file path (applies limit if provided)
    - Yahoo Finance if input is a symbol

    Args:
        input_path: Either file path or trading symbol

    Returns:
        DataFrame with OHLCV data and calculated returns
    """
    try:
        # If input is a file path
        if input_path.endswith(".csv"):
            logger.info(f"Loading data from file: {input_path}{f' (limit: {limit})' if limit else ''}")
            # Use nrows parameter if limit is provided
            data = pd.read_csv(input_path, nrows=limit)

            # Convert date/timestamp column if exists and set as index
            date_col = None
            if "timestamp" in data.columns:
                date_col = "timestamp"
            elif "date" in data.columns:
                date_col = "date"
            elif "Date" in data.columns:
                date_col = "Date"
            
            if date_col:
                logger.info(f"Using column '{date_col}' as DatetimeIndex.")
                data[date_col] = pd.to_datetime(data[date_col])
                data = data.set_index(date_col)
            else:
                logger.warning("No standard date/timestamp column found ('timestamp', 'date', 'Date'). Index will be RangeIndex.")

        # Else treat as symbol for yfinance
        else:
            logger.info(f"Downloading {input_path} data from Yahoo Finance")
            data = yf.download(tickers=input_path, period="3y", interval="1d", auto_adjust=True, prepost=False)

            if data.empty:
                raise ValueError("No data returned from Yahoo Finance")

        # Calculate basic features
        if "Close" in data.columns:
            data["returns"] = data["Close"].pct_change()
            data["log_returns"] = np.log(data["Close"]).diff()

        logger.info(f"Successfully loaded {len(data)} rows")
        return data

    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning: drop NaNs and duplicates.

    Args:
        df: Input DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    original_rows = len(df)
    df = df.dropna()
    df = df.drop_duplicates()
    rows_dropped = original_rows - len(df)
    if rows_dropped > 0:
        logger.info(f"Dropped {rows_dropped} rows during cleaning (NaNs/duplicates).")
    logger.info(f"Data cleaned. Shape: {df.shape}")
    return df

def prepare_link_4h_dataset(force_download: bool = False) -> pd.DataFrame:
    """Prepare LINK/USDT 4h dataset with technical indicators, LLM embeddings and MCP features.
    
    Args:
        force_download: Whether to force download fresh data
        
    Returns:
        Processed DataFrame with all features
    """
    # Paths
    raw_path = Path("data/raw/link_usdt_binance_4h.csv")
    processed_path = Path("data/processed/link_usdt_binance_4h_processed.parquet")
    
    # Download data if needed
    if force_download or not raw_path.exists():
        logger.info("Downloading fresh LINK/USDT 4h data")
        download_cmd = [
            "python", "-m", "utils.api_manager",
            "--token", "LINK/USDT",
            "--exchange", "binance",
            "--timeframe", "4h",
            "--start", (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
            "--end", datetime.now().strftime('%Y-%m-%d'),
            "--output", str(raw_path)
        ]
        subprocess.run(download_cmd, check=True)
    
    # Load and clean data
    df = load_raw_data(str(raw_path))
    df = clean_data(df)
    
    # Add instrument type
    df['instrument_type'] = 'link'
    
    # Apply feature engineering
    from utils.feature_engineering import apply_feature_pipeline
    df = apply_feature_pipeline(df)
    
    # Add label
    from utils.labeling import build_label
    df = build_label(df)
    
    # Add MCP features
    from utils.mcp_integration import MCPIntegration
    mcp = MCPIntegration()
    mcp_features = mcp.get_mcp_features("LINK/USDT")
    for i in range(128):
        df[f'mcp_{i}'] = np.random.normal(mcp_features[i], 0.1, len(df)) if i < len(mcp_features) else np.random.normal(0, 0.1, len(df))
    
    # Add LLM embeddings
    from utils.llm_integration import LLMIntegration
    llm = LLMIntegration()
    dates = pd.to_datetime(df.index).date if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['timestamp']).dt.date
    
    embeddings = []
    for date in dates:
        embedding = llm.get_cached_embedding("LINK", str(date))
        if embedding is None:
            embedding = np.random.normal(0, 0.1, 768)
            embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)
    
    embeddings_df = pd.DataFrame(embeddings, index=df.index, columns=[f'llm_{i}' for i in range(768)])
    df = pd.concat([df, embeddings_df], axis=1)
    
    # Save processed data
    df.to_parquet(processed_path)
    logger.info(f"Saved processed LINK data to {processed_path}")
    
    return df
