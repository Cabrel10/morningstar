import argparse
import os
import json
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict

from utils.api_manager import APIManager
from utils.feature_engineering import apply_feature_pipeline
from utils.market_regime import MarketRegimeDetector
from utils.llm_integration import LLMIntegration
from utils.mcp_integration import MCPIntegration
from utils.data_preparation import CryptoBERTEmbedder

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration Gemini API
GEMINI_API_KEY = "api"
GEMINI_API_BACKUP_KEYS = [
    "Api",
    "Api"
]

def setup_gemini_api():
    """Configure l'API Gemini avec gestion des clés de backup."""
    import google.generativeai as genai
    for key in [GEMINI_API_KEY] + GEMINI_API_BACKUP_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-pro')
            # Test rapide
            response = model.generate_content("Test de connexion Gemini API")
            if response and hasattr(response, 'text'):
                logger.info("Connexion Gemini API établie avec succès")
                return model
        except Exception as e:
            logger.warning(f"Échec de la clé Gemini API {key[:10]}...: {e}")
            continue
    raise RuntimeError("Aucune clé Gemini API fonctionnelle")

def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Valide et nettoie les features techniques."""
    # 1. Identification des colonnes
    base_cols = ['open', 'high', 'low', 'close', 'volume']
    tech_cols = [col for col in df.columns if col.startswith('tech_')]
    llm_cols = [col for col in df.columns if col.startswith(('llm_', 'mcp_', 'regime'))]
    
    # 2. Vérification du nombre d'indicateurs
    n_tech = len(tech_cols)
    if n_tech != 38:
        raise ValueError(f"Nombre incorrect d'indicateurs techniques: {n_tech} (attendu: 38)")
    
    # 3. Vérification de la variance
    zero_var_cols = [col for col in tech_cols if df[col].std() == 0]
    if zero_var_cols:
        logger.warning(f"Colonnes à variance nulle détectées: {zero_var_cols}")
        for col in zero_var_cols:
            df = df.drop(columns=[col])
    
    # 4. Gestion des NaN uniquement sur les indicateurs techniques
    tech_na_counts = df[tech_cols].isna().sum()
    if tech_na_counts.any():
        logger.info(f"Remplissage de {tech_na_counts.sum()} NaN dans les indicateurs techniques")
        df[tech_cols] = df[tech_cols].fillna(method='ffill').fillna(method='bfill')
    
    # 5. Vérification finale
    remaining_na = df[tech_cols].isna().sum().sum()
    if remaining_na > 0:
        raise ValueError(f"Il reste {remaining_na} NaN après nettoyage")
    
    return df

def get_market_context(model, symbol: str, date: datetime) -> str:
    """Récupère le contexte de marché via Gemini."""
    try:
        prompt = f"""Analyse le marché de {symbol} pour la date {date.strftime('%Y-%m-%d')}.
        Fournis un résumé concis incluant:
        - Tendance générale
        - Événements majeurs
        - Sentiment du marché
        Limite ta réponse à 3-4 phrases."""
        
        response = model.generate_content(prompt)
        return response.text[:500] if response and hasattr(response, 'text') else ""
    except Exception as e:
        logger.error(f"Erreur Gemini API: {e}")
        return ""

def calculate_trading_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les métriques de trading (volatilité, stop-loss, take-profit)."""
    # Volatilité sur 24h
    df['volatility'] = df['close'].pct_change().rolling(24).std()
    
    # ATR pour stop-loss et take-profit dynamiques
    df['TR'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(14).mean()
    
    # Stop-loss et take-profit basés sur l'ATR
    df['stop_loss'] = df['close'] - 2 * df['ATR']
    df['take_profit'] = df['close'] + 3 * df['ATR']
    
    # Signal basé sur le régime et la volatilité
    df['signal'] = 0  # Neutre par défaut
    df.loc[(df['regime'] == 1) & (df['volatility'] < df['volatility'].quantile(0.7)), 'signal'] = 1  # Long
    df.loc[(df['regime'] == 3) & (df['volatility'] > df['volatility'].quantile(0.3)), 'signal'] = -1  # Short
    
    return df

def main():
    """Point d'entrée principal du script."""
    parser = argparse.ArgumentParser(description="Génère un dataset ETH complet avec features avancées")
    parser.add_argument('--start-date', required=True, help="Date de début (YYYY-MM-DD)")
    parser.add_argument('--end-date', required=True, help="Date de fin (YYYY-MM-DD)")
    parser.add_argument('--interval', default='1h', help="Intervalle temporel")
    parser.add_argument('--output-path', required=True, help="Chemin de sortie du dataset")
    parser.add_argument('--exchange', default='binance', help="Exchange source")
    parser.add_argument('--symbol', default='ETH/USDT', help="Paire de trading")
    args = parser.parse_args()

    try:
        # 1. Initialisation des composants
        logger.info("Initialisation des composants...")
        api_manager = APIManager({'exchange': args.exchange})
        hmm_detector = MarketRegimeDetector(n_states=4)
        llm_integration = LLMIntegration()
        mcp_integration = MCPIntegration()
        cryptobert = CryptoBERTEmbedder(model_name="ElKulako/cryptobert")
        gemini_model = setup_gemini_api()

        # 2. Téléchargement OHLCV
        logger.info(f"Téléchargement données {args.symbol} de {args.start_date} à {args.end_date}")
        df = api_manager.fetch_ohlcv_data(
            args.exchange, args.symbol, args.interval,
            args.start_date, args.end_date
        )
        if df is None or df.empty:
            raise RuntimeError("Échec du téléchargement OHLCV")

        # 3. Features techniques (38 indicateurs)
        logger.info("Calcul des indicateurs techniques...")
        df = apply_feature_pipeline(df)
        df = validate_features(df)  # Validation des features

        # 4. Régimes de marché HMM
        logger.info("Détection des régimes de marché...")
        regimes, proba = hmm_detector.fit_predict(df[['close', 'volume']])
        df['regime'] = regimes
        for i in range(proba.shape[1]):
            df[f'regime_proba_{i+1}'] = proba[:, i]

        # 5. Contexte et embeddings LLM
        logger.info("Génération des embeddings LLM...")
        contexts, embeddings = [], []
        for date in df.index:
            context = get_market_context(gemini_model, args.symbol, date)
            contexts.append(context)
            try:
                emb = cryptobert.embed_text(context)
            except Exception as e:
                logger.error(f"Erreur embedding: {e}")
                emb = np.zeros(768)
            embeddings.append(emb)
        
        df['llm_context_summary'] = contexts
        for i in range(768):
            df[f'llm_embedding_{i}'] = [e[i] for e in embeddings]

        # 6. Features MCP
        logger.info("Calcul des features MCP...")
        mcp_features = mcp_integration.get_features(df)
        for i in range(mcp_features.shape[1]):
            df[f'mcp_{i+1}'] = mcp_features[:, i]

        # 7. Métriques de trading
        logger.info("Calcul des métriques de trading...")
        df = calculate_trading_metrics(df)

        # 8. Sauvegarde
        logger.info(f"Sauvegarde du dataset dans {args.output_path}")
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        df.to_parquet(args.output_path)
        
        # Résumé
        tech_cols = [col for col in df.columns if col.startswith('tech_')]
        mcp_cols = [col for col in df.columns if col.startswith('mcp_')]
        logger.info(f"""
        Dataset généré avec succès:
        - Lignes: {len(df)}
        - Features techniques: {len(tech_cols)}
        - Features MCP: {len(mcp_cols)}
        - Dimension embedding: 768
        - Régimes détectés: {len(df['regime'].unique())}
        - Mémoire utilisée: {df.memory_usage().sum() / 1024**2:.1f} MB
        """)

    except Exception as e:
        logger.error(f"Erreur lors de la génération du dataset: {e}")
        raise
        
        # Résumé
        tech_cols = [col for col in df.columns if col.startswith('tech_')]
        mcp_cols = [col for col in df.columns if col.startswith('mcp_')]
        logger.info(f"""
        Dataset généré avec succès:
        - Lignes: {len(df)}
        - Features techniques: {len(tech_cols)}
        - Features MCP: {len(mcp_cols)}
        - Dimension embedding: 768
        - Régimes détectés: {len(df['regime'].unique())}
        - Mémoire utilisée: {df.memory_usage().sum() / 1024**2:.1f} MB
        """)

    except Exception as e:
        logger.error(f"Erreur lors de la génération du dataset: {e}")
        raise

if __name__ == "__main__":
    main()


def main():
    parser = argparse.ArgumentParser(description="Génère un dataset ETH complet avec features avancées.")
    parser.add_argument('--start-date', required=True)
    parser.add_argument('--end-date', required=True)
    parser.add_argument('--interval', default='1h')
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--exchange', default='binance')
    parser.add_argument('--symbol', default='ETH/USDT')
    parser.add_argument('--gemini-api-key', default=os.getenv('GEMINI_API_KEY'))
    args = parser.parse_args()

    # 1. Téléchargement OHLCV
    api_config = {'exchange': args.exchange}
    api_manager = APIManager(api_config)
    print(f"Téléchargement OHLCV {args.symbol} {args.start_date} -> {args.end_date} ({args.interval})")
    df = api_manager.fetch_ohlcv_data(
        args.exchange, args.symbol, args.interval, args.start_date, args.end_date
    )
    if df is None or df.empty:
        print("Erreur: données OHLCV introuvables.")
        return

    # 2. Feature engineering (38 indicateurs techniques)
    df = apply_feature_pipeline(df)

    # 3. Régimes de marché HMM
    hmm = MarketRegimeDetector(n_states=4)
    hmm.fit(df)
    regimes, proba = hmm.predict(df)
    df['regime'] = regimes
    for i in range(proba.shape[1]):
        df[f'regime_proba_{i+1}'] = proba[:, i]

    # 4. Génération des embeddings LLM (Gemini + CryptoBERT)
    llm = LLMIntegration()
    cryptobert = CryptoBERTEmbedder(model_name="ElKulako/cryptobert")
    summaries, embeddings = [], []
    for date in df['timestamp']:
        date_str = str(date.date())
        # 1. Récupère contexte Gemini (API)
        context = fetch_gemini_context('ETH', date_str, args.gemini_api_key)
        # 2. Embedding CryptoBERT
        try:
            emb = cryptobert.embed_text(context)
        except Exception as e:
            print(f"Erreur embedding LLM: {e}")
            emb = np.zeros(768)
        summaries.append(context)
        embeddings.append(emb)
    df['llm_context_summary'] = summaries
    df['llm_embedding'] = embeddings

    # 5. Génération des features MCP (128-dim)
    mcp = MCPIntegration()
    mcp_features = mcp.get_features(df)
    for i in range(mcp_features.shape[1]):
        df[f'mcp_{i+1}'] = mcp_features[:, i]

    # 6. Colonnes de trading (exemple: signal, volatilité, stop-loss, take-profit)
    df['signal'] = np.random.randint(-1, 2, size=len(df))
    df['volatility'] = df['close'].pct_change().rolling(24).std().fillna(0)
    df['stop_loss'] = df['close'] * (1 - 0.02)
    df['take_profit'] = df['close'] * (1 + 0.03)

    # 7. Sauvegarde
    df.to_parquet(args.output_path)
    print(f"Dataset ETH enrichi sauvegardé: {args.output_path}")

if __name__ == "__main__":
    main()
