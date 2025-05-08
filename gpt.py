# gpt.py
# Script de collecte OHLCV pour 5 paires sur Binance via ccxt
# Collecte bougies 1m pour une liste de 5 symboles et une plage de dates

import os
import time
import ccxt
import pandas as pd
from datetime import datetime
import logging

# === CONFIGURATION ===
# Restreint aux 5 premières paires
symbols = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT'
]
start_date = '2024-01-01T00:00:00Z'  # ISO 8601
end_date   = '2025-02-02T00:00:00Z'
timeframe  = '1m'
limit      = 1000       # max OHLCV par requête
max_retries = 5
pause_on_fail = 5        # secondes entre retries
output_dir = 'data'

# === LOGGING ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger()

# === INITIALISATION EXCHANGE ===
exchange = ccxt.binance({
    'enableRateLimit': True,
    'timeout': 30000,
})

# === FONCTION DE RÉCUP OHLCV AVEC RETRIES ===
def fetch_ohlcv_retry(symbol, since_ts, limit):
    for attempt in range(1, max_retries + 1):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=limit)
        except (ccxt.RequestTimeout, ccxt.NetworkError) as e:
            logger.warning(f"[{symbol}] Tentative {attempt}/{max_retries} échouée: {e}")
            time.sleep(pause_on_fail)
        except Exception as e:
            logger.error(f"[{symbol}] Erreur inattendue: {e}")
            time.sleep(pause_on_fail)
    raise RuntimeError(f"[{symbol}] Échec après {max_retries} tentatives")

# === BOUCLE PRINCIPALE ===
if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    since_ms = exchange.parse8601(start_date)
    end_ms   = exchange.parse8601(end_date)

    for symbol in symbols:
        logger.info(f"Début collecte pour {symbol}")
        all_ohlcv = []
        ts = since_ms
        # boucle de collecte
        while ts < end_ms:
            ohlcv = fetch_ohlcv_retry(symbol, ts, limit)
            if not ohlcv:
                logger.info(f"[{symbol}] Pas de données, arrêt.")
                break
            all_ohlcv.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            ts = last_ts + 60 * 1000  # incrément 1 minute
            if ts <= last_ts:
                ts = last_ts + 1
        # création du DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)
        df = df[df['timestamp'] < end_dt]
        # sauvegarde
        fname = os.path.join(output_dir, f"{symbol.replace('/','')}_1m.csv")
        df.to_csv(fname, index=False)
        logger.info(f"Données {symbol} sauvegardées dans {fname} ({len(df)} lignes)")

    logger.info("Collecte terminée pour toutes les paires.")
