# Instructions pour configurer les clés API

Pour exploiter pleinement le pipeline de trading crypto amélioré, vous devez configurer les clés API suivantes dans un fichier `.env` à la racine du projet.

## Étapes à suivre

1. Créez un fichier nommé `.env` à la racine du projet (`/home/morningstar/Desktop/crypto_robot/Morningstar/.env`)
2. Ajoutez les clés API suivantes au fichier :

```
# Exchange API Keys
BINANCE_API_KEY=votre_clé_binance
BINANCE_SECRET_KEY=votre_clé_secrète_binance

# API Gemini pour l'analyse de sentiment
GEMINI_API_KEY=votre_clé_gemini

# API CoinMarketCap pour les informations de marché
COINMARKETCAP_API_KEY=votre_clé_coinmarketcap

# Clés API Google (pour Gemini)
GOOGLE_API_KEY_1=votre_première_clé_google
GOOGLE_API_KEY_2=votre_deuxième_clé_google
GOOGLE_API_KEY_3=votre_troisième_clé_google
```

## Comment obtenir les clés API

### Gemini API (Google)
1. Visitez [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Connectez-vous avec votre compte Google
3. Créez une nouvelle clé API
4. Copiez la clé et ajoutez-la à votre fichier `.env`

### CoinMarketCap API
1. Visitez [CoinMarketCap Developer Portal](https://coinmarketcap.com/api/)
2. Créez un compte ou connectez-vous
3. Obtenez une clé API gratuite (plan Basic)
4. Copiez la clé et ajoutez-la à votre fichier `.env`

### Binance API
1. Connectez-vous à votre compte Binance
2. Allez dans Paramètres > API Management
3. Créez une nouvelle clé API
4. Copiez la clé API et la clé secrète dans votre fichier `.env`

## Utilisation limitée sans clés API

Si vous ne souhaitez pas configurer toutes les clés API immédiatement, vous pouvez exécuter le pipeline avec des fonctionnalités limitées :

```bash
python scripts/collect_enriched_data.py --symbols BTC/USDT,ETH/USDT --start-date 2023-01-01 --end-date 2023-01-31 --use-hmm
```

Cette commande exécutera le pipeline sans l'analyse de sentiment ni les embeddings CryptoBERT, mais avec la détection de régime HMM qui fonctionne sans clés API externes.
