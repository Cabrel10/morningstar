# Guide de Déploiement du Modèle Monolithique Morningstar

Ce document détaille les étapes pour déployer et gérer le modèle monolithique Morningstar en production.

## Prérequis

- Docker et Docker Compose
- Python 3.10 ou 3.11
- Accès aux clés API des exchanges (Binance/Bitget)
- 8 Go RAM minimum, 16 Go recommandés
- Accès à Prometheus/Grafana pour la surveillance

## Construction de l'Image Docker

```bash
# Construction de l'image
docker build -t morningstar-monolith:latest .

# Construction avec tag spécifique (pour versioning)
docker build -t morningstar-monolith:v2.0.0 .
```

## Démarrage du Système

### Environnement de Staging (testnet)

```bash
# Démarrer en mode staging (testnet)
docker-compose -f docker-compose.staging.yml up -d

# Vérifier les logs
docker-compose -f docker-compose.staging.yml logs -f

# Exécuter un test de trading de 2 heures
docker-compose -f docker-compose.staging.yml exec monolith python -m ultimate.scripts.run_simulation --testnet --hours 2
```

### Environnement de Production (mainnet)

```bash
# Démarrer en mode production
docker-compose -f docker-compose.production.yml up -d

# Vérifier les logs
docker-compose -f docker-compose.production.yml logs -f
```

## Arrêter le Système

```bash
# Arrêter les conteneurs
docker-compose -f docker-compose.production.yml down

# Arrêter et supprimer les volumes (pour un redémarrage propre)
docker-compose -f docker-compose.production.yml down -v
```

## Procédure de Rollback

En cas de problème avec la version actuelle:

```bash
# Arrêter le système actuel
docker-compose -f docker-compose.production.yml down

# Modifier le tag de l'image dans docker-compose.production.yml
# De: morningstar-monolith:v2.0.0
# À: morningstar-monolith:v2.0.0-rc0 (version précédente stable)

# Redémarrer avec la version précédente
docker-compose -f docker-compose.production.yml up -d
```

## Structure des Logs

Les logs sont organisés dans les répertoires suivants:

- `/ultimate/logs/backtest/` - Logs des backtests
- `/ultimate/logs/live/` - Logs du trading en direct
- `/ultimate/logs/api/` - Logs de l'API

Formats de logs:
- Production: `%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(thread)d - %(message)s`
- Développement: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

Les logs anciens sont automatiquement archivés après 30 jours dans le sous-dossier `archives/`.

## Endpoint Metrics pour Prometheus

Le système expose un endpoint `/metrics` sur le port 8888 pour intégration avec Prometheus:

```
# HELP morningstar_predictions_count Nombre total de prédictions
# TYPE morningstar_predictions_count counter
morningstar_predictions_count 15896

# HELP morningstar_trades_executed Nombre de trades exécutés
# TYPE morningstar_trades_executed counter
morningstar_trades_executed 142

# HELP morningstar_prediction_latency_milliseconds Latence des prédictions en ms
# TYPE morningstar_prediction_latency_milliseconds histogram
morningstar_prediction_latency_milliseconds_bucket{le="50"} 1524
morningstar_prediction_latency_milliseconds_bucket{le="100"} 13982
morningstar_prediction_latency_milliseconds_bucket{le="200"} 15880

# HELP morningstar_equity_percentage Équité en pourcentage du capital initial
# TYPE morningstar_equity_percentage gauge
morningstar_equity_percentage 108.45

# HELP morningstar_sl_hit_rate Taux de hit des stop loss (%)
# TYPE morningstar_sl_hit_rate gauge
morningstar_sl_hit_rate 22.5

# HELP morningstar_tp_hit_rate Taux de hit des take profit (%)
# TYPE morningstar_tp_hit_rate gauge
morningstar_tp_hit_rate 42.8
```

## Dashboard Grafana

Des tableaux de bord Grafana prédéfinis sont disponibles dans le dossier `/ultimate/monitoring/grafana_dashboards/`.

Pour importer un dashboard:
1. Ouvrir Grafana
2. Cliquer sur "+" > "Import"
3. Copier-coller le contenu du fichier JSON du dashboard 
4. Sélectionner la source de données Prometheus

## Alertes via Telegram

Le bot Telegram est configuré pour envoyer des alertes dans les situations suivantes:

1. **Drawdown excessif**: Alerte quand le drawdown dépasse 5% (critique > 10%)
2. **Latence d'inférence**: Alerte quand la latence moyenne dépasse 200ms
3. **Erreurs d'API**: Alerte quand plus de 5 erreurs sont détectées en 5 minutes

Configuration du bot dans le fichier `/ultimate/config/telegram_config.json`:

```json
{
    "bot_token": "YOUR_BOT_TOKEN",
    "chat_id": "YOUR_CHAT_ID",
    "notify_trades": true,
    "notify_signals": true,
    "notify_errors": true,
    "notify_performance": true,
    "performance_interval": "1h"
}
```

## Commandes Telegram pour Contrôle à Distance

Les commandes suivantes sont disponibles via le bot Telegram:

- `/status` - Affiche l'état actuel du système
- `/pause` - Met en pause le trading
- `/resume` - Reprend le trading après une pause
- `/position` - Affiche les positions actuelles
- `/balance` - Affiche le solde actuel
- `/performance` - Affiche les performances
- `/close_all` - Ferme toutes les positions (urgence)
- `/shutdown` - Arrête le système proprement 