# Utilisation en ligne de commande (CLI)

## Commandes principales

### Standardiser un dataset
```bash
python scripts/standardize_datasets.py --input data/BTC.csv --output data/BTC_std.csv
```

### Entraîner le modèle
```bash
python scripts/create_morningstar_model.py --train --config config/config.yaml
```

### Lancer le trading live
```bash
python app.py --live
```

### Backtesting
```bash
python app.py --backtest --pair BTC/USDT
```

## Exemples de workflow
- Standardisation → Entraînement → Backtest → Trading live

## Sauvegarde & Finetuning
- Les modèles et scalers sont sauvegardés automatiquement dans `models/`
- Pour le finetuning : `--finetune --checkpoint chemin`

*Voir la section avancée pour les options expertes.*
