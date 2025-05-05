# Plan d'Action Détaillé - Logique Live v2 (Indicateurs Techniques)

**Objectif :** Rendre `live/executor.py` fonctionnel pour un trading simulé (`dry-run`) réaliste en mode **Long/Flat uniquement**, intégrant :
*   Calcul des indicateurs techniques live.
*   Gestion d'état de position (savoir si on est 'long' ou 'flat').
*   Dimensionnement des ordres basé sur un **risque fixe par trade et la distance au Stop Loss (calculé via l'ATR)**.
*   Placement systématique d'ordres **Stop Loss (basé sur l'ATR)** et **Take Profit (basé sur un ratio Risk/Reward)** après une entrée en position.
*   Clôture de la position sur signal de vente.

## Diagramme du Flux Logique

```mermaid
graph TD
    A[Start: Donnée OHLCV Reçue (WebSocket)] --> B(LiveDataPreprocessor);
    B -- Buffer suffisant? --> C{Calcul Indicateurs Tech (apply_feature_pipeline)};
    C -- Indicateurs OK (incl. ATR) --> D{Préparation Input Modèle (Tech only + ATR)};
    D -- Input Prêt --> E{MorningstarModel.predict (Tête Signal)};
    E -- Prédiction (Signal) --> F(LiveExecutor);

    subgraph "LiveExecutor: Gestion Cycle de Trade (Long/Flat)"
        F -- Signal Achat/Vente --> G{État Actuel?};

        %% --- Branche Entrée Long ---
        G -- Actuellement Flat? --> H{Signal?};
        H -- Signal Achat --> I[Récupérer: Prix Actuel, Dernier ATR, Solde Compte];
        I --> J{Calculer Niveau SL (Prix Actuel - 1.5 * ATR)};
        J --> K{Calculer Niveau TP (Prix Actuel + 2 * SL_distance)};
        K --> L{Calculer Taille Ordre (Solde * Risque% / SL_distance)};
        L -- Détails Trade OK --> M{Passer Ordre MARKET BUY (ccxt)};
        M -- Ordre Entrée OK --> N{Placer Ordre STOP_MARKET SL (ccxt)};
        N -- Ordre SL OK --> O{Placer Ordre TAKE_PROFIT_MARKET TP (ccxt)};
        O -- Ordres Limites OK --> P[MAJ État: Position='long', Taille, Prix Entrée, IDs SL/TP];
        P --> Q[Log Entrée + Metrics];
        Q --> R[Fin Cycle];

        %% --- Branche Sortie Long ---
        G -- Actuellement Long? --> S{Signal?};
        S -- Signal Vente --> T[Annuler Ordres SL/TP Actifs (ccxt)];
        T -- Annulation OK --> U{Passer Ordre MARKET SELL (Clôture) (ccxt)};
        U -- Ordre Clôture OK --> V[Calculer P&L];
        V --> W[MAJ État: Position='flat', Vider IDs SL/TP];
        W --> X[Log Sortie + Metrics P&L];
        X --> R;

        %% --- Cas Ignorés ---
        H -- Signal Vente --> R;  // Ignorer Vente si Flat
        S -- Signal Achat --> R;  // Ignorer Achat si Long

        %% --- Gestion Erreurs ---
        M -- Erreur Ordre Entrée --> Y[Log Erreur + Metrics];
        N -- Erreur Ordre SL --> Y;
        O -- Erreur Ordre TP --> Y;
        T -- Erreur Annulation --> Y;
        U -- Erreur Ordre Clôture --> Y;
        Y --> R;
    end

    subgraph "utils/live_preprocessing.py"
        style B fill:#ccf,stroke:#333,stroke-width:1px
        style C fill:#ccf,stroke:#333,stroke-width:1px
        style D fill:#ccf,stroke:#333,stroke-width:1px
        B; C; D;
    end

    subgraph "model/architecture/"
        style E fill:#fcc,stroke:#333,stroke-width:1px
        E;
    end

    subgraph "live/executor.py"
        style F fill:#cfc,stroke:#333,stroke-width:1px
        F; G; H; I; J; K; L; M; N; O; P; Q; S; T; U; V; W; X; Y;
    end

    style J fill:#f9f,stroke:#333,stroke-width:2px
    style K fill:#f9f,stroke:#333,stroke-width:2px
    style L fill:#f9f,stroke:#333,stroke-width:2px
    style N fill:#f9f,stroke:#333,stroke-width:2px
    style O fill:#f9f,stroke:#333,stroke-width:2px
    style P fill:#f9f,stroke:#333,stroke-width:2px
    style T fill:#f9f,stroke:#333,stroke-width:2px
    style V fill:#f9f,stroke:#333,stroke-width:2px
    style W fill:#f9f,stroke:#333,stroke-width:2px
```

## Actions Détaillées Concrètes

1.  **`utils/live_preprocessing.py` :**
    *   **Valider `MIN_BUFFER_SIZE`** : Vérifier la période la plus longue des indicateurs dans `utils.feature_engineering.py` et ajuster la constante si nécessaire.
    *   **Exposer l'ATR** : Modifier `get_model_input` pour qu'il retourne également la dernière valeur de l'ATR calculée, en plus du dictionnaire d'inputs pour le modèle. `return model_inputs, last_atr_value`

2.  **`live/executor.py` :**
    *   **Ajouter Attributs d'État** :
        ```python
        self.current_position_size: float = 0.0
        self.entry_price: Optional[float] = None
        self.position_side: Optional[str] = None # 'long' ou None
        self.active_sl_order_id: Optional[str] = None
        self.active_tp_order_id: Optional[str] = None
        self.last_known_balance: Dict[str, float] = {} # Pour stocker le solde
        self.risk_per_trade_pct: float = self.config.get('live_trading', {}).get('risk_per_trade_pct', 0.01) # Ex: 1%
        self.atr_sl_multiplier: float = self.config.get('live_trading', {}).get('atr_sl_multiplier', 1.5)
        self.rr_ratio_tp: float = self.config.get('live_trading', {}).get('rr_ratio_tp', 2.0)
        self.dry_run: bool = False # Sera mis à jour par l'appelant (ex: CLI)
        ```
    *   **Initialiser `dry_run`** : Modifier `__init__` pour accepter un argument `dry_run` et le stocker dans `self.dry_run`.
    *   **Récupérer Solde (Moins Fréquemment)** : Créer une méthode `_update_balance` qui appelle `self.client.fetch_balance()` et stocke le résultat dans `self.last_known_balance`. L'appeler dans `__init__` et peut-être périodiquement (ex: toutes les heures) ou avant une série de trades. Utiliser `self.last_known_balance` dans la logique de trading.
    *   **Nouvelle Méthode `_calculate_trade_details`** :
        ```python
        def _calculate_trade_details(self, current_price: float, atr: float) -> Optional[Dict]:
            if atr <= 0: return None # Ne peut pas calculer sans ATR valide
            quote_currency = self.symbol.split('/')[1] # Ex: 'USDT'
            balance = self.last_known_balance.get(quote_currency, {}).get('free', 0)
            if balance <= 0: return None # Pas de fonds

            sl_distance_points = atr * self.atr_sl_multiplier
            sl_price = current_price - sl_distance_points
            tp_price = current_price + (sl_distance_points * self.rr_ratio_tp)

            risk_amount_quote = balance * self.risk_per_trade_pct
            # Gérer division par zéro si sl_distance_points est trop petit
            if sl_distance_points < 1e-8: return None
            order_size_base = risk_amount_quote / sl_distance_points

            # TODO: Ajouter vérifications (taille min/max ordre, step size...) via self.client.load_markets()

            return {
                'sl_price': sl_price,
                'tp_price': tp_price,
                'order_size': order_size_base # En devise de base (ex: BTC)
            }
        ```
    *   **Refactoriser `execute_trades` en `_handle_signal`** (ou similaire) : Cette méthode sera appelée dans la boucle `run` après avoir reçu une prédiction. (Voir le pseudo-code détaillé dans la discussion précédente).
    *   **Modifier la boucle `run`** :
        *   Récupérer `model_input, last_atr` depuis `self.preprocessor.get_model_input()`.
        *   Récupérer le `current_price` (ex: `latest_ohlcv[4]`).
        *   Appeler `await self._handle_signal(predictions, current_price, last_atr)` au lieu de `self.execute_trades`.
        *   Assurer que les méthodes `ccxt` appelées sont `await` si le client est asynchrone.