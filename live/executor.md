# Spécification : Exécuteur d'Ordres Live

**Objectif :** Définir les responsabilités, l'interface et le comportement du module `Executor` (`live/executor.py`), chargé de l'interaction avec les API des exchanges pour passer des ordres et gérer les positions en temps réel.

---

## 1. Vue d'ensemble

L'`Executor` est le composant qui traduit les décisions de trading prises par le `trading_workflow.py` en actions concrètes sur les plateformes d'échange (Bitget, KuCoin, Binance, etc.). Il utilise la bibliothèque `ccxt` et le module `utils/api_manager.py` pour communiquer avec les exchanges de manière standardisée.

---

## 2. Responsabilités Principales

*   **Connexion aux Exchanges**: Établir et maintenir la connexion aux exchanges configurés via `ccxt` (en utilisant les clés API fournies par `api_manager`).
*   **Passage d'Ordres**:
    *   Exécuter différents types d'ordres : `market`, `limit`, `stop_loss`, `take_profit`, `stop_loss_limit`, `take_profit_limit` (selon support de l'exchange et de `ccxt`).
    *   Gérer les paramètres spécifiques aux ordres (paire, type, côté (achat/vente), montant, prix limite, prix stop).
*   **Gestion des Ordres Ouverts**:
    *   Récupérer la liste des ordres ouverts.
    *   Annuler des ordres ouverts.
    *   Modifier des ordres (si supporté par l'exchange et `ccxt`).
*   **Gestion des Positions**:
    *   Récupérer les positions actuelles (notamment pour les marchés futures/margin).
    *   Calculer la taille des positions, le P&L réalisé/non réalisé.
*   **Gestion du Portefeuille**:
    *   Récupérer les soldes des différents actifs sur les exchanges.
*   **Gestion des Erreurs API**:
    *   Intercepter et gérer les erreurs courantes des API d'exchange (limites de taux, erreurs d'authentification, fonds insuffisants, erreurs de paramètre d'ordre, maintenance...).
    *   Implémenter des stratégies de retry si approprié.
*   **Standardisation**: Fournir une interface cohérente pour interagir avec différents exchanges, masquant autant que possible les spécificités de chaque API grâce à `ccxt`.

---

## 3. Interface Attendue (Exemple Classe)

```python
# Exemple de structure de classe (pseudo-code)
import ccxt

class OrderExecutor:
    def __init__(self, config: dict, api_manager):
        """
        Initialise l'Executor.
        Args:
            config: Dictionnaire de configuration (exchanges cibles, etc.).
            api_manager: Instance de ApiManager pour obtenir les connexions ccxt.
        """
        self.config = config
        self.api_manager = api_manager
        self.exchanges = {} # Dictionnaire pour stocker les instances ccxt connectées
        self._connect_exchanges()

    def _connect_exchanges(self):
        """Établit la connexion aux exchanges configurés."""
        for ex_name in self.config.get('active_exchanges', []):
            try:
                self.exchanges[ex_name] = self.api_manager.get_exchange_connection(ex_name)
                # Vérifier la connexion (ex: fetch balance)
                self.exchanges[ex_name].fetch_balance()
                log.info(f"Connecté à l'exchange: {ex_name}")
            except Exception as e:
                log.error(f"Échec de la connexion à {ex_name}: {e}")

    def place_order(self, exchange_name: str, symbol: str, order_type: str, side: str, amount: float, price: float = None, params: dict = {}) -> dict:
        """
        Place un ordre sur un exchange spécifique.
        Args:
            exchange_name: Nom de l'exchange (ex: 'binance').
            symbol: Paire de trading (ex: 'BTC/USDT').
            order_type: Type d'ordre ('market', 'limit'...).
            side: 'buy' or 'sell'.
            amount: Quantité à trader.
            price: Prix pour les ordres limit.
            params: Paramètres additionnels spécifiques à ccxt/exchange.
        Returns:
            Dictionnaire représentant l'ordre placé (format ccxt) ou lève une exception.
        """
        if exchange_name not in self.exchanges:
            raise ValueError(f"Exchange {exchange_name} non connecté.")
        exchange = self.exchanges[exchange_name]
        try:
            log.info(f"Placement ordre: {exchange_name} {symbol} {side} {amount} @ {price or 'MARKET'}")
            order = exchange.create_order(symbol, order_type, side, amount, price, params)
            log.info(f"Ordre placé avec succès: ID {order.get('id')}")
            return order
        except ccxt.InsufficientFunds as e:
            log.error(f"Fonds insuffisants pour l'ordre: {e}")
            raise # Relancer l'exception pour gestion par le workflow
        except ccxt.NetworkError as e:
            log.error(f"Erreur réseau ccxt: {e}")
            # Implémenter retry ?
            raise
        except ccxt.ExchangeError as e:
            log.error(f"Erreur exchange ccxt: {e}")
            raise
        except Exception as e:
            log.error(f"Erreur inconnue lors du placement de l'ordre: {e}")
            raise

    def cancel_order(self, exchange_name: str, order_id: str, symbol: str = None) -> dict:
        """Annule un ordre ouvert."""
        # ... implémentation avec exchange.cancel_order(order_id, symbol) ...

    def fetch_open_orders(self, exchange_name: str, symbol: str = None) -> list:
        """Récupère les ordres ouverts."""
        # ... implémentation avec exchange.fetch_open_orders(symbol) ...

    def fetch_balance(self, exchange_name: str) -> dict:
        """Récupère les soldes du compte."""
        # ... implémentation avec exchange.fetch_balance() ...

    def fetch_positions(self, exchange_name: str, symbols: list = None) -> list:
        """Récupère les positions ouvertes (futures/margin)."""
        # ... implémentation avec exchange.fetch_positions(symbols) si supporté ...

    # ... autres méthodes nécessaires (fetch_order, fetch_trades, etc.) ...

```

---

## 4. Interaction avec les Autres Modules

*   **Workflow -> Executor**: Le workflow appelle les méthodes de l'`Executor` pour placer/annuler des ordres et obtenir des informations sur le portefeuille/positions.
*   **Executor -> ApiManager**: Utilise l'`ApiManager` pour obtenir les instances `ccxt` authentifiées.
*   **Executor -> Logging**: Enregistre toutes les actions et les erreurs importantes.

---

## 5. Considérations

*   **Gestion des Erreurs et Retries**: Implémenter une logique de gestion des erreurs robuste, notamment pour les limites de taux d'API et les erreurs réseau temporaires. Des retries avec backoff exponentiel peuvent être nécessaires.
*   **Spécificités des Exchanges**: Bien que `ccxt` standardise beaucoup de choses, certains exchanges ont des comportements ou des paramètres spécifiques (`params` dans `create_order`). Ceux-ci doivent être gérés si nécessaire.
*   **Concurrence**: Si le système doit gérer plusieurs ordres ou paires en parallèle, s'assurer que l'accès à l'API est géré correctement pour éviter les problèmes de limites de taux.
*   **Sécurité**: La gestion des clés API est déléguée à `ApiManager`, mais l'`Executor` doit s'assurer qu'il utilise les connexions de manière sécurisée.
*   **Tests**: Tester intensivement en environnement de paper trading avant de passer en live. Utiliser le mocking pour tester la logique sans interagir réellement avec les exchanges.

---

Cette spécification guide l'implémentation de `live/executor.py`.
