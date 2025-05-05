# Spécification : Gestionnaire d'API (ApiManager)

**Objectif :** Définir les responsabilités, l'interface et le comportement du module `ApiManager` (`utils/api_manager.py`), chargé de centraliser et de gérer les interactions avec toutes les API externes utilisées par le projet (Exchanges, LLM, News, Social Media...).

---

## 1. Vue d'ensemble

L'`ApiManager` sert de point d'accès unique et standardisé pour toutes les communications avec les services externes. Il gère l'authentification (chargement sécurisé des clés API), la gestion des connexions, l'envoi des requêtes et le traitement initial des réponses (gestion basique des erreurs).

Il utilise des bibliothèques spécifiques comme `ccxt` pour les exchanges, `openai` pour OpenAI, `tweepy` pour Twitter, etc.

---

## 2. Responsabilités Principales

*   **Gestion des Clés API**:
    *   Charger de manière sécurisée les clés API et autres secrets depuis le fichier `config/secrets.env` en utilisant `python-dotenv`. Ne jamais coder les clés en dur.
*   **Gestion des Connexions Exchanges (`ccxt`)**:
    *   Instancier et configurer les objets `ccxt` pour les exchanges spécifiés dans `config/config.yaml` (Bitget, KuCoin, Binance...).
    *   Fournir des méthodes pour récupérer une instance `ccxt` authentifiée pour un exchange donné.
    *   Gérer potentiellement un pool de connexions ou la réutilisation des instances.
*   **Interaction API LLM (ex: OpenAI)**:
    *   Configurer le client API OpenAI (ou autre LLM) avec la clé API.
    *   Fournir une méthode pour envoyer un prompt au LLM et récupérer la réponse.
    *   Gérer les paramètres de l'appel API (modèle à utiliser, température, max tokens...).
*   **Interaction API News/Social (Optionnel)**:
    *   Configurer les clients pour les API News (ex: NewsAPI) ou Social Media (ex: Tweepy).
    *   Fournir des méthodes pour rechercher/récupérer des news ou des tweets pertinents selon des critères définis.
*   **Gestion des Erreurs Communes**:
    *   Intercepter les erreurs courantes liées aux API (erreurs réseau, limites de taux, erreurs d'authentification, erreurs spécifiques au service).
    *   Logger les erreurs de manière informative.
    *   Potentiellement implémenter une logique de retry simple (à utiliser avec prudence, surtout pour les appels LLM coûteux ou les ordres).
*   **Standardisation**: Offrir une interface interne cohérente pour accéder aux différentes API, même si les bibliothèques sous-jacentes sont différentes.

---

## 3. Interface Attendue (Exemple Classe)

```python
# Exemple de structure de classe (pseudo-code)
import ccxt
import openai
import os
from dotenv import load_dotenv

class ApiManager:
    def __init__(self, config: dict):
        """
        Initialise l'ApiManager.
        Args:
            config: Dictionnaire de configuration.
        """
        self.config = config
        self._load_secrets()
        self.exchange_connections = {} # Cache pour les connexions ccxt
        self._setup_llm_client()
        # self._setup_news_client() # Si utilisé
        # self._setup_social_client() # Si utilisé

    def _load_secrets(self):
        """Charge les secrets depuis .env."""
        load_dotenv(dotenv_path=self.config['paths']['secrets_env_path'])
        self.api_keys = {
            'binance': {'apiKey': os.getenv('BINANCE_API_KEY'), 'secret': os.getenv('BINANCE_SECRET_KEY')},
            'kucoin': {'apiKey': os.getenv('KUCOIN_API_KEY'), 'secret': os.getenv('KUCOIN_SECRET_KEY'), 'password': os.getenv('KUCOIN_API_PASSWORD')},
            'bitget': {'apiKey': os.getenv('BITGET_API_KEY'), 'secret': os.getenv('BITGET_SECRET_KEY'), 'password': os.getenv('BITGET_API_PASSWORD')},
            'openai': os.getenv('OPENAI_API_KEY'),
            # ... autres clés ...
        }
        # Vérifier que les clés nécessaires sont présentes

    def get_exchange_connection(self, exchange_name: str) -> ccxt.Exchange:
        """Retourne une instance ccxt connectée et authentifiée."""
        if exchange_name not in self.api_keys:
            raise ValueError(f"Clés API non trouvées pour l'exchange: {exchange_name}")

        if exchange_name not in self.exchange_connections:
            try:
                exchange_class = getattr(ccxt, exchange_name)
                self.exchange_connections[exchange_name] = exchange_class(self.api_keys[exchange_name])
                log.info(f"Instance ccxt créée pour {exchange_name}")
                # On pourrait ajouter un test de connexion ici (ex: fetch_balance)
            except AttributeError:
                raise ValueError(f"Exchange ccxt non supporté: {exchange_name}")
            except Exception as e:
                log.error(f"Erreur lors de la création de l'instance ccxt pour {exchange_name}: {e}")
                raise
        return self.exchange_connections[exchange_name]

    def _setup_llm_client(self):
        """Configure le client pour l'API LLM."""
        if self.api_keys.get('openai'):
            openai.api_key = self.api_keys['openai']
            log.info("Client API OpenAI configuré.")
        else:
            log.warning("Clé API OpenAI non trouvée. Fonctionnalités LLM désactivées.")
            # Gérer le cas où la clé n'est pas là

    def query_llm(self, prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 150, temperature: float = 0.7) -> str:
        """Envoie un prompt au LLM et retourne la réponse textuelle."""
        if not openai.api_key:
             raise ConnectionError("Client API OpenAI non configuré.")
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except openai.error.RateLimitError as e:
            log.error(f"Erreur RateLimit OpenAI: {e}")
            # Implémenter backoff/retry ?
            raise
        except Exception as e:
            log.error(f"Erreur lors de la requête LLM: {e}")
            raise

    # --- Méthodes pour News/Social (Exemples) ---
    # def search_news(self, query: str, ...) -> list: ...
    # def search_tweets(self, query: str, ...) -> list: ...

```

---

## 4. Interaction avec les Autres Modules

*   **Executor (`live/executor.py`) -> ApiManager**: Obtient les connexions `ccxt` pour interagir avec les exchanges.
*   **Workflow (`workflows/trading_workflow.py`) -> ApiManager**: Récupère les données de marché via les connexions `ccxt` fournies et interroge le LLM.
*   **Data Pipelines (`data/pipelines/`) -> ApiManager**: Peuvent utiliser l'ApiManager pour récupérer les données brutes initiales.
*   **Monitoring (`live/monitoring.py`) -> ApiManager**: Peut l'utiliser pour les requêtes LLM manuelles depuis le dashboard.
*   **ApiManager -> Config**: Lit les chemins et potentiellement certains paramètres API. Charge les secrets depuis `.env`.

---

## 5. Considérations

*   **Sécurité**: La principale responsabilité est la gestion sécurisée des clés API. Le fichier `.env` ne doit jamais être commit.
*   **Gestion des Limites de Taux (Rate Limiting)**: Les API externes ont des limites sur le nombre d'appels par période. L'ApiManager (ou les modules qui l'utilisent) doit être conscient de ces limites et potentiellement implémenter une logique pour les respecter (throttling, backoff). `ccxt` gère cela partiellement.
*   **Gestion des Erreurs**: Standardiser la gestion des erreurs API pour que les modules appelants puissent réagir de manière appropriée.
*   **Abstractions**: Trouver le bon équilibre entre fournir une interface simple et exposer suffisamment de fonctionnalités/paramètres des API sous-jacentes.

---

Cette spécification guide l'implémentation de `utils/api_manager.py`.
