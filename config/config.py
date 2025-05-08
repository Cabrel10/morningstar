import os
import yaml
from dotenv import load_dotenv

class Config:
    def __init__(self):
        # Obtenir le chemin du répertoire du fichier config.py
        config_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construire les chemins absolus
        secrets_path = os.path.join(config_dir, 'secrets.env')
        yaml_path = os.path.join(config_dir, 'config.yaml')
        
        load_dotenv(secrets_path)
        
        # Charger la config YAML
        try:
            with open(yaml_path) as f:
                self.yaml_config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"ERREUR: Le fichier de configuration {yaml_path} n'a pas été trouvé.")
            print("Veuillez vous assurer qu'il existe et qu'il est correctement placé.")
            # Tenter de charger depuis un chemin relatif comme fallback, au cas où
            # le script est exécuté depuis la racine du projet.
            # Cela peut aider dans certains contextes d'exécution, mais la méthode ci-dessus est plus robuste.
            try:
                alt_yaml_path = 'config/config.yaml' # Chemin relatif depuis la racine
                with open(alt_yaml_path) as f:
                    self.yaml_config = yaml.safe_load(f) or {}
                print(f"INFO: Fichier de configuration chargé depuis {alt_yaml_path} (fallback).")
            except FileNotFoundError:
                print(f"ERREUR: Tentative de fallback pour charger {alt_yaml_path} a également échoué.")
                self.yaml_config = {} # Initialiser avec un dictionnaire vide pour éviter d'autres erreurs
            
        # Config Redis
        self.redis = type('', (), {})()
        self.redis.host = os.getenv('REDIS_HOST', 'localhost')
        self.redis.port = int(os.getenv('REDIS_PORT', 6379))
        self.redis.db = int(os.getenv('REDIS_DB', 0))
        self.redis.cache_ttl = self.yaml_config.get('llm', {}).get('cache_ttl', 86400)
        
        # Config LLM
        self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

    def get_config(self, key_path, default=None):
        """
        Récupère une valeur de configuration en utilisant un chemin de clé de type 'a.b.c'.
        Retourne une valeur par défaut si la clé n'est pas trouvée.
        """
        keys = key_path.split('.')
        value = self.yaml_config
        try:
            for key in keys:
                if isinstance(value, dict):
                    value = value[key]
                else: # Gérer le cas où un segment de chemin n'est pas un dictionnaire
                    return default
            return value
        except (KeyError, TypeError):
            return default
