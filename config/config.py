import os
import yaml
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv('config/secrets.env')
        
        # Charger la config YAML
        with open('config/config.yaml') as f:
            self.yaml_config = yaml.safe_load(f) or {}
            
        # Config Redis
        self.redis = type('', (), {})()
        self.redis.host = os.getenv('REDIS_HOST', 'localhost')
        self.redis.port = int(os.getenv('REDIS_PORT', 6379))
        self.redis.db = int(os.getenv('REDIS_DB', 0))
        self.redis.cache_ttl = self.yaml_config.get('llm', {}).get('cache_ttl', 86400)
        
        # Config LLM
        self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
