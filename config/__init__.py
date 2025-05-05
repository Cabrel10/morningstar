import os
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / "config.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config or {}
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}

# Load configuration when module is imported
config = load_config()

def get_config(key: str, default=None) -> Any:
    """Get a specific configuration value"""
    keys = key.split(".")
    value = config
    
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default
