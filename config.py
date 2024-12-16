import json
from pathlib import Path
import logging


def load_config(config_path: str) -> dict:
    """Loads configuration parameters from a JSON file."""
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading config file {config_path}: {e}")
        return {
            'sa_weight': 2,
            'qed_weight': 1,
            'docking_weight': 1.0
        }  # default configuration

    return config_data
