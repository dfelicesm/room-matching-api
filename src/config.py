from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

RAW_PATH = config["data"]["raw_path"]
PRECOMPUTED_PATH = config["data"]["precomputed_path"]

TOP_K = config["matching"]["top_k"]

API_HOST = config["api"]["host"]
API_PORT = config["api"]["port"]
