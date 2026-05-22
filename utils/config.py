import json
import os
import yaml

def load_model_paths(config_path="config/model_paths.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found at: {config_path}")
    
    with open(config_path, "r") as f:
        return json.load(f)

def load_dataset_config(dataset_name: str, config_path="config/datasets.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    if dataset_name not in configs:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in {config_path}"
        )

    return configs[dataset_name]
