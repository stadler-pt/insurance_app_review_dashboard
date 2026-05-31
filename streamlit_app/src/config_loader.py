from pathlib import Path
import json
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STREAMLIT_ROOT = Path(__file__).resolve().parents[1]

def load_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_app_config() -> dict:
    return load_json(STREAMLIT_ROOT / 'config' / 'app_config.json')

def get_label_metadata() -> dict:
    return load_json(STREAMLIT_ROOT / 'config' / 'label_metadata.json')

def resolve_project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path