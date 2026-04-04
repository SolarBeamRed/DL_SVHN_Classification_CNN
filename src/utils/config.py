from pathlib import Path

BASE_DIR = Path().resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'checkpoints' / 'final_model.pt'