from src.utils.config import MODEL_DIR
from src.models.model import TunedModel3
from torchinfo import summary
from pathlib import Path
from rich.console import Console
import torch
console = Console()

def summarise_model():
    if not MODEL_DIR.exists():
        console.print(f'Model not found in {MODEL_DIR.parent}. Make sure to run train.py first', style='red bold')
        return

    model = TunedModel3()
    summary(model, input_size=(1,3,28,28))

if __name__=='__main__': summarise_model()