from src.utils.config import MODEL_DIR
from src.data.load_data import load_data
from src.models.model import TunedModel3
import torch

def evaluate_model():
    if not MODEL_DIR.exists():
        print(f'Model not found in {MODEL_DIR.parent}. Make sure to run training and verify name of .pt file before evaluating model.\n[model should be named "final_model.pt"]')
        return

    _, _ , loader_test = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TunedModel3()
    state_dict = torch.load(MODEL_DIR, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    correct_preds = 0
    total_samples = 0
    model.eval()
    for X, y in loader_test:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.no_grad():
            preds = model(X).argmax(dim=1)
            correct_preds += (y == preds).sum().item()
            total_samples += y.size(0)

    print('accuracy on test data: ', correct_preds / total_samples)

if __name__ == "__main__": evaluate_model()