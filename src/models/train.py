from src.models.model import TunedModel3
from src.data.load_data import load_data
from src.utils.config import MODEL_DIR
import torch
import torch.nn as nn
from tqdm import trange

def train_model():
    loader_train, loader_val, _ = load_data()

    model = TunedModel3()
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0034, weight_decay=4.85e-5, betas=(0.75, 0.95))
    epochs = 80
    best_loss = float('inf')
    patience = 10
    counter = 0
    min_delta = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f'Using device: {device}')

    train_losses = []
    val_losses = []

    for epoch in trange(epochs, desc='Training', total=epochs):
        model.train()

        training_loss_epoch = 0
        for X, y in loader_train:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimiser.zero_grad()

            logits = model(X)
            loss = loss_fn(logits, y)
            training_loss_epoch += loss.item()

            loss.backward()
            optimiser.step()
        train_losses.append(training_loss_epoch / len(loader_train))

        # Early stopping
        model.eval()

        val_loss_epoch = 0
        for X, y in loader_val:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.no_grad():
                logits = model(X)
                val_loss_epoch += loss_fn(logits, y).item()
        val_loss_epoch /= len(loader_val)
        val_losses.append(val_loss_epoch)

        if val_loss_epoch < best_loss - min_delta:
            best_loss = val_loss_epoch
            best_epoch = epoch
            counter = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimiser.state_dict(),
                'best_loss': best_loss,
                'counter': counter
            }, MODEL_DIR.parent / 'train_checkpoint.pth')
        else:
            counter += 1
            if counter >= patience:
                print(
                    f'Early stopping triggered, validation loss has not improved after {patience} rounds.\nBest epoch: {best_epoch}\nBest loss: {best_loss}')
                break

    #Restoring best model
    checkpoint = torch.load(MODEL_DIR.parent / 'train_checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    torch.save(model.state_dict(), MODEL_DIR)
    print(f'Model has been trained. Saved model in {MODEL_DIR.parent}')

if __name__ == "__main__": train_model()