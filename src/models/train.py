"""Training loop for probes on cached features."""
import copy

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn


def train_probe(model, X_train, y_train, X_val, y_val, *, epochs=300, lr=1e-3,
                weight_decay=0.05, batch_size=64, patience=40, seed=0, device="cuda"):
    """Train with class-weighted cross-entropy and keep the weights with the best validation macro F1.

    Returns the trained model (best epoch) and that best validation macro F1.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)

    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)

    # Rare classes weigh more, like class_weight="balanced" in scikit-learn
    counts = torch.bincount(y_train).float()
    loss_fn = nn.CrossEntropyLoss(weight=counts.sum() / (len(counts) * counts), label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_f1, best_state, epochs_without_gain = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(X_train), device=device)
        for start in range(0, len(X_train), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).argmax(dim=1).cpu().numpy()
        val_f1 = f1_score(y_val, val_pred, average="macro")

        if val_f1 > best_f1:
            best_f1, best_state, epochs_without_gain = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_f1


@torch.no_grad()
def predict(model, X, device="cuda"):
    """Predicted class index for each clip."""
    model.eval()
    return model(torch.tensor(X, dtype=torch.float32, device=device)).argmax(dim=1).cpu().numpy()