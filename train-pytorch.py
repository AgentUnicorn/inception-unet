import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import functional as F
from tqdm import tqdm

import read_data  # your data reading module
from unetV2 import UNetPlusInception  # your PyTorch model file

# Dice coefficient and loss (same as before)
smooth = 1.0


def dice_coef(y_true, y_pred):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2.0 * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def save_history(history, filename):
    with open(filename, "wb") as f:
        pickle.dump(history, f)


class SatelliteDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert numpy arrays to torch tensors and transpose to (C, H, W)
        x = torch.tensor(self.images[idx], dtype=torch.float32).permute(2, 0, 1)
        y = torch.tensor(self.masks[idx], dtype=torch.float32).permute(2, 0, 1)
        return x, y


def train_model(
    model,
    train_loader,
    valid_loader,
    optimizer,
    criterion,
    device,
    epochs=100,
    patience=10,
    save_path="model_weights.pth",
):
    best_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_dice_score = 0.0
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            train_dice_score += dice_coef(y_batch, outputs).item() * x_batch.size(0)

        train_loss /= len(train_loader.dataset)
        train_dice_score /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_dice_score = 0.0
        with torch.no_grad():
            for x_val, y_val in valid_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item() * x_val.size(0)
                val_dice_score += dice_coef(y_val, outputs).item() * x_val.size(0)

        val_loss /= len(valid_loader.dataset)
        val_dice_score /= len(valid_loader.dataset)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice_score:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice_score:.4f}"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice_score)
        history["val_dice"].append(val_dice_score)

        # Early stopping and checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return history


def main():
    MASSACHUSETTS_PATH = "datasets/Massachusetts/"
    TRAINING_SET = 1
    MODEL_NAME = "UNETV2"  # Only UNETV2 implemented here

    # Load data
    train_path = MASSACHUSETTS_PATH + "train/"
    x_train, y_train = read_data.read(train_path, 110)

    if TRAINING_SET == 2:
        index = 75 * 49
        x_train = x_train[:index]
        y_train = y_train[:index]

    print("len train", len(x_train))

    valid_path = MASSACHUSETTS_PATH + "validation/"
    x_valid, y_valid = read_data.read(valid_path, 4)
    print("len valid", len(x_valid))

    # Create datasets and loaders
    train_dataset = SatelliteDataset(x_train, y_train)
    valid_dataset = SatelliteDataset(x_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=5, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetPlusInception()
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-5)
    criterion = dice_coef_loss

    save_weights_path = f"results/{MODEL_NAME}_weights.pth"
    history_path = f"results/{MODEL_NAME}_history.pkl"

    history = train_model(
        model,
        train_loader,
        valid_loader,
        optimizer,
        criterion,
        device,
        epochs=100,
        patience=10,
        save_path=save_weights_path,
    )

    save_history(history, history_path)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
