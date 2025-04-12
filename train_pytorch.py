import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F

import read_data

# Dice coefficient and loss function for PyTorch
def dice_coef(y_true, y_pred, smooth=1):
    # Ensure the sizes are the same
    y_pred = F.interpolate(y_pred, size=y_true.shape[2:], mode='bilinear', align_corners=True)
    intersection = torch.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# UNet Model definition in PyTorch
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Upsample the output to match the input size
        )
    
    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2


# Assuming your data loading function (read_data.read) works similarly for PyTorch
class CustomDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        
        return x, y

# Load your data (assuming `read_data.read` returns numpy arrays)
MASSACHUSETTS_PATH = "datasets/Massachusetts/"
TRAINING_SET = 1
MODEL_NAME = 'UNET'  # or 'INCEPTION' or 'UNETV2'

path = MASSACHUSETTS_PATH + 'train/'
x_train, y_train = read_data.read(path, 110)

if 2 == TRAINING_SET:
    index = 75 * 49
    x_train = x_train[0:index,:,:,:]
    y_train = y_train[0:index,:,:,:]

path = MASSACHUSETTS_PATH + 'validation/'
x_valid, y_valid = read_data.read(path, 4)

# Convert to PyTorch tensors and create DataLoader
transform = transforms.ToTensor()

train_dataset = CustomDataset(x_train, y_train, transform=transform)
valid_dataset = CustomDataset(x_valid, y_valid, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=5)

# Initialize the model
model = UNet().cuda()

# Optimizer and Loss function
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = dice_coef_loss

# Training Loop
epochs = 100
save_weights_path = "results/" + MODEL_NAME
model_name = MODEL_NAME + ".pth"

# Model checkpoints and early stopping
best_loss = float('inf')
patience = 10
no_improvement_count = 0

# Save history function
def save_history(hist, filename):
    with open(filename, 'wb') as file_pi:
        pickle.dump(hist, file_pi)

history = {'train_loss': [], 'valid_loss': []}

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(labels, outputs)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)

    # Validation Loop
    model.eval()
    running_valid_loss = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(labels, outputs)
            running_valid_loss += loss.item()

    avg_valid_loss = running_valid_loss / len(valid_loader)
    history['valid_loss'].append(avg_valid_loss)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Validation Loss: {avg_valid_loss:.4f}")
    sys.stdout.flush()
    # Checkpoint and Early Stopping
    if avg_valid_loss < best_loss:
        best_loss = avg_valid_loss
        no_improvement_count = 0

        if not os.path.exists(save_weights_path):
            os.makedirs(save_weights_path)

        torch.save(model.state_dict(), save_weights_path + model_name)
    else:
        no_improvement_count += 1
        if no_improvement_count >= patience:
            print("Early stopping triggered.")
            break

# Save the training history
save_history(history, save_weights_path + MODEL_NAME + ".history")


