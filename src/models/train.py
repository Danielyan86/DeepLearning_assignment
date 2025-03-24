"""
Training script for rice disease classification models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import argparse
from pathlib import Path
import sys

# Add the project root directory to Python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.data.data_loader import get_data_loaders
from src.models.pretrained_cnn import PretrainedCNN
from src.visualization.visualizer import Visualizer


def get_device():
    """
    Get the appropriate device for training (M1 GPU, CPU, or CUDA).
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, save_dir
):
    """
    Train the model.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        save_dir: Directory to save results
    """
    device = get_device()
    model = model.to(device)

    # Enable mixed precision training for better performance
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0

    for epoch in range(num_epochs):
        # Train epoch
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # Validate
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = total_loss / len(val_loader)
        val_acc = 100.0 * correct / total

        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

        # Plot learning curves
        Visualizer.plot_learning_curves(
            train_losses,
            val_losses,
            train_accs,
            val_accs,
            save_path=os.path.join(save_dir, "learning_curves.png"),
        )


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize model
    model = PretrainedCNN(num_classes=3)

    # Get data loaders with optimized num_workers for M1
    train_loader, val_loader = get_data_loaders(
        args.data_dir,
        transform=model.get_transforms(train=True),
        batch_size=args.batch_size,
    )

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train model
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        args.num_epochs,
        args.save_dir,
    )


if __name__ == "__main__":
    main()
