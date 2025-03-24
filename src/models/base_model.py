"""
Base model class for all model implementations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple


class BaseModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        raise NotImplementedError

    def train_epoch(
        self, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer
    ) -> Tuple[float, float]:
        """
        Train the model for one epoch.

        Args:
            train_loader: DataLoader for training data
            criterion: Loss function
            optimizer: Optimizer

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return total_loss / len(train_loader), 100.0 * correct / total

    def validate(
        self, val_loader: DataLoader, criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Validate the model.

        Args:
            val_loader: DataLoader for validation data
            criterion: Loss function

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return total_loss / len(val_loader), 100.0 * correct / total

    def get_predictions(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get model predictions for a dataset.

        Args:
            data_loader: DataLoader for the dataset

        Returns:
            Tuple of (predictions, true labels)
        """
        self.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                outputs = self(inputs)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(targets.numpy())

        return np.array(all_preds), np.array(all_labels)

    def get_class_probabilities(self, data_loader: DataLoader) -> np.ndarray:
        """
        Get class probabilities for all samples.

        Args:
            data_loader: DataLoader for the dataset

        Returns:
            Array of class probabilities
        """
        self.eval()
        all_probs = []

        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                outputs = self(inputs)
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_probs)
