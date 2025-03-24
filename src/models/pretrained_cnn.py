"""
Pre-trained CNN model (ResNet50) for rice disease classification.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from .base_model import BaseModel


class PretrainedCNN(BaseModel):
    def __init__(self, num_classes: int, freeze_backbone: bool = True):
        """
        Initialize the pre-trained CNN model.

        Args:
            num_classes: Number of output classes
            freeze_backbone: Whether to freeze the backbone layers
        """
        super().__init__(num_classes)

        # Load pre-trained ResNet50 with updated API
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer with a more sophisticated classifier
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def get_transforms(train: bool = True):
        """
        Get the image transforms.

        Args:
            train: Whether these are training transforms

        Returns:
            torchvision.transforms.Compose object
        """
        if train:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
