"""
Data loading and preprocessing utilities for the rice plant disease dataset.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class RiceDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        """
        Args:
            root_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on an image
            is_train (bool): Whether this is the training set
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train

        # Get all image paths and labels
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.images = []
        self.labels = []

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.images.append(os.path.join(cls_dir, img_name))
                self.labels.append(self.class_to_idx[cls])

        # Split into train and validation sets
        if is_train:
            self.images, _, self.labels, _ = train_test_split(
                self.images,
                self.labels,
                test_size=0.2,
                random_state=42,
                stratify=self.labels,
            )
        else:
            _, self.images, _, self.labels = train_test_split(
                self.images,
                self.labels,
                test_size=0.2,
                random_state=42,
                stratify=self.labels,
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loaders(root_dir, transform, batch_size=32):
    """
    Create train and validation data loaders.

    Args:
        root_dir (string): Directory with all the images
        transform (callable): Transform to be applied on images
        batch_size (int): Batch size for the data loaders

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_dataset = RiceDiseaseDataset(root_dir, transform=transform, is_train=True)
    val_dataset = RiceDiseaseDataset(root_dir, transform=transform, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader
