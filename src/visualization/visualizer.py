"""
Visualization utilities for model training and evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple
import torch
from PIL import Image
import os


class Visualizer:
    @staticmethod
    def plot_learning_curves(
        train_losses: List[float],
        val_losses: List[float],
        train_accs: List[float],
        val_accs: List[float],
        save_path: str = None,
    ):
        """
        Plot learning curves for loss and accuracy.

        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            train_accs: List of training accuracies
            val_accs: List of validation accuracies
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot losses
        ax1.plot(train_losses, label="Training Loss")
        ax1.plot(val_losses, label="Validation Loss")
        ax1.set_title("Learning Curves - Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        # Plot accuracies
        ax2.plot(train_accs, label="Training Accuracy")
        ax2.plot(val_accs, label="Validation Accuracy")
        ax2.set_title("Learning Curves - Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_misclassified_examples(
        model,
        data_loader,
        class_names: List[str],
        num_examples: int = 3,
        save_path: str = None,
    ):
        """
        Plot the worst misclassified examples for each class.

        Args:
            model: Trained model
            data_loader: DataLoader for the dataset
            class_names: List of class names
            num_examples: Number of examples to plot per class
            save_path: Optional path to save the plot
        """
        model.eval()
        all_probs = []
        all_labels = []
        all_images = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(model.device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(targets.numpy())
                all_images.extend(inputs.cpu())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_images = np.array(all_images)

        fig, axes = plt.subplots(
            len(class_names),
            num_examples,
            figsize=(5 * num_examples, 5 * len(class_names)),
        )

        for i, class_name in enumerate(class_names):
            class_indices = np.where(all_labels == i)[0]
            class_probs = all_probs[class_indices]

            # Find worst misclassified examples
            misclassified = np.where(np.argmax(class_probs, axis=1) != i)[0]
            if len(misclassified) > 0:
                worst_indices = misclassified[
                    np.argsort(
                        class_probs[misclassified, i]
                        - np.max(class_probs[misclassified, :], axis=1)
                    )[:num_examples]
                ]

                for j, idx in enumerate(worst_indices):
                    img = all_images[class_indices[idx]]
                    pred_class = np.argmax(class_probs[idx])

                    # Denormalize and convert to PIL Image
                    img = img.permute(1, 2, 0).numpy()
                    img = (img * 255).astype(np.uint8)

                    axes[i, j].imshow(img)
                    axes[i, j].set_title(
                        f"True: {class_name}\nPred: {class_names[pred_class]}"
                    )
                    axes[i, j].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        save_path: str = None,
    ):
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_path: Optional path to save the plot
        """
        cm = np.zeros((len(class_names), len(class_names)))
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        if save_path:
            plt.savefig(save_path)
        plt.close()
