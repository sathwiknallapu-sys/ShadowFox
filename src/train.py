"""
train.py
--------
Trains SimpleCNN on the CIFAR-10 dataset to classify images into 10
elementary categories (airplane, automobile/car, bird, cat, deer, dog,
frog, horse, ship, truck).

Usage
-----
    python src/train.py                     # normal training run
    python src/train.py --epochs 25 --batch-size 128
    python src/train.py --smoke-test         # fast dry-run using random
                                              # data to verify the whole
                                              # pipeline works, without
                                              # needing to download the
                                              # dataset (useful to sanity
                                              # check your environment).

Outputs
-------
    models/cifar10_cnn.pth        -> trained model weights
    models/training_curves.png    -> loss/accuracy plot
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")  # allow saving plots without a display
import matplotlib.pyplot as plt

from model import SimpleCNN
from utils import CLASS_NAMES, train_transform, test_transform, get_device

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def get_dataloaders(batch_size: int, smoke_test: bool = False):
    """Return (train_loader, test_loader).

    In normal mode this downloads CIFAR-10 via torchvision (only needed
    once; it is cached locally afterwards). In --smoke-test mode it
    generates small random tensors instead, so the training loop can be
    verified end-to-end without any internet access or download wait.
    """
    if smoke_test:
        n_train, n_test = 256, 64
        x_train = torch.randn(n_train, 3, 32, 32)
        y_train = torch.randint(0, 10, (n_train,))
        x_test = torch.randn(n_test, 3, 32, 32)
        y_test = torch.randint(0, 10, (n_test,))
        train_ds = TensorDataset(x_train, y_train)
        test_ds = TensorDataset(x_test, y_test)
    else:
        from torchvision.datasets import CIFAR10

        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        train_ds = CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        test_ds = CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """Run a single training or evaluation epoch. Returns (avg_loss, accuracy)."""
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def plot_curves(history, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["test_loss"], label="Test loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train accuracy")
    axes[1].plot(epochs, history["test_acc"], label="Test accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train an image classifier on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--smoke-test", action="store_true",
                         help="Run a tiny fast pass with random data to verify the pipeline works.")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(args.batch_size, smoke_test=args.smoke_test)
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    model = SimpleCNN(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    best_path = os.path.join(MODELS_DIR, "cifar10_cnn.pth")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        elapsed = time.time() - start
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
              f"{elapsed:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "test_accuracy": best_acc,
            }, best_path)
            print(f"  -> New best model saved (test_acc={best_acc:.4f})")

    print(f"\nTraining complete. Best test accuracy: {best_acc:.4f}")
    print(f"Best model weights saved to: {best_path}")

    if not args.smoke_test:
        plot_curves(history, os.path.join(MODELS_DIR, "training_curves.png"))


if __name__ == "__main__":
    main()
