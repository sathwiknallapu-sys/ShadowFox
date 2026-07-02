"""
model.py
--------
Defines the Convolutional Neural Network (CNN) architecture used for
image classification into elementary classes such as "cat", "dog",
"car", "airplane", "bird", "deer", "dog", "frog", "horse", "ship",
"truck" (the 10 classes of the CIFAR-10 dataset).

The architecture is intentionally simple (a few conv blocks + a
classifier head) so it is fast to train on a laptop CPU/GPU while
still reaching a respectable accuracy (~70-75% test accuracy after
~15-20 epochs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """A small CNN for 32x32 RGB image classification (CIFAR-10 style)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # --- Convolutional feature extractor ---
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
        )

        # --- Classifier head ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Quick sanity check: run a random batch through the network and
    # make sure the output shape is correct.
    model = SimpleCNN(num_classes=10)
    dummy_input = torch.randn(4, 3, 32, 32)  # batch of 4 RGB 32x32 images
    output = model(dummy_input)
    print("Model instantiated successfully.")
    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)  # should be [4, 10]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {n_params:,}")
