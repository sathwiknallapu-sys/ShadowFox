"""
utils.py
--------
Shared constants and helper functions used by both train.py and
predict.py: the class label names, the image pre-processing
("transform") pipeline, and a small helper to load the trained model.
"""

import torch
from torchvision import transforms

# CIFAR-10 class names, in the exact order the dataset provides labels.
# Note this already covers the example classes from the task brief:
# "cat", "dog", "car" (called "automobile" in CIFAR-10), etc.
CLASS_NAMES = [
    "airplane",
    "automobile (car)",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Mean/std used to normalize CIFAR-10 images (standard values).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Transform applied to training images (with light augmentation).
train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
)

# Transform applied to test/validation images (no augmentation).
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
)

# Transform used at inference time on an arbitrary user-supplied image:
# resize to 32x32 first, then the same normalization as test_transform.
inference_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
)


def get_device() -> torch.device:
    """Return CUDA/MPS device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon GPUs
        return torch.device("mps")
    return torch.device("cpu")
