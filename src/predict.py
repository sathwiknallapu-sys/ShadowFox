"""
predict.py
----------
Load the trained model and classify a single image (or all images in a
folder) into one of the 10 CIFAR-10 classes: airplane, automobile
(car), bird, cat, deer, dog, frog, horse, ship, truck.

Usage
-----
    python src/predict.py --image path/to/photo.jpg
    python src/predict.py --folder path/to/folder_of_images/
"""

import argparse
import os

import torch
import torch.nn.functional as F
from PIL import Image

from model import SimpleCNN
from utils import CLASS_NAMES, inference_transform, get_device

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "cifar10_cnn.pth")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_model(model_path: str, device: torch.device) -> SimpleCNN:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'. "
            "Run `python src/train.py` first to train and save a model."
        )
    checkpoint = torch.load(model_path, map_location=device)
    model = SimpleCNN(num_classes=len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded model (reported test accuracy: {checkpoint.get('test_accuracy', 'N/A')})")
    return model


def predict_image(model: SimpleCNN, image_path: str, device: torch.device, top_k: int = 3):
    image = Image.open(image_path).convert("RGB")
    tensor = inference_transform(image).unsqueeze(0).to(device)  # add batch dim

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_probs, top_idxs = torch.topk(probs, k=min(top_k, len(CLASS_NAMES)))

    print(f"\nImage: {image_path}")
    for prob, idx in zip(top_probs, top_idxs):
        print(f"  {CLASS_NAMES[idx]:<20s} {prob.item() * 100:6.2f}%")

    best_idx = top_idxs[0].item()
    return CLASS_NAMES[best_idx], top_probs[0].item()


def main():
    parser = argparse.ArgumentParser(description="Classify image(s) with the trained CNN.")
    parser.add_argument("--image", type=str, help="Path to a single image file.")
    parser.add_argument("--folder", type=str, help="Path to a folder of images.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained model weights.")
    parser.add_argument("--top-k", type=int, default=3, help="Show top-k predicted classes.")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Provide either --image <path> or --folder <path>.")

    device = get_device()
    model = load_model(args.model, device)

    if args.image:
        predict_image(model, args.image, device, top_k=args.top_k)

    if args.folder:
        files = sorted(
            f for f in os.listdir(args.folder)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        if not files:
            print(f"No images found in {args.folder}")
        for fname in files:
            predict_image(model, os.path.join(args.folder, fname), device, top_k=args.top_k)


if __name__ == "__main__":
    main()
