# Image Classification with CNN (TensorFlow/PyTorch) — ShadowFox AIML Internship

**Task (Beginner Level):** Train a straightforward image classification model
that can categorize images into elementary classes such as *cat*, *dog*,
*car*, etc., using a prominent deep learning library.

## Overview

This project trains a Convolutional Neural Network (CNN) in **PyTorch** on
the **CIFAR-10** dataset, which contains 60,000 32x32 color images across
exactly the kind of elementary classes described in the task brief:

`airplane, automobile (car), bird, cat, deer, dog, frog, horse, ship, truck`

The project includes:
- A from-scratch CNN architecture (`src/model.py`)
- A full training pipeline with data augmentation, checkpointing, and
  loss/accuracy plots (`src/train.py`)
- An inference script to classify any new image you provide
  (`src/predict.py`)
- Shared utilities for preprocessing (`src/utils.py`)

## Project Structure

```
image-classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── model.py       # CNN architecture
│   ├── train.py       # Training script (also has a --smoke-test mode)
│   ├── predict.py      # Inference on new images
│   └── utils.py        # Class names, transforms, device helper
├── data/                # (created automatically) CIFAR-10 dataset cache
├── models/              # Trained weights + training curve plot land here
├── sample_images/       # Put your own test images here
└── screenshots/         # Put your submission screenshots here
```

---

## 1. Setup in VS Code

1. **Install prerequisites**: Python 3.9+ and VS Code with the Python
   extension installed.

2. **Open the project folder** in VS Code:
   `File → Open Folder... → image-classifier`

3. **Open a terminal inside VS Code** (`` Ctrl+` ``) and create a virtual
   environment:

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. In VS Code, select the `.venv` interpreter: `Ctrl+Shift+P` →
   *Python: Select Interpreter* → choose the one inside `.venv`.

---

## 2. Train the model

```bash
cd src
python train.py --epochs 15 --batch-size 64
```

- The first run will automatically **download CIFAR-10** (~170 MB) into
  `../data/` — this requires an internet connection and happens only once.
- Training on CPU takes roughly 3-8 minutes per epoch (faster with a GPU).
  15 epochs typically reaches **~72-78% test accuracy** with this
  architecture.
- The best checkpoint (by test accuracy) is saved automatically to
  `../models/cifar10_cnn.pth`.
- A loss/accuracy plot is saved to `../models/training_curves.png` — take a
  screenshot of this for your submission.

**Quick sanity check (optional):** to verify your environment/pipeline works
before committing to a full training run, use the fast smoke-test mode
(uses random data, finishes in seconds, no download required):

```bash
python train.py --smoke-test --epochs 2
```

---

## 3. Classify your own images

Drop any `.jpg`/`.png` photo into `sample_images/`, then run:

```bash
python predict.py --image ../sample_images/your_photo.jpg
```

Or classify a whole folder at once:

```bash
python predict.py --folder ../sample_images/
```

Example output:

```
Loaded model (reported test accuracy: 0.7634)

Image: ../sample_images/dog.jpg
  dog                   91.42%
  cat                    4.15%
  horse                  2.03%
```

This is your **proof-of-work screenshot** moment — capture the terminal
output and a couple of test images for your LinkedIn video/submission.

---

## 4. Push the completed project to your `ShadowFox` GitHub repository

Per the internship pre-requisites, all tasks go into one repository named
**`ShadowFox`**. From inside the `image-classifier` folder:

```bash
# 1. Initialize git (skip if the ShadowFox repo already exists locally)
git init

# 2. Stage and commit your work
git add .
git commit -m "Beginner Task: CNN image classifier (CIFAR-10) with TensorFlow/PyTorch"

# 3. Point this repo at your GitHub "ShadowFox" repository
#    (create it first on github.com if you haven't already — public repo,
#    no README/gitignore initialization needed since we already have ours)
git remote add origin https://github.com/<your-username>/ShadowFox.git

# 4. Push
git branch -M main
git push -u origin main
```

If you're adding this task to an **existing** ShadowFox repo that already has
other tasks in it, instead put this project in its own subfolder (e.g.
`ShadowFox/beginner-image-classification/`) and just `git add`, `commit`,
and `push` as usual from the repo root.

> Note: `models/*.pth` (the trained weights, which can be 5-20MB+) is
> excluded via `.gitignore` by default to keep the repo lightweight, since
> it's regenerable by running `train.py`. If you want the trained weights
> in the repo too (e.g., so reviewers don't have to retrain), remove the
> `models/*.pth` line from `.gitignore` before committing, or use
> [Git LFS](https://git-lfs.com/) for large files.

---

## 5. What to include in your submission

- [ ] Screenshot of `train.py` running and completing (showing accuracy)
- [ ] Screenshot of `training_curves.png`
- [ ] Screenshot of `predict.py` correctly classifying a sample image
- [ ] Link to your GitHub `ShadowFox` repo with this project pushed
- [ ] LinkedIn video walking through the code and results
- [ ] LinkedIn profile updated with "ShadowFox AIML Intern"

## Notes on extending this project

- Swap in a deeper architecture (ResNet-style blocks) or use transfer
  learning (`torchvision.models.resnet18(pretrained=True)`) for higher
  accuracy.
- Train on your own custom dataset (e.g., real photos of cats/dogs/cars)
  by replacing the CIFAR-10 loader in `train.py` with
  `torchvision.datasets.ImageFolder` pointed at a folder structured as
  `data/train/<class_name>/*.jpg`.
- Add a small Flask/Streamlit web UI around `predict.py` for a live demo.
