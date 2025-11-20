# -*- coding: utf-8 -*-
"""test.py — Evaluate Tuned Improved CNN on KMNIST test data."""

import sys
import os
import subprocess
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pytz")

# ---------------------------------------------------------------------
# Minimal dependency check
# ---------------------------------------------------------------------
def install_if_missing(module_name, pip_name=None):
    """
    Try to import a module; if it fails, install the corresponding pip package.
    """
    if pip_name is None:
        pip_name = module_name
    try:
        __import__(module_name)
    except ImportError:
        print(f"[INFO] Installing missing package: {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


# module_name -> pip_package
REQUIRED = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "gdown": "gdown",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
}

for module_name, pip_name in REQUIRED.items():
    install_if_missing(module_name, pip_name)

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import gdown
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ---------------------------------------------------------------------
# Download / load test data
# ---------------------------------------------------------------------
# These should point to the KMNIST test .npz files used during training.
TEST_IMGS_URL = "https://drive.google.com/uc?id=1XxqC-7zch8Gr6-uHioGwgFpCH-Fc3c3g"
TEST_LABELS_URL = "https://drive.google.com/uc?id=1Q1zLTzR3GXgbMx2VjvGlx_XVku_kSS_v"

if not os.path.exists("kmnist-test-imgs.npz"):
    print("[INFO] Downloading kmnist-test-imgs.npz from Google Drive...")
    gdown.download(TEST_IMGS_URL, "kmnist-test-imgs.npz", quiet=False)

if not os.path.exists("kmnist-test-labels.npz"):
    print("[INFO] Downloading kmnist-test-labels.npz from Google Drive...")
    gdown.download(TEST_LABELS_URL, "kmnist-test-labels.npz", quiet=False)

if not os.path.exists("best_tuned_improved_cnn.pth"):
    raise FileNotFoundError(
        "best_tuned_improved_cnn.pth not found in the current directory.\n"
        "Train the TunedImprovedCNN first and save the best model as best_tuned_improved_cnn.pth."
    )

test_imgs = np.load("kmnist-test-imgs.npz", allow_pickle=True)["arr_0"]
test_labels = np.load("kmnist-test-labels.npz", allow_pickle=True)["arr_0"]

test_imgs_tensor = torch.tensor(test_imgs, dtype=torch.float32).unsqueeze(1) / 255.0
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

test_dataset = TensorDataset(test_imgs_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ---------------------------------------------------------------------
# Model definition (must match training code)
# ---------------------------------------------------------------------
class ImprovedCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.res_conv = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.res_conv is not None:
            identity = self.res_conv(identity)
        out = out + identity
        return self.relu(out)


class TunedImprovedCNN(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super().__init__()
        self.block1 = ImprovedCNNBlock(1, 32)
        self.pool = nn.MaxPool2d(2, 2)
        self.block2 = ImprovedCNNBlock(32, 64)
        self.block3 = ImprovedCNNBlock(64, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------
# Load trained weights
# ---------------------------------------------------------------------
model = TunedImprovedCNN(dropout_rate=0.4).to(DEVICE)
state_dict = torch.load("best_tuned_improved_cnn.pth", map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="weighted")
recall = recall_score(all_labels, all_preds, average="weighted")
f1 = f1_score(all_labels, all_preds, average="weighted")

print("\n==== Test Set Metrics ====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds))

# ---------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Test Set Confusion Matrix")

# If running headless, just save the figure
if matplotlib.get_backend().lower() == "agg":
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
else:
    plt.show()

metrics_df = pd.DataFrame(
    [
        {
            "Dataset": "KMNIST Test",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
        }
    ]
)

print("\nAggregated Metrics Table:")
print(metrics_df.to_string(index=False))
