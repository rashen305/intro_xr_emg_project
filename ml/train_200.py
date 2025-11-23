# train_single_subject_myo_fixed.py
import os
import sys

# Workaround for Windows DLL loading issue
# Set environment variable before importing torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, stft, detrend

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ===========================
# Config
# ===========================
FS = 200                      # Sampling rate
WINDOW_SIZE = 256
STRIDE = 50
NPERSEG = 128
NOVERLAP = 64

BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = {"rest": 0, "pinch": 1}

# ===========================
# New Files for Raymond (200 Hz)
# ===========================
DATA_FILES = [
    ("../myo/samples/raymond_arm_90_deg_200hz.csv", LABELS["rest"]),
    ("../myo/samples/raymond_arm_90_deg_pinch_200hz.csv", LABELS["pinch"]),
    ("../myo/samples/raymond_arm_down_200hz.csv", LABELS["rest"]),
    ("../myo/samples/raymond_arm_down_pinch_200hz.csv", LABELS["pinch"]),
    ("../myo/samples/raymond_bending_arm.csv", LABELS["rest"]),
    ("../myo/samples/raymond_bending_arm_pinch.csv", LABELS["pinch"]),
    ("../myo/samples/raymond_swing_arm.csv", LABELS["rest"])
]

# ===========================
# Preprocessing
# ===========================
def preprocess(path):
    df = pd.read_csv(path)

    # Keep only EMG1–EMG8 columns
    emg_cols = [f"emg{i}" for i in range(1, 9)]
    data = df[emg_cols].values.astype(np.float64)

    # detrend + DC removal
    data = detrend(data, axis=0, type='constant')
    data = data - np.mean(data, axis=0, keepdims=True)

    # notch 60 Hz
    b, a = iirnotch(60.0 / (FS / 2), 30)
    data = filtfilt(b, a, data, axis=0)

    # bandpass 20–90 Hz
    b, a = butter(4, [20/(FS/2), 90/(FS/2)], btype='band')
    data = filtfilt(b, a, data, axis=0)

    # sliding windows
    windows = []
    N = len(data)
    for start in range(0, N - WINDOW_SIZE + 1, STRIDE):
        windows.append(data[start:start + WINDOW_SIZE])
    windows = np.array(windows)

    # STFT per window per channel
    specs = []
    for w in windows:
        ch_specs = []
        for ch in range(w.shape[1]):
            f, t, Zxx = stft(
                w[:, ch], fs=FS,
                nperseg=NPERSEG,
                noverlap=NOVERLAP,
                boundary=None
            )
            ch_specs.append(np.abs(Zxx))
        specs.append(np.array(ch_specs))

    return np.array(specs).astype(np.float32)

# ===========================
# Dataset
# ===========================
class EMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

# ===========================
# CNN Model
# ===========================
class CNNmodel(nn.Module):
    def __init__(self, in_channels=8, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, (5,3), padding=(2,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2,1))

        self.conv2 = nn.Conv2d(32, 64, (3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2,1))

        self.conv3 = nn.Conv2d(64, 128, (3,3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(128, 64)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.global_pool(self.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

# ===========================
# Training (Single Subject)
# ===========================
def train_single_subject():
    print("\nLoading & preprocessing 200 Hz data...")

    X_list, y_list = [], []

    for path, label in DATA_FILES:
        X_proc = preprocess(path)
        X_list.append(X_proc)
        y_list.append(np.full(len(X_proc), label))

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    print("Data shape (windows, channels, freq_bins, time_steps):", X.shape)

    # Log + Normalize
    X = np.log1p(X)
    mean = X.mean(axis=(0,2,3), keepdims=True)
    std = X.std(axis=(0,2,3), keepdims=True) + 1e-8
    X = (X - mean) / std

    # Random 80/20 split
    N = len(X)
    idx = np.random.permutation(N)
    train_idx = idx[:int(0.8*N)]
    test_idx  = idx[int(0.8*N):]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    train_dataset = EMGDataset(X_train, y_train)
    test_dataset  = EMGDataset(X_test,  y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Model
    model = CNNmodel().to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0
    best_preds = None
    best_labels = None

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS+1):
        # ---- train ----
        model.train()
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)

        train_acc = correct / total

        # ---- test ----
        model.eval()
        correct = 0
        total = 0
        preds_all = []
        labels_all = []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                preds = out.argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
                preds_all.append(preds.cpu().numpy())
                labels_all.append(yb.cpu().numpy())

        test_acc = correct / total

        print(f"Epoch {epoch:02d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_preds = np.concatenate(preds_all)
            best_labels = np.concatenate(labels_all)

    print("\nBest Test Acc:", best_acc)

    # Confusion Matrix
    cm = confusion_matrix(best_labels, best_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["rest", "pinch"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Raymond 200 Hz — Confusion Matrix")
    plt.show()

    save_path = "train_single_subject_myo_model_swing.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Pretrained model weights saved to '{save_path}'")

# ===========================
# Main
# ===========================
if __name__ == "__main__":
    train_single_subject()
