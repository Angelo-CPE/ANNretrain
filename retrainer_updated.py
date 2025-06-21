# -*- coding: utf-8 -*-
"""
Retrainer with automated train/validation/test split and unknown-class detection.
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from skimage.feature import hog
from scipy.signal import hilbert
import time

# ---- Configuration ----
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))        # should contain 'flawed' and 'not flawed' subfolders
TEST_SIZE = 0.1               # proportion for test split
VAL_SIZE = 0.2                # proportion of remaining for validation
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
UNKNOWN_THRESHOLD = 0.7       # softmax confidence threshold

def load_data_from_directory(directory):
    """
    Load images from `directory` expecting two subfolders ('flawed', 'not flawed'),
    apply grayscale->resize->crop->HOG+LMD feature extraction.
    Returns features X and labels y.
    """
    CAM_W, CAM_H = 128, 128  # final image size
    X, y = [], []
    for label in os.listdir(directory):
        lbl = label.lower()
        if lbl not in ('flawed', 'not flawed'):
            continue
        path = os.path.join(directory, label)
        if not os.path.isdir(path):
            continue
        for fname in os.listdir(path):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(path, fname)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError("cv2.imread returned None")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # center-crop to square
                h, w = gray.shape
                side = min(h, w)
                x0, y0 = (w-side)//2, (h-side)//2
                crop = gray[y0:y0+side, x0:x0+side]
                resized = cv2.resize(crop, (CAM_W, CAM_H))
                # augmentation
                if np.random.rand() < 0.5:
                    resized = cv2.flip(resized, 1)
                factor = 0.8 + np.random.rand() * 0.4
                resized = np.clip(resized * factor, 0, 255).astype(np.uint8)
                # HOG
                hog_f = hog(resized,
                            pixels_per_cell=(16,16),
                            cells_per_block=(2,2),
                            feature_vector=True,
                            block_norm='L2')
                # LMD: envelope + frequency
                signal = np.mean(resized, axis=0)
                analytic = hilbert(signal)
                envelope = np.abs(analytic)
                phase = np.unwrap(np.angle(analytic))
                freq = np.diff(phase)/(2*np.pi)
                # pad/truncate to envelope length
                if freq.shape[0] < envelope.shape[0]:
                    freq = np.pad(freq, (0, envelope.shape[0]-freq.shape[0]))
                else:
                    freq = freq[:envelope.shape[0]]
                features = np.concatenate([hog_f, envelope, freq])
                if features.shape[0] != 2020:
                    features = np.resize(features, 2020)
                X.append(features)
                y.append(1 if lbl == 'flawed' else 0)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    return np.array(X), np.array(y)

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return (torch.tensor(self.features[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long))

class ANNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 2)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop1(x)
        x = torch.relu(self.fc2(x))
        x = self.drop2(x)
        x = torch.relu(self.fc3(x))
        return self.out(x)

def main():
    # Load and split
    X, y = load_data_from_directory(ROOT_DIR)
    # first split out test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    # then split train/val from remaining
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, random_state=RANDOM_STATE, stratify=y_tmp)
    print(f"Samples -> train:{len(y_train)}, val:{len(y_val)}, test:{len(y_test)}")

    # DataLoaders
    train_loader = DataLoader(CustomDataset(X_train,y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(CustomDataset(X_val,  y_val),   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(CustomDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ANNModel(input_dim=X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    train_acc, train_loss = [], []
    val_acc,   val_loss   = [], []
    start = time.time()
    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss.append(running_loss/len(train_loader))
        train_acc.append(correct/total)

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                outputs = model(feats)
                loss = criterion(outputs, labels)
                v_loss += loss.item()
                preds = outputs.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)
        val_loss.append(v_loss/len(val_loader))
        val_acc.append(v_correct/v_total)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{EPOCHS} - "
                  f"Train Acc: {train_acc[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}")

    duration = time.time() - start
    print(f"Training took {duration/60:.1f}m")

    # Save model
    torch.save(model.state_dict(), "ANN_model.pth")
    print("Model weights saved to ANN_model.pth")

    # Evaluate on test set with unknown detection
    model.eval()
    y_true_known, y_pred_known = [], []
    unknown_count, total = 0, 0
    with torch.no_grad():
        for feats, labels in test_loader:
            feats, labels = feats.to(device), labels.to(device)
            outputs = model(feats)
            probs = torch.softmax(outputs, dim=1)
            maxp, preds = probs.max(dim=1)
            for i in range(len(labels)):
                total += 1
                if maxp[i] < UNKNOWN_THRESHOLD:
                    unknown_count += 1
                else:
                    y_true_known.append(labels[i].cpu().item())
                    y_pred_known.append(preds[i].cpu().item())
    print(f"Unrecognized (not wheels or low-confidence): {unknown_count}/{total}")
    if y_true_known:
        print("Accuracy on recognized samples: "
              f"{accuracy_score(y_true_known,y_pred_known)*100:.2f}%")
        print("Confusion Matrix:\n", confusion_matrix(y_true_known,y_pred_known))
        print("Classification Report:\n",
              classification_report(y_true_known,y_pred_known, zero_division=1))
    else:
        print("No high-confidence samples for evaluation.")

    # Plot metrics
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc,   label='Val Acc')
    plt.legend(), plt.title('Accuracy')
    plt.subplot(1,2,2)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss,   label='Val Loss')
    plt.legend(), plt.title('Loss')
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    print("Saved training_metrics.png")

if __name__ == "__main__":
    main()
