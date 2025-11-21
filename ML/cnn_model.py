import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np 

class CNNmodel(nn.Module):
    """
    CNN model architecture for EMG classification using Spectrogram features.
    This class acts as a training module, handling its own architecture, 
    optimizer, loss function, and epoch execution.
    """
    def __init__(self, in_channels=8, num_classes=2, learning_rate=1e-3, device="cpu"):
        super().__init__()

        # --- Architecture Definition ---
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
        
        # --- Training Components Initialization ---
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = device
        
        # Move model to device
        self.to(self.device)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the network."""
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.global_pool(self.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

    
    # ==========================================================
    # --- Training Step Methods (Matching neuralnet.py style) ---
    # ==========================================================
    
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the CrossEntropy loss given model output (logits) and target labels."""
        return self.criterion(output, target)

    def backward(self, loss: torch.Tensor) -> None:
        """Zeroes gradients and performs the backward pass."""
        self.optimizer.zero_grad()
        loss.backward()

    def step(self) -> None:
        """Updates model weights using the optimizer."""
        self.optimizer.step()

    
    # ==========================================================
    # --- Epoch Execution Methods (High-level train/test) ---
    # ==========================================================
    
    def train_epoch(self, dataloader: DataLoader) -> tuple[float, float]:
        """Runs a single training epoch over the entire dataloader."""
        self.train() # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in dataloader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            
            # Forward pass
            out = self(xb)
            
            # Optimization steps
            loss = self.compute_loss(out, yb)
            self.backward(loss)
            self.step()

            # Metrics aggregation
            running_loss += loss.item() * yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def test_epoch(self, dataloader: DataLoader) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Runs a single evaluation epoch over the entire dataloader.
        
        Returns: test_accuracy, predicted_labels (numpy array), true_labels (numpy array)
        """
        self.eval() # Set model to evaluation mode
        correct = 0
        total = 0
        preds_all = []
        labels_all = []

        with torch.no_grad():
            for xb, yb in dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                
                out = self(xb)
                preds = out.argmax(1)
                
                correct += (preds == yb).sum().item()
                total += yb.size(0)
                preds_all.append(preds.cpu().numpy())
                labels_all.append(yb.cpu().numpy())

        test_acc = correct / total
        
        # Concatenate and return predictions/labels for confusion matrix
        final_preds = np.concatenate(preds_all)
        final_labels = np.concatenate(labels_all)
        
        return test_acc, final_preds, final_labels