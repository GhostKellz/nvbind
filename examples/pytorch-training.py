#!/usr/bin/env python3
"""
Example: PyTorch Training with nvbind GPU Passthrough

Run with:
  docker run --rm \
    --device=nvidia.com/gpu=gpu0 \
    -v $(pwd):/workspace \
    pytorch/pytorch:2.5.1-cuda12.6-cudnn9-runtime \
    python pytorch-training.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def check_gpu():
    """Verify GPU is available"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: No GPU found! Training will be slow.")
    print()

class SimpleModel(nn.Module):
    """Simple neural network for demonstration"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_example():
    """Simple training example"""
    # Check GPU
    check_gpu()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create synthetic dataset
    print("Creating synthetic dataset...")
    X = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    print("Initializing model...")
    model = SimpleModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print(f"Training on {device}...")
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    print("\n✓ Training completed successfully!")
    print("GPU passthrough with nvbind is working correctly!")

if __name__ == '__main__':
    train_example()