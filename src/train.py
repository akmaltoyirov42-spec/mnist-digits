import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import time

from model import DigitCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BATCH = 128
EPOCHS = 5
LR = 1e-3

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),   # MNIST stats
])

train_set = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
test_set  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=BATCH, shuffle=False, num_workers=0)

print(f"Train: {len(train_set)}   Test: {len(test_set)}")


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


model = DigitCNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print(f"\nTraining {EPOCHS} epochs...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    running = 0.0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)

    train_loss = running / len(train_set)
    test_acc = evaluate(model, test_loader)
    print(f"Epoch {epoch}/{EPOCHS}  loss={train_loss:.4f}  test_acc={test_acc:.4f}  ({time.time()-t0:.1f}s)")

torch.save(model.state_dict(), MODEL_DIR / "digit_cnn.pt")
print(f"\nSaved -> {MODEL_DIR / 'digit_cnn.pt'}")
