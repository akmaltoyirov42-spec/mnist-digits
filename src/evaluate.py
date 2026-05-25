import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

from model import DigitCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

test_set = datasets.MNIST("data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

model = DigitCNN().to(DEVICE)
model.load_state_dict(torch.load("models/digit_cnn.pt", map_location=DEVICE))
model.eval()

# collect predictions
all_preds, all_true = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        pred = model(x).argmax(dim=1).cpu().numpy()
        all_preds.extend(pred)
        all_true.extend(y.numpy())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)

acc = (all_preds == all_true).mean()
print(f"Test accuracy: {acc:.4f}")
print()
print(classification_report(all_true, all_preds, digits=3))


# confusion matrix
fig, ax = plt.subplots(figsize=(8, 7))
cm = confusion_matrix(all_true, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=range(10), yticklabels=range(10), ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — accuracy {acc:.2%}", fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT / "confusion_matrix.png", dpi=130)
plt.close()
print("Saved confusion_matrix.png")


# sample predictions grid — mix of right and wrong
wrong_idx = np.where(all_preds != all_true)[0]
right_idx = np.where(all_preds == all_true)[0]

# 12 correct + 4 wrong (so portfolio shows both)
np.random.seed(7)
pick_right = np.random.choice(right_idx, size=12, replace=False)
pick_wrong = np.random.choice(wrong_idx, size=min(4, len(wrong_idx)), replace=False)
picks = np.concatenate([pick_right, pick_wrong])

fig, axes = plt.subplots(4, 4, figsize=(9, 9))
for ax, idx in zip(axes.flat, picks):
    img, _ = test_set[idx]
    img = img.squeeze().numpy() * 0.3081 + 0.1307   # unnormalize
    pred, true = all_preds[idx], all_true[idx]
    color = "green" if pred == true else "red"
    ax.imshow(img, cmap="gray")
    ax.set_title(f"pred {pred}  /  true {true}", color=color, fontsize=10)
    ax.axis("off")

plt.suptitle("Sample Predictions (red = mistake)", fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT / "sample_predictions.png", dpi=130)
plt.close()
print("Saved sample_predictions.png")


# per-digit accuracy bar
per_class = []
for d in range(10):
    mask = all_true == d
    per_class.append((all_preds[mask] == d).mean())

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(range(10), per_class, color="#3498db", edgecolor="white")
for i, v in enumerate(per_class):
    ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylim(0.9, 1.01)
ax.set_xticks(range(10))
ax.set_xlabel("Digit")
ax.set_ylabel("Accuracy")
ax.set_title("Per-digit accuracy", fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT / "per_digit_accuracy.png", dpi=130)
plt.close()
print("Saved per_digit_accuracy.png")

print("\nDone — plots in output/")
