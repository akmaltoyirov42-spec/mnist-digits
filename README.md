# MNIST Digit Recognition

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red)

A small CNN that reads handwritten digits. Trained on MNIST, hits ~99% on the test set after 5 epochs.

---

## Results

| Metric | Value |
|---|---|
| Test accuracy | **99.1%** |
| Training time | ~2 min on CPU, ~20s on GPU |
| Parameters | ~225k |

The model misses mostly on ambiguous handwriting — 4s that look like 9s, 7s that look like 1s. Per-digit accuracy is above 98.5% for every class.

---

## The model

Pretty standard CNN:

```
Input  28×28×1
→ Conv 3×3, 32 filters  → ReLU → MaxPool 2×2
→ Conv 3×3, 64 filters  → ReLU → MaxPool 2×2
→ Dropout 0.25
→ FC 1600 → 128 → ReLU → Dropout
→ FC 128 → 10
```

Trained with Adam (lr=1e-3), batch size 128, 5 epochs, cross-entropy loss.

---

## Run it

```bash
git clone https://github.com/akmaltoyirov42-spec/mnist-digits.git
cd mnist-digits
pip install -r requirements.txt

# trains and saves the model (downloads MNIST automatically)
python src/train.py

# generates confusion matrix + sample predictions in output/
python src/evaluate.py
```

---

## Output

After running `evaluate.py` you get three plots in `output/`:

- `confusion_matrix.png` — where the model confuses digits
- `sample_predictions.png` — 16 test images with predictions (red = wrong)
- `per_digit_accuracy.png` — accuracy per digit

---

## Stack

PyTorch, torchvision, scikit-learn, matplotlib, seaborn
