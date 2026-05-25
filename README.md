# mnist digit recognition

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red)

i'm new to deep learning so i started with the classic — train a CNN to read handwritten digits. got ~99% on the test set which i was pretty happy with.

---

## results

| | |
|---|---|
| test accuracy | **99.1%** |
| training time | ~2 min on CPU, ~20 sec on GPU |
| parameters | ~225k |

the mistakes are mostly weird handwriting — 4s that look like 9s, 7s that look like 1s. fair enough.

---

## the model

just a basic CNN, nothing fancy:

```
input 28x28
-> conv 3x3, 32 filters -> relu -> maxpool
-> conv 3x3, 64 filters -> relu -> maxpool
-> dropout
-> fc 128 -> relu -> dropout
-> fc 10
```

adam optimizer, lr=1e-3, batch size 128, 5 epochs, cross-entropy loss.

---

## run it

```bash
git clone https://github.com/akmaltoyirov42-spec/mnist-digits.git
cd mnist-digits
pip install -r requirements.txt

# downloads MNIST automatically and trains
python src/train.py

# makes confusion matrix + sample predictions in output/
python src/evaluate.py
```

---

## what i learned

- dropout actually helps (without it i overfit by epoch 3)
- LBFGS is way more efficient than SGD for small problems but i used adam here anyway
- the gram matrix idea from style transfer doesn't apply here but that's the next project
- training on GPU is like 6x faster, but for MNIST CPU is fine

---

PyTorch, torchvision, scikit-learn, matplotlib, seaborn
