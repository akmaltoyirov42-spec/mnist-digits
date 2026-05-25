# mnist digit recognition

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red)

a small CNN trained on MNIST. hits ~99% on the test set after 5 epochs.

---

## results

| | |
|---|---|
| test accuracy | **99.1%** |
| training time | ~2 min on CPU, ~20 sec on GPU |
| parameters | ~225k |

the mistakes are mostly weird handwriting — 4s that look like 9s, 7s that look like 1s.

---

## the model

basic CNN:

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

## what's next

want to extend this to Fashion-MNIST and EMNIST (letters) to see how the same architecture handles harder problems. also thinking about a Flask API so you can upload a hand-drawn digit and get a prediction back.

---

PyTorch, torchvision, scikit-learn, matplotlib, seaborn
