# Character-Level Language Model — From Bigrams to Neural Networks (MLP)

This project extends the basic bigram model into a more powerful **neural network-based language model**, capable of handling **larger context sizes** and learning richer patterns in text.

---

## 📂 Files

* `mlp.ipynb` — Multi-layer neural network implementation
* `names.txt` — Dataset used for training

---

## 🧠 What this project does

The model learns to predict the next character given a **sequence of previous characters (context)**.

Unlike the bigram model (which only uses one previous character), this implementation supports:

> **Flexible context size (n-grams)** — bigram, trigram, or even larger context windows.

---

## ⚙️ Workflow

```text
Names → Character mapping → Context creation → Embedding
→ Neural Network (MLP) → Softmax → Probabilities
→ Loss → Backpropagation → Weight updates
```

---

## 🔍 Key Improvements Over Bigram

* Supports **larger context (trigram, 4-gram, etc.)**
* No exponential growth of tables
* Learns patterns instead of memorizing counts
* More expressive and scalable

---

## ⚠️ Why Not Use Count-Based Models?

Increasing context in count-based models leads to exponential growth:

```text
Bigram   → 27² = 729
Trigram  → 27³ = 19,683
4-gram   → 27⁴ = 531,441
```

This becomes impractical very quickly.

Neural networks solve this by **learning representations instead of storing massive tables**.

---

## 🧠 Key Concepts I Learned

### 1. Variable Context Size (n-grams)

* The model can now take multiple previous characters as input
* Context is no longer fixed to 1 (bigram)
* Easily extendable to any size

---

### 2. Train / Validation Split

* Data is split into:

  * Training set
  * Validation set
* Helps detect **overfitting**
* Ensures model generalization

---

### 3. Cross Entropy Loss

* Combines:

  * Log Softmax
  * Negative Log Likelihood

> More numerically stable than computing them separately.

---

### 4. Mini-Batch Training

* Instead of using full dataset:

  * Train on small random batches
* Benefits:

  * Faster training
  * More stable updates
  * Better generalization

---

### 5. Learning Rate Scheduling

Instead of fixed learning rate, we adjust it during training:

```python
lr = 0.1 if i < 100000 else 0.01
```

* High learning rate → faster learning initially
* Lower learning rate → fine-tuning later

---

### 6. Learning Rate Visualization

* Observed how loss changes over time
* Helps in selecting better learning rate

---

### 7. Embeddings vs One-Hot Encoding

One-hot encoding:

* Sparse representation
* No relationship between characters
* All characters equally distant

Embeddings:

* Dense representation
* Learn relationships between characters
* Similar characters form clusters in vector space

> This allows the model to capture patterns and similarities in language.

---

### 8. Tensor Reshaping (`view` vs `unbind`)

* `unbind` creates new tensors → more memory usage
* `view` reshapes tensor without copying data

> Using `view` is more efficient when reshaping inputs for the network.

---

## 🚀 What I Learned Overall

* How increasing context improves language modeling
* Why count-based models don’t scale
* How neural networks learn representations instead of memorizing
* Importance of training strategies (mini-batch, LR scheduling)
* How embeddings capture semantic structure

---

## 📌 Learning Source

This project was implemented while learning neural networks and language modeling concepts from the makemore series by Andrej Karpathy.
The code follows Karpathy's implementation closely. I focused on understanding each concept deeply, which is documented in the sections above.

---

## ▶️ How to Run

1. Open the notebook in Jupyter or VS Code
2. Run all cells
3. Observe training loss and generated names

---

## 🔮 Next Steps

* Add deeper networks (multiple layers)
* Experiment with different embedding sizes
* Improve sampling (temperature scaling)
* Move towards sequence models (RNNs / Transformers)

---
