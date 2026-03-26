# Bigram Language Model — From Counts to Neural Networks

This project implements a **character-level bigram language model** in two different ways:

1. A **count-based approach** using frequency matrices
2. A **neural network approach** using PyTorch

The goal is to understand how models can learn to generate text (names) by predicting the next character based on the current one.

---

## 📂 Files

* `bigram.ipynb` — Builds the model using raw counts and probability matrices
* `nn_bigram.ipynb` — Reimplements the same idea using a neural network
* `names.txt` — Dataset used for training

---

## 🧠 What this project does

At its core, the model learns:

> “Given one character, what is the most likely next character?”

It then uses these learned probabilities to generate new names by sampling one character at a time.

---

## ⚙️ Workflow (Bigram)

```text
Names → Character mapping → Bigram pairs → Count matrix → Probabilities → Sampling
```

---

## ⚙️ Workflow (Neural Network)

```text
Input character → One-hot encoding → Linear layer → Softmax → Probabilities
→ Loss calculation → Backpropagation → Weight updates
```

---

## 🔍 Key Insight

While implementing both versions, I realized:

> A single-layer neural network with softmax can represent the same behavior as the bigram count model.

The neural network is essentially **learning the same probability table** that the count-based approach computes directly.

---

## Why we need Neural Network implementation
# ⚠️ Limitations of Bigram Models

Bigram models are limited because they only consider one previous character to predict the next. This restricts their ability to capture longer patterns in names.

When we try to increase the context (e.g., trigrams or higher n-grams), the size of the count matrix grows exponentially.

For example:
- Bigram: 27 × 27 = 729
- Trigram: 27 × 27 × 27 = 19,683

This rapid growth makes count-based methods inefficient and difficult to manage for larger contexts.

To overcome this, neural networks are used. They can learn patterns from larger contexts without explicitly storing massive tables, making them more scalable and powerful.

## 📊 Bigram vs Neural Network

| Aspect      | Count-based Model | Neural Network           |
| ----------- | ----------------- | ------------------------ |
| Approach    | Direct counting   | Learned through training |
| Training    | Not required      | Uses gradient descent    |
| Flexibility | Limited           | Can be extended further  |

---

## 🚀 What I Learned

* How probability distributions can be built from raw data
* Why log probabilities are used for stable training
* How neural networks approximate statistical models
* How sampling from probabilities generates realistic outputs
* Why log likelihood or negative log likelihood is used as loss function

---

## 📌 Learning Source

This project was implemented while learning neural network and language model concepts from the makemore series by Andrej Karpathy. The code follows his implementation closely — I restructured it by changing variable names for better understanding and it also gives a proper name convention and also wrapped the forward pass, loss computation, backward pass, and update step (in neural net implementation) into separate functions for clarity. The explanations in this README are written in my own words to document what I understood

---

## ▶️ How to Run

1. Open the notebooks in Jupyter or VS Code
2. Run all cells in order
3. Observe generated names

---
