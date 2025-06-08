# Neural Machine Translation Demo

A PyTorch-based English→French sequence‑to‑sequence model with Bahdanau attention.

## 📋 Overview

* Trains a single-layer GRU encoder/decoder with attention.
* Supports teacher forcing in training and greedy decoding in validation.
* Computes corpus BLEU and tracks loss/perplexity via TensorBoard.
* Visualizes an attention heatmap for the first validation batch.

## ⚙️ Requirements

* Python 3.8+
* PyTorch
* scikit-learn
* HuggingFace `datasets` & `evaluate` (or fallback: NLTK)
* Matplotlib

## 🚀 Installation

```bash
git clone <repo-url>
cd your-project
pip install -r requirements.txt
```

## ▶️ Running

```bash
python NLP_05.py            # trains & validates model
tensorboard --logdir runs   # launch TensorBoard dashboard
```

## 📂 Project Structure

```
├── data.csv                # parallel EN↔FR corpus
├── NLP_05.py               # main training & evaluation script
├── runs/                   # TensorBoard logs
├── README.md               # this file
```

## 📈 Results

* Training loss & perplexity curves in TensorBoard
* Validation loss & BLEU score per epoch
* Example attention heatmap displayed at validation

---

Feel free to tweak hyperparameters at the top of `NLP_05.py` for your experiments.
