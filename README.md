# ✍️ TGNHTR — Temporal Graph Network for Handwriting Recognition

> Graph-based Handwriting Text Recognition using **Graph Attention Networks**, **BiLSTM** and **CTC decoding**

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9+-blue">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.x-orange">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## Overview

**TGNHTR (Temporal Graph Network for Handwriting Text Recognition)** — модель для распознавания рукописного текста, которая рассматривает изображение **не как последовательность пикселей, а как графовую структуру**.

Каждый фрагмент изображения становится **узлом графа**, а связи между ними формируют пространственно-временной контекст.

Архитектура объединяет:

* 🧠 **Graph Attention Network (GAT)** — извлечение зависимостей между частями текста
* 🔁 **BiLSTM** — восстановление последовательности чтения
* ✍️ **CTC Loss** — обучение без посимвольного выравнивания

---

## Pipeline

```text
Image
  ↓
Graph Construction
(nodes + edges)
  ↓
GAT × 3
  ↓
Temporal Gather
  ↓
BiLSTM
  ↓
CTC Decoder
  ↓
Recognized Text
```

---

## Architecture

| Component     | Description                                             |
| ------------- | ------------------------------------------------------- |
| `node_proj`   | Проекция признаков узлов в пространство `node_channels` |
| `GATConv × 3` | Три слоя Graph Attention Network (Spektral)             |
| `tf.gather`   | Преобразование графа в последовательность               |
| `BiLSTM`      | Двунаправленное чтение последовательности               |
| `Dense`       | Классификатор по словарю символов                       |
| `CTC Loss`    | Обучение без выровненной разметки                       |


---

## Installation

```bash
pip install tensorflow spektral editdistance tqdm tf_keras
```

### Requirements

* Python **3.9+**
* TensorFlow **2.x**
* Keras 2 compatibility:

```bash
export TF_USE_LEGACY_KERAS=1
```

---

## Training

Запуск обучения:

```bash
python train.py
```

### Default configuration

| Parameter     | Value |
| ------------- | ----: |
| Epochs        |    5 |
| Batch size    |    16 |
| Optimizer     |  Adam |
| Learning rate |  1e-3 |

### Model initialization

```python
model = TGNHTR(
    vocab_size=61,
    node_channels=64,
    attn_heads=8,
    lstm_units=64,
    dropout_rate=0.2,
)
```

---

## Evaluation

Метрики считаются через `evaluate()`.

| Metric  | Meaning              |
| ------- | -------------------- |
| **CER** | Character Error Rate |
| **WER** | Word Error Rate      |

