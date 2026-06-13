# 🛡️ Network Intrusion Detection System using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-NSL--KDD-orange)
![ML](https://img.shields.io/badge/ML-scikit--learn-red?logo=scikit-learn)

A machine learning–based **Network Intrusion Detection System (NIDS)** that classifies network traffic as **Normal** or **Attack** using the NSL-KDD benchmark dataset.

This project was built as a final assignment for the **Introduction to Open Source Software** course at Sejong University.

---

## 📌 Project Description

Modern networks face constant threats from malicious actors. An **Intrusion Detection System (IDS)** monitors network traffic and raises alerts when suspicious activity is detected.

This project trains three machine learning classifiers on the **NSL-KDD** dataset — an improved version of the classic KDD Cup 1999 dataset — to distinguish between normal traffic and various attack types (DoS, Probe, R2L, U2R).

---

## ✨ Features

| Feature | Details |
|---|---|
| Dataset | NSL-KDD (auto-downloaded) |
| Binary Classification | Normal (0) vs Attack (1) |
| Models Trained | Logistic Regression, Decision Tree, Random Forest |
| Visualizations | 6 charts saved as PNG |
| Best Model Saved | `models/best_model.pkl` (joblib) |
| CLI Prediction | Interactive terminal interface |

---

## 📂 Project Structure

```
network-intrusion-detection/
│
├── data/
│   ├── KDDTrain+.txt          ← NSL-KDD training set (auto-downloaded)
│   └── KDDTest+.txt           ← NSL-KDD testing set  (auto-downloaded)
│
├── models/
│   ├── best_model.pkl         ← Saved best classifier
│   ├── scaler.pkl             ← StandardScaler for new inputs
│   └── feature_cols.pkl       ← Feature column order
│
├── outputs/
│   ├── attack_distribution.png
│   ├── protocol_distribution.png
│   ├── heatmap.png
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   └── feature_importance.png
│
├── main.py                    ← Entry point (run this)
├── train.py                   ← Training pipeline + visualizations
├── predict.py                 ← Prediction CLI
├── requirements.txt           ← Python dependencies
├── README.md                  ← This file
└── LICENSE                    ← MIT License
```

---

## 🔧 Installation

### 1. Clone or download this repository

```bash
git clone https://github.com/your-username/nids-ml.git
cd nids-ml
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> The NSL-KDD dataset will be **automatically downloaded** the first time you run training.

---

## 🚀 Usage

### Train models and generate all visualizations

```bash
python main.py --train
```

This will:
- Download the NSL-KDD dataset (if not already present)
- Preprocess and encode features
- Train Logistic Regression, Decision Tree, and Random Forest
- Print accuracy reports
- Save all 6 charts to `outputs/`
- Save the best model to `models/best_model.pkl`

---

### Run the interactive prediction CLI

```bash
python main.py --predict
```

Choose from:
- **Option 1** — Run predefined demo samples (Normal HTTP vs DoS Attack)
- **Option 2** — Enter your own network values manually

---

### Train and then predict in one command

```bash
python main.py --all
```

---

## 📊 Sample Output

### Model Accuracy Comparison

| Model | Accuracy |
|---|---|
| Logistic Regression | 75.39% |
| Decision Tree | 76.34% |
| **Random Forest** | **77.07%** ✅ (Best) |

> **Note on test accuracy:** The NSL-KDD `KDDTest+` set intentionally contains novel attack types not seen during training, making it a realistic and challenging benchmark. This is why test accuracy is lower than what you'd see on a held-out portion of the training data — it tests generalization to *unseen* attack categories.

### Visualizations Generated

| Chart | Description |
|---|---|
| `attack_distribution.png` | Count of Normal vs Attack records |
| `protocol_distribution.png` | TCP / UDP / ICMP traffic breakdown |
| `heatmap.png` | Feature correlation matrix (top 15 features) |
| `confusion_matrix.png` | True vs Predicted classifications (best model) |
| `model_comparison.png` | Accuracy of all 3 models side-by-side |
| `feature_importance.png` | Top 15 most important features (Random Forest) |

---

## 🔄 Project Workflow

```
[Input: NSL-KDD .txt files]
          ↓
[Data Loading & Column Assignment]
          ↓
[Preprocessing]
  • Multi-class label → Binary (Normal / Attack)
  • Label encoding of categorical columns
  • StandardScaler normalization
          ↓
[Train / Test Split  (80% / 20%)]
          ↓
[Model Training]
  • Logistic Regression
  • Decision Tree
  • Random Forest
          ↓
[Evaluation]
  • Accuracy, Precision, Recall, F1
  • Confusion Matrix
          ↓
[Best Model Saved → models/best_model.pkl]
          ↓
[Visualizations → outputs/*.png]
          ↓
[Interactive CLI for live predictions]
```

---

## 📦 Dependencies

| Library | Version | Purpose |
|---|---|---|
| `numpy` | 1.26.4 | Numerical computing |
| `pandas` | 2.2.2 | Data manipulation |
| `scikit-learn` | 1.5.0 | ML models & preprocessing |
| `matplotlib` | 3.9.0 | Chart generation |
| `seaborn` | 0.13.2 | Statistical visualizations |
| `joblib` | 1.4.2 | Model persistence |

---

## 📄 Dataset

**NSL-KDD** is a widely-used benchmark for network intrusion detection research. It improves upon the original KDD Cup 1999 dataset by removing duplicate records and balancing class distribution.

- Source: [University of New Brunswick — CIC](https://www.unb.ca/cic/datasets/nsl.html)
- Auto-download mirror: [defcom17/NSL_KDD on GitHub](https://github.com/defcom17/NSL_KDD)
- Features: 41 network traffic attributes
- Labels: `normal` + 22 attack categories → converted to **binary** (Normal / Attack)

---

## 📝 Why README.md and requirements.txt Matter in Open Source

**README.md** is the front door of any open source project. It tells contributors and users:
- What the project does
- How to install and run it
- What the expected output looks like

Without a good README, even excellent code is unusable by others.

**requirements.txt** ensures **reproducibility** — anyone who clones the project can instantly install the exact same library versions with one command (`pip install -r requirements.txt`), eliminating "works on my machine" problems.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

MIT is one of the most permissive open source licenses: anyone may use, copy, modify, and distribute this software, provided the original copyright notice is retained.

---

## 👤 Author
SOURAV MD SHARIFUL ISLAM
**Sejong University — Introduction to Open Source Software**

