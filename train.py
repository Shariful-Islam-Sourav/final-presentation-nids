"""
train.py — Network Intrusion Detection System
==============================================
Handles data loading, preprocessing, model training,
evaluation, and visualization for the NIDS project.

Dataset: NSL-KDD (downloaded automatically if not found)
Models:  Logistic Regression, Decision Tree, Random Forest
"""

import os
import sys
import warnings
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (saves PNGs without a display)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

# NSL-KDD column names (41 features + label + difficulty)
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

# Categorical features that need encoding
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# Output directories
OUTPUT_DIR = "outputs"
MODEL_DIR  = "models"
DATA_DIR   = "data"

# URLs for the NSL-KDD dataset (GitHub mirror — no login required)
TRAIN_URL = (
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
)
TEST_URL  = (
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
)

TRAIN_FILE = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_FILE  = os.path.join(DATA_DIR, "KDDTest+.txt")


# ─────────────────────────────────────────────
#  STEP 1 — DOWNLOAD DATASET
# ─────────────────────────────────────────────

def download_dataset():
    """Download NSL-KDD train and test files if they do not exist locally."""
    os.makedirs(DATA_DIR, exist_ok=True)

    for url, filepath in [(TRAIN_URL, TRAIN_FILE), (TEST_URL, TEST_FILE)]:
        if os.path.exists(filepath):
            print(f"  [✓] Found: {filepath}")
        else:
            print(f"  [↓] Downloading {os.path.basename(filepath)} ...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"  [✓] Saved to {filepath}")
            except Exception as e:
                print(f"  [✗] Download failed: {e}")
                print("      Please download the NSL-KDD dataset manually from:")
                print("      https://www.unb.ca/cic/datasets/nsl.html")
                sys.exit(1)


# ─────────────────────────────────────────────
#  STEP 2 — LOAD DATASET
# ─────────────────────────────────────────────

def load_dataset():
    """
    Load the NSL-KDD dataset from disk.
    Returns: (train DataFrame, test DataFrame)
    """
    download_dataset()

    train_df = pd.read_csv(TRAIN_FILE, header=None, names=COLUMNS)
    test_df  = pd.read_csv(TEST_FILE,  header=None, names=COLUMNS)

    print(f"\n  Training samples : {len(train_df):,}")
    print(f"  Testing  samples : {len(test_df):,}")
    print(f"  Features         : {len(COLUMNS) - 2}")       # exclude label + difficulty

    return train_df, test_df


# ─────────────────────────────────────────────
#  STEP 3 — PREPROCESS DATA
# ─────────────────────────────────────────────

def preprocess(train_df, test_df):
    """
    Clean and encode the data.
    - Convert multi-class labels → binary (Normal / Attack)
    - Label-encode categorical columns
    - Scale numeric features with StandardScaler
    - Drop the 'difficulty' column (not a feature)

    Returns: X_train, X_test, y_train, y_test, feature_names, scaler
    """
    # --- Combine for consistent encoding across train/test ---
    train_df = train_df.copy()
    test_df  = test_df.copy()

    # Drop difficulty column — not a feature
    train_df.drop(columns=["difficulty"], inplace=True)
    test_df.drop(columns=["difficulty"],  inplace=True)

    # --- Binary label: 'normal' → 0, everything else → 1 (Attack) ---
    train_df["label"] = train_df["label"].apply(lambda x: 0 if x.strip() == "normal" else 1)
    test_df["label"]  = test_df["label"].apply(lambda x: 0 if x.strip() == "normal" else 1)

    # --- Encode categorical columns ---
    le = LabelEncoder()
    combined = pd.concat([train_df, test_df], axis=0)   # fit on all values

    for col in CATEGORICAL_COLS:
        combined[col] = le.fit_transform(combined[col].astype(str))

    # Split back
    n_train = len(train_df)
    train_df = combined.iloc[:n_train].copy()
    test_df  = combined.iloc[n_train:].copy()

    # --- Separate features and labels ---
    feature_cols = [c for c in train_df.columns if c != "label"]
    X_train = train_df[feature_cols].values.astype(np.float64)
    X_test  = test_df[feature_cols].values.astype(np.float64)
    y_train = train_df["label"].values
    y_test  = test_df["label"].values

    # --- Scale numeric features ---
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"\n  Class balance (train) — Normal: {(y_train == 0).sum():,} | Attack: {(y_train == 1).sum():,}")

    return X_train, X_test, y_train, y_test, feature_cols, scaler


# ─────────────────────────────────────────────
#  STEP 4 — TRAIN MODELS
# ─────────────────────────────────────────────

def train_models(X_train, y_train):
    """
    Train three classifiers on the preprocessed data.
    Returns a dict: { model_name: trained_model }
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }

    trained = {}
    for name, model in models.items():
        print(f"  Training {name} ...", end=" ", flush=True)
        model.fit(X_train, y_train)
        trained[name] = model
        print("done ✓")

    return trained


# ─────────────────────────────────────────────
#  STEP 5 — EVALUATE MODELS
# ─────────────────────────────────────────────

def evaluate_models(trained_models, X_test, y_test):
    """
    Evaluate each model on the test set.
    Returns: results dict { name: {accuracy, report, cm} }
    """
    results = {}

    print()
    for name, model in trained_models.items():
        y_pred   = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report   = classification_report(y_test, y_pred, target_names=["Normal", "Attack"])
        cm       = confusion_matrix(y_test, y_pred)

        results[name] = {"accuracy": accuracy, "report": report, "cm": cm, "y_pred": y_pred}

        print(f"  ── {name} ──")
        print(f"     Accuracy : {accuracy * 100:.2f}%")
        print(report)

    return results


# ─────────────────────────────────────────────
#  STEP 6 — SAVE BEST MODEL
# ─────────────────────────────────────────────

def save_best_model(trained_models, results, scaler, feature_cols):
    """
    Save the best-performing model (by accuracy) using joblib.
    Also saves the scaler and feature column names for use in predict.py.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Find best model
    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_model = trained_models[best_name]

    model_path   = os.path.join(MODEL_DIR, "best_model.pkl")
    scaler_path  = os.path.join(MODEL_DIR, "scaler.pkl")
    columns_path = os.path.join(MODEL_DIR, "feature_cols.pkl")

    joblib.dump(best_model,   model_path)
    joblib.dump(scaler,       scaler_path)
    joblib.dump(feature_cols, columns_path)

    print(f"\n  [✓] Best model saved  : {best_name} (Accuracy: {results[best_name]['accuracy']*100:.2f}%)")
    print(f"      Model  → {model_path}")
    print(f"      Scaler → {scaler_path}")

    return best_name


# ─────────────────────────────────────────────
#  STEP 7 — VISUALIZATIONS
# ─────────────────────────────────────────────

# Shared color palette for consistency across plots
PALETTE   = ["#4C72B0", "#DD8452"]
STYLE_CFG = dict(style="whitegrid", palette="muted")

def _save(fig, filename):
    """Save a matplotlib figure to the outputs/ folder."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved → {path}")


def plot_attack_distribution(train_df):
    """Bar chart: Normal vs Attack traffic count."""
    labels_col = train_df["label"].apply(lambda x: "Normal" if str(x).strip() == "normal" else "Attack")
    counts = labels_col.value_counts()

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values, color=["#4C72B0", "#DD8452"], width=0.5, edgecolor="white")

    # Add value labels on bars
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 200,
                f"{bar.get_height():,}",
                ha="center", fontsize=12, fontweight="bold")

    ax.set_title("Attack vs Normal Traffic Distribution", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Traffic Type", fontsize=12)
    ax.set_ylabel("Number of Records", fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    fig.tight_layout()
    _save(fig, "attack_distribution.png")


def plot_protocol_distribution(train_df):
    """Bar chart: Protocol type distribution."""
    counts = train_df["protocol_type"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4C72B0", "#55A868", "#DD8452"]
    bars = ax.bar(counts.index, counts.values, color=colors[:len(counts)], width=0.5, edgecolor="white")

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 200,
                f"{bar.get_height():,}",
                ha="center", fontsize=11, fontweight="bold")

    ax.set_title("Protocol Type Distribution", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Protocol", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    fig.tight_layout()
    _save(fig, "protocol_distribution.png")


def plot_correlation_heatmap(X_train_df):
    """Heatmap of feature correlations (top 15 numeric features)."""
    # Use only numeric columns
    numeric_df = X_train_df.select_dtypes(include=[np.number])

    # Pick top-15 most correlated columns with 'label' for a readable heatmap
    corr_matrix = numeric_df.corr()
    top_cols = (corr_matrix["label"].abs()
                .sort_values(ascending=False)
                .head(15).index.tolist())
    sub_corr = corr_matrix.loc[top_cols, top_cols]

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        sub_corr, annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.5, ax=ax, annot_kws={"size": 7}
    )
    ax.set_title("Feature Correlation Heatmap (Top 15 Features)", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    _save(fig, "heatmap.png")


def plot_confusion_matrix(results, best_name, y_test):
    """Confusion matrix for the best model."""
    cm = results[best_name]["cm"]

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    _save(fig, "confusion_matrix.png")


def plot_model_comparison(results):
    """Bar chart comparing accuracy of all three models."""
    names = list(results.keys())
    accs  = [results[n]["accuracy"] * 100 for n in names]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, accs, color=colors, width=0.5, edgecolor="white")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{acc:.2f}%",
                ha="center", fontsize=12, fontweight="bold")

    ax.set_title("Model Accuracy Comparison", fontsize=15, fontweight="bold", pad=15)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    _save(fig, "model_comparison.png")


def plot_feature_importance(trained_models, feature_cols, top_n=15):
    """Horizontal bar chart of top-N Random Forest feature importances."""
    rf_model    = trained_models["Random Forest"]
    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]

    top_features    = [feature_cols[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.Blues_r(np.linspace(0.3, 0.85, top_n))
    ax.barh(top_features[::-1], top_importances[::-1], color=colors, edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score", fontsize=12)
    fig.tight_layout()
    _save(fig, "feature_importance.png")


def generate_all_visualizations(train_df, trained_models, results, best_name,
                                 X_train_df, y_test, feature_cols):
    """Generate and save all 6 required visualizations."""
    print("\n  Generating visualizations ...")
    plot_attack_distribution(train_df)
    plot_protocol_distribution(train_df)
    plot_correlation_heatmap(X_train_df)
    plot_confusion_matrix(results, best_name, y_test)
    plot_model_comparison(results)
    plot_feature_importance(trained_models, feature_cols)


# ─────────────────────────────────────────────
#  PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def run_training():
    """
    Full training pipeline:
      1. Load data  →  2. Preprocess  →  3. Train  →
      4. Evaluate   →  5. Save model  →  6. Visualize
    """
    print("\n" + "=" * 60)
    print("  Network Intrusion Detection System — Training Pipeline")
    print("=" * 60)

    # ── 1. Load ──
    print("\n[1/6] Loading dataset ...")
    train_df, test_df = load_dataset()

    # ── 2. Preprocess ──
    print("\n[2/6] Preprocessing data ...")
    X_train, X_test, y_train, y_test, feature_cols, scaler = preprocess(train_df, test_df)

    # Build a DataFrame version of X_train for the heatmap (needs column names + label)
    train_processed = pd.DataFrame(X_train, columns=feature_cols)
    train_processed["label"] = y_train

    # ── 3. Train ──
    print("\n[3/6] Training models ...")
    trained_models = train_models(X_train, y_train)

    # ── 4. Evaluate ──
    print("\n[4/6] Evaluating models ...")
    results = evaluate_models(trained_models, X_test, y_test)

    # ── 5. Save ──
    print("\n[5/6] Saving best model ...")
    best_name = save_best_model(trained_models, results, scaler, feature_cols)

    # ── 6. Visualize ──
    print("\n[6/6] Generating visualizations ...")
    generate_all_visualizations(
        train_df, trained_models, results, best_name,
        train_processed, y_test, feature_cols
    )

    print("\n" + "=" * 60)
    print("  Training complete!  All outputs saved to outputs/ and models/")
    print("=" * 60 + "\n")

    return trained_models, results, best_name
