"""
predict.py — Network Intrusion Detection System
================================================
Loads the saved best model and provides:
  1. A programmatic predict() function
  2. An interactive CLI where users can enter network values manually

Usage:
    python predict.py
"""

import os
import sys
import numpy as np
import joblib

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

MODEL_DIR    = "models"
MODEL_PATH   = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_cols.pkl")

# Label mapping
LABEL_MAP = {0: "✅  NORMAL Traffic", 1: "🚨  ATTACK Detected!"}
COLOR_MAP = {0: "\033[92m", 1: "\033[91m"}   # green / red ANSI codes
RESET     = "\033[0m"

# ─────────────────────────────────────────────
#  LOAD MODEL ARTIFACTS
# ─────────────────────────────────────────────

def load_model():
    """
    Load the saved model, scaler, and feature column list from disk.
    Exits with a helpful message if they are not found.
    """
    for path in [MODEL_PATH, SCALER_PATH, COLUMNS_PATH]:
        if not os.path.exists(path):
            print(f"\n  [✗] File not found: {path}")
            print("      Please run training first:  python main.py --train\n")
            sys.exit(1)

    model       = joblib.load(MODEL_PATH)
    scaler      = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(COLUMNS_PATH)

    print(f"  [✓] Loaded model  : {MODEL_PATH}")
    print(f"  [✓] Loaded scaler : {SCALER_PATH}")
    print(f"  [✓] Features      : {len(feature_cols)}")

    return model, scaler, feature_cols


# ─────────────────────────────────────────────
#  PREDICT FUNCTION (programmatic use)
# ─────────────────────────────────────────────

def predict(feature_values, model=None, scaler=None, feature_cols=None):
    """
    Predict a single network traffic record.

    Parameters
    ----------
    feature_values : list or np.ndarray
        A list of numeric feature values in the same order as feature_cols.
    model, scaler, feature_cols : pre-loaded objects (loaded if None)

    Returns
    -------
    tuple: (label_int, label_str, confidence_pct)
    """
    if model is None or scaler is None or feature_cols is None:
        model, scaler, feature_cols = load_model()

    arr     = np.array(feature_values, dtype=np.float64).reshape(1, -1)
    arr_sc  = scaler.transform(arr)
    pred    = model.predict(arr_sc)[0]
    proba   = model.predict_proba(arr_sc)[0]
    conf    = proba[pred] * 100

    return int(pred), LABEL_MAP[pred], round(conf, 2)


# ─────────────────────────────────────────────
#  PREDEFINED SAMPLE INPUTS  (for demo / CLI)
# ─────────────────────────────────────────────

# These samples mimic typical NSL-KDD records.
# Values must match the 41 numeric features (after encoding) in order.
SAMPLE_INPUTS = {
    "normal_http": {
        "description": "Normal HTTP web browsing traffic",
        # duration, protocol_type(tcp=2), service(http=10), flag(SF=9),
        # src_bytes, dst_bytes, land, wrong_frag, urgent, hot,
        # num_failed_logins, logged_in, num_compromised, root_shell, su_attempted,
        # num_root, num_file_creations, num_shells, num_access_files, num_outbound_cmds,
        # is_host_login, is_guest_login, count, srv_count,
        # serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate,
        # same_srv_rate, diff_srv_rate, srv_diff_host_rate,
        # dst_host_count, dst_host_srv_count,
        # dst_host_same_srv_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rate,
        # dst_host_srv_diff_host_rate, dst_host_serror_rate, dst_host_srv_serror_rate,
        # dst_host_rerror_rate, dst_host_srv_rerror_rate
        "values": [0, 2, 10, 9, 215, 45076, 0, 0, 0, 0,
                   0, 1, 0, 0, 0,
                   0, 0, 0, 0, 0,
                   0, 0, 1, 1,
                   0.0, 0.0, 0.0, 0.0,
                   1.0, 0.0, 0.0,
                   0, 0,
                   0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0,
                   0.0, 0.0],
    },
    "neptune_dos": {
        "description": "DoS Neptune flood attack (SYN flood)",
        "values": [0, 2, 10, 5, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0,
                   0, 0, 511, 511,
                   1.0, 1.0, 0.0, 0.0,
                   1.0, 0.0, 0.0,
                   255, 255,
                   1.0, 0.0, 0.01,
                   0.0, 1.0, 1.0,
                   0.0, 0.0],
    },
}


# ─────────────────────────────────────────────
#  INTERACTIVE CLI
# ─────────────────────────────────────────────

def _print_header():
    print("\n" + "=" * 60)
    print("   Network Intrusion Detection System — Prediction CLI")
    print("=" * 60)


def _print_result(pred_int, pred_str, confidence, description=""):
    color = COLOR_MAP.get(pred_int, "")
    if description:
        print(f"\n  Input    : {description}")
    print(f"  Result   : {color}{pred_str}{RESET}")
    print(f"  Confidence: {confidence:.2f}%")
    print("─" * 60)


def run_demo_predictions(model, scaler, feature_cols):
    """Run the two predefined sample inputs for demonstration."""
    print("\n  [Demo] Running predefined sample predictions ...\n")
    for key, sample in SAMPLE_INPUTS.items():
        values = sample["values"]
        if len(values) != len(feature_cols):
            print(f"  [!] Sample '{key}' has {len(values)} values but model expects {len(feature_cols)}. Skipping.")
            continue
        pred_int, pred_str, conf = predict(values, model, scaler, feature_cols)
        _print_result(pred_int, pred_str, conf, sample["description"])


def run_interactive_cli(model, scaler, feature_cols):
    """
    Interactive mode: prompt the user to enter values for key features,
    fill the rest with neutral defaults, then predict.
    """
    print("\n  [Interactive Mode]")
    print("  Enter values for the key network features below.")
    print("  Press ENTER to use the default value shown in brackets.\n")

    # Simplified prompt: only ask for the most intuitive features
    prompts = [
        ("duration",          "Connection duration (seconds)",           0),
        ("src_bytes",         "Bytes sent from source",                  0),
        ("dst_bytes",         "Bytes sent to destination",               0),
        ("logged_in",         "Is user logged in? (1=Yes, 0=No)",        0),
        ("count",             "Connections to same host in 2 sec",       1),
        ("serror_rate",       "SYN error rate (0.0 – 1.0)",              0.0),
        ("rerror_rate",       "REJ error rate (0.0 – 1.0)",              0.0),
        ("same_srv_rate",     "Same service rate (0.0 – 1.0)",           1.0),
        ("dst_host_count",    "Destination host connection count",       1),
        ("dst_host_serror_rate", "Dst host SYN error rate (0.0–1.0)",   0.0),
    ]

    # Build a default record (neutral/normal baseline)
    defaults = {col: 0.0 for col in feature_cols}
    defaults.update({
        "protocol_type": 2,   # tcp
        "service":       10,  # http
        "flag":          9,   # SF (normal)
        "same_srv_rate": 1.0,
        "dst_host_same_srv_rate": 1.0,
    })

    user_values = dict(defaults)

    for col, description, default in prompts:
        if col not in feature_cols:
            continue
        try:
            raw = input(f"  {description} [{default}]: ").strip()
            user_values[col] = float(raw) if raw else float(default)
        except ValueError:
            print(f"  [!] Invalid input — using default ({default})")
            user_values[col] = float(default)

    # Build ordered feature vector
    values = [user_values[col] for col in feature_cols]
    pred_int, pred_str, conf = predict(values, model, scaler, feature_cols)
    _print_result(pred_int, pred_str, conf, "User-entered traffic record")


def run_prediction_cli():
    """Main entry point for prediction mode."""
    _print_header()

    print("\n  Loading saved model ...")
    model, scaler, feature_cols = load_model()

    while True:
        print("\n  Choose an option:")
        print("    [1] Run demo predictions (predefined samples)")
        print("    [2] Enter custom network values manually")
        print("    [3] Exit")
        choice = input("\n  Your choice: ").strip()

        if choice == "1":
            run_demo_predictions(model, scaler, feature_cols)
        elif choice == "2":
            run_interactive_cli(model, scaler, feature_cols)
        elif choice == "3":
            print("\n  Goodbye!\n")
            break
        else:
            print("  [!] Invalid choice. Please enter 1, 2, or 3.")


# ─────────────────────────────────────────────
#  STANDALONE ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_prediction_cli()
