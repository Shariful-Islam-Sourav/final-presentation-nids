"""
main.py — Network Intrusion Detection System
=============================================
Entry point for the NIDS project.

Usage
-----
    python main.py --train        # Train models + generate all visualizations
    python main.py --predict      # Run interactive prediction CLI
    python main.py --all          # Train first, then run prediction CLI
    python main.py                # Show help menu
"""

import sys
import argparse

from train   import run_training
from predict import run_prediction_cli


# ─────────────────────────────────────────────
#  BANNER
# ─────────────────────────────────────────────

BANNER = r"""
  ███╗   ██╗██╗██████╗ ███████╗
  ████╗  ██║██║██╔══██╗██╔════╝
  ██╔██╗ ██║██║██║  ██║███████╗
  ██║╚██╗██║██║██║  ██║╚════██║
  ██║ ╚████║██║██████╔╝███████║
  ╚═╝  ╚═══╝╚═╝╚═════╝ ╚══════╝
  Network Intrusion Detection System
  Using Machine Learning  |  NSL-KDD Dataset
  ─────────────────────────────────────────
"""


def print_banner():
    print(BANNER)


def print_help():
    """Display usage instructions."""
    print_banner()
    print("  USAGE:")
    print("    python main.py --train       Train models & generate visualizations")
    print("    python main.py --predict     Run interactive prediction CLI")
    print("    python main.py --all         Train, then run prediction CLI")
    print()
    print("  EXAMPLES:")
    print("    python main.py --train")
    print("    python main.py --predict")
    print()
    print("  OUTPUT:")
    print("    outputs/   → All generated charts (PNG)")
    print("    models/    → Saved best model + scaler")
    print()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Network Intrusion Detection System using Machine Learning",
        add_help=False,
    )
    parser.add_argument("--train",   action="store_true", help="Train models")
    parser.add_argument("--predict", action="store_true", help="Run prediction CLI")
    parser.add_argument("--all",     action="store_true", help="Train then predict")
    parser.add_argument("--help",    action="store_true", help="Show help")

    args = parser.parse_args()

    print_banner()

    if args.help or len(sys.argv) == 1:
        print_help()
        return

    if args.train or args.all:
        run_training()

    if args.predict or args.all:
        run_prediction_cli()


if __name__ == "__main__":
    main()
