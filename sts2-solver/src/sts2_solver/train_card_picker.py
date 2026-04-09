"""Train the XGBoost card picker model from collected data.

Usage:
    python -m sts2_solver.train_card_picker --data card_pick_data.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .card_picker_xgb import (
    CardPickerXGB,
    FEATURE_NAMES,
    MODEL_DIR,
    records_to_training_data,
)
from .data_loader import load_cards


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost card picker")
    parser.add_argument("--data", type=str, default="card_pick_data.json",
                        help="Path to card pick records JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Output model path (default: card_picker_model/card_picker.json)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: data file not found: {data_path}")
        print("Run collect_card_picks first:")
        print("  python -m sts2_solver.collect_card_picks --games 500")
        sys.exit(1)

    output_path = Path(args.output) if args.output else MODEL_DIR / "card_picker.json"

    print(f"Loading card database...")
    card_db = load_cards()

    print(f"Converting records from {data_path}...")
    X, y = records_to_training_data(data_path, card_db)
    print(f"  Training samples: {len(X)}")
    print(f"  Features per sample: {len(FEATURE_NAMES)}")
    print(f"  Positive labels (picked+won): {(y > 0.5).sum()}")
    print(f"  Label stats: mean={y.mean():.3f}, std={y.std():.3f}, "
          f"min={y.min():.3f}, max={y.max():.3f}")

    if len(X) == 0:
        print("Error: no training samples generated. Check that card names in the")
        print("collected data match the card database.")
        sys.exit(1)

    if len(X) < 100:
        print("Warning: very few samples. Consider collecting more games.")

    print(f"\nTraining XGBoost model...")
    picker = CardPickerXGB()
    picker.train(X, y, save_path=output_path)

    # Print feature importances
    if picker.model is not None:
        importances = picker.model.feature_importances_
        ranked = sorted(zip(FEATURE_NAMES, importances),
                        key=lambda x: x[1], reverse=True)
        print(f"\nTop 15 feature importances:")
        for name, imp in ranked[:15]:
            bar = "#" * int(imp * 100)
            print(f"  {name:30s} {imp:.4f} {bar}")

    print(f"\nModel saved to {output_path}")
    print(f"The model will be auto-loaded by the simulator on next run.")


if __name__ == "__main__":
    main()
