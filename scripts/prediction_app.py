import sys
import argparse
from pathlib import Path
import glob
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

MODEL_ROOT = Path(r"F:/steam_market_project/data/model_training_out")
DEFAULT_VARIANT = "raw_fe_liquipedia"
DATA_ROOT = Path(r"F:/steam_market_project/data/model_subset_csv_raw_fe_liquipedia")

TARGET_COL = "y_avg_7d_fwd"
DATE_COL = "date"
ITEM_COL = "item_name"

# ──────────────────────────────────────────────

def find_latest_row_for_item(item_name: str, data_dirs=None):
    if data_dirs is None:
        data_dirs = [DATA_ROOT]

    candidates = []

    for dr in data_dirs:
        pattern = dr / "subset_*.csv"
        for fpath in sorted(glob.glob(str(pattern)), reverse=True):
            try:
                df = pd.read_csv(fpath, usecols=[DATE_COL, ITEM_COL], nrows=10)
                if ITEM_COL not in df.columns:
                    continue
                if item_name not in df[ITEM_COL].values:
                    continue

                df = pd.read_csv(fpath, low_memory=False)
                df_item = df[df[ITEM_COL].astype(str).str.strip() == item_name.strip()]
                if df_item.empty:
                    continue

                df_item = df_item.copy()
                df_item[DATE_COL] = pd.to_datetime(df_item[DATE_COL], errors='coerce')
                latest = df_item.loc[df_item[DATE_COL].idxmax()]
                candidates.append(latest)

            except Exception as e:
                print(f"Skip {fpath.name}  ──  {type(e).__name__}", file=sys.stderr)

    if not candidates:
        return None

    df_cand = pd.DataFrame(candidates)
    df_cand[DATE_COL] = pd.to_datetime(df_cand[DATE_COL])
    latest_row = df_cand.loc[df_cand[DATE_COL].idxmax()]
    return latest_row.to_frame().T


def prepare_row_for_model(row: pd.DataFrame, meta, add_na_flags=True):
    row = row.copy()

    feats = meta.get('feat_cols')
    imp   = meta.get('imputer')
    print("Imputer n_features_in_:", imp.n_features_in_)
    print("Feat_cols:", len(feats))
    if feats is None:
        raise KeyError("No 'feat_cols' in model file")
    if imp is None:
        raise KeyError("No 'imputer' in model file")

    n_expected = len(feats)
    print(f"Model expects {n_expected} features")
    print(f"First 6: {feats[:6]} ...")
    print(f"Last 6:  {feats[-6:]} ...")

    # Create row with EXACTLY the expected columns
    row_aligned = pd.DataFrame(index=[0], columns=feats, dtype=float)

    # Copy values from the original row
    for col in feats:
        if col in row.columns:
            try:
                val = row[col].iloc[0]
                row_aligned.at[0, col] = pd.to_numeric(val, errors='coerce')
            except:
                row_aligned.at[0, col] = np.nan
        else:
            row_aligned.at[0, col] = np.nan

    # Add isna_ columns only for features the model actually expects
    if add_na_flags:
        for c in feats:
            if not c.startswith("isna_"):
                na_col = f"isna_{c}"
                if na_col in feats:
                    row_aligned[na_col] = pd.isna(row_aligned[c]).astype('int8')

    print(f"Final row shape before numpy: {row_aligned.shape}")

    # Safety check
    if row_aligned.shape[1] != n_expected:
        missing = [c for c in feats if c not in row_aligned.columns]
        print("MISSING COLUMNS:", missing)
        raise ValueError(f"Shape mismatch after alignment: {row_aligned.shape[1]} vs {n_expected}")

    X = row_aligned[feats].values.astype(np.float32)
    print(f"X shape: {X.shape}")

    X_imp = imp.transform(X)
    print("Imputation successful")

    return X_imp


def main():
    variant = DEFAULT_VARIANT
    add_na_flags = True

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Predict next 7-day avg price for CS2 skin")
        parser.add_argument("item", nargs="+", help="Item name")
        parser.add_argument("--variant", "-v", default=DEFAULT_VARIANT)
        parser.add_argument("--no-na-flags", action="store_false", dest="na_flags")
        args = parser.parse_args()
        item_name = " ".join(args.item).strip()
        variant = args.variant
        add_na_flags = args.na_flags
    else:
        item_name = input("Enter item name (example: AK-47 | Redline (Field-Tested)): ").strip()
        if not item_name:
            print("No item name entered. Exiting.")
            return 0

    print(f"\nItem: {item_name}")
    print(f"Variant: {variant}")
    print("Searching for latest data...")

    model_dir = MODEL_ROOT / variant
    if not model_dir.exists():
        print(f"Model folder not found: {model_dir}", file=sys.stderr)
        return 1

    model_path = model_dir / "model_xgboost_raw.joblib"
    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 1

    print(f"→ Loading model: {model_path.name}")

    try:
        bundle = joblib.load(model_path)
        meta = bundle
        print("Loaded keys:", list(meta.keys()))
    except Exception as e:
        print(f"Cannot load model: {e}", file=sys.stderr)
        return 1

    row = find_latest_row_for_item(item_name)
    if row is None:
        print(f"Item '{item_name}' not found in any subset_*.csv", file=sys.stderr)
        return 1

    print(f"Latest date found: {row[DATE_COL].iloc[0]}")

    try:
        X_ready = prepare_row_for_model(row, meta, add_na_flags=add_na_flags)
    except Exception as e:
        print(f"Error preparing features: {e}", file=sys.stderr)
        return 1

    model = meta.get('model')
    if model is None:
        print("No 'model' key found in loaded file", file=sys.stderr)
        return 1

    try:
        pred = model.predict(X_ready)[0]
        print(f"\nPredicted 7-day forward average price:")
        print(f"  {pred:.2f} €")
    except Exception as e:
        print(f"Prediction failed: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)