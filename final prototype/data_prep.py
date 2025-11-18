import pandas as pd
import numpy as np

# Keep track of mappings for undoing transformations
_one_hot_map = {}
_binary_map = {}
_orig_order = None

def preprocess_csv(
    input_path,
    columns_to_transform=None,
    drop_original=True,
    sep=','
):
    global _one_hot_map, _binary_map, _orig_order

    df = pd.read_csv(input_path, sep=sep)
    _orig_order = df.columns.tolist()

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    if columns_to_transform is None:
        columns_to_transform = df.select_dtypes(include=['object']).columns.tolist()

    one_hot = []
    drop_columns = []
    _one_hot_map.clear()
    _binary_map.clear()

    for col in columns_to_transform:
        if col not in df.columns:
            print(f"column '{col}' not found, skipping")
            continue

        unique_vals = df[col].dropna().unique()
        n_unique = len(unique_vals)

        if n_unique == 2:
            # Binary column — map to 0/1
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            reverse = {0: unique_vals[0], 1: unique_vals[1]}
            _binary_map[col] = reverse
            df[col] = df[col].map(mapping)
            #print(f"encoded binary column '{col}' to 0/1")

        elif n_unique > 2:
            # Multi-category — one-hot encode
            dummies = pd.get_dummies(df[col], prefix=col).astype(int)
            df = pd.concat([df, dummies], axis=1)
            if drop_original:
                df = df.drop(columns=[col])
            #print(f"encoded '{col}' into {list(dummies.columns)}")
            one_hot.append(list(dummies.columns))
            _one_hot_map[col] = list(dummies.columns)

        else:
            #print(f'{col} has one unique value, produced dataset will not include this column')
            drop_columns.append(col)

    # Drop unwanted columns
    df = df.drop(columns=[c for c in drop_columns if c in df.columns])
    # Drop non-numeric columns that remain
    df = df.select_dtypes(include=[np.number])
    # Fill missing values (mean imputation)
    df = df.fillna(df.mean())

    return df, one_hot


def undo_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Reverse preprocessing using the stored _one_hot_map and _binary_map
    global _one_hot_map, _binary_map, _orig_order

    df = df.copy()

    # Reconstruct one-hot encoded categorical columns
    for col, dummies in _one_hot_map.items():
        present = [c for c in dummies if c in df.columns]
        if not present:
            print(f"Warning: none of {dummies} found in df for '{col}'")
            continue

        idx = df[present].to_numpy().argmax(axis=1)
        labels = [name[len(col)+1:] for name in present]  # remove prefix_
        df[col] = [labels[i] for i in idx]
        df = df.drop(columns=present)

    # Reconstruct binary categorical columns
    for col, reverse in _binary_map.items():
        if col in df.columns:
            df[col] = df[col].map(reverse)

    # Reorder to match original CSV order
    if _orig_order is not None:
        present = [c for c in _orig_order if c in df.columns]
        extras = [c for c in df.columns if c not in present]
        df = df[present + extras]

    return df


if __name__ == "__main__":
    df_proc, one_hot = preprocess_csv(
        input_path="data/student-merged.csv",
        sep=';',
        columns_to_transform=None
    )

    print("\n--- Encoded Preview ---")
    print(df_proc.head())

    df_restored = undo_preprocess(df_proc)
    print("\n--- Decoded Preview ---")
    print(df_restored.head())
