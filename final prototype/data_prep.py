import pandas as pd
import numpy as np

def preprocess_csv(
    input_path,
    columns_to_transform=None,
    drop_original=True, # should it drop the original column, should be true but customizable
    sep=',' # delimiter for csv
):
    df = pd.read_csv(input_path, sep=sep)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    if columns_to_transform is None:
        columns_to_transform = df.select_dtypes(include=['object']).columns.tolist()

    one_hot = []
    drop_columns = []

    for col in columns_to_transform:
        if col not in df.columns:
            print(f"column '{col}' not found, skipping")
            continue

        unique_vals = df[col].dropna().unique()
        n_unique = len(unique_vals)

        if n_unique == 2:
            # Binary column — map to 0/1
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            df[col] = df[col].map(mapping)
            print(f"encoded binary column '{col}' to 0/1")

        elif n_unique > 2:
            # Multi-category — one-hot encode
            dummies = pd.get_dummies(df[col], prefix=col).astype(int)
            df = pd.concat([df, dummies], axis=1)
            if drop_original:
                df = df.drop(columns=[col])
            print(f"encoded '{col}' into {list(dummies.columns)}")
            one_hot = one_hot + [list(dummies.columns)]

        else:
            print(f'{col} has one unique value, produced dataset will not include this column')
            drop_columns = drop_columns + [col]

    # Drop unwanted columns
    df = df.drop(columns=[c for c in drop_columns if c in df.columns])
    # Drop non-numeric columns that remain
    df = df.select_dtypes(include=[np.number])
    # Fill missing values (mean imputation)
    df = df.fillna(df.mean())

    return (df, one_hot)

if __name__ == "__main__":
    df = preprocess_csv(
        input_path="data/student-merged.csv",
        sep=';',
        columns_to_transform=None
    )
    print(df.head())
