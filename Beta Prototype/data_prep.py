import pandas as pd

def preprocess_csv(
    input_path,
    output_path=None,
    columns_to_transform=None,
    drop_original=True, # should it drop the original column, should be true but customizable
    sep=',' # delimiter for csv
):
    df = pd.read_csv(input_path, sep=sep)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    if columns_to_transform is None:
        columns_to_transform = df.select_dtypes(include=['object']).columns.tolist()

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
            print(f"encoded '{col}' into {len(dummies.columns)} columns")

        else:
            print(f'{col} already unique, you should drop it from your dataset for generation')

    if output_path:
        df.to_csv(output_path, index=False)

    return df

if __name__ == "__main__":
    df = preprocess_csv(
        input_path="student-performance/data/student-merged.csv",
        output_path="student-performance/data/processed_data.csv",
        sep=';',
        columns_to_transform=None
    )
    print(df.head())
