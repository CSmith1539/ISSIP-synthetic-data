import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, entropy

from collections import Counter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings("ignore") # Suppress minor sklearn warnings

# Column information
_COLUMN_INFO = {
    'initialized': False,
    'numerical_cols': [],
    'binary_cols': [],
    'categorical_cols': []
}

# Parses and identifies numeric, binary, and categorical columns
def setup_column_info(df, force=False):
    global _COLUMN_INFO
    if _COLUMN_INFO['initialized'] and force == False:
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    binary_cols = [c for c in num_cols if df[c].nunique() == 2]
    numerical_cols = [c for c in num_cols if c not in binary_cols]
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    _COLUMN_INFO.update({
        'initialized': True,
        'numerical_cols': numerical_cols,
        'binary_cols': binary_cols,
        'categorical_cols': categorical_cols
    })

# Runs train_test_split, but is safe in the event there is 1 unique value
def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    cls_counts = Counter(y)
    if min(cls_counts.values()) < 2:
        # Cannot stratify
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
    else:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

# Preprocesses the original and synthetic data for later tests
def load_and_preprocess_data(df_orig, df_synth, target_cols):
    global _COLUMN_INFO

    if isinstance(target_cols, str):
        target_cols = [target_cols]
    else:
        target_cols = list(target_cols)

    ALL_FEATURES = [col for col in df_orig.columns if col not in target_cols]
    NUMERICAL_FEATURES = [col for col in _COLUMN_INFO['numerical_cols'] if col not in target_cols]
    CATEGORICAL_FEATURES = [
        col for col in _COLUMN_INFO['binary_cols'] + _COLUMN_INFO['categorical_cols']
        if col not in target_cols
    ]

    # Extract X and y from ORIGINAL dataset
    X = df_orig[ALL_FEATURES]
    y = df_orig[target_cols]

    # Split original data only
    X_orig, X_test, y_orig, y_test = safe_train_test_split(X, y)

    X_synth = df_synth[ALL_FEATURES]
    y_synth = df_synth[target_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )

    preprocessor.fit(X_orig)

    X_orig_proc = preprocessor.transform(X_orig)
    X_synth_proc = preprocessor.transform(X_synth)
    X_test_proc = preprocessor.transform(X_test)

    return df_orig, df_synth, X_orig_proc, y_orig, X_synth_proc, y_synth, X_test_proc, y_test

def run_ml_utility_test(model_class, X_orig_train, y_orig_train,
                        X_synth_train, y_synth_train, 
                        X_test, y_test, name, **kwargs):

    model_orig = model_class(**kwargs)
    model_orig.fit(X_orig_train, y_orig_train)
    y_pred_orig = model_orig.predict(X_test)
    rmse_orig = np.sqrt(mean_squared_error(y_test, y_pred_orig))

    model_synth = model_class(**kwargs)
    model_synth.fit(X_synth_train, y_synth_train)
    y_pred_synth = model_synth.predict(X_test)
    rmse_synth = np.sqrt(mean_squared_error(y_test, y_pred_synth))

    rmse_gap = abs(rmse_orig - rmse_synth)

    return {
        'Model': name,
        'RMSE_Orig': rmse_orig,
        'RMSE_Synth': rmse_synth,
        'RMSE_Gap': rmse_gap
    }


def get_column_info():
    if not _COLUMN_INFO['initialized']:
        raise RuntimeError("Column info not initialized. Call setup_column_info(df_orig) first.")
    return _COLUMN_INFO['numerical_cols'], _COLUMN_INFO['binary_cols'], _COLUMN_INFO['categorical_cols']

def test_privacy(X_orig_proc, X_synth_proc):
    #Calculates the minimum Nearest Neighbor distance.
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean').fit(X_orig_proc)
    distances, _ = nn.kneighbors(X_synth_proc)
    min_distance = distances.min()

    #print(f"Minimum Nearest Neighbor Distance: {min_distance:.6f}")

    if min_distance > 1e-6:
        return True
    else:
        return False

# Fidelity of the distribution of the synthetic data
def column_distribution_score(df_orig, df_synth, num_cols, cat_cols):
    # Compare marginal distributions of numeric and categorical columns
    scores = []
    for col in num_cols:
        stat, _ = ks_2samp(df_orig[col], df_synth[col])
        scores.append(1 - stat)  # Smaller KS → better
    for col in cat_cols:
        p = df_orig[col].value_counts(normalize=True)
        q = df_synth[col].value_counts(normalize=True).reindex(p.index, fill_value=0)
        jsd = 0.5 * (entropy(p, (p + q)/2) + entropy(q, (p + q)/2))
        scores.append(1 - jsd)
    return np.mean(scores) if scores else 0.0

# Measure preservation of correlations in the data
def correlation_score(df_orig, df_synth, num_cols):
    if len(num_cols) < 2:
        return 1.0  # trivial case

    corr_orig = df_orig[num_cols].corr().fillna(0)
    corr_synth = df_synth[num_cols].corr().fillna(0)
    diff = np.abs(corr_orig - corr_synth).values
    return 1 - np.mean(diff)  # smaller difference → higher score

# Run the utility test on selection of models
def test_utility_all_models(X_orig_proc, y_orig,
                            X_synth_proc, y_synth,
                            X_test_proc, y_test):

    models = [
        (LinearRegression, "Linear Regression", {}),
        (RandomForestRegressor, "Random Forest", {'random_state': 42})
       # (KNeighborsRegressor, "KNN Regressor", {})
    ]

    results = [
        run_ml_utility_test(mc, X_orig_proc, y_orig,
                            X_synth_proc, y_synth,
                            X_test_proc, y_test,
                            name, **kwargs)
        for mc, name, kwargs in models
    ]
    
    df_results = pd.DataFrame(results)

    print(df_results)
    mean = df_results['RMSE_Gap'].mean()

    # Lower is better
    return 1 - mean / 100

# Get a simple 0-100 score on the synthetic data compared to the original
def get_optimization_score(df, sf, target_columns, force=False):
    df_orig = df
    df_synth = sf

    setup_column_info(df_orig, force)
    num_cols, bin_cols, cat_cols = get_column_info()

    # distribution fidelity
    dist_score = column_distribution_score(df_orig, df_synth, num_cols, cat_cols)

    # correlation fidelity
    corr_score = correlation_score(df_orig, df_synth, num_cols)

    # usability
    df_orig_raw, df_synth_raw, X_orig_proc, y_orig, X_synth_proc, y_synth, X_test_proc, y_test = load_and_preprocess_data(df_orig, df_synth, target_columns)
    utility_score = test_utility_all_models(X_orig_proc, y_orig, X_synth_proc, y_synth, X_test_proc, y_test)

    # final score
    final_score = 100 * (0.3 * dist_score + 0.3 * corr_score + 0.4 * utility_score)
    print(f"Scores → Dist: {dist_score:.3f}, Corr: {corr_score:.3f}, Utility: {utility_score:.3f}, Final: {final_score:.2f}")
    return final_score
