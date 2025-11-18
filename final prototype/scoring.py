import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore") # Suppress minor sklearn warnings

# Column information
_COLUMN_INFO = {
    'initialized': False,
    'numerical_cols': [],
    'binary_cols': [],
    'categorical_cols': []
}

def setup_column_info(df):
    #Identify and store column types from the original dataset
    global _COLUMN_INFO
    if _COLUMN_INFO['initialized']:
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

def get_column_info():
    if not _COLUMN_INFO['initialized']:
        raise RuntimeError("Column info not initialized. Call setup_column_info(df_orig) first.")
    return _COLUMN_INFO['numerical_cols'], _COLUMN_INFO['binary_cols'], _COLUMN_INFO['categorical_cols']

def test_privacy(X_orig_proc, X_synth_proc):
    #Calculates the minimum Nearest Neighbor distance.
    print("\n\n--- Running Test 2: Privacy (Req. #2) ---")
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

# Distinguishibility testing, ie, can a model accurately predict if the data is synthetic
# Closer the model's guesses get to 50/50 guesses, the higher the score
def indistinguishability_score(X_orig, X_synth):
    #print(np.mean(X_orig, axis=0))
    #print(np.mean(X_synth, axis=0))
    #print(np.std(X_orig, axis=0))
    #print(np.std(X_synth, axis=0))

    n = min(len(X_orig), len(X_synth))
    X_orig, X_synth = X_orig[:n], X_synth[:n]

    X_comb = np.vstack([X_orig, X_synth])
    y = np.array([0]*n + [1]*n)

    X_train, X_test, y_train, y_test = train_test_split(
        X_comb, y, test_size=0.1, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=40, max_depth=5, min_samples_leaf=15, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred)
    return 1 - abs(auc - 0.5) * 2

# Get a simple 0-100 score on the synthetic data compared to the original
def get_optimization_score(df, sf):
    df_orig = df
    df_synth = sf

    setup_column_info(df_orig)
    num_cols, bin_cols, cat_cols = get_column_info()

    # distribution fidelity
    dist_score = column_distribution_score(df_orig, df_synth, num_cols, cat_cols)

    # correlation fidelity
    corr_score = correlation_score(df_orig, df_synth, num_cols)

    # indistinguishability
    indist_score = indistinguishability_score(df_orig, df_synth)

    # final score
    final_score = 100 * (0.3 * dist_score + 0.3 * corr_score + 0.4 * indist_score)
    print(f"Scores → Dist: {dist_score:.3f}, Corr: {corr_score:.3f}, Indist: {indist_score:.3f}, Final: {final_score:.2f}")
    return final_score
