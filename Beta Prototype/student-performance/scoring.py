import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings

ORIGINAL_PATH = "data/processed_data.csv"
SYNTHETIC_PATH = "output/synthetic_output.csv"

warnings.filterwarnings("ignore") # Suppress minor sklearn warnings

# Column information
_COLUMN_INFO = {
    'initialized': False,
    'numerical_cols': [],
    'binary_cols': [],
    'categorical_cols': []
}

def setup_column_info(df):
    """Identify and store column types from the original dataset."""
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

# Measure correlations in the data
def correlation_score(df_orig, df_synth, num_cols):
    """
    Measures how well inter-feature correlations are preserved.
    Uses Pearson correlation on numeric columns.
    """
    if len(num_cols) < 2:
        return 1.0  # trivial case

    corr_orig = df_orig[num_cols].corr().fillna(0)
    corr_synth = df_synth[num_cols].corr().fillna(0)
    diff = np.abs(corr_orig - corr_synth).values
    return 1 - np.mean(diff)  # smaller difference → higher score

# Distinguishibility testing, ie, can a model accurately predict if the data is synthetic
def indistinguishability_score(X_orig, X_synth):
    """
    Train a classifier to distinguish real from synthetic samples.
    Perfect indistinguishability → AUC = 0.5 → score = 1.0
    """
    X_comb = np.vstack([X_orig, X_synth])
    y = np.array([0]*len(X_orig) + [1]*len(X_synth))

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_comb, y)
    y_pred = clf.predict_proba(X_comb)[:, 1]
    auc = roc_auc_score(y, y_pred)
    return 1 - abs(auc - 0.5) * 2


# Get numeric grading for synthetic data
def get_optimization_score():
    #Compute a single 0-100 score comparing synthetic to original
    df_orig = pd.read_csv(ORIGINAL_PATH)
    df_synth = pd.read_csv(SYNTHETIC_PATH)

    setup_column_info(df_orig)
    num_cols, bin_cols, cat_cols = get_column_info()

    # --- (1) Distribution fidelity
    dist_score = column_distribution_score(df_orig, df_synth, num_cols, cat_cols)

    # --- (2) Correlation fidelity
    corr_score = correlation_score(df_orig, df_synth, num_cols)

    # --- (3) Indistinguishability
    usable_cols = num_cols + bin_cols
    scaler = StandardScaler()
    X_orig = scaler.fit_transform(df_orig[usable_cols].dropna())
    X_synth = scaler.transform(df_synth[usable_cols].dropna())
    indist_score = indistinguishability_score(X_orig, X_synth)

    # Weighted final score
    final_score = 100 * (0.4 * dist_score + 0.3 * corr_score + 0.3 * indist_score)
    print(f"Scores → Dist: {dist_score:.3f}, Corr: {corr_score:.3f}, Indist: {indist_score:.3f}, Final: {final_score:.2f}")
    return final_score

if __name__ == "__main__":
    get_optimization_score()
