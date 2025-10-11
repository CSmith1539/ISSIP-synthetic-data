import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore") # Suppress minor sklearn warnings

# Define file paths
ORIGINAL_PATH = '../data/processed_data.csv'
SYNTHETIC_PATH = '../output/synthetic_output.csv'

# --------------------------------------------------------------------------------------
# PART A: Data Loading and Preprocessing
# --------------------------------------------------------------------------------------

def load_and_preprocess_data():
    """
    Loads both datasets, defines features/target, and prepares all necessary
    processed components for the tests.
    """
    print("Loading and preprocessing data...")
    df_orig = pd.read_csv(ORIGINAL_PATH, sep=',')
    df_synth = pd.read_csv(SYNTHETIC_PATH, sep=',')

    # --- 1. Define Features and Target (Target: G3_Pass) ---
    TARGET_COLUMN = 'G3_Pass'

    df_orig[TARGET_COLUMN] = (df_orig['G3'] >= 10).astype(int)
    df_synth[TARGET_COLUMN] = (df_synth['G3'] >= 10).astype(int)

    df_orig_features = df_orig.drop(columns=['G1', 'G2', 'G3'], errors='ignore')
    df_synth_features = df_synth.drop(columns=['G1', 'G2', 'G3'], errors='ignore')

    NUMERICAL_FEATURES = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
    ALL_FEATURES = [col for col in df_orig_features.columns if col != TARGET_COLUMN]
    CATEGORICAL_FEATURES = [f for f in ALL_FEATURES if f not in NUMERICAL_FEATURES]
    
    # --- 2. Split Original Data (D_train, D_test) ---
    X_orig, X_test, y_orig, y_test = train_test_split(
        df_orig_features[ALL_FEATURES],
        df_orig[TARGET_COLUMN],
        test_size=0.2,
        random_state=42,
        stratify=df_orig[TARGET_COLUMN]
    )

    X_synth = df_synth_features[ALL_FEATURES].head(len(X_orig))
    y_synth = df_synth_features[TARGET_COLUMN].head(len(X_orig))

    # --- 3. Create and Fit Preprocessor Pipeline ---
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

    print("✅ Data setup complete.")
    return df_orig, df_synth, X_orig_proc, y_orig, X_synth_proc, y_synth, X_test_proc, y_test

# Load all data components
df_orig_raw, df_synth_raw, X_orig_proc, y_orig, X_synth_proc, y_synth, X_test_proc, y_test = load_and_preprocess_data()


# --------------------------------------------------------------------------------------
# PART B: Validation Tests
# --------------------------------------------------------------------------------------

# --- Test 1: Statistical Fidelity (Requirement #1) ---
def test_fidelity(df_orig, df_synth):
    """Calculates and compares multiple statistical measures for the G3 column."""
    print("\n\n--- Running Test 1: Statistical Fidelity (Req. #1) ---")
    
    stats_orig = df_orig['G3'].describe()
    stats_synth = df_synth['G3'].describe()

    comparison_df = pd.DataFrame({'Original': stats_orig, 'Synthetic': stats_synth})
    comparison_df['Deviation (%)'] = abs(comparison_df['Synthetic'] - comparison_df['Original']) / comparison_df['Original'] * 100
    
    print("Comparison of G3 Statistical Properties:")
    print(comparison_df.round(2))

    mean_deviation = comparison_df.loc['mean', 'Deviation (%)']
    if mean_deviation <= 5.0:
        print("\n✅ PASS: Mean deviation is within the 5.0% threshold.")
    else:
        print(f"\n❌ FAIL: Mean deviation is {mean_deviation:.2f}%, which exceeds the 5.0% threshold.")

# --- Test 2: Privacy (Requirement #2) ---
def test_privacy(X_orig_proc, X_synth_proc):
    """Calculates the minimum Nearest Neighbor distance."""
    print("\n\n--- Running Test 2: Privacy (Req. #2) ---")
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean').fit(X_orig_proc)
    distances, _ = nn.kneighbors(X_synth_proc)
    min_distance = distances.min()

    print(f"Minimum Nearest Neighbor Distance: {min_distance:.6f}")

    if min_distance > 1e-6:
        print("✅ PASS: No exact record match found.")
    else:
        print("❌ FAIL: Exact record correspondence detected.")

# --- Test 3: ML Utility (Requirement #3) ---
def run_ml_utility_test(model_class, X_orig_train, y_orig_train, X_synth_train, y_synth_train, X_test, y_test, name, **kwargs):
    """Trains a model on both sets and calculates the Accuracy Gap."""
    model_orig = model_class(**kwargs)
    model_orig.fit(X_orig_train, y_orig_train)
    y_pred_orig = model_orig.predict(X_test)
    acc_orig = accuracy_score(y_test, y_pred_orig)

    model_synth = model_class(**kwargs)
    model_synth.fit(X_synth_train, y_synth_train)
    y_pred_synth = model_synth.predict(X_test)
    acc_synth = accuracy_score(y_test, y_pred_synth)
    
    acc_gap = abs(acc_orig - acc_synth)
    is_pass = "✅ PASS" if acc_gap <= 0.02 else "❌ FAIL"

    return {'Model': name, 'Acc_Orig': acc_orig, 'Acc_Synth': acc_synth, 'Acc_Gap': acc_gap, 'Status': is_pass}

def test_utility_all_models(X_orig_proc, y_orig, X_synth_proc, y_synth, X_test_proc, y_test):
    """Runs the utility test for all three models and compiles results."""
    print("\n\n--- Running Test 3: ML Utility (Req. #3) ---")
    
    models = [
        (LogisticRegression, "Logistic Regression", {'random_state': 42}),
        (RandomForestClassifier, "Random Forest", {'random_state': 42}),
        (KNeighborsClassifier, "K-Nearest Neighbors", {})
    ]
    
    results = [run_ml_utility_test(mc, X_orig_proc, y_orig, X_synth_proc, y_synth, X_test_proc, y_test, name, **kwargs) for mc, name, kwargs in models]
    df_results = pd.DataFrame(results)

    print("Criterion: Accuracy Gap (ΔAccuracy) must be <= 0.02")
    print(df_results.to_markdown(index=False, floatfmt=".4f"))
    
    if all(df_results['Acc_Gap'] <= 0.02):
        print("\n✅ OVERALL PASS: All three models meet the utility criterion.")
    else:
        print("\n❌ OVERALL FAIL: At least one model exceeded the 0.02 Accuracy Gap.")
    
    return df_results

# --------------------------------------------------------------------------------------
# PART C: Visualization
# --------------------------------------------------------------------------------------
def generate_visualizations(df_orig, df_synth, ml_results_df):
    """Generates and saves plots for the test results."""
    print("\n\n--- Generating Result Visualizations ---")

    # 1. Distribution Plot for G3
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_orig['G3'], label='Original Data', color='blue', shade=True)
    sns.kdeplot(df_synth['G3'], label='Synthetic Data', color='red', shade=True)
    plt.title('Distribution of Final Grades (G3) - Original vs. Synthetic')
    plt.xlabel('Final Grade (G3)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('g3_distribution_comparison.png')
    print("✅ Saved G3 distribution plot to g3_distribution_comparison.png")

    # 2. Bar Chart for ML Accuracy
    df_melted = ml_results_df.melt(id_vars='Model', value_vars=['Acc_Orig', 'Acc_Synth'], var_name='Dataset', value_name='Accuracy')
    df_melted['Dataset'] = df_melted['Dataset'].replace({'Acc_Orig': 'Original', 'Acc_Synth': 'Synthetic'})
    
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x='Model', y='Accuracy', hue='Dataset', data=df_melted, palette=['#4c72b0', '#dd8452'])
    plt.title('Machine Learning Model Efficacy Comparison')
    plt.ylabel('Accuracy Score')
    plt.xlabel('')
    plt.ylim(0, 1.0)
    plt.legend(title='Trained On')
    
    # Add accuracy labels on top of bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2%}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.savefig('ml_accuracy_comparison.png')
    print("✅ Saved ML accuracy bar chart to ml_accuracy_comparison.png")

# --------------------------------------------------------------------------------------
# FINAL EXECUTION
# --------------------------------------------------------------------------------------
test_fidelity(df_orig_raw, df_synth_raw)
test_privacy(X_orig_proc, X_synth_proc)
ml_results = test_utility_all_models(X_orig_proc, y_orig, X_synth_proc, y_synth, X_test_proc, y_test)
generate_visualizations(df_orig_raw, df_synth_raw, ml_results)

print("\n\nScript finished.")