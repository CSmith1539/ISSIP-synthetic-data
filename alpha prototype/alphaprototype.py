import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np

# A dictionary to hold the file paths for easy access
file_paths = {
    "Original": "alzheimers_disease_data.csv",
    "Gemini": "gemini-synthetic-dataset.csv",
    "Claude": "claude_dataset.csv",
    "ChatGPT": "chat_gpt_dataset.csv",
    "Copilot": "copilot_dataset.csv",
    "DuckDuckGo": "duckduck-synthetic-dataset.csv"
}

# An empty dictionary to store the final accuracy results
results = {}
target = 'Diagnosis'

# Loop through each file defined in the dictionary
for name, path in file_paths.items():
    try:
        df = pd.read_csv(path)
        # Clean potential leading/trailing spaces from column headers
        df.columns = df.columns.str.strip()

        # --- CONDITIONAL FEATURE SELECTION ---
        if name == "Original":
            # Use columns that exist in the original dataset
            features = ['Age', 'Gender', 'Ethnicity', 'EducationLevel', 'MMSE', 'SystolicBP', 'DiastolicBP']
            categorical_features = ['Gender', 'Ethnicity']
            numerical_features = ['Age', 'EducationLevel', 'MMSE', 'SystolicBP', 'DiastolicBP']
        else:
            # Use columns that exist in the synthetic datasets
            features = ['Age', 'Gender', 'Ethnicity', 'EducationLevel', 'MMSE', 'APOE4', 'CDR']
            categorical_features = ['Gender', 'Ethnicity']
            numerical_features = ['Age', 'EducationLevel', 'MMSE', 'APOE4', 'CDR']
        # --- END CONDITIONAL SECTION ---

        # Create the preprocessor inside the loop with the correct features for each file type
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        # --- Data Cleanup: Handle known inconsistencies in some synthetic datasets ---
        if name in ["Copilot", "DuckDuckGo"]:
            # Convert the 0/1 in the Diagnosis column to standard FALSE/TRUE values
            df[target] = df[target].apply(lambda x: True if x == 1 else False)
        
        # Ensure the target column is the correct boolean type
        df[target] = df[target].astype(bool)
        
        # Define the features (X) and the target (y)
        X = df[features]
        y = df[target]

        # Split data into a training set (80%) and a testing set (20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Create the full machine learning pipeline
        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('classifier', LogisticRegression(random_state=42))])

        # Train the model on the training data
        model_pipeline.fit(X_train, y_train)

        # Use the trained model to make predictions on the unseen test data
        y_pred = model_pipeline.predict(X_test)

        # Calculate the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = f"{accuracy:.2%}"

    except Exception as e:
        results[name] = f"Error: {e}"

# --- Print the Final Results ---
print("Machine Learning Efficacy Results:")
print("Model trained to predict 'Diagnosis'")
print("-" * 40)
for name, accuracy in results.items():
    print(f"{name:<12}: {accuracy}")
print("-" * 40)