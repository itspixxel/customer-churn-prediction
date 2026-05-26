import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Import the functions wrote in data_preprocessing.py
from data_preprocessing import load_and_clean_data, get_preprocessing_pipeline

def train_model():
    print("Starting model training workflow...")

    # 1. Load and clean the data
    data_path = os.path.join('data', 'telco_customer_churn.csv')
    df = load_and_clean_data(data_path)

    # 2. Split into features (X) and target (y)
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # 3. Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split successfully. Training rows: {len(X_train)}, Testing rows: {len(X_test)}")

    # 4. Get our preprocessing assembly lines
    preprocessor = get_preprocessing_pipeline(X_train)

    # 5. Combine Preprocessing AND Model into one unified pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=21))
    ])

    # 6. Train the entire pipeline
    print("Training the Random Forest model (this might take a few seconds)...")
    full_pipeline.fit(X_train, y_train)

    # 7. Evaluate the model
    print("\nEvaluating model performance on unseen test data:")
    predictions = full_pipeline.predict(X_test)
    probabilities = full_pipeline.predict_proba(X_test)[:, 1]

    # Print the classification report (Precision, Recall, F1-Score)
    print("\nClassification Report:")
    print(f"ROC AUC Score: {roc_auc_score(y_test, probabilities):.4f}")

    # 8. Save the entire pipeline artifact
    model_output_path = os.path.join('models', 'churn_pipeline.joblib')
    joblib.dump(full_pipeline, model_output_path)
    print(f"\n Success! Complete pipeline saved to: {model_output_path}")

if __name__ == "__main__":
    train_model()