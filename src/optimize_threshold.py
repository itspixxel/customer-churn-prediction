import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from data_preprocessing import load_and_clean_data, train_test_split

def find_best_threshold():
    
    # 1. Load model and test data
    model = joblib.load(os.path.join('models', 'churn_pipeline.joblib'))
    df = load_and_clean_data(os.path.join('data', 'telco_customer_churn.csv'))

    X = df.drop(columns=['Churn'])
    y = df['Churn']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)

    # 2. Get the raw churn probabilities for the test set
    probabilities = model.predict_proba(X_test)[:, 1]

    # 3. Calculate precisions, recalls, and thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_test, probabilities)

    # 4. Calculate F1-score for every threshold point
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10) # Small epsilon added to denominator to prevent division by zero errors

    # 5. Find the index of the highest F1-score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"Optimal Threshold found: {best_threshold:.4f}")
    print(f"Max achievable F1-Score at this threshold: {best_f1:.4f}")
    print(f"Corresponding Precision: {precisions[best_idx]:.4f}")
    print(f"Corresponding Recall: {recalls[best_idx]:.4f}")

if __name__ == "__main__":
    find_best_threshold()