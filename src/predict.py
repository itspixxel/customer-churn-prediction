import os
import joblib
import pandas as pd

def predict_single_customer():
    
    # 1. Path to our saved pipeline
    model_path = os.path.join('models', 'churn_pipeline.joblib')

    if not os.path.exists(model_path):
        print(f" Error: Model artifact not found at {model_path}. Run train.py first.")
        return
    
    # 2. Load the entire end-to-end pipeline
    print("Loading trained pipeline artifact...")
    pipeline = joblib.load(model_path)

    # 3. Simulate a raw data payload coming from a user or API
    raw_customer_data = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.25,
        'TotalCharges': 170.50
    }

    # 4. Convert the single dictionary into a 1-row Pandas DataFrame
    input_df = pd.DataFrame([raw_customer_data])

    # 5. Run inference
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    # 6. Display results
    print("\n--- Prediction Results ---")
    if prediction == 1:
        print(f"Status: RISK OF CHURN")
    else:
        print(f"Stats: LIKELY TO STAY")

    print(f"Churn Probability: {probability * 100:.2f}%")

if __name__ == "__main__":
    predict_single_customer()