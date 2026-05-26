import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Initialize the FastAPI application
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="A production-ready API to predict the probability of customer churn.",
    version="1.0"
)

# 2. Define the expected structure of incoming JSON data using Pydantic.
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# 3. Load our ML Pipeline artifact globally when the app starts
MODEL_PATH = os.path.join('models', 'churn_pipeline.joblib')
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Pipeline artifact not found at {MODEL_PATH}. Run train.py first.")

# 4. Create a basic health-check route
@app.get("/")
def home():
    return {"message": "Telco Churn Prediction API is live! Navigate to /docs for interactive testing."}

# 5. Create the live inference endpoint
@app.post("/predict")
def predict_churn(customer: CustomerData):
    try: 
        # Convert incoming Pydantic validation object directly into a dictionary, then wrap it into a single-row Pandas DataFrame for our Scikit-Learn pipeline.
        input_df = pd.DataFrame([customer.model_dump()])

        # Run inference using our serialized pipeline
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        # Return a clean JSON response to the client
        return {
            "churn_prediction": prediction,
            "churn_probability": round(probability, 4),
            "status": "RISK OF CHURN" if prediction == 1 else "LIKELY TO STAY"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")