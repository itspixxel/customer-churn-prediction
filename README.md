# 📊 Telco Customer Churn Predictor

An end-to-end, production-grade machine learning microservice that predicts customer churn risk using an optimized Random Forest classifier. This project moves away from monolithic Jupyter notebooks, utilizing a completely decoupled architecture with a serialized Scikit-Learn preprocessing pipeline, a validation-backed FastAPI gateway, and an interactive Streamlit presentation layer.

## 🏗️ System Architecture

The application is built using a multi-tier microservice architecture to ensure a strict separation of concerns:

```text
[ Presentation Layer ]      [ API Gateway Layer ]       [ Machine Learning Engine ]
   Streamlit (UI)     ──►     FastAPI Server     ──►     Serialized Pipeline (.joblib)
 (User Inputs Form)        (Pydantic Validation)        (Imputer + Scaler + RF Model)
```

1. **Machine Learning Pipeline:** Encapsulates handling of missing values, numeric feature scaling (`StandardScaler`), and categorical encoding (`OneHotEncoder`) into an immutable, unified Scikit-Learn `Pipeline` artifact to eliminate data leakage between training and live inference.
2. **API Gateway:** A lightweight FastAPI server running Uvicorn that provides high-throughput inference endpoints and utilizes Pydantic schemas for strict request payload validation.
3. **Business Logic Layer:** Implements a custom decision threshold optimized via a Precision-Recall curve to maximize the model's F1-Score, balancing proactive customer retention costs against false-alarm rates.

## 📁 Repository Structure

```text
telco-churn-prediction/
├── data/                       # Dataset storage (git-ignored in production)
│   └── telco_customer_churn.csv
├── models/                     # Serialized production model artifacts
│   └── churn_pipeline.joblib
├── src/                        # Modular application source code
│   ├── data_preprocessing.py   # ETL and data pipeline definitions
│   ├── train.py                # Model training and serialization workflow
│   ├── optimize_threshold.py   # Decision threshold optimization utility
│   ├── predict.py              # Local simulation/inference runner
│   ├── main.py                 # FastAPI microservice implementation
│   └── ui.py                   # Streamlit dashboard interface
├── notebooks/                  # Sandbox environments for EDA
│   └── 01_eda.ipynb
├── requirements.txt            # Project dependencies
└── README.md
```

## 🚀 Getting Started

### 1. Environment Setup

Clone the repository and spin up an isolated virtual environment:

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# Install dependencies
pip install -r requirements.txt
```

### 2. Execution Pipeline

To train, optimize, and serve the application, execute the following modules from the project root:

* **Train and Serialize the Engine:**
```bash
python src/train.py
```


* **Optimize the Decision Threshold:**
```bash
python src/optimize_threshold.py
```


* **Launch the Backend API Server:**
```bash
uvicorn src.main:app --reload
```


* **Launch the Frontend Dashboard (In a separate terminal):**
```bash
streamlit run src/ui.py
```