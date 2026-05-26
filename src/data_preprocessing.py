import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_clean_data(filepath):
    """
    Loads the Telco CSV, fixes data types, removes unneeded columns, and maps the target variable.
    """

    df = pd.read_csv(filepath)

    # 1. Drop customerID since it has no predictive power
    df.drop(columns=['customerID'], errors='ignore', inplace=True)

    # 2. Fix the TotalCharges empty spaces found in EDA
    df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

    # 3. Encode our target variable 'Churn' to binary integers (0 and 1)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df

def get_preprocessing_pipeline(X):
    """
    Creates an end-to-end ColumnTransformer that automatically handles imputing and scaling/encoding for new data.
    """

    # Dynamically grab column names based on data types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Assembly line for numerical data (tenure, MonthlyCharges, TotalCharges)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Assembly line for categorial data (Contract, InternetService, etc.)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # Combine both assembly lines into a single master preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor