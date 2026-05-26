import streamlit as st
import requests

# Set page configuration for a wider, cleaner dashboard layout
st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")

st.title("📊 Telco Customer Churn Risk Analyzer")
st.markdown("Enter a customer's account metrics below to evaluate their operational risk profile in real-time.")

st.divider()

# Organize the inputs into 3 distinct visual columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen Status", ["No", "Yes"])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

with col2:
    st.subheader("📱 Core Services")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service Provider", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security Add-on", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup Add-on", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection Plan", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support Add-on", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV Service", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies Service", ["No", "Yes", "No internet service"])

with col3:
    st.subheader("💳 Financials & Account")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    tenure = st.slider("Account Tenure (Months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
    total_charges = st.number_input("Total Lifetime Charges ($)", min_value=0.0, value=780.0)

st.divider()

# Trigger analysis when button is pressed
if st.button("Evaluate Churn Risk Profile", type="primary", use_container_width=True):
    
    # Map visual UI labels to the exact string names expected by our Pydantic schema
    payload = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges)
    }
    
    try:
        # Send an HTTP POST request to our running FastAPI microservice backend
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            prob = result["churn_probability"]
            
            # --- CUSTOM BUSINESS THRESHOLD ENGINE ---
            # Instead of accepting the default 50% limit from FastAPI, 
            # we implement our tuned threshold directly in the presentation layer!
            CUSTOM_THRESHOLD = 0.32  
            
            st.subheader("🔮 Analysis Engine Output")
            
            # Display risk metrics cleanly based on our custom business rule threshold
            if prob >= CUSTOM_THRESHOLD:
                st.error(f"🚨 ALERT: HIGH CHURN RISK INDICATION (Probability: {prob * 100:.2f}%)")
                st.markdown(f"**Operational Threshold Notice:** This account has crossed our target risk threshold of {CUSTOM_THRESHOLD * 100:.0f}%. Proactive account retention workflows should be initiated.")
            else:
                st.success(f"✅ STATUS NORMAL: ACCOUNT STABLE (Probability: {prob * 100:.2f}%)")
                st.markdown(f"The structural risk profile for this customer remains within nominal parameters.")
        else:
            st.error(f"Backend API returned an error status code: {response.status_code}")
            st.json(response.json())
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Communication Failure: Could not connect to the FastAPI backend. Is your Uvicorn server running on port 8000?")