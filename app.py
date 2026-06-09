import streamlit as st
import pandas as pd
import joblib

model = joblib.load("retail_fraud_detection_pipeline.pkl")

st.set_page_config(page_title="Retail Fraud Detection App", layout="centered")

st.title("Retail Fraud Detection App")
st.write("Enter transaction details to predict whether the transaction is fraudulent.")

transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
payment_method = st.selectbox("Payment Method", ["UPI", "Credit Card", "Debit Card", "Net Banking", "Wallet"])
device_type = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])
location = st.selectbox("Location", ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata"])
merchant_category = st.selectbox("Merchant Category", ["Electronics", "Fashion", "Grocery", "Travel", "Food", "Entertainment"])

is_international = st.selectbox("International Transaction?", [0, 1])
transaction_frequency_24h = st.slider("Transaction Frequency in 24h", 0, 20, 2)
avg_transaction_amount_7d = st.number_input("Average Transaction Amount 7 Days", min_value=0.0, value=800.0)
failed_transaction_count_24h = st.slider("Failed Transaction Count 24h", 0, 10, 0)
account_age_days = st.number_input("Account Age in Days", min_value=0, value=365)

previous_fraud_flag = st.selectbox("Previous Fraud Flag", [0, 1])
unusual_amount_flag = st.selectbox("Unusual Amount Flag", [0, 1])
unusual_location_flag = st.selectbox("Unusual Location Flag", [0, 1])
multiple_transactions_short_time = st.selectbox("Multiple Transactions in Short Time", [0, 1])
high_risk_device_flag = st.selectbox("High Risk Device Flag", [0, 1])
velocity_flag = st.selectbox("Velocity Flag", [0, 1])

transaction_hour = st.slider("Transaction Hour", 0, 23, 12)
transaction_day = st.slider("Transaction Day", 1, 31, 15)
transaction_month = st.slider("Transaction Month", 1, 12, 6)
transaction_dayofweek = st.slider("Transaction Day of Week", 0, 6, 2)

input_data = pd.DataFrame({
    "transaction_amount": [transaction_amount],
    "payment_method": [payment_method],
    "device_type": [device_type],
    "location": [location],
    "merchant_category": [merchant_category],
    "is_international": [is_international],
    "transaction_frequency_24h": [transaction_frequency_24h],
    "avg_transaction_amount_7d": [avg_transaction_amount_7d],
    "failed_transaction_count_24h": [failed_transaction_count_24h],
    "account_age_days": [account_age_days],
    "previous_fraud_flag": [previous_fraud_flag],
    "unusual_amount_flag": [unusual_amount_flag],
    "unusual_location_flag": [unusual_location_flag],
    "multiple_transactions_short_time": [multiple_transactions_short_time],
    "high_risk_device_flag": [high_risk_device_flag],
    "velocity_flag": [velocity_flag],
    "transaction_hour": [transaction_hour],
    "transaction_day": [transaction_day],
    "transaction_month": [transaction_month],
    "transaction_dayofweek": [transaction_dayofweek]
})

if st.button("Predict Fraud"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"Fraudulent Transaction Detected! Probability: {probability:.2f}")
    else:
        st.success(f"Transaction Looks Safe. Fraud Probability: {probability:.2f}")