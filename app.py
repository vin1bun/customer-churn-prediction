
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load model, scaler and feature names ---
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

# --- Page config ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📱",
    layout="wide"
)

# --- Title ---
st.title("📱 Customer Churn Predictor")
st.markdown("**Predict if a customer will leave — and what they are worth**")
st.divider()

# --- Sidebar inputs ---
st.sidebar.header("👤 Customer Details")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 0, 120, 65)
contract = st.sidebar.selectbox("Contract Type",
    ["Month-to-month", "One year", "Two year"])
payment = st.sidebar.selectbox("Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"])
internet = st.sidebar.selectbox("Internet Service",
    ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines",
    ["Yes", "No", "No phone service"])
streaming_tv = st.sidebar.selectbox("Streaming TV",
    ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies",
    ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No"])
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# --- Feature Engineering ---
services = [online_security, online_backup, device_protection,
            tech_support, streaming_tv, streaming_movies]
services_count = sum([1 for s in services if s == "Yes"])
total_charges = monthly_charges * max(tenure, 1)
avg_monthly_spend = total_charges / max(tenure, 1)
is_new_customer = 1 if tenure < 12 else 0
charge_per_service = monthly_charges / max(services_count, 1)

# --- Build input dict ---
input_dict = {
    "gender": 1 if gender == "Male" else 0,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": 1 if partner == "Yes" else 0,
    "Dependents": 1 if dependents == "Yes" else 0,
    "tenure": tenure,
    "PhoneService": 1 if phone_service == "Yes" else 0,
    "OnlineSecurity": 1 if online_security == "Yes" else 0,
    "OnlineBackup": 1 if online_backup == "Yes" else 0,
    "DeviceProtection": 1 if device_protection == "Yes" else 0,
    "TechSupport": 1 if tech_support == "Yes" else 0,
    "StreamingTV": 1 if streaming_tv == "Yes" else 0,
    "StreamingMovies": 1 if streaming_movies == "Yes" else 0,
    "PaperlessBilling": 1 if paperless == "Yes" else 0,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "avg_monthly_spend": avg_monthly_spend,
    "services_count": services_count,
    "is_new_customer": is_new_customer,
    "charge_per_service": charge_per_service,
    "MultipleLines_No phone service": 1 if multiple_lines == "No phone service" else 0,
    "MultipleLines_Yes": 1 if multiple_lines == "Yes" else 0,
    "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
    "InternetService_No": 1 if internet == "No" else 0,
    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,
    "PaymentMethod_Credit card (automatic)": 1 if payment == "Credit card (automatic)" else 0,
    "PaymentMethod_Electronic check": 1 if payment == "Electronic check" else 0,
    "PaymentMethod_Mailed check": 1 if payment == "Mailed check" else 0,
}

# --- Create DataFrame with correct feature order ---
input_df = pd.DataFrame([input_dict])[feature_names]

# --- Scale numeric columns ---
scale_cols = ["tenure", "MonthlyCharges", "TotalCharges",
              "avg_monthly_spend", "charge_per_service"]
input_df[scale_cols] = scaler.transform(input_df[scale_cols])

# --- Predict ---
prob = model.predict_proba(input_df)[0][1]
prediction = int(prob >= 0.5)

# --- Calculate CLV ---
clv = monthly_charges * (1 / max(prob, 0.01)) * 12

# --- Priority Level based on CLV ---
if clv > 1500:
    priority = "🔴 HIGH PRIORITY"
elif clv > 800:
    priority = "🟡 MEDIUM PRIORITY"
else:
    priority = "🟢 LOW PRIORITY"

# --- Show Results ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🎯 Churn Prediction")
    if prediction == 1:
        st.error("🔴 HIGH CHURN RISK")
    else:
        st.success("🟢 LOW CHURN RISK")
    st.metric("Churn Probability", f"{prob*100:.1f}%")

with col2:
    st.subheader("💰 Customer Value")
    st.metric("Customer Lifetime Value", f"${clv:,.0f}")
    st.metric("Monthly Revenue", f"${monthly_charges}")
    st.markdown(f"**Retention Priority:** {priority}")

with col3:
    st.subheader("💡 Retention Action")
    if prediction == 1:
        if contract == "Month-to-month":
            st.warning("👉 Offer discounted annual contract upgrade")
        if payment == "Electronic check":
            st.warning("👉 Encourage switch to automatic payment")
        if internet == "Fiber optic":
            st.warning("👉 Offer loyalty discount on fiber plan")
        if is_new_customer:
            st.warning("👉 Trigger onboarding support program")
        if services_count < 2:
            st.warning("👉 Offer bundled services discount")
    else:
        st.info("✅ Customer appears loyal — maintain engagement")

st.divider()

# --- CLV Insight Banner ---
if prediction == 1:
    st.error(f"⚠️ This customer represents **\${clv:,.0f}** in lifetime value at risk. {priority}")

st.caption("Built by Vineet | Customer Churn Prediction Project")
