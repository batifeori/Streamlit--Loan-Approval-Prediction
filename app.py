# ===============================
# IMPORT LIBRARIES
# ===============================
import streamlit as st
import numpy as np
import pickle

# ===============================
# LOAD MODEL & SCALER
# ===============================
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

# ===============================
# TITLE & DESCRIPTION
# ===============================
st.title("💼 Loan Approval Prediction System")
st.write("""
This application predicts whether a loan will be **Approved or Rejected** 
based on applicant financial and personal information.
""")

st.markdown("---")

# ===============================
# USER INPUTS
# ===============================

st.subheader("📋 Enter Applicant Details")

# Layout in columns
col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    income_annum = st.number_input("Annual Income", min_value=0, value=50000)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=200000)
    loan_term = st.number_input("Loan Term (months)", min_value=0, value=12)

with col2:
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=650)
    residential_assets_value = st.number_input("Residential Assets Value", min_value=0, value=0)
    commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, value=0)
    luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, value=0)

bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=0)

# Categorical inputs
st.subheader("📊 Applicant Profile")

col3, col4 = st.columns(2)

with col3:
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col4:
    # if gender exists in your dataset, otherwise remove this
    gender = st.selectbox("Gender", ["Male", "Female"])

# ===============================
# ENCODING INPUTS (MUST MATCH TRAINING)
# ===============================

education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
gender = 1 if gender == "Male" else 0

# ===============================
# PREDICTION BUTTON
# ===============================
if st.button("🔍 Predict Loan Status"):

    # Create feature array (ORDER MUST MATCH TRAINING DATASET)
    features = np.array([[no_of_dependents,
                          education,
                          self_employed,
                          income_annum,
                          loan_amount,
                          loan_term,
                          cibil_score,
                          residential_assets_value,
                          commercial_assets_value,
                          luxury_assets_value,
                          bank_asset_value,
                          gender]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Prediction
    prediction = model.predict(features_scaled)

    # ===============================
    # OUTPUT
    # ===============================
    st.markdown("---")
    st.subheader("📢 Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")