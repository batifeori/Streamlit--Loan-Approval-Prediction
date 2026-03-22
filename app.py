import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Loan Approval Dashboard",
    page_icon="💳",
    layout="wide"
)

# ==============================
# LOAD MODEL (FIXED)
# ==============================
@st.cache_resource
def load_model():
    possible_paths = [
        "loan_model.pkl",
        "loan_approval_model.pkl",
        "model.pkl",
        "models/loan_model.pkl"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return joblib.load(path)

    st.error(
        "❌ Model file not found.\n\n"
        "Make sure your .pkl file is uploaded to GitHub and matches one of these names:\n"
        f"{possible_paths}"
    )
    return None

model = load_model()

# ==============================
# HEADER
# ==============================
st.markdown("""
<h1 style='text-align: center; color: #2E86C1;'>💳 Loan Approval Dashboard</h1>
<p style='text-align: center;'>Smart Prediction • Risk Analysis</p>
""", unsafe_allow_html=True)

st.divider()

# ==============================
# INPUT SECTION
# ==============================
st.subheader("📋 Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    no_of_dependents = st.number_input("Dependents", min_value=0)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    income_annum = st.number_input("Income", min_value=0)

with col2:
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Term", min_value=1)
    cibil_score = st.number_input("CIBIL Score", 300, 900)

with col3:
    residential_assets_value = st.number_input("Residential Assets", min_value=0)
    commercial_assets_value = st.number_input("Commercial Assets", min_value=0)
    luxury_assets_value = st.number_input("Luxury Assets", min_value=0)
    bank_asset_value = st.number_input("Bank Assets", min_value=0)

# ==============================
# ENCODING FUNCTION
# ==============================
def encode_inputs(education, self_employed):
    education_map = {"Graduate": 0, "Not Graduate": 1}
    self_employed_map = {"No": 0, "Yes": 1}

    return education_map[education], self_employed_map[self_employed]

# ==============================
# PREDICTION
# ==============================
st.divider()

if st.button("🔍 Predict Loan Status"):

    if model is None:
        st.error("⚠️ Model not loaded. Check your .pkl file.")
    else:
        try:
            education_enc, self_emp_enc = encode_inputs(education, self_employed)

            input_data = np.array([[
                no_of_dependents,
                education_enc,
                self_emp_enc,
                income_annum,
                loan_amount,
                loan_term,
                cibil_score,
                residential_assets_value,
                commercial_assets_value,
                luxury_assets_value,
                bank_asset_value
            ]])

            prediction = model.predict(input_data)[0]

            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_data)[0]

            st.divider()

            if prediction == 1:
                st.success("✅ Loan Approved")
                st.balloons()
            else:
                st.error("❌ Loan Rejected")

            if prob is not None:
                st.metric("Approval Probability", f"{prob[1]*100:.2f}%")

            # ==============================
            # VISUALIZATION
            # ==============================
            st.subheader("📊 Financial Overview")

            total_assets = (
                residential_assets_value +
                commercial_assets_value +
                luxury_assets_value +
                bank_asset_value
            )

            labels = ["Income", "Loan", "Assets"]
            values = [income_annum, loan_amount, total_assets]

            fig, ax = plt.subplots()
            ax.bar(labels, values)
            ax.set_title("Financial Comparison")

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction error: {e}")