# =========================================
# IMPORT LIBRARIES
# =========================================
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Clear cache
st.cache_resource.clear()

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="AI Loan Approval System",
    page_icon="💳",
    layout="wide"
)

# =========================================
# CONSTANTS
# =========================================
CURRENCY = "₹"

# =========================================
# CUSTOM STYLE
# =========================================
st.markdown("""
<style>

.main-title{
font-size:42px;
font-weight:bold;
color:#1F4E79;
}

.subtitle{
font-size:18px;
color:gray;
margin-bottom:10px;
}

.metric-box{
background-color:#f5f7fa;
padding:10px;
border-radius:8px;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODEL
# =========================================
@st.cache_resource
def load_model():

    model_path = Path("loan_model.pkl")

    if model_path.exists():
        return joblib.load(model_path)
    else:
        st.error("❌ loan_model.pkl not found")
        return None

model = load_model()

# =========================================
# HEADER
# =========================================
st.markdown('<p class="main-title">💳 AI Loan Approval System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine Learning Powered Credit Risk Evaluation</p>', unsafe_allow_html=True)

st.divider()

# =========================================
# SIDEBAR APPLICATION FORM
# =========================================
st.sidebar.header("Loan Application")

no_of_dependents = st.sidebar.slider("Dependents", 0, 10, 1)

education = st.sidebar.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.sidebar.selectbox(
    "Self Employed",
    ["No", "Yes"]
)

income_annum = st.sidebar.number_input(
    "Annual Income",
    min_value=0
)

loan_amount = st.sidebar.number_input(
    "Loan Amount Requested",
    min_value=0
)

loan_term = st.sidebar.slider(
    "Loan Term (Months)",
    6,
    360,
    12
)

cibil_score = st.sidebar.slider(
    "CIBIL Score",
    300,
    900,
    650
)

st.sidebar.subheader("Assets")

residential_assets_value = st.sidebar.number_input("Residential Assets", 0)
commercial_assets_value = st.sidebar.number_input("Commercial Assets", 0)
luxury_assets_value = st.sidebar.number_input("Luxury Assets", 0)
bank_asset_value = st.sidebar.number_input("Bank Assets", 0)

interest_rate = st.sidebar.slider(
    "Interest Rate (%)",
    1.0,
    20.0,
    8.5
)

predict_button = st.sidebar.button("Evaluate Loan")

# =========================================
# FINANCIAL SUMMARY
# =========================================
st.subheader("Applicant Financial Summary")

total_assets = (
    residential_assets_value +
    commercial_assets_value +
    luxury_assets_value +
    bank_asset_value
)

loan_income_ratio = 0
if income_annum > 0:
    loan_income_ratio = loan_amount / income_annum

col1, col2, col3, col4 = st.columns(4)

col1.metric("Annual Income", f"{CURRENCY}{income_annum:,.0f}")
col2.metric("Loan Requested", f"{CURRENCY}{loan_amount:,.0f}")
col3.metric("Total Assets", f"{CURRENCY}{total_assets:,.0f}")
col4.metric("Loan/Income Ratio", f"{loan_income_ratio:.2f}")

st.divider()

# =========================================
# EMI CALCULATOR
# =========================================
def calculate_emi(P, r, n):

    r = r / (12 * 100)

    if r == 0:
        return P / n

    emi = P * r * (1+r)**n / ((1+r)**n - 1)

    return emi

emi = 0

if loan_amount > 0:

    emi = calculate_emi(
        loan_amount,
        interest_rate,
        loan_term
    )

    st.subheader("Estimated Monthly Repayment")

    st.metric("Monthly EMI", f"{CURRENCY}{emi:,.2f}")

st.divider()

# =========================================
# PREDICTION
# =========================================
if predict_button:

    if model is None:
        st.error("Model not loaded")

    else:

        try:

            # Encode categorical variables
            education_encoded = 1 if education == "Graduate" else 0
            self_employed_encoded = 1 if self_employed == "Yes" else 0

            # Feature order must match training dataset
            input_data = pd.DataFrame([[
                no_of_dependents,
                education_encoded,
                self_employed_encoded,
                income_annum,
                loan_amount,
                loan_term,
                cibil_score,
                residential_assets_value,
                commercial_assets_value,
                luxury_assets_value,
                bank_asset_value
            ]], columns=[
                "no_of_dependents",
                "education",
                "self_employed",
                "income_annum",
                "loan_amount",
                "loan_term",
                "cibil_score",
                "residential_assets_value",
                "commercial_assets_value",
                "luxury_assets_value",
                "bank_asset_value"
            ])

            prediction = model.predict(input_data)[0]

            probability = None
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(input_data)[0][1]

            st.subheader("Loan Decision")

            colA, colB = st.columns(2)

            with colA:
                if prediction == 1:
                    st.success("✅ Loan Approved")
                else:
                    st.error("❌ Loan Rejected")

            with colB:
                if probability is not None:
                    st.metric(
                        "Approval Probability",
                        f"{probability*100:.2f}%"
                    )

            st.divider()

            # =========================================
            # FINANCIAL COMPARISON CHART
            # =========================================
            st.subheader("Financial Analysis")

            fig, ax = plt.subplots()

            categories = ["Income", "Loan", "Assets"]
            values = [income_annum, loan_amount, total_assets]

            ax.bar(categories, values)

            ax.set_title("Financial Comparison")

            st.pyplot(fig)

            # =========================================
            # ASSET DISTRIBUTION
            # =========================================
            st.subheader("Asset Distribution")

            fig2, ax2 = plt.subplots()

            assets = [
                residential_assets_value,
                commercial_assets_value,
                luxury_assets_value,
                bank_asset_value
            ]

            labels = [
                "Residential",
                "Commercial",
                "Luxury",
                "Bank"
            ]

            ax2.pie(
                assets,
                labels=labels,
                autopct="%1.1f%%"
            )

            st.pyplot(fig2)

            # =========================================
            # CREDIT RISK ANALYSIS
            # =========================================
            st.subheader("Credit Risk Assessment")

            if cibil_score >= 750:
                st.success("Low Credit Risk")

            elif cibil_score >= 650:
                st.warning("Moderate Credit Risk")

            else:
                st.error("High Credit Risk")

            # =========================================
            # LOAN AFFORDABILITY
            # =========================================
            st.subheader("Loan Affordability")

            if loan_income_ratio < 0.3:
                st.success("Loan is financially safe")

            elif loan_income_ratio < 0.5:
                st.warning("Loan is moderate risk")

            else:
                st.error("Loan may be difficult to repay")

            # =========================================
            # APPLICANT SUMMARY
            # =========================================
            st.subheader("Application Summary")

            st.write(f"""
### Applicant Financial Summary

Annual Income: {CURRENCY}{income_annum:,.0f}

Loan Requested: {CURRENCY}{loan_amount:,.0f}

Total Assets: {CURRENCY}{total_assets:,.0f}

CIBIL Score: {cibil_score}

Estimated Monthly Payment: {CURRENCY}{emi:,.2f}
""")

        except Exception as e:

            st.error(f"Prediction Error: {e}")