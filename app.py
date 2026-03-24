# =========================================
# IMPORTS
# =========================================
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

st.cache_resource.clear()

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="SmartLoan AI | Credit Intelligence",
    page_icon="💳",
    layout="wide"
)

CURRENCY = "₹"

# =========================================
# CUSTOM UI STYLE
# =========================================
st.markdown("""
<style>
.main-header {
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(90deg, #1f77b4, #00c6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.card {
    padding: 15px;
    border-radius: 12px;
    background-color: #111827;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODEL
# =========================================
@st.cache_resource
def load_model():
    path = Path("loan_model.pkl")
    if path.exists():
        return joblib.load(path)
    return None

model = load_model()

# =========================================
# HEADER
# =========================================
st.markdown('<div class="main-header">💳 SmartLoan AI</div>', unsafe_allow_html=True)
st.caption("AI-powered Credit Risk & Loan Approval Engine")
st.divider()

# =========================================
# SIDEBAR
# =========================================
st.sidebar.header("📋 Application Form")

no_of_dependents = st.sidebar.slider("Dependents", 0, 10, 1)
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["No", "Yes"])

income_annum = st.sidebar.number_input("Annual Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)

loan_term = st.sidebar.slider("Loan Term (Months)", 6, 360, 12)
cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, 650)

st.sidebar.subheader("🏠 Assets")
residential_assets_value = st.sidebar.number_input("Residential", 0)
commercial_assets_value = st.sidebar.number_input("Commercial", 0)
luxury_assets_value = st.sidebar.number_input("Luxury", 0)
bank_asset_value = st.sidebar.number_input("Bank Balance", 0)

interest_rate = st.sidebar.slider("Interest Rate (%)", 1.0, 20.0, 8.5)

predict_button = st.sidebar.button("🚀 Evaluate Loan")

# =========================================
# SUMMARY
# =========================================
total_assets = sum([
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
])

loan_income_ratio = loan_amount / income_annum if income_annum > 0 else 0

st.subheader("📊 Financial Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Income", f"{CURRENCY}{income_annum:,.0f}")
col2.metric("Loan", f"{CURRENCY}{loan_amount:,.0f}")
col3.metric("Assets", f"{CURRENCY}{total_assets:,.0f}")
col4.metric("Debt Ratio", f"{loan_income_ratio:.2f}")

# =========================================
# INPUT COMPLETENESS BAR
# =========================================
inputs_filled = sum([
    income_annum > 0,
    loan_amount > 0,
    total_assets > 0
])

st.progress(inputs_filled / 3)
st.caption("Application Completeness")

st.divider()

# =========================================
# EMI
# =========================================
def calculate_emi(P, r, n):
    r = r / (12 * 100)
    return P*r*(1+r)**n / ((1+r)**n - 1) if r > 0 else P/n

if loan_amount > 0:
    emi = calculate_emi(loan_amount, interest_rate, loan_term)
    st.metric("💰 Monthly EMI", f"{CURRENCY}{emi:,.2f}")
else:
    emi = 0

st.divider()

# =========================================
# PREDICTION
# =========================================
if predict_button:

    if income_annum <= 0 or loan_amount <= 0 or total_assets == 0:
        st.warning("⚠️ Please complete all required financial inputs")
        st.stop()

    if model is None:
        st.error("Model not found")
        st.stop()

    try:
        # Encoding
        edu = 1 if education == "Graduate" else 0
        emp = 1 if self_employed == "Yes" else 0

        input_data = pd.DataFrame([[
            no_of_dependents, edu, emp,
            income_annum, loan_amount, loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value
        ]], columns=[
            "no_of_dependents","education","self_employed",
            "income_annum","loan_amount","loan_term",
            "cibil_score",
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value"
        ])

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.subheader("🏦 Loan Decision")

        if pred == 1:
            st.success("✅ Approved")
        else:
            st.error("❌ Rejected")

        st.metric("Approval Probability", f"{prob*100:.2f}%")

        # =========================================
        # RISK SCORE VISUAL
        # =========================================
        risk_score = (1 - prob) * 100
        st.progress(prob)
        st.caption(f"Risk Score: {risk_score:.1f}%")

        # =========================================
        # CHARTS
        # =========================================
        st.subheader("📊 Financial Analysis")

        fig, ax = plt.subplots()
        ax.bar(["Income", "Loan", "Assets"],
               [income_annum, loan_amount, total_assets])
        st.pyplot(fig)

        # PIE FIXED
        assets = [
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value
        ]

        if sum(assets) > 0:
            fig2, ax2 = plt.subplots()
            ax2.pie(assets, labels=["Res","Com","Lux","Bank"], autopct="%1.1f%%")
            st.pyplot(fig2)
        else:
            st.info("No asset distribution available")

        # =========================================
        # INSIGHTS
        # =========================================
        st.subheader("🧠 AI Insights")

        if cibil_score > 750:
            st.success("Excellent credit behavior")
        elif cibil_score > 650:
            st.warning("Average credit profile")
        else:
            st.error("High credit risk")

        if loan_income_ratio > 0.5:
            st.error("High repayment burden")
        elif loan_income_ratio > 0.3:
            st.warning("Moderate repayment risk")
        else:
            st.success("Healthy financial position")

        # =========================================
        # DOWNLOAD REPORT
        # =========================================
        report = f"""
Loan Decision Report

Income: {income_annum}
Loan: {loan_amount}
Assets: {total_assets}
CIBIL: {cibil_score}

Approval Probability: {prob*100:.2f}%
EMI: {emi:.2f}
"""

        st.download_button("📄 Download Report", report)

    except Exception as e:
        st.error(f"Error: {e}")