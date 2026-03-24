# =========================================
# IMPORTS
# =========================================
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

st.cache_resource.clear()

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Loan Approval System",
    page_icon="🏦",
    layout="wide"
)

CURRENCY = "₹"

# =========================================
# UI
# =========================================
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 48px;
    font-weight: 700;
}
.subtitle {
    text-align: center;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🏦 Loan Approval System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Credit Risk Evaluation Platform</div>', unsafe_allow_html=True)

st.divider()

# =========================================
# LOAD MODEL
# =========================================
@st.cache_resource
def load_model():
    path = Path("loan_model.pkl")
    return joblib.load(path) if path.exists() else None

model = load_model()

# =========================================
# SIDEBAR INPUT
# =========================================
st.sidebar.header("📋 Application Form")

no_of_dependents = st.sidebar.slider("Dependents", 0, 10, 1)
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["No", "Yes"])

income_annum = st.sidebar.number_input("Annual Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)

loan_term = st.sidebar.slider("Loan Term (Months)", 6, 360, 12)
cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, 650)

st.sidebar.subheader("Assets")
residential_assets_value = st.sidebar.number_input("Residential", 0)
commercial_assets_value = st.sidebar.number_input("Commercial", 0)
luxury_assets_value = st.sidebar.number_input("Luxury", 0)
bank_asset_value = st.sidebar.number_input("Bank", 0)

interest_rate = st.sidebar.slider("Interest Rate (%)", 1.0, 20.0, 8.5)

predict_button = st.sidebar.button("🚀 Evaluate Loan")

# =========================================
# SUMMARY
# =========================================
total_assets = (
    residential_assets_value +
    commercial_assets_value +
    luxury_assets_value +
    bank_asset_value
)

loan_income_ratio = loan_amount / income_annum if income_annum > 0 else 0

st.subheader("📊 Financial Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Income", f"{CURRENCY}{income_annum:,.0f}")
col2.metric("Loan", f"{CURRENCY}{loan_amount:,.0f}")
col3.metric("Assets", f"{CURRENCY}{total_assets:,.0f}")
col4.metric("Debt Ratio", f"{loan_income_ratio:.2f}")

st.divider()

# =========================================
# EMI
# =========================================
def calculate_emi(P, r, n):
    r = r / (12 * 100)
    return P*r*(1+r)**n / ((1+r)**n - 1) if r > 0 else P/n

emi = calculate_emi(loan_amount, interest_rate, loan_term) if loan_amount > 0 else 0

if emi > 0:
    st.metric("💰 Monthly EMI", f"{CURRENCY}{emi:,.2f}")

st.divider()

# =========================================
# PREDICTION
# =========================================
if predict_button:

    if income_annum <= 0 or loan_amount <= 0:
        st.warning("⚠️ Please enter valid Income & Loan Amount")
        st.stop()

    if total_assets == 0:
        st.warning("⚠️ Please enter asset details")
        st.stop()

    if model is None:
        st.error("Model not found")
        st.stop()

    try:
        # ✅ FIXED INPUT (NO MANUAL ENCODING)
        input_data = pd.DataFrame([{
            "no_of_dependents": no_of_dependents,
            "education": education,
            "self_employed": self_employed,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets_value,
            "commercial_assets_value": commercial_assets_value,
            "luxury_assets_value": luxury_assets_value,
            "bank_asset_value": bank_asset_value
        }])

        # 🔍 DEBUG (REMOVE LATER)
        st.write("Input Data Sent to Model:")
        st.write(input_data)

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.subheader("🏦 Loan Decision")

        if pred == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        st.metric("Approval Probability", f"{prob*100:.2f}%")

    except Exception as e:
        st.error(f"Error: {e}")