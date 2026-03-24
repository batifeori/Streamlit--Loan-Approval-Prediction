# IMPORTS
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

st.cache_resource.clear()

# PAGE CONFIG
st.set_page_config(
    page_title="Loan Approval System",
    page_icon="🏦",
    layout="wide"
)

CURRENCY = "₹"

# CUSTOM UI
st.markdown("""
<style>

.title {
    text-align:center;
    font-size:48px;
    font-weight:700;
}

.subtitle {
    text-align:center;
    color:gray;
    margin-bottom:25px;
}

.approved-box{
    background-color:#e6ffed;
    padding:25px;
    border-radius:10px;
    border:2px solid #00c853;
    text-align:center;
    font-size:28px;
    font-weight:600;
}

.rejected-box{
    background-color:#ffe6e6;
    padding:25px;
    border-radius:10px;
    border:2px solid red;
    text-align:center;
    font-size:28px;
    font-weight:600;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🏦 Loan Approval System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Credit Risk Evaluation Platform</div>', unsafe_allow_html=True)

st.divider()

# LOAD MODEL
@st.cache_resource
def load_model():
    path = Path("loan_model.pkl")
    if path.exists():
        return joblib.load(path)
    return None

model = load_model()

# SIDEBAR INPUT
st.sidebar.header("📋 Loan Application")

no_of_dependents = st.sidebar.slider("Dependents", 0, 10, 1)

education = st.sidebar.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.sidebar.selectbox(
    "Self Employed",
    ["No", "Yes"]
)

income_annum = st.sidebar.number_input("Annual Income", min_value=0)

loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)

loan_term = st.sidebar.slider("Loan Term (Months)", 6, 360, 12)

cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, 650)

st.sidebar.subheader("Assets")

residential_assets_value = st.sidebar.number_input("Residential Assets", 0)
commercial_assets_value = st.sidebar.number_input("Commercial Assets", 0)
luxury_assets_value = st.sidebar.number_input("Luxury Assets", 0)
bank_asset_value = st.sidebar.number_input("Bank Assets", 0)

interest_rate = st.sidebar.slider("Interest Rate (%)", 1.0, 20.0, 8.5)

predict_button = st.sidebar.button("🚀 Evaluate Loan")

# SUMMARY CALCULATIONS
total_assets = (
    residential_assets_value
    + commercial_assets_value
    + luxury_assets_value
    + bank_asset_value
)

loan_income_ratio = loan_amount / income_annum if income_annum > 0 else 0

st.subheader("📊 Financial Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Income", f"{CURRENCY}{income_annum:,.0f}")
col2.metric("Loan", f"{CURRENCY}{loan_amount:,.0f}")
col3.metric("Assets", f"{CURRENCY}{total_assets:,.0f}")
col4.metric("Debt Ratio", f"{loan_income_ratio:.2f}")

st.divider()

# EMI CALCULATION
def calculate_emi(P, r, n):

    r = r / (12 * 100)

    if r == 0:
        return P / n

    emi = P * r * (1 + r)**n / ((1 + r)**n - 1)

    return emi


emi = calculate_emi(loan_amount, interest_rate, loan_term) if loan_amount > 0 else 0

if emi > 0:
    st.metric("💰 Monthly EMI", f"{CURRENCY}{emi:,.2f}")

st.divider()

# PREDICTION
if predict_button:

    if income_annum <= 0 or loan_amount <= 0:
        st.warning("⚠️ Please enter valid Income & Loan Amount")
        st.stop()

    if total_assets == 0:
        st.warning("⚠️ Please enter asset details")
        st.stop()

    if model is None:
        st.error("❌ Model file not found")
        st.stop()

    try:

        input_data = pd.DataFrame([[
            no_of_dependents,
            education,
            self_employed,
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

        input_data["loan_income_ratio"] = (
            input_data["loan_amount"] / input_data["income_annum"]
        ).replace([float("inf"), -float("inf")], 0).fillna(0)

        proba = model.predict_proba(input_data)[0]

        probability = proba[1]

        threshold = 0.5

        prediction = int(probability >= threshold)

        st.subheader("🏦 Loan Decision")

        # APPROVED
        if prediction == 1:

            st.balloons()

            st.markdown(
                '<div class="approved-box">🎉 CONGRATULATIONS! <br> Your Loan Has Been Approved 🎉</div>',
                unsafe_allow_html=True
            )

            st.success("🏆 The applicant meets the credit requirements.")

        # REJECTED
        else:

            st.markdown(
                '<div class="rejected-box">❌ Loan Application Rejected</div>',
                unsafe_allow_html=True
            )

            st.error("The applicant does not meet approval criteria.")

        st.metric("Approval Probability", f"{probability*100:.2f}%")

        # INSIGHTS
        st.subheader("🧠 Decision Insights")

        if probability > 0.8:
            st.success("Very strong approval likelihood")

        elif probability > 0.6:
            st.info("Good chances of approval")

        elif probability > 0.4:
            st.warning("Borderline applicant")

        else:
            st.error("High risk applicant")

        if loan_income_ratio > 0.5:
            st.error("⚠️ High loan burden relative to income")

        elif loan_income_ratio > 0.3:
            st.warning("Moderate financial stress")

        else:
            st.success("Healthy financial ratio")

        if cibil_score < 650:
            st.error("Low credit score impacts approval")

        # CHARTS
        st.subheader("📊 Financial Analysis")

        fig, ax = plt.subplots()

        ax.bar(
            ["Income", "Loan", "Assets"],
            [income_annum, loan_amount, total_assets]
        )

        st.pyplot(fig)

        assets = [
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value
        ]

        if sum(assets) > 0:

            fig2, ax2 = plt.subplots()

            ax2.pie(
                assets,
                labels=["Residential", "Commercial", "Luxury", "Bank"],
                autopct="%1.1f%%"
            )

            st.pyplot(fig2)

    except Exception as e:

        st.error(f"⚠️ Error: {e}")