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
# PREMIUM UI DESIGN
# =========================================
st.markdown("""
<style>

/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Header */
.header {
    text-align: center;
    padding: 20px;
}

.title {
    font-size: 50px;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #94a3b8;
    font-size: 18px;
}

/* Card UI */
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
}

/* Metrics */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 15px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #38bdf8);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: 600;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #020617;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# HEADER
# =========================================
st.markdown("""
<div class="header">
    <div class="title">🏦 Loan Approval System</div>
    <div class="subtitle">Smart AI-powered Credit Risk Evaluation</div>
</div>
""", unsafe_allow_html=True)

st.divider()

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
# SUMMARY SECTION
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
        st.warning("⚠️ Enter valid Income & Loan Amount")
        st.stop()

    if total_assets == 0:
        st.warning("⚠️ Enter asset values")
        st.stop()

    if model is None:
        st.error("Model not found")
        st.stop()

    try:
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
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        st.metric("Approval Probability", f"{prob*100:.2f}%")

        # =========================
        # INSIGHTS
        # =========================
        st.subheader("🧠 AI Insights")

        if prob > 0.8:
            st.success("Excellent candidate")
        elif prob > 0.6:
            st.info("Good approval chances")
        elif prob > 0.4:
            st.warning("Borderline case")
        else:
            st.error("High risk")

        # =========================
        # CHARTS
        # =========================
        st.subheader("📊 Financial Analysis")

        fig, ax = plt.subplots()
        ax.bar(["Income", "Loan", "Assets"],
               [income_annum, loan_amount, total_assets])
        st.pyplot(fig)

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

    except Exception as e:
        st.error(f"Error: {e}")