# =========================================
# IMPORT LIBRARIES
# =========================================
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Loan Approval AI",
    page_icon="💳",
    layout="wide"
)

# =========================================
# CUSTOM UI STYLE
# =========================================
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: bold;
    color: #2E86C1;
}
.subtitle {
    font-size: 18px;
    color: gray;
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
    else:
        st.error("❌ loan_model.pkl not found")
        return None

model = load_model()

# =========================================
# HEADER
# =========================================
st.markdown('<p class="main-title">💳 Loan Approval AI System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Smart Credit Risk Evaluation</p>', unsafe_allow_html=True)

st.divider()

# =========================================
# INPUT SECTION
# =========================================
st.subheader("📋 Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    no_of_dependents = st.number_input("Dependents", 0, step=1)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    income_annum = st.number_input("Annual Income", 0)

with col2:
    loan_amount = st.number_input("Loan Amount", 0)
    loan_term = st.number_input("Loan Term (Months)", 1)
    cibil_score = st.number_input("CIBIL Score", 300, 900)

with col3:
    residential_assets_value = st.number_input("Residential Assets", 0)
    commercial_assets_value = st.number_input("Commercial Assets", 0)
    luxury_assets_value = st.number_input("Luxury Assets", 0)
    bank_asset_value = st.number_input("Bank Assets", 0)

st.divider()

# =========================================
# PREDICTION
# =========================================
if st.button("🔍 Predict Loan Status"):

    if model is None:
        st.error("Model not loaded")

    else:
        try:
            # Create DataFrame (must match training data)
            input_df = pd.DataFrame([{
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

            # Prediction
            prediction = model.predict(input_df)[0]

            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df)[0][1]

            st.divider()

            # =========================================
            # RESULT DISPLAY
            # =========================================
            st.subheader("📊 Result")

            colA, colB, colC = st.columns(3)

            total_assets = (
                residential_assets_value +
                commercial_assets_value +
                luxury_assets_value +
                bank_asset_value
            )

            with colA:
                if prediction == 1:
                    st.success("✅ Approved")
                else:
                    st.error("❌ Rejected")

            with colB:
                if prob is not None:
                    st.metric("Approval Probability", f"{prob*100:.2f}%")

            with colC:
                st.metric("Total Assets", f"{total_assets:,}")

            # =========================================
            # VISUAL
            # =========================================
            st.subheader("📈 Financial Overview")

            fig, ax = plt.subplots()
            ax.bar(
                ["Income", "Loan", "Assets"],
                [income_annum, loan_amount, total_assets]
            )
            ax.set_title("Financial Comparison")

            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"Error: {e}")