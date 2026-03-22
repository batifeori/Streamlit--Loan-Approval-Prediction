import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Loan Approval Dashboard",
    page_icon="💳",
    layout="wide"
)

# ==============================
# LOAD MODEL + COLUMNS
# ==============================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("loan_approval_model.pkl")
        columns = joblib.load("columns.pkl")  # IMPORTANT
        return model, columns
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

model, train_cols = load_assets()

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
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
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
# PREDICTION
# ==============================
st.divider()

if st.button("🔍 Predict Loan Status"):

    if model is None or train_cols is None:
        st.error("Model or columns not loaded!")
    else:
        try:
            # ✅ RAW INPUT (no manual encoding)
            input_dict = {
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
            }

            input_df = pd.DataFrame([input_dict])

            # ✅ Apply SAME encoding as training
            input_df = pd.get_dummies(input_df)

            # ✅ Add missing columns
            for col in train_cols:
                if col not in input_df:
                    input_df[col] = 0

            # ✅ Remove extra columns & order correctly
            input_df = input_df[train_cols]

            # ==============================
            # PREDICT
            # ==============================
            prediction = model.predict(input_df)[0]

            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df)[0]

            st.divider()

            # ==============================
            # RESULT
            # ==============================
            if prediction == "Approved":
                st.success("✅ Loan Approved")
                st.balloons()
            else:
                st.error("❌ Loan Rejected")

            if prob is not None:
                st.metric("Approval Probability", f"{max(prob)*100:.2f}%")

            # ==============================
            # VISUAL
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