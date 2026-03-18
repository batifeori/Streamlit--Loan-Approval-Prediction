import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Loan Approval AI Dashboard",
    page_icon="💳",
    layout="wide"
)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("loan_model.pkl", "rb"))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>
        💳 Loan Approval AI Dashboard
    </h1>
    <p style='text-align: center;'>
        Smart Prediction • Risk Analysis • Decision Support System
    </p>
""", unsafe_allow_html=True)

st.divider()

# -------------------------------
# INPUT SECTION
# -------------------------------
st.subheader("📋 Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    income_annum = st.number_input("Annual Income", min_value=0)

with col2:
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Term (months)", min_value=1)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)

with col3:
    residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
    commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
    luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
    bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

# -------------------------------
# ENCODING
# -------------------------------
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# -------------------------------
# PREDICTION BUTTON
# -------------------------------
st.divider()

if st.button("🔍 Predict Loan Status", use_container_width=True):

    if model is None:
        st.error("Model not loaded properly.")
    else:
        try:
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
                                  bank_asset_value]])

            with st.spinner("Analyzing application..."):
                prediction = model.predict(features)

                # Probability
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(features)[0][1]
                else:
                    prob = None

            st.divider()

            # -------------------------------
            # RESULT
            # -------------------------------
            if prediction[0] == 1:
                st.success("✅ Loan Approved")
                st.balloons()
            else:
                st.error("❌ Loan Rejected")

            if prob is not None:
                st.metric("Approval Probability", f"{prob*100:.2f}%")

            # -------------------------------
            # DATA VISUALIZATION
            # -------------------------------
            st.subheader("📊 Financial Overview")

            labels = ["Income", "Loan", "Total Assets"]
            total_assets = (residential_assets_value +
                            commercial_assets_value +
                            luxury_assets_value +
                            bank_asset_value)

            values = [income_annum, loan_amount, total_assets]

            fig, ax = plt.subplots()
            ax.bar(labels, values)
            ax.set_title("Financial Comparison")

            st.pyplot(fig)

            # -------------------------------
            # RISK ANALYSIS
            # -------------------------------
            st.subheader("⚠️ Risk Analysis")

            risks = []

            if income_annum > 0:
                ratio = loan_amount / income_annum
                if ratio > 2:
                    risks.append("Loan amount is too high compared to income")

            if cibil_score < 600:
                risks.append("Low credit score")

            if loan_term < 6:
                risks.append("Loan term is too short")

            if total_assets < loan_amount:
                risks.append("Assets are lower than requested loan")

            if risks:
                for r in risks:
                    st.warning(f"❗ {r}")
            else:
                st.success("No major risks detected")

            # -------------------------------
            # SUGGESTIONS
            # -------------------------------
            st.subheader("💡 Recommendations")

            suggestions = []

            if income_annum > 0 and loan_amount / income_annum > 2:
                suggestions.append("Reduce loan amount or increase income")

            if cibil_score < 700:
                suggestions.append("Improve your credit score")

            if total_assets < loan_amount:
                suggestions.append("Increase assets or reduce loan")

            if loan_term < 12:
                suggestions.append("Consider longer loan term")

            if suggestions:
                for s in suggestions:
                    st.info(f"✔ {s}")
            else:
                st.success("Strong application profile")

            # -------------------------------
            # FEATURE IMPORTANCE (if supported)
            # -------------------------------
            if hasattr(model, "feature_importances_"):
                st.subheader("📈 Feature Importance")

                feature_names = [
                    "Dependents", "Education", "Self Employed",
                    "Income", "Loan Amount", "Loan Term",
                    "CIBIL Score", "Residential Assets",
                    "Commercial Assets", "Luxury Assets", "Bank Assets"
                ]

                importance = model.feature_importances_

                fig2, ax2 = plt.subplots()
                ax2.barh(feature_names, importance)
                ax2.set_title("Model Feature Importance")

                st.pyplot(fig2)

        except Exception as e:
            st.error(f"Prediction error: {e}")