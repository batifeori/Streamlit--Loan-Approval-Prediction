import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Approval Decision System",
    page_icon="💳",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        return pickle.load(open("loan_model.pkl", "rb"))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("loan_data.csv")
    except:
        return None

df = load_data()

# ---------------- HEADER ----------------
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>
        💳 Loan Approval Decision Support System
    </h1>
    <p style='text-align: center;'>
        Predict • Analyze • Interpret • Decision Support
    </p>
""", unsafe_allow_html=True)

st.divider()

# ---------------- CURRENCY SELECTOR ----------------
currency = st.selectbox(
    "💱 Select Currency",
    ["GBP (£)", "USD ($)", "EUR (€)", "INR (₹)"]
)

if "GBP" in currency:
    symbol = "£"
elif "USD" in currency:
    symbol = "$"
elif "EUR" in currency:
    symbol = "€"
else:
    symbol = "₹"

# ---------------- INPUT SECTION ----------------
st.subheader("📋 Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    dependents = st.number_input("Number of Dependents", min_value=0, step=1)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    income = st.number_input("Annual Income", min_value=0)

with col2:
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Term (months)", min_value=1)
    cibil = st.number_input("CIBIL Score", min_value=300, max_value=900)

with col3:
    residential = st.number_input("Residential Assets", min_value=0)
    commercial = st.number_input("Commercial Assets", min_value=0)
    luxury = st.number_input("Luxury Assets", min_value=0)
    bank = st.number_input("Bank Balance", min_value=0)

# Encode categorical
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

total_assets = residential + commercial + luxury + bank

# ---------------- VALIDATION ----------------
errors = []

if income < 1000:
    errors.append("Income is unrealistically low")

if loan_amount <= 0:
    errors.append("Loan amount must be greater than 0")

if income > 0 and loan_amount > income * 10:
    errors.append("Loan amount too high compared to income")

if loan_term < 6:
    errors.append("Loan term too short")

if cibil < 300 or cibil > 900:
    errors.append("Invalid CIBIL score")

# ---------------- PREDICTION ----------------
st.divider()

if st.button("🔍 Predict Loan Status", use_container_width=True):

    if errors:
        for e in errors:
            st.error(f"⚠️ {e}")
        st.stop()

    if model is None:
        st.error("Model not loaded.")
        st.stop()

    try:
        features = np.array([[dependents, education, self_employed,
                              income, loan_amount, loan_term, cibil,
                              residential, commercial, luxury, bank]])

        with st.spinner("Analyzing application..."):
            prediction = model.predict(features)

            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(features)[0][1]

        st.divider()

        # ---------------- RESULT ----------------
        if prediction[0] == 1:
            st.success("✅ Loan Approved")
            st.balloons()
        else:
            st.error("❌ Loan Rejected")

        # ---------------- PROBABILITY ----------------
        if prob is not None:
            st.subheader("🎯 Approval Confidence")

            st.progress(float(prob))

            if prob > 0.75:
                st.success(f"High probability ({prob*100:.1f}%)")
            elif prob > 0.5:
                st.warning(f"Moderate probability ({prob*100:.1f}%)")
            else:
                st.error(f"Low probability ({prob*100:.1f}%)")

        # ---------------- METRICS ----------------
        st.subheader("📌 Key Metrics")

        m1, m2, m3 = st.columns(3)
        m1.metric("Income", f"{symbol}{income:,.0f}")
        m2.metric("Loan", f"{symbol}{loan_amount:,.0f}")
        m3.metric("CIBIL Score", cibil)

        # ---------------- FINANCIAL VISUAL ----------------
        st.subheader("📊 Financial Overview")

        labels = ["Income", "Loan", "Assets"]
        values = [income, loan_amount, total_assets]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_title(f"Financial Comparison ({symbol})")
        ax.set_ylabel(f"Amount ({symbol})")

        st.pyplot(fig)

        # ---------------- DATASET VISUALS ----------------
        if df is not None:

            st.subheader("📊 Dataset Insights")

            colA, colB = st.columns(2)

            with colA:
                fig1, ax1 = plt.subplots()
                df["loan_status"].value_counts().plot(kind="bar", ax=ax1)
                ax1.set_title("Loan Approval Distribution")
                st.pyplot(fig1)

            with colB:
                fig2, ax2 = plt.subplots()
                for status in df["loan_status"].unique():
                    subset = df[df["loan_status"] == status]
                    ax2.scatter(subset["income_annum"],
                                subset["loan_amount"],
                                label=status)
                ax2.legend()
                ax2.set_xlabel("Income")
                ax2.set_ylabel("Loan Amount")
                st.pyplot(fig2)

            # ---------------- COMPARISON ----------------
            st.subheader("📊 Applicant vs Dataset Average")

            avg_income = df["income_annum"].mean()
            avg_loan = df["loan_amount"].mean()

            labels = ["Your Income", "Avg Income", "Your Loan", "Avg Loan"]
            values = [income, avg_income, loan_amount, avg_loan]

            fig3, ax3 = plt.subplots()
            ax3.bar(labels, values)
            ax3.set_ylabel(f"Amount ({symbol})")

            st.pyplot(fig3)

        # ---------------- RISK ANALYSIS ----------------
        st.subheader("⚠️ Risk Analysis")

        risks = []
        dti = loan_amount / income if income > 0 else 0

        if dti > 3:
            risks.append("Very high debt-to-income ratio")
        elif dti > 2:
            risks.append("Moderate debt burden")

        if cibil < 600:
            risks.append("Low credit score")

        if total_assets < loan_amount:
            risks.append("Assets less than loan amount")

        if risks:
            for r in risks:
                st.warning(f"❗ {r}")
        else:
            st.success("No major financial risks detected")

        # ---------------- RECOMMENDATIONS ----------------
        st.subheader("💡 Recommendations")

        suggestions = []

        if dti > 2:
            suggestions.append("Reduce loan amount")

        if cibil < 700:
            suggestions.append("Improve credit score")

        if total_assets < loan_amount:
            suggestions.append("Increase assets or savings")

        if loan_term < 12:
            suggestions.append("Consider longer repayment term")

        if suggestions:
            for s in suggestions:
                st.info(f"✔ {s}")
        else:
            st.success("Strong application profile")

        # ---------------- FEATURE IMPORTANCE ----------------
        if hasattr(model, "feature_importances_"):

            st.subheader("📈 Feature Importance")

            feature_names = [
                "Dependents", "Education", "Self Employed",
                "Income", "Loan Amount", "Loan Term",
                "CIBIL", "Residential", "Commercial",
                "Luxury", "Bank"
            ]

            importance = model.feature_importances_

            fig4, ax4 = plt.subplots()
            ax4.barh(feature_names, importance)
            ax4.set_title("Model Feature Importance")

            st.pyplot(fig4)

    except Exception as e:
        st.error(f"Prediction error: {e}")