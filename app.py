# =========================================
# IMPORT LIBRARIES
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
    page_title="AI Loan Approval System",
    page_icon="💳",
    layout="wide"
)

# =========================================
# CURRENCY (INDIAN RUPEE)
# =========================================
symbol = "₹"

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
        st.error("loan_model.pkl not found")
        return None

model = load_model()

# =========================================
# HEADER
# =========================================
st.markdown('<p class="main-title">💳 AI Loan Approval System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Fintech Credit Risk Evaluation Dashboard</p>', unsafe_allow_html=True)

st.divider()

# =========================================
# SIDEBAR APPLICATION
# =========================================
st.sidebar.header("Loan Application")

no_of_dependents = st.sidebar.slider("Dependents",0,10,1)

education = st.sidebar.selectbox(
"Education",
["Graduate","Not Graduate"]
)

self_employed = st.sidebar.selectbox(
"Self Employed",
["No","Yes"]
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
"Credit Score",
300,
900,
650
)

st.sidebar.subheader("Assets")

residential_assets_value = st.sidebar.number_input("Residential Assets",0)
commercial_assets_value = st.sidebar.number_input("Commercial Assets",0)
luxury_assets_value = st.sidebar.number_input("Luxury Assets",0)
bank_asset_value = st.sidebar.number_input("Bank Assets",0)

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
total_assets = (
residential_assets_value +
commercial_assets_value +
luxury_assets_value +
bank_asset_value
)

loan_income_ratio = 0

if income_annum > 0:
    loan_income_ratio = loan_amount / income_annum

st.subheader("Financial Dashboard")

col1,col2,col3,col4 = st.columns(4)

col1.metric("Annual Income",f"{symbol}{income_annum:,.0f}")
col2.metric("Loan Requested",f"{symbol}{loan_amount:,.0f}")
col3.metric("Total Assets",f"{symbol}{total_assets:,.0f}")
col4.metric("Loan / Income Ratio",f"{loan_income_ratio:.2f}")

st.divider()

# =========================================
# EMI CALCULATOR
# =========================================
def calculate_emi(P,r,n):

    r = r/(12*100)

    emi = P*r*(1+r)**n / ((1+r)**n -1)

    return emi

if loan_amount>0:

    emi = calculate_emi(
        loan_amount,
        interest_rate,
        loan_term
    )

    st.subheader("Loan Repayment Estimate")

    st.metric("Monthly EMI",f"{symbol}{emi:,.2f}")

st.divider()

# =========================================
# PREDICTION
# =========================================
if predict_button:

    try:

        # Convert categorical features exactly like training
        education_val = 1 if education == "Graduate" else 0
        employed_val = 1 if self_employed == "Yes" else 0

        # Feature order must match training dataset
        input_data = [[
            no_of_dependents,
            education_val,
            employed_val,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value
        ]]

        prediction = model.predict(input_data)[0]

        prob=None

        if hasattr(model,"predict_proba"):
            prob=model.predict_proba(input_data)[0][1]

        st.subheader("Loan Decision")

        colA,colB=st.columns(2)

        with colA:

            if prediction==1:
                st.success("Loan Approved")

            else:
                st.error("Loan Rejected")

        with colB:

            if prob:
                st.metric("Approval Probability",f"{prob*100:.2f}%")

        # =========================================
        # FINANCIAL CHART
        # =========================================
        st.subheader("Financial Comparison")

        fig,ax=plt.subplots()

        categories=["Income","Loan","Assets"]
        values=[income_annum,loan_amount,total_assets]

        ax.bar(categories,values)

        st.pyplot(fig)

        # =========================================
        # ASSET DISTRIBUTION
        # =========================================
        st.subheader("Asset Portfolio")

        fig2,ax2=plt.subplots()

        assets=[
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
        ]

        labels=[
        "Residential",
        "Commercial",
        "Luxury",
        "Bank"
        ]

        ax2.pie(assets,labels=labels,autopct="%1.1f%%")

        st.pyplot(fig2)

    except Exception as e:

        st.error(f"Prediction Error: {e}")