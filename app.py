# =========================================
# IMPORT LIBRARIES
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
    page_title="AI Loan Approval System",
    page_icon="💳",
    layout="wide"
)

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
# CURRENCY SELECTOR
# =========================================
currency_options = {
"Indian Rupee": "₹",
"British Pound": "£",
"US Dollar": "$",
"Euro": "€"
}

currency = st.sidebar.selectbox(
"Select Currency",
list(currency_options.keys())
)

symbol = currency_options[currency]

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

        input_df = pd.DataFrame([{

        "no_of_dependents":no_of_dependents,
        "education":education,
        "self_employed":self_employed,
        "income_annum":income_annum,
        "loan_amount":loan_amount,
        "loan_term":loan_term,
        "cibil_score":cibil_score,
        "residential_assets_value":residential_assets_value,
        "commercial_assets_value":commercial_assets_value,
        "luxury_assets_value":luxury_assets_value,
        "bank_asset_value":bank_asset_value

        }])

        prediction = model.predict(input_df)[0]

        probability=None

        if hasattr(model,"predict_proba"):
            probability=model.predict_proba(input_df)[0][1]

        st.subheader("Loan Decision")

        colA,colB=st.columns(2)

        with colA:

            if prediction==1 or prediction=="Approved":

                st.success("Loan Approved")

            else:

                st.error("Loan Rejected")

        with colB:

            if probability:

                st.metric(
                "Approval Probability",
                f"{probability*100:.2f}%"
                )

        # =========================================
        # LOAN ELIGIBILITY SCORE
        # =========================================
        score = 0

        if cibil_score>750:
            score+=40
        elif cibil_score>650:
            score+=25

        if loan_income_ratio<0.3:
            score+=30
        elif loan_income_ratio<0.5:
            score+=15

        if total_assets>loan_amount:
            score+=30
        elif total_assets>loan_amount*0.5:
            score+=15

        st.subheader("Loan Eligibility Score")

        st.progress(score)

        st.write(f"Eligibility Score: **{score}/100**")

        st.divider()

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

        ax2.pie(
        assets,
        labels=labels,
        autopct="%1.1f%%"
        )

        st.pyplot(fig2)

        # =========================================
        # RISK INTERPRETATION
        # =========================================
        st.subheader("Credit Risk Analysis")

        if cibil_score>=750:
            st.success("Low Risk Borrower")

        elif cibil_score>=650:
            st.warning("Moderate Risk Borrower")

        else:
            st.error("High Risk Borrower")

        # =========================================
        # DECISION EXPLANATION
        # =========================================
        st.subheader("Decision Explanation")

        reasons=[]

        if cibil_score>=700:
            reasons.append("Strong credit score")

        if loan_income_ratio<0.4:
            reasons.append("Healthy income compared to loan")

        if total_assets>loan_amount:
            reasons.append("Strong asset backing")

        if len(reasons)==0:
            reasons.append("High financial risk factors")

        for r in reasons:
            st.write("•",r)

    except Exception as e:

        st.error(f"Prediction Error: {e}")