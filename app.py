========================================= 

IMPORT LIBRARIES 

========================================= 

import streamlit as st import pandas as pd import joblib import matplotlib.pyplot as plt import numpy as np from pathlib import Path 

st.cache_resource.clear() 

========================================= 

PAGE CONFIG 

========================================= 

st.set_page_config( page_title="AI Loan Approval System", page_icon="💳", layout="wide" ) 

========================================= 

CUSTOM STYLE 

========================================= 

st.markdown(""" 

""", unsafe_allow_html=True) 

========================================= 

LOAD MODEL 

========================================= 

@st.cache_resource def load_model(): path = Path("loan_model.pkl") 

if path.exists(): 
   return joblib.load(path) 
 
else: 
   st.error("loan_model.pkl not found") 
   return None 
 

model = load_model() 

========================================= 

HEADER 

========================================= 

st.markdown(' 

💳 AI Loan Approval System 

', unsafe_allow_html=True) st.markdown(' 

Machine Learning Powered Credit Risk Evaluation 

', unsafe_allow_html=True) 

st.divider() 

========================================= 

SIDEBAR APPLICATION FORM 

========================================= 

st.sidebar.header("Loan Application") 

no_of_dependents = st.sidebar.slider("Dependents", 0, 10, 1) 

education = st.sidebar.selectbox( "Education", ["Graduate","Not Graduate"] ) 

self_employed = st.sidebar.selectbox( "Self Employed", ["No","Yes"] ) 

income_annum = st.sidebar.number_input( "Annual Income", min_value=0 ) 

loan_amount = st.sidebar.number_input( "Loan Amount Requested", min_value=0 ) 

loan_term = st.sidebar.slider( "Loan Term (Months)", 6, 360, 12 ) 

cibil_score = st.sidebar.slider( "Cibil Score", 300, 900, 650 ) 

st.sidebar.subheader("Assets") 

residential_assets_value = st.sidebar.number_input("Residential Assets",0) commercial_assets_value = st.sidebar.number_input("Commercial Assets",0) luxury_assets_value = st.sidebar.number_input("Luxury Assets",0) bank_asset_value = st.sidebar.number_input("Bank Assets",0) 

interest_rate = st.sidebar.slider( "Interest Rate (%)", 1.0, 20.0, 8.5 ) 

predict_button = st.sidebar.button("Evaluate Loan") 

========================================= 

FINANCIAL SUMMARY 

========================================= 

st.subheader("Applicant Financial Summary") 

total_assets = ( residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value ) 

loan_income_ratio = 0 

if income_annum > 0: loan_income_ratio = loan_amount / income_annum 

col1,col2,col3,col4 = st.columns(4) 

col1.metric("Annual Income",f"£{income_annum:,.0f}") col2.metric("Loan Requested",f"£{loan_amount:,.0f}") col3.metric("Total Assets",f"₹{total_assets:,.0f}") col4.metric("Loan/Income Ratio",f"{loan_income_ratio:.2f}") 

st.divider() 

========================================= 

EMI CALCULATOR 

========================================= 

def calculate_emi(P,r,n): 

r = r / (12*100) 
 
emi = P * r * (1+r)**n / ((1+r)**n -1) 
 
return emi 
 

if loan_amount > 0: 

emi = calculate_emi( 
   loan_amount, 
   interest_rate, 
   loan_term 
) 
 
st.subheader("Estimated Monthly Repayment") 
 
st.metric("Monthly EMI",f"£{emi:,.2f}") 
 

st.divider() 

========================================= 

PREDICTION 

========================================= 

if predict_button: 

if model is None: 
 
   st.error("Model not loaded") 
 
else: 
 
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
 
       st.divider() 
 
       # ========================================= 
       # FINANCIAL VISUALIZATION 
       # ========================================= 
       st.subheader("Financial Analysis") 
 
       fig,ax=plt.subplots() 
 
       categories=["Income","Loan","Assets"] 
 
       values=[ 
       income_annum, 
       loan_amount, 
       total_assets 
       ] 
 
       ax.bar(categories,values) 
 
       ax.set_title("Financial Comparison") 
 
       st.pyplot(fig) 
 
       plt.close() 
 
       # ========================================= 
       # ASSET DISTRIBUTION 
       # ========================================= 
       st.subheader("Asset Distribution") 
 
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
       # CREDIT RISK INTERPRETATION 
       # ========================================= 
       st.subheader("Credit Risk Assessment") 
 
       if cibil_score >=750: 
 
           st.success("Low Credit Risk") 
 
       elif cibil_score >=650: 
 
           st.warning("Moderate Credit Risk") 
 
       else: 
 
           st.error("High Credit Risk") 
 
       # ========================================= 
       # LOAN AFFORDABILITY 
       # ========================================= 
       st.subheader("Loan Affordability") 
 
       if loan_income_ratio <0.3: 
 
           st.success("Loan is financially safe") 
 
       elif loan_income_ratio <0.5: 
 
           st.warning("Loan is moderate risk") 
 
       else: 
 
           st.error("Loan may be difficult to repay") 
 
       # ========================================= 
       # APPLICANT SUMMARY 
       # ========================================= 
       st.subheader("Application Summary") 
 
       st.write(f""" 
 

Applicant Financial Summary 

Annual Income: ₹{income_annum:,.0f} 

Loan Requested: ₹{loan_amount:,.0f} 

Total Assets: ₹{total_assets:,.0f} 

Cibil Score: {cibil_score} 

Estimated Monthly Payment: ₹{emi:,.2f} """) 

  except Exception as e: 
 
       st.error(f"Prediction Error: {e}") 
 