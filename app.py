import streamlit as st
import numpy as np
import pickle

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Loan Approval Dashboard",
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
        💳 Loan Approval Prediction Dashboard
    </h1>
    <p style='text-align: center;'>
        Enter applicant details to predict loan approval status
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
# PREDICTION
# -------------------------------
st.divider()

if st.button("🔍 Predict Loan Status", use_container_width=True):

    if model is None:
        st.error("Model not loaded properly.")
    else:
        try:
            # Feature array (⚠️ must match training order)
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

            st.divider()

            # -------------------------------
            # RESULT DISPLAY
            # -------------------------------
            if prediction[0] == 1:
                st.success("✅ Loan Approved")
                st.balloons()
            else:
                st.error("❌ Loan Rejected")

        except Exception as e:
            st.error(f"Prediction error: {e}")