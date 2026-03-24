# =========================================
# PREDICTION
# =========================================
if predict_button:

    # ✅ STRICT VALIDATION
    if income_annum <= 0 or loan_amount <= 0:
        st.warning("⚠️ Please enter valid Income & Loan Amount")
        st.stop()

    if total_assets == 0:
        st.warning("⚠️ Please enter asset details")
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

        # 🔥 MODEL PREDICTION
        prob = model.predict_proba(input_data)[0][1]

        # =========================================
        # 🔥 RULE-BASED CORRECTION (IMPORTANT FIX)
        # =========================================
        reject_flag = False

        if income_annum < 100000:
            reject_flag = True

        if loan_income_ratio > 0.6:
            reject_flag = True

        if cibil_score < 600:
            reject_flag = True

        if total_assets < loan_amount * 0.2:
            reject_flag = True

        # =========================================
        # FINAL DECISION (MODEL + RULES)
        # =========================================
        if prob >= 0.6 and not reject_flag:
            final_decision = 1
        else:
            final_decision = 0

        st.subheader("🏦 Loan Decision")

        if final_decision == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        st.metric("Approval Probability", f"{prob*100:.2f}%")

        # =========================================
        # INSIGHTS
        # =========================================
        st.subheader("🧠 Decision Insights")

        if prob > 0.8:
            st.success("Very strong approval likelihood")
        elif prob > 0.6:
            st.info("Good chances, but depends on bank policies")
        elif prob > 0.4:
            st.warning("Borderline case")
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

        # =========================================
        # CHARTS
        # =========================================
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