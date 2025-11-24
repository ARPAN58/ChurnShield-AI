import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# 1. Load the model and column names
model = joblib.load('churn_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="ChurnShield AI", layout="wide")

st.title("📊 ChurnShield: Explainable Customer Retention")
st.markdown("""
This tool predicts if a customer will leave (churn) and **explains why** using SHAP values.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Customer Profile")

def user_input_features():
    # We'll collect key features that usually impact churn the most
    tenure = st.sidebar.slider('Tenure (Months)', 0, 72, 12)
    monthly_charges = st.sidebar.number_input('Monthly Charges ($)', min_value=0.0, value=65.0)
    total_charges = st.sidebar.number_input('Total Charges ($)', min_value=0.0, value=tenure * 65.0)
    
    contract = st.sidebar.selectbox('Contract Type', [0, 1, 2], format_func=lambda x: ['Month-to-month', 'One year', 'Two year'][x])
    tech_support = st.sidebar.selectbox('Tech Support?', [0, 1], format_func=lambda x: ['No', 'Yes'][x])
    online_security = st.sidebar.selectbox('Online Security?', [0, 1], format_func=lambda x: ['No', 'Yes'][x])
    fiber_optic = st.sidebar.selectbox('Fiber Optic Internet?', [0, 1], format_func=lambda x: ['No', 'Yes'][x])
    
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'TechSupport': tech_support,
        'OnlineSecurity': online_security,
        'InternetService': fiber_optic # Assuming this maps to a specific encoded column
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Preprocessing ---
# Align input with model training columns
# We fill missing columns with 0 (representing 'No' or 'Average' depending on encoding)
# In a real production app, you would ask for all inputs.
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0 

# Reorder columns to match the training order strictly
input_df = input_df[model_columns]

# --- Main Section ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Prediction")
    if st.button('Analyze Customer Risk'):
        # Predict
        prediction_prob = model.predict_proba(input_df)
        churn_risk = prediction_prob[0][1]
        
        # Visual Gauge
        if churn_risk > 0.5:
            st.error(f"High Churn Risk: {churn_risk:.1%}")
            st.markdown("**Action:** 🚨 Immediate Retention Offer Required")
        else:
            st.success(f"Low Churn Risk: {churn_risk:.1%}")
            st.markdown("**Action:** ✅ Monitor Normally")

        # Save the risk for the explanation section
        st.session_state['churn_risk'] = churn_risk
        st.session_state['calculated'] = True

with col2:
    st.subheader("Why this prediction? (Explainable AI)")
    
    if 'calculated' in st.session_state and st.session_state['calculated']:
        with st.spinner('Calculating SHAP values...'):
            
            # --- SHAP EXPLANATION CORE ---
            # 1. Initialize the explainer with the XGBoost model
            explainer = shap.TreeExplainer(model)
            
            # 2. Calculate SHAP values for this specific instance
            shap_values = explainer.shap_values(input_df)
            
            # 3. Create the Waterfall Plot
            # We use explainer(input_df) to get an Explanation object for the waterfall plot
            explanation = explainer(input_df)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            # The waterfall plot shows how each feature pushes the probability from the base value
            shap.plots.waterfall(explanation[0], show=False)
            
            # Display in Streamlit
            st.pyplot(fig)
            
            st.info("""
            **How to read this chart:**
            * **Red bars (→)** push the risk **HIGHER**.
            * **Blue bars (←)** push the risk **LOWER**.
            * The length of the bar is the strength of the impact.
            """)
    else:
        st.write("Click 'Analyze Customer Risk' to see the breakdown.")
