import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards

# Netflix-style theme
st.set_page_config(
    page_title="ChurnFlix",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Netflix-inspired CSS
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #141414;
        color: #e5e5e5;
    }
    
    /* Netflix Red */
    :root {
        --netflix-red: #e50914;
        --netflix-dark: #141414;
        --netflix-light: #e5e5e5;
        --netflix-gray: #808080;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #141414;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #e50914;
        border-radius: 4px;
    }
    
    /* Buttons */
    .stButton>button {
        background: var(--netflix-red);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 4px;
        transition: all 0.3s;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        background: #f40612;
        transform: scale(1.05);
    }
    
    /* Cards */
    .card {
        background: #181818;
        border-radius: 6px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #333;
        transition: all 0.3s;
    }
    
    .card:hover {
        transform: scale(1.03);
        border-color: var(--netflix-red);
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #333 !important;
        color: #fff !important;
        border-radius: 4px !important;
        padding: 8px 16px !important;
        transition: all 0.3s !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--netflix-red) !important;
        color: white !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #000 !important;
    }
    
    /* Sliders */
    .stSlider .st-ae {
        color: var(--netflix-red) !important;
    }
    
    /* Select boxes */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #333;
        border-radius: 4px;
    }
    
    .stSelectbox div[data-baseweb="select"] div {
        color: white !important;
    }
    
    /* Number input */
    .stNumberInput input {
        background-color: #333 !important;
        color: white !important;
        border: 1px solid #555 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load('churn_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, model_columns

model, model_columns = load_model()

# Netflix-style header
st.markdown("""
<div style="background: linear-gradient(to bottom, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0) 100%), url('https://assets.nflxext.com/ffe/siteui/vlv3/9d3533b2-0e2b-40b2-95e0-ecd7979cc88b/9c9a7f0f-4c0a-4ce2-8c9a-3d3c3c3c3c3c/IN-en-20240311-popsignuptwoweeks-perspective_alpha_website_small.jpg');
            background-size: cover;
            height: 60vh;
            position: relative;
            margin-bottom: 40px;">
    <div style="position: absolute; top: 0; left: 0; right: 0; padding: 20px 60px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h1 style="color: var(--netflix-red); font-size: 3em; font-weight: 900; margin: 0;">CHURNFLIX</h1>
            <div style="display: flex; gap: 20px;">
                <button style="background: none; border: none; color: white; font-weight: bold; cursor: pointer;">Sign In</button>
                <button style="background: var(--netflix-red); color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; cursor: pointer;">Sign Up</button>
            </div>
        </div>
    </div>
    <div style="position: absolute; bottom: 20%; left: 60px; max-width: 40%;">
        <h1 style="font-size: 3.5em; margin: 0 0 20px 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">Customer Churn Prediction</h1>
        <p style="font-size: 1.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">Predict which customers are at risk of leaving and take action before it's too late.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Main content
st.markdown("<h2 style='color: white; margin-bottom: 20px;'>📊 Customer Analysis Dashboard</h2>", unsafe_allow_html=True)

# Two-column layout
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: white;'>👤 Customer Profile</h3>", unsafe_allow_html=True)
    
    # Using tabs for better organization
    tab1, tab2 = st.tabs(["Basic Info", "Services"])
    
    with tab1:
        st.markdown("<h4 style='color: white;'>Basic Information</h4>", unsafe_allow_html=True)
        tenure = st.slider('📅 Tenure (Months)', 0, 72, 12, help="How long the customer has been with the company")
        monthly_charges = st.number_input('💵 Monthly Charges ($)', min_value=0.0, value=65.0, step=5.0)
        total_charges = st.number_input('💰 Total Charges ($)', min_value=0.0, value=tenure * 65.0, step=10.0)
    
    with tab2:
        st.markdown("<h4 style='color: white;'>Service Details</h4>", unsafe_allow_html=True)
        contract = st.selectbox(
            '📝 Contract Type', 
            [0, 1, 2], 
            format_func=lambda x: ['Month-to-month', 'One year', 'Two year'][x],
            help="Longer contracts typically have lower churn rates"
        )
        tech_support = st.selectbox(
            '🛠️ Tech Support', 
            [0, 1], 
            format_func=lambda x: ['No', 'Yes'][x],
            help="Customers with tech support are less likely to churn"
        )
        online_security = st.selectbox(
            '🔒 Online Security', 
            [0, 1], 
            format_func=lambda x: ['No', 'Yes'][x],
            help="Additional security services can reduce churn"
        )
        fiber_optic = st.selectbox(
            '🌐 Internet Type', 
            [0, 1], 
            format_func=lambda x: ['Standard', 'Fiber Optic'][x],
            help="Fiber optic customers may have different churn patterns"
        )
    
    # Analyze button
    analyze_btn = st.button("🔍 Analyze Customer Risk", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    

# Prepare input data
input_data = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Contract': contract,
    'TechSupport': tech_support,
    'OnlineSecurity': online_security,
    'InternetService': fiber_optic
}

input_df = pd.DataFrame(input_data, index=[0])

# Align input with model training columns
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0 
input_df = input_df[model_columns]

with col2:
    st.markdown("<div class='card' style='min-height: 80vh;'>", unsafe_allow_html=True)
    
    if analyze_btn or 'calculated' in st.session_state:
        # Make prediction
        prediction_prob = model.predict_proba(input_df)
        churn_risk = prediction_prob[0][1]
        
        # Save to session state
        st.session_state['churn_risk'] = churn_risk
        st.session_state['calculated'] = True
        
        # Risk level card
        if churn_risk > 0.7:
            risk_color = "#e50914"
            risk_level = "High Risk"
            emoji = "🔴"
        elif churn_risk > 0.4:
            risk_color = "#e6b800"
            risk_level = "Medium Risk"
            emoji = "🟡"
        else:
            risk_color = "#4CAF50"
            risk_level = "Low Risk"
            emoji = "🟢"
        
        st.markdown(f"""
        <div style="background: #1a1a1a; padding: 20px; border-radius: 8px; border-left: 5px solid {risk_color}; margin-bottom: 30px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="margin: 0 0 5px 0; color: {risk_color};">{emoji} {risk_level}</h2>
                    <p style="color: var(--netflix-gray); margin: 0;">Churn Probability: <span style="color: white; font-weight: bold;">{churn_risk*100:.1f}%</span></p>
                </div>
                <div style="font-size: 2.5em;">{emoji}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = churn_risk * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Risk Score", 'font': {'size': 18, 'color': 'white'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': risk_color},
                'bgcolor': "#1a1a1a",
                'borderwidth': 2,
                'bordercolor': "#333",
                'steps': [
                    {'range': [0, 40], 'color': "#4CAF50"},
                    {'range': [40, 70], 'color': "#FFC107"},
                    {'range': [70, 100], 'color': "#e50914"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.75,
                    'value': churn_risk * 100
                }
            },
            number = {'font': {'size': 28, 'color': 'white'}},
            delta = {'reference': 50, 'increasing': {'color': "#e50914"}, 'decreasing': {'color': "#4CAF50"}}
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=60, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "white", 'family': "Arial"},
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("### 🎯 Recommended Actions")
        
        if churn_risk > 0.7:
            st.markdown("""
            <div style="background: rgba(229, 9, 20, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #e50914; margin-bottom: 20px;">
                <h4 style="margin-top: 0; color: #e50914;">🚨 Immediate Action Required</h4>
                <ul style="margin-bottom: 0;">
                    <li>Offer personalized retention deal (e.g., 30% off for 6 months)</li>
                    <li>Schedule 1:1 call with customer success manager</li>
                    <li>Consider free month or service upgrade</li>
                    <li>Send personalized email from account executive</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif churn_risk > 0.4:
            st.markdown("""
            <div style="background: rgba(255, 193, 7, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #FFC107; margin-bottom: 20px;">
                <h4 style="margin-top: 0; color: #FFC107;">⚠️ Monitor Closely</h4>
                <ul style="margin-bottom: 0;">
                    <li>Send satisfaction survey to understand concerns</li>
                    <li>Offer loyalty discount (e.g., 15% off for 3 months)</li>
                    <li>Check for service issues or complaints</li>
                    <li>Schedule check-in call</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(76, 175, 80, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; margin-bottom: 20px;">
                <h4 style="margin-top: 0; color: #4CAF50;">✅ Low Risk - Growth Opportunity</h4>
                <ul style="margin-bottom: 0;">
                    <li>Consider upsell opportunities to premium plans</li>
                    <li>Request referral or testimonial</li>
                    <li>Send appreciation message or small gift</li>
                    <li>Offer to be a beta tester for new features</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature importance
        st.markdown("### 🔍 Key Factors")
        
        # Mock feature importance (replace with actual SHAP values)
        features = [
            ("Contract Length", 0.32, "Longer contracts reduce churn"),
            ("Tenure", 0.28, "Loyal customers are less likely to leave"),
            ("Tech Support", 0.18, "Customers with support churn less"),
            ("Monthly Charges", 0.12, "Higher charges may increase churn"),
            ("Online Security", 0.10, "Security features reduce churn")
        ]
        
        for feature, value, desc in features:
            st.markdown(f"""
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: white;">{feature}</span>
                    <span style="color: var(--netflix-gray);">{value*100:.0f}%</span>
                </div>
                <div style="height: 6px; background: #333; border-radius: 3px; overflow: hidden;">
                    <div style="width: {value*100}%; height: 6px; background: {risk_color}; border-radius: 3px;"></div>
                </div>
                <div style="color: var(--netflix-gray); font-size: 0.85em; margin-top: 3px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 60vh; text-align: center; color: var(--netflix-gray);">
            <div style="font-size: 5em; margin-bottom: 20px;">🔍</div>
            <h2>Ready to Analyze</h2>
            <p>Fill in the customer details on the left and click 'Analyze Customer Risk' to see the prediction.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="margin-top: 60px; padding: 20px 0; border-top: 1px solid #333; color: var(--netflix-gray);">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>© 2023 ChurnFlix - All Rights Reserved</div>
        <div style="display: flex; gap: 15px;">
            <a href="#" style="color: var(--netflix-gray); text-decoration: none;">Privacy</a>
            <a href="#" style="color: var(--netflix-gray); text-decoration: none;">Terms</a>
            <a href="#" style="color: var(--netflix-gray); text-decoration: none;">Help Center</a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
