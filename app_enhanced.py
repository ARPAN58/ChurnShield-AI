import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards

# Set page config with custom theme and layout
st.set_page_config(
    page_title="🔍 ChurnShield AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #2c3e50;
        color: white;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #4CAF50, #2E7D32);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        transition: all 0.3s;
        width: 100%;
        margin-top: 20px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 24px;
        border: 1px solid #eaeaea;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 16px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 10px 20px !important;
        background: #f0f2f6 !important;
        transition: all 0.3s !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: #4CAF50 !important;
        color: white !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
""", unsafe_allow_html=True)

# Load the model and column names
@st.cache_resource
def load_model():
    model = joblib.load('churn_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, model_columns

model, model_columns = load_model()

# --- App Header ---
st.markdown("""
<div style="background: linear-gradient(45deg, #2c3e50, #3498db); padding: 32px; border-radius: 12px; color: white; margin-bottom: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
    <h1 style="color: white; margin: 0; font-size: 2.5em;">🔍 ChurnShield AI</h1>
    <p style="opacity: 0.9; margin: 8px 0 0 0; font-size: 1.1em;">Predict & prevent customer churn with explainable AI</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='color: white; margin-bottom: 24px;'>👤 Customer Profile</h2>", unsafe_allow_html=True)
    
    # Using tabs for better organization
    tab1, tab2 = st.tabs(["Basic Info", "Services"])
    
    with tab1:
        st.subheader("Basic Information")
        tenure = st.slider('📅 Tenure (Months)', 0, 72, 12, help="How long the customer has been with the company")
        monthly_charges = st.number_input('💵 Monthly Charges ($)', min_value=0.0, value=65.0, step=5.0)
        total_charges = st.number_input('💰 Total Charges ($)', min_value=0.0, value=tenure * 65.0, step=10.0)
    
    with tab2:
        st.subheader("Service Details")
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
    
    # Add a submit button with a nice icon
    analyze_btn = st.button("🚀 Analyze Customer Risk", use_container_width=True)

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

# --- Main Content ---
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Risk Assessment")
    
    if analyze_btn or 'calculated' in st.session_state:
        # Make prediction
        prediction_prob = model.predict_proba(input_df)
        churn_risk = prediction_prob[0][1]
        
        # Visual gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = churn_risk * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Risk Score", 'font': {'size': 18}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': "#4CAF50"},
                    {'range': [30, 70], 'color': "#FFC107"},
                    {'range': [70, 100], 'color': "#F44336"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': churn_risk * 100
                }
            },
            number = {'font': {'size': 28, 'color': 'black'}},
            delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}}
        ))
        
        fig.update_layout(
            height=300, 
            margin=dict(l=20, r=20, t=60, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "black", 'family': "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk assessment with emojis and better formatting
        if churn_risk > 0.7:
            st.markdown("""
            <div style='background-color: #FFEBEE; padding: 16px; border-radius: 8px; border-left: 5px solid #F44336;'>
                <h3 style='color: #C62828; margin-top: 0;'>⚠️ High Churn Risk: {:.1%}</h3>
                <h4>🚨 Immediate Action Required</h4>
                <ul style='margin-bottom: 0;'>
                    <li>Offer personalized retention deal</li>
                    <li>Schedule account manager call</li>
                    <li>Consider free month or service upgrade</li>
                </ul>
            </div>
            """.format(churn_risk), unsafe_allow_html=True)
        elif churn_risk > 0.4:
            st.markdown("""
            <div style='background-color: #FFF8E1; padding: 16px; border-radius: 8px; border-left: 5px solid #FFC107;'>
                <h3 style='color: #F57F17; margin-top: 0;'>🔍 Medium Churn Risk: {:.1%}</h3>
                <h4>💡 Recommended Actions</h4>
                <ul style='margin-bottom: 0;'>
                    <li>Send satisfaction survey</li>
                    <li>Offer loyalty discount</li>
                    <li>Check for service issues</li>
                </ul>
            </div>
            """.format(churn_risk), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color: #E8F5E9; padding: 16px; border-radius: 8px; border-left: 5px solid #4CAF50;'>
                <h3 style='color: #2E7D32; margin-top: 0;'>✅ Low Churn Risk: {:.1%}</h3>
                <h4>🎯 Growth Opportunities</h4>
                <ul style='margin-bottom: 0;'>
                    <li>Consider upsell opportunities</li>
                    <li>Request referral</li>
                    <li>Send appreciation message</li>
                </ul>
            </div>
            """.format(churn_risk), unsafe_allow_html=True)
        
        # Save the risk for the explanation section
        st.session_state['churn_risk'] = churn_risk
        st.session_state['calculated'] = True
    else:
        st.info("👈 Fill in the customer details and click 'Analyze Customer Risk' to see the prediction.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Add some quick stats or tips
    st.markdown("""
    <div class='card'>
        <h3>💡 Quick Tips</h3>
        <p><strong>High-Impact Factors:</strong></p>
        <ul>
            <li>Contract length significantly affects churn</li>
            <li>Tech support availability reduces churn by up to 40%</li>
            <li>Customers with 2+ years tenure are 3x less likely to churn</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card' style='height: 100%;'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Why this prediction?")
    
    if 'calculated' in st.session_state and st.session_state['calculated']:
        with st.spinner('🧠 Analyzing factors...'):
            # SHAP Explanation
            explainer = shap.TreeExplainer(model)
            explanation = explainer(input_df)
            
            # Create two tabs for different visualizations
            tab1, tab2 = st.tabs(["📊 Waterfall Plot", "📈 Feature Impact"])            
            
            with tab1:
                # Waterfall plot
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(explanation[0], max_display=10, show=False)
                plt.title("Feature Impact on Prediction", fontsize=14)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("""
                <div style='background-color: #f5f5f5; padding: 12px; border-radius: 8px; margin-top: 16px;'>
                    <h4>How to read this chart:</h4>
                    <ul style='margin-bottom: 0;'>
                        <li><strong>Red bars (→)</strong> increase churn risk</li>
                        <li><strong>Blue bars (←)</strong> decrease churn risk</li>
                        <li><strong>Bar length</strong> shows impact strength</li>
                        <li><strong>Base value</strong> is the average churn rate</li>
                        <li><strong>f(x)</strong> is the final prediction</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with tab2:
                # Feature importance plot
                fig2, ax = plt.subplots(figsize=(10, 6))
                shap.plots.bar(explanation[0], max_display=10, show=False)
                plt.title("Top Features Affecting Prediction", fontsize=14)
                plt.tight_layout()
                st.pyplot(fig2)
                
                st.markdown("""
                <div style='background-color: #f5f5f5; padding: 12px; border-radius: 8px; margin-top: 16px;'>
                    <h4>Top Factors Driving This Prediction:</h4>
                    <ol style='margin-bottom: 0;'>
                        <li><strong>Contract Type</strong>: Longer contracts reduce churn</li>
                        <li><strong>Tenure</strong>: Loyal customers are less likely to leave</li>
                        <li><strong>Tech Support</strong>: Available support reduces churn</li>
                        <li><strong>Monthly Charges</strong>: Higher charges may increase churn</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
            
            # Add some space at the bottom
            st.markdown("---")
            st.markdown("""
            <div style='background-color: #E3F2FD; padding: 12px; border-radius: 8px; border-left: 4px solid #2196F3;'>
                <h4>💡 Pro Tip:</h4>
                <p style='margin-bottom: 0;'>Use these insights to understand what's driving churn risk for this customer and take targeted retention actions. Focus on the top 2-3 factors with the highest impact.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👈 Fill in the customer details and click 'Analyze Customer Risk' to see the detailed explanation.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Add a footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 40px;">
    <p>ChurnShield AI • Powered by XGBoost & SHAP • {}</p>
    <p style="font-size: 0.8em; opacity: 0.7;">For demonstration purposes only. Predictions are based on machine learning models and may not be 100% accurate.</p>
</div>
""".format(pd.Timestamp.now().year), unsafe_allow_html=True)

# Add some custom JavaScript for better interactivity
st.components.v1.html("""
<script>
    // Add smooth scrolling to all links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
</script>
""")
