"""
🫀 Heart Disease Risk Assessment System
Professional Medical Dashboard
Created by: Partha Sarathi
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# =============================================
# PAGE CONFIGURATION
# =============================================
st.set_page_config(
    page_title="Cardiovascular Risk Assessment | Partha Sarathi",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# PROFESSIONAL CSS STYLING
# =============================================
st.markdown("""
<style>
    /* Global Settings */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }
    
    /* Backgrounds */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    
    /* Custom Header */
    .dashboard-header {
        background-color: #ffffff;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.01), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .header-title {
        font-size: 1.875rem;
        color: #0f172a;
        margin: 0;
    }
    
    .header-subtitle {
        color: #64748b;
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }
    
    .creator-tag {
        background-color: #f1f5f9;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        color: #475569;
        font-size: 0.75rem;
        font-weight: 500;
        border: 1px solid #e2e8f0;
    }
    
    /* Input Cards */
    .input-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        height: 100%;
    }
    
    .section-title {
        color: #3b82f6; /* Professional Blue */
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 0.5rem;
    }
    
    /* Custom Button */
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        padding: 0.6rem 2rem;
        font-weight: 500;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.1), 0 2px 4px -1px rgba(37, 99, 235, 0.06);
        transition: all 0.2s;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.2);
    }
    
    /* Result Section */
    .report-container {
        border-top: 4px solid #3b82f6;
        background-color: white;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .risk-label {
        font-size: 0.875rem;
        color: #475569; /* Darker gray for better visibility */
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    .risk-value-low { color: #059669; font-size: 2.5rem; font-weight: 700; }
    .risk-value-med { color: #d97706; font-size: 2.5rem; font-weight: 700; }
    .risk-value-high { color: #dc2626; font-size: 2.5rem; font-weight: 700; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #0f172a;
    }
    
    /* Adjust Sliders */
    div[class*="stSlider"] > label {
        color: #475569;
        font-weight: 500;
        font-size: 0.875rem;
    }
    
    /* Footer */
    .footer {
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
        padding-top: 1.5rem;
        text-align: center;
        color: #94a3b8;
        font-size: 0.875rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        color: #64748b;
        padding-bottom: 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        color: #2563eb;
        border-bottom: 2px solid #2563eb;
    }
    
</style>
""", unsafe_allow_html=True)

# =============================================
# LOAD ASSETS
# =============================================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('heart_disease_scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# =============================================
# HEADER
# =============================================
st.markdown("""
<div class="dashboard-header">
    <div>
        <h1 class="header-title">Cardiovascular Risk Assessment</h1>
        <p class="header-subtitle">Framingham Heart Study Predictive Model • Logistic Regression</p>
    </div>
    <div class="creator-tag">
        Project by Partha Sarathi
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================
# SIDEBAR
# =============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=60)
    st.markdown("### Clinical Dashboard")
    st.markdown("Enter patient vitals and demographics to generate a 10-year CHD risk profile.")
    
    st.markdown("---")
    st.markdown("##### ⚙️ Model Specs")
    st.caption("Algorithm: Logistic Regression")
    st.caption("Training Accuracy: 86.60%")
    st.caption("Validation: 1060 samples")
    
    st.markdown("---")
    st.info("ℹ️ This system is a support tool for the College Project 2026. Not for clinical use.")

# =============================================
# MAIN INTERFACE
# =============================================

if model:
    # Use Tabs for Organization
    tab_predict, tab_stats, tab_info = st.tabs(["Patient Evaluation", "Model Performance", "Study Details"])
    
    with tab_predict:
        # Input Section
        with st.container():
            col_left, col_mid, col_right = st.columns(3)
            
            # --- COLUMN 1: DEMOGRAPHICS & LIFESTYLE ---
            with col_left:
                st.markdown('<div class="input-card"><div class="section-title">01. Patient Profile</div>', unsafe_allow_html=True)
                
                gender = st.selectbox("Gender", ["Male", "Female"])
                male = 1 if gender == "Male" else 0
                
                age = st.number_input("Age", 30, 80, 50, help="Patient age in years")
                
                smoker_status = st.radio("Smoking History", ["Non-Smoker", "Current Smoker"], horizontal=True)
                currentSmoker = 1 if smoker_status == "Current Smoker" else 0
                
                if currentSmoker:
                    cigsPerDay = st.slider("Cigarettes / Day", 1, 70, 10)
                else:
                    cigsPerDay = 0
                
                st.markdown('</div>', unsafe_allow_html=True)

            # --- COLUMN 2: CLINICAL VITALS ---
            with col_mid:
                st.markdown('<div class="input-card"><div class="section-title">02. Clinical Vitals</div>', unsafe_allow_html=True)
                
                sysBP = st.number_input("Systolic BP (mmHg)", 80, 250, 120)
                diaBP = st.number_input("Diastolic BP (mmHg)", 40, 140, 80)
                
                bmi = st.slider("BMI (kg/m²)", 15.0, 50.0, 25.0, format="%.1f")
                heartRate = st.slider("Resting Heart Rate (bpm)", 40, 120, 72)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            # --- COLUMN 3: LAB RESULTS ---
            with col_right:
                st.markdown('<div class="input-card"><div class="section-title">03. Lab Results & History</div>', unsafe_allow_html=True)
                
                totChol = st.number_input("Total Cholesterol (mg/dL)", 100, 600, 200)
                glucose = st.number_input("Glucose (mg/dL)", 40, 400, 85)
                
                st.markdown('<div class="section-title" style="margin-top:1rem;">04. Medical History</div>', unsafe_allow_html=True)
                
                h_col1, h_col2 = st.columns(2)
                with h_col1:
                    bp_meds = st.checkbox("BP Meds")
                    diabetes_hist = st.checkbox("Diabetes")
                with h_col2:
                    stroke_hist = st.checkbox("Prior Stroke")
                    hyp_hist = st.checkbox("Hypertension")
                
                # Convert booleans to int
                BPMeds = 1 if bp_meds else 0
                diabetes = 1 if diabetes_hist else 0
                prevalentStroke = 1 if stroke_hist else 0
                prevalentHyp = 1 if hyp_hist else 0
                
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # PREDICTION ACTION
        col_space1, col_action, col_space2 = st.columns([1, 2, 1])
        with col_action:
            predict_btn = st.button("Generate Risk Assessment Report")

        # LOGIC
        if predict_btn:
            input_df = pd.DataFrame({
                'male': [male], 'age': [age], 'currentSmoker': [currentSmoker], 
                'cigsPerDay': [cigsPerDay], 'BPMeds': [BPMeds], 'prevalentStroke': [prevalentStroke],
                'prevalentHyp': [prevalentHyp], 'diabetes': [diabetes], 'totChol': [totChol],
                'sysBP': [sysBP], 'diaBP': [diaBP], 'BMI': [bmi], 'heartRate': [heartRate],
                'glucose': [glucose]
            })
            
            scaled_input = scaler.transform(input_df)
            prob = model.predict_proba(scaled_input)[0][1] * 100
            
            # --- RESULTS SECTION ---
            st.markdown("---")
            st.markdown("### Assessment Report")
            
            r_col1, r_col2 = st.columns([1, 2])
            
            with r_col1:
                # Determine Styling
                if prob < 20:
                    status_color = "#059669" # Green
                    status_text = "LOW RISK"
                    rec_text = "Patient falls within the low-risk category. Maintain healthy lifestyle."
                elif prob < 50:
                    status_color = "#d97706" # Amber
                    status_text = "ELEVATED RISK"
                    rec_text = "Moderate risk factors identified. Lifestyle modification recommended."
                else:
                    status_color = "#dc2626" # Red
                    status_text = "HIGH RISK"
                    rec_text = "Significant risk factors present. Clinical intervention may be required."
                    
                st.markdown(f"""
                <div class="report-container" style="border-top-color: {status_color};">
                    <p class="risk-label">10-Year CHD Probability</p>
                    <p style="color: {status_color}; font-size: 3.5rem; font-weight: 700; margin: 0;">{prob:.1f}%</p>
                    <p style="color: {status_color}; font-weight: 600; font-size: 1.2rem; margin-top: 0;">{status_text}</p>
                    <p style="color: #64748b; margin-top: 1rem; font-size: 0.95rem;">{rec_text}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with r_col2:
                # Professional Gauge Plot
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    number={'font': {'color': "#0f172a"}}, # Force dark text for number
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#334155", 'tickfont': {'color': "#334155"}}, # Dark ticks
                        'bar': {'color': status_color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#cbd5e1",
                        'steps': [
                            {'range': [0, 20], 'color': "#ecfdf5"}, # Light Green
                            {'range': [20, 50], 'color': "#fffbeb"}, # Light Amber
                            {'range': [50, 100], 'color': "#fef2f2"}  # Light Red
                        ],
                    }
                ))
                fig.update_layout(
                    height=300, 
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'family': "Inter", 'color': "#0f172a"} # Global chart font color dark
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab_stats:
        st.markdown("### Model Performance Metrics")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Overall Accuracy", "86.60%", "+1.2%")
        m_col2.metric("ROC-AUC Score", "0.728", "-0.01")
        m_col3.metric("Dataset Size", "4,238", "Patients")
        
        st.markdown("### Feature Importance Analysis")
        # Reuse feature data
        feature_importance = pd.DataFrame({
            'Factor': ['Age', 'Cigarettes/Day', 'Systolic BP', 'Gender (Male)', 'Glucose'],
            'Coefficient': [0.545, 0.290, 0.290, 0.220, 0.188]
        })
        
        fig_bar = px.bar(feature_importance, x='Coefficient', y='Factor', orientation='h',
                        title="Top 5 Predictive Risk Factors", text_auto=True)
        fig_bar.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font={'color': '#0f172a', 'family': 'Inter'}, # Force dark font
            xaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
            height=300
        )
        fig_bar.update_traces(marker_color='#3b82f6')
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_info:
        st.markdown("### About the Framingham Heart Study")
        st.write("""
        The Framingham Heart Study is a long-term, ongoing cardiovascular cohort study of residents of the city of Framingham, Massachusetts. 
        The study began in 1948 with 5,209 adult subjects from Framingham, and is now on its third generation of participants.
        """)
        
        st.info("Project created by Partha Sarathi for Data Science coursework.")

else:
    st.error("Model file/scaler not found. Please verify the file paths.")

# =============================================
# FOOTER
# =============================================
st.markdown("""
<div class="footer">
    © 2026 Partha Sarathi • Framingham Heart Study Model • v1.0.0
</div>
""", unsafe_allow_html=True)
