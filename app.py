import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time
import re
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
# Note: Install reportlab for PDF export feature using: pip install reportlab

# ----------------------------
# Custom Class: FeatureEngineer
# ----------------------------
class FeatureEngineer:
    def __init__(self):
        self.mean_amount = None
        self.std_amount = None
    
    def fit(self, X, y=None):
        if 'Amount' in X.columns:
            self.mean_amount = X['Amount'].mean()
            self.std_amount = X['Amount'].std()
        return self
    
    def transform(self, X):
        X_new = X.copy()
        if 'Time' in X_new.columns:
            X_new['Hour'] = X_new['Time'] // 3600 % 24
            X_new['Hour_sin'] = np.sin(2 * np.pi * X_new['Hour'] / 24)
            X_new['Hour_cos'] = np.cos(2 * np.pi * X_new['Hour'] / 24)
            X_new['IsMorning'] = ((X_new['Hour'] >= 6) & (X_new['Hour'] < 12)).astype(int)
            X_new['IsAfternoon'] = ((X_new['Hour'] >= 12) & (X_new['Hour'] < 18)).astype(int)
            X_new['IsEvening'] = ((X_new['Hour'] >= 18) & (X_new['Hour'] < 22)).astype(int)
            X_new['IsNight'] = ((X_new['Hour'] >= 22) | (X_new['Hour'] < 6)).astype(int)
            X_new = X_new.drop(['Time', 'Hour'], axis=1)
        if 'Amount' in X_new.columns:
            X_new['Amount_Log'] = np.log1p(X_new['Amount'])
            X_new['Amount_Zscore'] = (X_new['Amount'] - self.mean_amount) / self.std_amount
            X_new['IsSmallTxn'] = (X_new['Amount'] <= 5).astype(int)
            X_new['IsMediumTxn'] = ((X_new['Amount'] > 5) & (X_new['Amount'] <= 100)).astype(int)
            X_new['IsLargeTxn'] = (X_new['Amount'] > 100).astype(int)
            X_new = X_new.drop(['Amount'], axis=1)
        v_columns = [col for col in X_new.columns if col.startswith('V')]
        if v_columns:
            X_new['V_Sum'] = X_new[v_columns].sum(axis=1)
            X_new['V_Mean'] = X_new[v_columns].mean(axis=1)
            X_new['V_Std'] = X_new[v_columns].std(axis=1)
            X_new['V_Kurtosis'] = X_new[v_columns].kurtosis(axis=1)
            X_new['V_Skew'] = X_new[v_columns].skew(axis=1)
            if all(col in X_new.columns for col in ['V1', 'V3', 'V4', 'V10', 'V11']):
                X_new['V1_to_V3'] = X_new['V1'] / (X_new['V3'] + 1e-8)
                X_new['V4_to_V10'] = X_new['V4'] / (X_new['V10'] + 1e-8)
                X_new['V11_to_V4'] = X_new['V11'] / (X_new['V4'] + 1e-8)
        return X_new

# ----------------------------
# Theme Styles (Global)
# ----------------------------
theme_styles = {
    "cyberpunk": {
        "bg": "#0A0F1E",
        "card_bg": "#0A0F1E/80",
        "text_color": "#E0FFFF",
        "accent": "#FF2E63",
        "font": "'Orbitron', sans-serif",
        "bg_effect": """
            background: linear-gradient(transparent 50%, rgba(0, 245, 255, 0.05) 50%);
            background-size: 100% 4px;
            animation: scanline 10s linear infinite;
        """,
        "card_effect": "animation: fadeIn 0.5s ease-in, glitch 2s infinite;"
    },
    "matrix": {
        "bg": "#0A0F1E",
        "card_bg": "#0A0F1E/80",
        "text_color": "#CCFFCC",
        "accent": "#00FF00",
        "font": "'Source Code Pro', monospace",
        "bg_effect": """
            background: #0A0F1E;
            animation: digitalRain 20s linear infinite;
        """,
        "card_effect": "animation: fadeIn 0.5s ease-in;"
    },
    "holographic": {
        "bg": "#111827",
        "card_bg": "#1E3A8A/80",
        "text_color": "#BFDBFE",
        "accent": "#60A5FA",
        "font": "'Exo', sans-serif",
        "bg_effect": """
            background: linear-gradient(45deg, rgba(30, 58, 138, 0.1), rgba(96, 165, 250, 0.1));
            animation: particleGlow 15s ease-in-out infinite;
        """,
        "card_effect": "animation: fadeIn 0.5s ease-in;"
    },
    "retro_wave": {
        "bg": "#1A0B2E",
        "card_bg": "#0A1C3A/80",
        "text_color": "#FFD1DC",
        "accent": "#FF007F",
        "font": "'Press Start 2P', cursive",
        "bg_effect": """
            background: linear-gradient(0deg, #1A0B2E, #2E1A47);
            background-size: 100% 4px;
            animation: retroScan 8s linear infinite;
        """,
        "card_effect": "animation: fadeIn 0.5s ease-in, neonPulse 2s infinite;"
    },
    "quantum_core": {
        "bg": "#0D1117",
        "card_bg": "#0A1C3A/80",
        "text_color": "#A5F3FC",
        "accent": "#2DD4BF",
        "font": "'IBM Plex Sans', sans-serif",
        "bg_effect": """
            background: linear-gradient(45deg, rgba(13, 17, 23, 0.9), rgba(45, 212, 191, 0.05));
            animation: quantumWave 12s ease-in-out infinite;
        """,
        "card_effect": "animation: fadeIn 0.5s ease-in, quantumPulse 3s infinite;"
    }
}

# Theme display name mapping
theme_display_map = {
    "Cyberpunk": "cyberpunk",
    "Matrix": "matrix",
    "Holographic": "holographic",
    "Retro Wave": "retro_wave",
    "Quantum Core": "quantum_core"
}
theme_options = list(theme_display_map.keys())

# ----------------------------
# Custom CSS with Tailwind and Google Fonts
# ----------------------------
def local_css(theme="cyberpunk"):
    style = theme_styles.get(theme, theme_styles["cyberpunk"])
    
    st.markdown(f"""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Source+Code+Pro:wght@400;700&family=Exo:wght@400;700&family=Press+Start+2P&family=IBM+Plex+Sans:wght@400;700&display=swap" rel="stylesheet">
    <style>
        /* Global Styling */
        .stApp {{
            background: {style['bg']};
            font-family: {style['font']};
            min-height: 100vh;
            color: {style['text_color']};
            position: relative;
            overflow: hidden;
        }}

        /* Background Effect */
        .stApp::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            {style['bg_effect']}
            pointer-events: none;
        }}

        /* Card Styling */
        .card {{
            @apply bg-{style['card_bg']} rounded-xl p-6 mb-6 border border-[{style['accent']}]/50 shadow-lg shadow-[{style['accent']}]/20;
            backdrop-filter: blur(10px);
            {style['card_effect']}
        }}

        /* Header Styling */
        .header {{
            @apply bg-gradient-to-r from-blue-900 to-purple-900 text-[{style['text_color']}] p-6 rounded-b-xl mb-6;
            border-bottom: 2px solid {style['accent']};
        }}

        /* Button Styling */
        .stButton>button {{
            @apply bg-[{style['accent']}] text-[{style['text_color']}] font-semibold py-2 px-6 rounded-lg border border-[{style['accent']}]/50;
            box-shadow: 0 0 10px {style['accent']}, 0 0 20px {style['accent']};
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            @apply transform scale-105;
            box-shadow: 0 0 15px {style['accent']}, 0 0 30px {style['accent']};
        }}

        /* Input Fields */
        .stNumberInput, .stTextArea {{
            @apply bg-gray-900 text-[{style['text_color']}] rounded-lg border border-[{style['accent']}]/50 p-3;
            box-shadow: 0 0 5px {style['accent']};
        }}
        .stNumberInput:focus, .stTextArea:focus {{
            @apply ring-2 ring-[{style['accent']}];
        }}

        /* Success and Error Messages */
        .success-message {{
            @apply bg-green-900/30 text-green-300 p-4 rounded-lg border-l-4 border-green-500 mb-4;
            box-shadow: 0 0 10px {style['accent']};
        }}
        .error-message {{
            @apply bg-red-900/30 text-red-300 p-4 rounded-lg border-l-4 border-[{style['accent']}] mb-4;
            box-shadow: 0 0 10px {style['accent']};
        }}

        /* Sidebar Styling */
        .sidebar .sidebar-content {{
            @apply bg-gray-900/80 rounded-xl p-6 border border-[{style['accent']}]/50;
            backdrop-filter: blur(10px);
        }}

        /* Animations */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes glitch {{
            0% {{ transform: translate(0); }}
            2% {{ transform: translate(-2px, 2px); }}
            4% {{ transform: translate(2px, -2px); }}
            6% {{ transform: translate(0); }}
            100% {{ transform: translate(0); }}
        }}
        @keyframes digitalRain {{
            0% {{ background-position: 0 0; }}
            100% {{ background-position: 0 -2000px; }}
        }}
        @keyframes particleGlow {{
            0% {{ opacity: 0.3; }}
            50% {{ opacity: 0.7; }}
            100% {{ opacity: 0.3; }}
        }}
        @keyframes retroScan {{
            0% {{ background-position: 0 0; }}
            100% {{ background-position: 0 -1000px; }}
        }}
        @keyframes neonPulse {{
            0%, 100% {{ opacity: 0.8; }}
            50% {{ opacity: 1; }}
        }}
        @keyframes scanline {{
            0% {{ background-position: 0 0; }}
            100% {{ background-position: 0 -1000px; }}
        }}
        @keyframes quantumWave {{
            0% {{ opacity: 0.4; transform: translateY(0); }}
            50% {{ opacity: 0.8; transform: translateY(-10px); }}
            100% {{ opacity: 0.4; transform: translateY(0); }}
        }}
        @keyframes quantumPulse {{
            0%, 100% {{ box-shadow: 0 0 10px {style['accent']}; }}
            50% {{ box-shadow: 0 0 20px {style['accent']}, 0 0 30px #7C3AED; }}
        }}
        .loading-pulse {{
            animation: glitch 1.5s infinite ease-in-out;
        }}

        /* Typography */
        h1 {{ @apply text-3xl font-bold text-[{style['text_color']}]; }}
        h2 {{ @apply text-2xl font-semibold text-[{style['text_color']}]; }}
        h3 {{ @apply text-xl font-medium text-[{style['accent']}]; }}

        /* Footer */
        .footer {{
            @apply text-center text-[{style['text_color']}] py-6;
        }}

        /* Progress Bar */
        .stProgress .st-bd {{
            @apply bg-[{style['accent']}];
            box-shadow: 0 0 10px {style['accent']};
        }}
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# Helper Functions for Visualization
# ----------------------------
def create_gauge_chart(score, threshold, accent_color="#FF2E63"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Threat Score", 'font': {'size': 24}},
        delta={'reference': threshold, 'increasing': {'color': accent_color}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "#E0FFFF"},
            'bar': {'color': accent_color},
            'bgcolor': "#0A0F1E",
            'borderwidth': 2,
            'bordercolor': "#A100F2",
            'steps': [
                {'range': [0, threshold], 'color': '#065F46'},
                {'range': [threshold, 1], 'color': '#4A0E1A'}],
            'threshold': {
                'line': {'color': accent_color, 'width': 4},
                'thickness': 0.75,
                'value': threshold}}))
    
    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor="#0A0F1E",
        font={'color': "#E0FFFF", 'family': "inherit"}
    )
    return fig

def create_model_confidence_chart(model_scores, accent_color="#FF2E63"):
    models = list(model_scores.keys())
    scores = list(model_scores.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=scores,
            y=models,
            orientation='h',
            text=[f"{score:.3f}" for score in scores],
            textposition='auto',
            marker=dict(
                color=[accent_color, '#A100F2', accent_color, '#FBBF24'],
                line=dict(color='#E0FFFF', width=1)
            )
        )
    ])
    
    fig.update_layout(
        title="Neural Confidence Matrix",
        xaxis_title="Confidence Score",
        yaxis_title="Module",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="#0A0F1E",
        font=dict(family="inherit", size=12, color="#E0FFFF")
    )
    return fig

def create_radar_chart(features, values, accent_color="#FF2E63"):
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=features,
        fill='toself',
        line=dict(color=accent_color),
        marker=dict(color=accent_color)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(values) * 1.2]),
            bgcolor="#0A0F1E",
        ),
        showlegend=False,
        height=300,
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor="#0A0F1E",
        font=dict(family="inherit", size=12, color="#E0FFFF")
    )
    return fig

def create_anomaly_breakdown_chart(features, values, accent_color="#FF2E63"):
    fig = go.Figure(data=[
        go.Bar(
            x=features,
            y=values,
            marker=dict(
                color=accent_color,
                line=dict(color='#00F5FF', width=1)
            )
        )
    ])
    
    fig.update_layout(
        title="Anomaly Score Breakdown",
        xaxis_title="Feature",
        yaxis_title="Contribution",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="#0A0F1E",
        plot_bgcolor="#0A0F1E",
        font=dict(family="inherit", size=12, color="#E0FFFF")
    )
    return fig

# ----------------------------
# PDF Export Function
# ----------------------------
def generate_pdf_report(predictions, ensemble_pred, threshold, input_df, model_scores_list, insights):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.black)
    
    c.drawString(50, 750, "CyberFraud Detection Report")
    c.drawString(50, 730, "Generated by Neural Ensemble v3.0")
    c.line(50, 720, 550, 720)
    
    y = 700
    for idx, pred in enumerate(predictions):
        y -= 20
        c.drawString(50, y, f"Packet {idx+1}: {'THREAT DETECTED' if pred else 'CLEAR'}")
        c.drawString(70, y-15, f"Threat Score: {ensemble_pred[idx]:.4f} (Threshold: {threshold:.4f})")
        c.drawString(70, y-30, f"Amount: ${input_df['Amount'].iloc[idx]:.2f}")
        c.drawString(70, y-45, f"Time: {input_df['Time'].iloc[idx]:.0f} seconds")
        c.drawString(70, y-60, f"Insight: {insights[idx]}")
        y -= 75
        for model, score in model_scores_list[idx].items():
            c.drawString(90, y, f"{model}: {score:.4f}")
            y -= 15
        y -= 20
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ----------------------------
# AI-Powered Insights Function
# ----------------------------
def generate_insight(prediction, input_fe, idx):
    if prediction[idx] == 1:
        reasons = []
        if input_fe['Amount_Zscore'].iloc[idx] > 2:
            reasons.append("Unusually high transaction amount")
        if abs(input_fe['V_Sum'].iloc[idx]) > 10:
            reasons.append("Anomalous feature vector sum")
        if input_fe['V_Skew'].iloc[idx] > 1:
            reasons.append("Skewed feature distribution")
        return "; ".join(reasons) if reasons else "High-risk pattern detected"
    return "Transaction within normal parameters"

# ----------------------------
# App Setup
# ----------------------------
st.set_page_config(
    page_title="CyberFraud Detection", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'transaction_log' not in st.session_state:
    st.session_state.transaction_log = []
if 'theme' not in st.session_state:
    st.session_state.theme = "cyberpunk"

# Apply CSS with Theme
local_css(st.session_state.theme)

# Header
st.markdown("""
<div class="header">
    <div class="flex items-center space-x-4">
        <span class="text-4xl">🛡️</span>
        <div>
            <h1 class="animate-pulse">CyberFraud Detection</h1>
            <p class="text-lg font-light">Neural Net Security Matrix</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2>Neural Core Status</h2>", unsafe_allow_html=True)
    
    with st.spinner('Initializing neural cores...'):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
            
        with open('ensemble_models.pkl', 'rb') as f:
            ensemble = pickle.load(f)
        xgb_model = ensemble['xgb_model']
        lgbm_model = ensemble['lgbm_model']
        catboost_model = ensemble['catboost_model']
        weights = ensemble['weights']
        threshold = ensemble['threshold']
        feature_engineer = ensemble['feature_engineer']
        preprocess_pipeline = ensemble['preprocess_pipeline']
        pca = ensemble['pca']
    
    st.markdown("""
    <div class="success-message">
        <h3>✅ Cores Online</h3>
        <p>System fully initialized.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3>Neural Modules</h3>", unsafe_allow_html=True)
    st.markdown("""
    - 🧠 XGBoost Matrix
    - ⚡ LightGBM Circuit
    - 🐾 CatBoost Neuron
    - 🔧 Data Synth Pipeline
    - 📉 Dimensionality Core
    """)
    
    st.markdown("<hr class='my-4 border-accent/50'>", unsafe_allow_html=True)
    st.markdown("<h3>System Intel</h3>", unsafe_allow_html=True)
    st.markdown("""
    Real-time fraud detection powered by advanced neural ensemble architecture.
    """)
    
    # Theme Toggle
    st.markdown("<h3>Interface Protocol</h3>", unsafe_allow_html=True)
    # Map current theme key to display name
    current_theme = next((display for display, key in theme_display_map.items() if key == st.session_state.theme), "Cyberpunk")
    theme = st.selectbox("Select Theme:", theme_options, index=theme_options.index(current_theme))
    if theme_display_map[theme] != st.session_state.theme:
        st.session_state.theme = theme_display_map[theme]
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# User Inputs
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h2>📡 Data Input Terminal</h2>", unsafe_allow_html=True)
st.markdown("<p>Transmit transaction data for neural analysis.</p>", unsafe_allow_html=True)

input_method = st.radio(
    "Select input protocol:", 
    ("Manual Input", "Paste Row Text", "Upload CSV File"),
    horizontal=True,
    label_visibility="collapsed"
)

features = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
input_df = None
validation_errors = []

if input_method == "Manual Input":
    with st.form("fraud_form_manual"):
        st.markdown("<h3>Data Entry</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            time_val = st.number_input("Time (seconds)", min_value=0, format="%d")
            if time_val < 0:
                validation_errors.append("Time cannot be negative.")
        with col2:
            amount_val = st.number_input("Amount ($)", min_value=0.0, format="%.2f")
            if amount_val < 0:
                validation_errors.append("Amount cannot be negative.")
        
        st.markdown("<h3 class='mt-4'>Feature Vectors (V1-V28)</h3>", unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["V1-V7", "V8-V14", "V15-V21", "V22-V28"])
        v_values = {}
        
        with tab1:
            cols = st.columns(3)
            for i in range(1, 8):
                v_values[f'V{i}'] = cols[(i-1) % 3].number_input(f"V{i}", format="%.5f")
                
        with tab2:
            cols = st.columns(3)
            for i in range(8, 15):
                v_values[f'V{i}'] = cols[(i-8) % 3].number_input(f"V{i}", format="%.5f")
        
        with tab3:
            cols = st.columns(3)
            for i in range(15, 22):
                v_values[f'V{i}'] = cols[(i-15) % 3].number_input(f"V{i}", format="%.5f")
        
        with tab4:
            cols = st.columns(3)
            for i in range(22, 29):
                v_values[f'V{i}'] = cols[(i-22) % 3].number_input(f"V{i}", format="%.5f")
        
        user_input = {'Time': time_val, 'Amount': amount_val, **v_values}
        
        submitted = st.form_submit_button("Execute Analysis", use_container_width=True)
        if submitted and validation_errors:
            st.markdown("""
            <div class="error-message">
                ❌ Input Validation Failed: {}
            </div>
            """.format(", ".join(validation_errors)), unsafe_allow_html=True)
        elif submitted:
            input_df = pd.DataFrame([user_input])

elif input_method == "Paste Row Text":
    st.markdown("<h3>Raw Data Stream</h3>", unsafe_allow_html=True)
    st.markdown("<p>Input: Time, Amount, V1-V28 (comma, tab, or space-separated)</p>", unsafe_allow_html=True)
    
    row_text = st.text_area("Data stream:", height=100)
    col1, col2 = st.columns([1, 3])
    
    with col1:
        submitted = st.button("Execute Analysis", use_container_width=True)
    
    if submitted and row_text:
        try:
            values = list(map(float, re.split(r'[\s,\t]+', row_text.strip())))
            if len(values) == len(features) + 1:
                values = values[:-1]
            if len(values) != len(features):
                validation_errors.append(f"Expected {len(features)} vectors, received {len(values)}.")
            if any(v < 0 for v in values[:2]):  # Check Time and Amount
                validation_errors.append("Time and Amount cannot be negative.")
            if validation_errors:
                st.markdown("""
                <div class="error-message">
                    ❌ Protocol Error: {}
                </div>
                """.format(", ".join(validation_errors)), unsafe_allow_html=True)
            else:
                input_df = pd.DataFrame([dict(zip(features, values))])
        except ValueError:
            st.markdown("""
            <div class="error-message">
                ❌ Data Corruption: Invalid vector format.
            </div>
            """, unsafe_allow_html=True)

elif input_method == "Upload CSV File":
    st.markdown("<h3>Data Uplink</h3>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Transmit CSV data:", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.markdown(f"""
            <div class="success-message">
                ✅ Uplink Complete: {len(df)} data packets received
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h4>Data Preview</h4>", unsafe_allow_html=True)
            st.dataframe(df.head(5), use_container_width=True)
            
            if 'Class' in df.columns:
                df = df.drop('Class', axis=1)
            missing_features = [f for f in features if f not in df.columns]
            
            if missing_features:
                st.markdown(f"""
                <div class="error-message">
                    ❌ Missing Vectors: {', '.join(missing_features)}
                </div>
                """, unsafe_allow_html=True)
            elif (df['Time'] < 0).any() or (df['Amount'] < 0).any():
                st.markdown("""
                <div class="error-message">
                    ❌ Invalid Data: Time and Amount cannot be negative.
                </div>
                """, unsafe_allow_html=True)
            else:
                input_df = df[features]
                submitted = st.button("Execute Analysis", use_container_width=True)
        except Exception as e:
            st.markdown(f"""
            <div class="error-message">
                ❌ Uplink Failure: {e}
            </div>
            """, unsafe_allow_html=True)

# Prediction and Features
if submitted and input_df is not None:
    # Get the current theme's style
    style = theme_styles.get(st.session_state.theme, theme_styles["cyberpunk"])
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2>🔬 Neural Analysis Output</h2>", unsafe_allow_html=True)
    
    with st.spinner('Processing data stream...'):
        batch_progress = st.progress(0)
        input_fe = feature_engineer.transform(input_df)
        input_preprocessed = preprocess_pipeline.transform(input_fe)
        input_pca = pca.transform(input_preprocessed)
        final_input = np.hstack((input_preprocessed, input_pca))

        xgb_pred = xgb_model.predict_proba(final_input)[:, 1]
        lgbm_pred = lgbm_model.predict_proba(final_input)[:, 1]
        catboost_pred = catboost_model.predict_proba(final_input)[:, 1]

        ensemble_pred = (weights[0] * xgb_pred + 
                         weights[1] * lgbm_pred + 
                         weights[2] * catboost_pred)

        prediction = (ensemble_pred >= threshold).astype(int)
        
        n_transactions = len(prediction)
        fraud_count = sum(prediction)
        model_scores_list = []
        insights = []
        
        for idx in range(n_transactions):
            batch_progress.progress((idx + 1) / n_transactions)
            time.sleep(0.05)  # Simulate processing delay for visual effect
        
        batch_progress.empty()
        
        if n_transactions > 1:
            st.markdown(f"""
            <h3>System Report</h3>
            <div class='grid grid-cols-3 gap-4'>
                <div class='bg-gray-900/50 p-4 rounded-lg border border-accent/50'>
                    <p class='font-semibold'>Total Packets</p>
                    <p class='text-2xl'>{n_transactions}</p>
                </div>
                <div class='bg-red-900/50 p-4 rounded-lg border border-accent/50'>
                    <p class='font-semibold'>Threats Detected</p>
                    <p class='text-2xl'>{fraud_count} ({fraud_count/n_transactions*100:.1f}%)</p>
                </div>
                <div class='bg-green-900/50 p-4 rounded-lg border border-accent/50'>
                    <p class='font-semibold'>Safe Packets</p>
                    <p class='text-2xl'>{n_transactions-fraud_count} ({(n_transactions-fraud_count)/n_transactions*100:.1f}%)</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        for idx in range(len(prediction)):
            col1, col2 = st.columns([3, 2])
            model_scores = {
                "XGBoost": xgb_pred[idx],
                "LightGBM": lgbm_pred[idx],
                "CatBoost": catboost_pred[idx],
                "Ensemble": ensemble_pred[idx]
            }
            model_scores_list.append(model_scores)
            insight = generate_insight(prediction, input_fe, idx)
            insights.append(insight)
            
            with col1:
                if prediction[idx] == 1:
                    st.markdown(f"""
                    <div class="error-message">
                        <h3>🚨 Packet {idx+1}: THREAT DETECTED</h3>
                        <p>High-risk anomaly flagged by neural core.</p>
                        <p><strong>Threat Score:</strong> {ensemble_pred[idx]:.4f} (Threshold: {threshold:.4f})</p>
                        <p><strong>Insight:</strong> {insight}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="success-message">
                        <h3>✅ Packet {idx+1}: CLEAR</h3>
                        <p>Data packet verified as secure.</p>
                        <p><strong>Threat Score:</strong> {ensemble_pred[idx]:.4f} (Threshold: {threshold:.4f})</p>
                        <p><strong>Insight:</strong> {insight}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if n_transactions == 1:
                    st.markdown("<h4>Packet Metadata</h4>", unsafe_allow_html=True)
                    details_col1, details_col2 = st.columns(2)
                    with details_col1:
                        st.metric("Time (seconds)", f"{input_df['Time'].iloc[0]:.0f}", delta_color="off")
                        hours = (input_df['Time'].iloc[0] // 3600) % 24
                        minutes = (input_df['Time'].iloc[0] % 3600) // 60
                        time_of_day = f"{int(hours):02d}:{int(minutes):02d}"
                        st.metric("Time of Day", time_of_day, delta_color="off")
                    with details_col2:
                        st.metric("Amount ($)", f"${input_df['Amount'].iloc[0]:.2f}", delta_color="off")
                        amount = input_df['Amount'].iloc[0]
                        size = "Small" if amount <= 5 else "Medium" if amount <= 100 else "Large"
                        st.metric("Packet Size", size, delta_color="off")
                    
                    # Risk Visualizer (Radar Chart)
                    st.markdown("<h4>Risk Vector Analysis</h4>", unsafe_allow_html=True)
                    selected_features = ['Amount_Zscore', 'V_Sum', 'V_Mean', 'V_Std', 'V_Skew']
                    feature_values = [abs(input_fe[feat].iloc[0]) for feat in selected_features]
                    radar_fig = create_radar_chart(selected_features, feature_values, accent_color=style['accent'])
                    st.plotly_chart(radar_fig, use_container_width=True)
                    
                    # Anomaly Score Breakdown
                    st.markdown("<h4>Anomaly Breakdown</h4>", unsafe_allow_html=True)
                    anomaly_features = selected_features
                    anomaly_values = [abs(input_fe[feat].iloc[0]) for feat in anomaly_features]
                    anomaly_fig = create_anomaly_breakdown_chart(anomaly_features, anomaly_values, accent_color=style['accent'])
                    st.plotly_chart(anomaly_fig, use_container_width=True)
            
            with col2:
                if n_transactions == 1:
                    gauge_fig = create_gauge_chart(ensemble_pred[idx], threshold, accent_color=style['accent'])
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    conf_fig = create_model_confidence_chart(model_scores, accent_color=style['accent'])
                    st.plotly_chart(conf_fig, use_container_width=True)
                else:
                    fraud_score = ensemble_pred[idx]
                    risk_level = "High Risk" if fraud_score >= threshold else "Low Risk"
                    st.metric("Amount", f"${input_df['Amount'].iloc[idx]:.2f}", delta_color="off")
                    st.metric("Threat Level", risk_level, delta_color="off")
                    st.progress(min(fraud_score * 1.25, 1.0))
                
            # Add to Transaction Log
            log_entry = {
                'Packet': idx + 1,
                'Time': input_df['Time'].iloc[idx],
                'Amount': input_df['Amount'].iloc[idx],
                'Fraud Score': ensemble_pred[idx],
                'Result': 'Threat' if prediction[idx] else 'Clear',
                'Insight': insight
            }
            st.session_state.transaction_log.append(log_entry)
            
            st.markdown("<hr class='my-4 border-accent/50'>", unsafe_allow_html=True)
        
        # PDF Export
        st.markdown("<h3>Export Analysis</h3>", unsafe_allow_html=True)
        pdf_buffer = generate_pdf_report(prediction, ensemble_pred, threshold, input_df, model_scores_list, insights)
        st.download_button(
            label="Download Report",
            data=pdf_buffer,
            file_name="CyberFraud_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

# Transaction History Log with Filter
if st.session_state.transaction_log:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2>📜 Transaction Log</h2>", unsafe_allow_html=True)
    
    # Filters
    st.markdown("<h3>Filter Data Stream</h3>", unsafe_allow_html=True)
    result_filter = st.multiselect("Result:", ["All", "Threat", "Clear"], default=["All"])
    amount_min, amount_max = st.slider("Amount Range ($):", 0.0, float(max([log['Amount'] for log in st.session_state.transaction_log], default=1000.0)), (0.0, 1000.0))
    
    log_df = pd.DataFrame(st.session_state.transaction_log)
    filtered_df = log_df.copy()
    if "All" not in result_filter:
        filtered_df = filtered_df[filtered_df['Result'].isin(result_filter)]
    filtered_df = filtered_df[(filtered_df['Amount'] >= amount_min) & (filtered_df['Amount'] <= amount_max)]
    
    st.dataframe(filtered_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>CyberFraud Detection | Neural Ensemble v3.0</p>
</div>
""", unsafe_allow_html=True)