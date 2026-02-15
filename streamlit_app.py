import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from models import DataLoader, Evaluator, Pipeline, ALL_MODELS

# ── Global style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)

# Intuitive semantic colors
CLR_BENIGN = "#22c55e"      # green — healthy / safe
CLR_MALIGNANT = "#ef4444"   # red   — danger / alert
CLR_ACTUAL = "#3b82f6"      # blue  — ground truth
CLR_PREDICTED = "#f97316"   # orange — model output

METRIC_COLORS = {
    "Accuracy":  "#22c55e",
    "AUC":       "#3b82f6",
    "Precision": "#f59e0b",
    "Recall":    "#ef4444",
    "F1 Score":  "#06b6d4",
    "MCC":       "#64748b",
}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Classification Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border-radius: 16px;
        padding: 20px 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        border-radius: 16px 16px 0 0;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
        line-height: 1.2 !important;
    }
    [data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        margin-bottom: 8px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 16px 32px;
        font-size: 1.6rem !important;
        font-weight: 900 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        min-height: 70px !important;
        width: 250px !important;
        max-width: 250px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%) !important;
        color: white !important;
    }
    .stTabs [data-baseweb="tab"]:not([aria-selected="true"]) {
        background: #f1f5f9 !important;
        color: #475569 !important;
        border: 1px solid #e2e8f0 !important;
    }
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background: #e2e8f0 !important;
        color: #1e293b !important;
    }
    /* Consistent heading sizes */
    h1 { font-size: 2.5rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.8rem !important; font-weight: 600 !important; }
    h3 { font-size: 1.4rem !important; font-weight: 600 !important; }
    /* Industry green button styles */
    button[kind="primary"] {
        background-color: #28A745 !important;
        border-color: #28A745 !important;
        color: white !important;
    }
    button[kind="primary"]:hover {
        background-color: #218838 !important;
        border-color: #218838 !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #28A745 !important;
        border-color: #28A745 !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #218838 !important;
        border-color: #218838 !important;
    }
    /* Center pie charts */
    .stPlotlyChart { text-align: center; }
    .matplotlib-figure { text-align: center; margin: 0 auto; }
    
    /* Modern card styling for info boxes */
    .element-container:has([data-testid="stInfo"]) {
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    [data-testid="stInfo"] {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%) !important;
        border-left: 4px solid #3b82f6 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    }
    [data-testid="stSuccess"] {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%) !important;
        border-left: 4px solid #22c55e !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    }
    [data-testid="stWarning"] {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%) !important;
        border-left: 4px solid #f59e0b !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    }
    [data-testid="stError"] {
        background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%) !important;
        border-left: 4px solid #ef4444 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Enhanced number display */
    .big-number {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        text-align: center !important;
        margin: 1rem 0 !important;
    }
    
    /* Modern interactive train button */
    div[data-testid="stSidebar"] .stButton > button[kind="primary"],
    .stSidebar .stButton > button[kind="primary"],
    button[kind="primary"] {
        width: 100% !important;
        padding: 20px 24px !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        margin: 15px auto !important;
        display: block !important;
        min-height: 70px !important;
        border-radius: 16px !important;
        text-align: center !important;
        line-height: 1.2 !important;
        box-sizing: border-box !important;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        border: 2px solid #059669 !important;
        color: white !important;
        position: relative !important;
        overflow: hidden !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    div[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
    .stSidebar .stButton > button[kind="primary"]:hover,
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        border-color: #047857 !important;
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4) !important;
    }
    
    div[data-testid="stSidebar"] .stButton > button[kind="primary"]:active,
    .stSidebar .stButton > button[kind="primary"]:active,
    button[kind="primary"]:active {
        transform: translateY(0) scale(0.98) !important;
        box-shadow: 0 2px 10px rgba(16, 185, 129, 0.3) !important;
    }
    
    /* Animated pulse effect */
    div[data-testid="stSidebar"] .stButton > button[kind="primary"]::before,
    .stSidebar .stButton > button[kind="primary"]::before,
    button[kind="primary"]::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent) !important;
        transition: left 0.5s !important;
    }
    
    div[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover::before,
    .stSidebar .stButton > button[kind="primary"]:hover::before,
    button[kind="primary"]:hover::before {
        left: 100% !important;
    }
    
    /* Increase spinner size and center it */
    .stSpinner {
        transform: scale(1.5) !important;
        margin: 20px auto !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }
    
    .stSpinner > div {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 10px !important;
        flex-direction: row-reverse !important;
    }
    
    .stSpinner > div::before {
        font-size: 2rem !important;
        margin-right: 8px !important;
    }
    
    /* Ensure spinner animation is on the right */
    .stSpinner > div > div[role="progressbar"] {
        order: -1 !important;
        margin-left: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# ML Classification Models Implementation and Evaluation")
st.markdown('<div style="font-size: 1.2rem; font-weight: 500; color: #64748b; line-height: 1.4; margin-top: -10px; margin-bottom: 20px;">Train, evaluate, and compare 6 ML models on the Wisconsin Breast Cancer dataset</div>', unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Settings")
st.sidebar.markdown("---")

# Dataset Selection
st.sidebar.markdown("###  Step 1: Dataset Selection")
dataset_source = st.sidebar.radio(
    "Choose dataset source:",
    ["SK Learn Built-in Dataset", "Upload Custom Dataset(.csv)"],
    index=0,
    help="Select between built-in sklearn dataset or upload your own CSV file"
)

# Custom CSV Upload Section (shown when Upload Custom Dataset is selected)
if dataset_source == "Upload Custom Dataset(.csv)":
    st.sidebar.markdown("#### Step 1.1: Upload Custom Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv", key="csv_upload", 
        help="Upload a CSV file with features and a target column for classification")
    
    # Simple supported dataset link
    st.sidebar.markdown("📖 [Kaggle-Supported Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)")
    
    
# Model Selection
st.sidebar.markdown("### Step 2: Model Selection")
model_options = ["All Models"] + list(ALL_MODELS.keys())
selected_model = st.sidebar.selectbox(
    "Select models to train:",
    options=model_options,
    index=0,
    help="Choose to train all models or select a specific one"
)

# Train Button
st.sidebar.markdown("---")
st.sidebar.markdown("### Step 3: Execute Training")
train_button = st.sidebar.button(
    "🚀 Train and Evaluate",
    type="primary",
    help="Start training the selected models and generate performance metrics"
)


# Session Info
st.sidebar.markdown("---")
st.sidebar.markdown("###  Information ℹ️")
st.sidebar.markdown("""
 **Available Models:** 6  
**Evaluation Metrics:** 6  
**Train/Test Split:** 85/15  
**Target:** Diagnosis(Benign/Malignant)
""")

# Initialize session state for loader and auto-run
if 'loader' not in st.session_state:
    st.session_state.loader = None
if 'auto_run_completed' not in st.session_state:
    st.session_state.auto_run_completed = False
if 'training_executed' not in st.session_state:
    st.session_state.training_executed = False

# ── Progress Indicator ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 Training and Evaluation Steps")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 25px 20px; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.1); transition: all 0.3s ease;">
        <div style="font-size: 2rem; font-weight: 800; color: #1e293b; margin-bottom: 12px; background: linear-gradient(135deg, #3b82f6, #1e40af); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">Step-1</div>
        <div style="font-size: 1.3rem; font-weight: 600; color: #475569; margin-bottom: 12px; letter-spacing: 0.5px;">DATASET SELECTION</div>
        <div style="font-size: 1.2rem; color: #64748b; line-height: 1.5; font-weight: 500;">Select dataset source from sidebar options</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 25px 20px; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.1); transition: all 0.3s ease;">
        <div style="font-size: 2rem; font-weight: 800; color: #1e293b; margin-bottom: 12px; background: linear-gradient(135deg, #10b981, #059669); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">Step-2</div>
        <div style="font-size: 1.3rem; font-weight: 600; color: #475569; margin-bottom: 12px; letter-spacing: 0.5px;">MODEL CONFIGURATION</div>
        <div style="font-size: 1.2rem; color: #64748b; line-height: 1.5; font-weight: 500;">Choose models to train from dropdown</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 25px 20px; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.1); transition: all 0.3s ease;">
        <div style="font-size: 2rem; font-weight: 800; color: #1e293b; margin-bottom: 12px; background: linear-gradient(135deg, #f59e0b, #d97706); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">Step-3</div>
        <div style="font-size: 1.3rem; font-weight: 600; color: #475569; margin-bottom: 12px; letter-spacing: 0.5px;">TRAINING EXECUTION</div>
        <div style="font-size: 1.2rem; color: #64748b; line-height: 1.5; font-weight: 500;">Click "Train and Evaluate" button</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _class_colors(target_labels: dict):
    """Return a list of colors mapped to sorted class keys (green=Benign, red=Malignant)."""
    keys = sorted(target_labels.keys()) if target_labels else [0, 1]
    colors = []
    for k in keys:
        label = target_labels.get(k, str(k)).lower() if target_labels else str(k)
        if "benign" in label:
            colors.append(CLR_BENIGN)
        elif "malignant" in label:
            colors.append(CLR_MALIGNANT)
        else:
            colors.append(sns.color_palette("tab10")[len(colors) % 10])
    return colors


def show_infographic_cards(df: pd.DataFrame, name: str, target_labels: dict):
    """Top-level summary cards with standardized styling."""
    st.markdown("### 📊 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    
    # Standardized metric cards with neutral colors
    with c1:
        st.markdown("""
        <div style="text-align: center; min-height: 180px; padding: 20px 16px; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 1.5rem; color: #64748b; margin-bottom: 12px;">📊</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #1e293b; margin-bottom: 8px;">{:,}</div>
            <div style="font-size: 0.875rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Samples</div>
        </div>
        """.format(df.shape[0]), unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div style="text-align: center; min-height: 180px; padding: 20px 16px; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 1.5rem; color: #64748b; margin-bottom: 12px;">🔧</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #1e293b; margin-bottom: 8px;">{}</div>
            <div style="font-size: 0.875rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Features</div>
        </div>
        """.format(df.shape[1] - 1), unsafe_allow_html=True)
    
    with c3:
        st.markdown("""
        <div style="text-align: center; min-height: 180px; padding: 20px 16px; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 1.5rem; color: #64748b; margin-bottom: 12px;">🎯</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #1e293b; margin-bottom: 8px;">{}</div>
            <div style="font-size: 0.875rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Classes</div>
        </div>
        """.format(df['target'].nunique()), unsafe_allow_html=True)
    
    with c4:
        if target_labels:
            labels_text = ", ".join(target_labels.values())
        else:
            labels_text = ", ".join(str(v) for v in sorted(df["target"].unique()))
        
        st.markdown("""
        <div style="text-align: center; min-height: 180px; padding: 20px 16px; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 1.5rem; color: #64748b; margin-bottom: 12px;">🏷️</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #1e293b; margin-bottom: 8px; line-height: 1.3;">{}</div>
            <div style="font-size: 0.875rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Labels</div>
        </div>
        """.format(labels_text), unsafe_allow_html=True)


def show_dataset_info(df: pd.DataFrame, name: str, target_labels: dict):
    """Dataset details and preview."""
    st.markdown(f"**Dataset:** {name}")
    
    # Simple Train/Test Summary
    st.markdown("### 📊 Train/Test Split Summary")
    total_samples = len(df)
    train_samples = int(total_samples * 0.85)
    test_samples = total_samples - train_samples
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; min-height: 180px; padding: 20px 16px; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 1.5rem; color: #64748b; margin-bottom: 12px;">🎯</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #1e293b; margin-bottom: 8px;">{:,}</div>
            <div style="font-size: 0.875rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Total Samples</div>
        </div>
        """.format(total_samples), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; min-height: 180px; padding: 20px 16px; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 1.5rem; color: #64748b; margin-bottom: 12px;">📚</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #1e293b; margin-bottom: 8px;">{:,}</div>
            <div style="font-size: 0.875rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Train Set (85%)</div>
        </div>
        """.format(train_samples), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; min-height: 180px; padding: 20px 16px; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 1.5rem; color: #64748b; margin-bottom: 12px;">🧪</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #1e293b; margin-bottom: 8px;">{:,}</div>
            <div style="font-size: 0.875rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Test Set (15%)</div>
        </div>
        """.format(test_samples), unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div style="font-size: 1.3rem; font-weight: 700; color: #1e293b; margin-bottom: 1rem;">Class Distribution</div>', unsafe_allow_html=True)
        dist = df["target"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(2.5, 2))
        labels = [target_labels.get(i, str(i)) for i in dist.index]
        # Modern green and light red colors
        chart_colors = ['#10b981', '#f87171']
        wedges, texts, autotexts = ax.pie(
            dist, labels=labels, autopct="%1.1f%%", colors=chart_colors[:len(dist)],
            startangle=90, textprops={"fontsize": 8, "color": "white", "weight": "600"},
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        for t in autotexts:
            t.set_fontweight("600")
            t.set_color("white")
        ax.set_title("Target Distribution", fontsize=8, fontweight="600", pad=4, color="#1e293b")
        ax.axis('equal')  # Ensure pie is circular
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with col2:
        st.markdown('<div style="font-size: 1.3rem; font-weight: 700; color: #1e293b; margin-bottom: 1rem;">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), width="stretch")


def show_model_metrics(results: dict, target_labels: dict):
    """Per-model metrics with confusion matrix, classification report, and actual vs predicted."""
    st.markdown("### Model Results")

    class_colors = _class_colors(target_labels)

    for name, result in results.items():
        metrics = result["metrics"]
        with st.expander(f"{name}", expanded=False):
            # Metric cards row
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            m2.metric("AUC Score", f"{metrics['AUC']:.4f}")
            m3.metric("Precision", f"{metrics['Precision']:.4f}")
            m4.metric("Recall", f"{metrics['Recall']:.4f}")
            m5.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
            m6.metric("MCC Score", f"{metrics['MCC']:.4f}")

            st.markdown("---")

            # Three-column layout
            col_cm, col_cr, col_avp = st.columns([1, 1, 1])

            with col_cm:
                st.markdown("**Confusion Matrix**")
                cm = result["confusion_matrix"]
                fig, ax = plt.subplots(figsize=(4.5, 3.8))
                labels = [target_labels.get(i, str(i)) for i in range(len(cm))]
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                            xticklabels=labels, yticklabels=labels,
                            cbar=False, annot_kws={"size": 12, "weight": "bold"},
                            linewidths=1, linecolor="white")
                ax.set_xlabel("Predicted", fontsize=10, fontweight="bold")
                ax.set_ylabel("Actual", fontsize=10, fontweight="bold")
                ax.set_title(f"{name}", fontsize=11, fontweight="bold", pad=8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with col_cr:
                st.markdown("**Classification Report**")
                st.code(result["classification_report"], language=None)

            with col_avp:
                st.markdown("**Actual vs Predicted**")
                y_test = result["y_test"]
                y_pred = result["y_pred"]
                classes = sorted(y_test.unique())
                actual_counts = [int((y_test == c).sum()) for c in classes]
                pred_counts = [int((y_pred == c).sum()) for c in classes]
                x_labels = [target_labels.get(c, str(c)) for c in classes]

                x = np.arange(len(classes))
                width = 0.32
                fig, ax = plt.subplots(figsize=(4.5, 3.8))
                bars1 = ax.bar(x - width / 2, actual_counts, width,
                               label="Actual", color=CLR_ACTUAL,
                               edgecolor="white", linewidth=0.8)
                bars2 = ax.bar(x + width / 2, pred_counts, width,
                               label="Predicted", color=CLR_PREDICTED,
                               edgecolor="white", linewidth=0.8)
                ax.set_xlabel("Classes", fontsize=10, fontweight="bold")
                ax.set_ylabel("Count", fontsize=10, fontweight="bold")
                ax.set_title("Class Distribution", fontsize=11, fontweight="bold", pad=8)
                ax.set_xticks(x)
                ax.set_xticklabels(classes)
                ax.legend(fontsize=9, loc="upper right")
                max_val = max(max(actual_counts), max(pred_counts))
                for bar in bars1 + bars2:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_val * 0.02,
                            str(int(bar.get_height())), ha="center", fontsize=9, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)


def show_comparison_table(results: dict) -> pd.DataFrame:
    """Styled comparison table with best values highlighted."""
    st.markdown("### Model Comparison")
    metrics_dict = {name: r["metrics"] for name, r in results.items()}
    comparison_df = pd.DataFrame(metrics_dict).T

    def highlight_max(s):
        is_max = s == s.max()
        return ["background-color: #dcfce7; font-weight: bold" if v else "" for v in is_max]

    styled = comparison_df.style.format("{:.4f}").apply(highlight_max, axis=0)
    st.dataframe(styled, width="stretch")
    return comparison_df


def show_visualisation(results: dict):
    """Performance bar charts — one per metric, each bar colored by its metric."""
    st.markdown("### Performance Visualization")

    metric_names = list(METRIC_COLORS.keys())
    model_names = list(results.keys())

    n_metrics = len(metric_names)
    cols = 2
    rows = (n_metrics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = axes.ravel()

    for i, metric in enumerate(metric_names):
        values = [results[m]["metrics"][metric] for m in model_names]
        color = METRIC_COLORS[metric]
        bars = axes[i].bar(model_names, values, color=color,
                           edgecolor="white", linewidth=0.8, alpha=0.85)
        axes[i].set_title(metric, fontsize=13, fontweight="bold",
                          color=color, pad=10)
        axes[i].set_ylim(0, 1.15)
        axes[i].set_ylabel("Score", fontsize=9)
        axes[i].tick_params(axis="x", rotation=45, labelsize=8)
        axes[i].grid(axis="y", alpha=0.3)
        avg = np.mean(values)
        axes[i].axhline(y=avg, color="#94a3b8", linestyle="--",
                        alpha=0.6, linewidth=0.8, label=f"Avg {avg:.3f}")
        axes[i].legend(fontsize=7, loc="lower right")
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.015,
                         f"{val:.3f}", ha="center", va="bottom",
                         fontsize=8, fontweight="bold")

    # Hide unused subplots
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Model Performance Across All Metrics",
                  fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def show_best_models(results: dict):
    """Best model summary cards."""
    st.markdown("### Best Model")
    metrics_dict = {name: r["metrics"] for name, r in results.items()}
    best_acc, best_acc_names = Evaluator.best_models(metrics_dict, "Accuracy")
    best_prec, best_prec_names = Evaluator.best_models(metrics_dict, "Precision")
    best_rec, best_rec_names = Evaluator.best_models(metrics_dict, "Recall")
    best_f1, best_f1_names = Evaluator.best_models(metrics_dict, "F1 Score")
    best_auc, best_auc_names = Evaluator.best_models(metrics_dict, "AUC")
    best_mcc, best_mcc_names = Evaluator.best_models(metrics_dict, "MCC")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        acc_models = "<br>".join([f"• {model}" for model in best_acc_names])
        st.markdown(f"""
        <div style='background-color: #dcfce7; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #22c55e; height: 220px; overflow: hidden; word-wrap: break-word;'>
        <div style='font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;'>Best Accuracy</div>
        <div style='font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0;'>{best_acc:.4f}</div>
        <div style='font-size: 0.9rem; margin-top: auto; line-height: 1.3;'>{acc_models}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        prec_models = "<br>".join([f"• {model}" for model in best_prec_names])
        st.markdown(f"""
        <div style='background-color: #dbeafe; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #3b82f6; height: 220px; overflow: hidden; word-wrap: break-word;'>
        <div style='font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;'>Best Precision</div>
        <div style='font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0;'>{best_prec:.4f}</div>
        <div style='font-size: 0.9rem; margin-top: auto; line-height: 1.3;'>{prec_models}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        rec_models = "<br>".join([f"• {model}" for model in best_rec_names])
        st.markdown(f"""
        <div style='background-color: #fef3c7; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b; height: 220px; overflow: hidden; word-wrap: break-word;'>
        <div style='font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;'>Best Recall</div>
        <div style='font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0;'>{best_rec:.4f}</div>
        <div style='font-size: 0.9rem; margin-top: auto; line-height: 1.3;'>{rec_models}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        f1_models = "<br>".join([f"• {model}" for model in best_f1_names])
        st.markdown(f"""
        <div style='background-color: #fee2e2; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ef4444; height: 220px; overflow: hidden; word-wrap: break-word;'>
        <div style='font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;'>Best F1 Score</div>
        <div style='font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0;'>{best_f1:.4f}</div>
        <div style='font-size: 0.9rem; margin-top: auto; line-height: 1.3;'>{f1_models}</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        auc_models = "<br>".join([f"• {model}" for model in best_auc_names])
        st.markdown(f"""
        <div style='background-color: #dcfce7; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #22c55e; height: 220px; overflow: hidden; word-wrap: break-word;'>
        <div style='font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;'>Best AUC</div>
        <div style='font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0;'>{best_auc:.4f}</div>
        <div style='font-size: 0.9rem; margin-top: auto; line-height: 1.3;'>{auc_models}</div>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        mcc_models = "<br>".join([f"• {model}" for model in best_mcc_names])
        st.markdown(f"""
        <div style='background-color: #f1f5f9; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #64748b; height: 220px; overflow: hidden; word-wrap: break-word;'>
        <div style='font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;'>Best MCC</div>
        <div style='font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0;'>{best_mcc:.4f}</div>
        <div style='font-size: 0.9rem; margin-top: auto; line-height: 1.3;'>{mcc_models}</div>
        </div>
        """, unsafe_allow_html=True)


def show_download(comparison_df: pd.DataFrame, key: str):
    """Download button."""
    st.download_button(
        label="Download Results as CSV",
        data=comparison_df.to_csv(),
        file_name=f"{key.lower().replace(' ', '_')}_results.csv",
        mime="text/csv",
        key=f"download_{key}",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(loader: DataLoader, label: str):
    target_labels = loader.target_labels

    # Set training execution flag
    st.session_state.training_executed = True

    # Show spinner for entire process
    with st.spinner("⚡ Training and evaluating models..."):
        # Prepare and train
        loader.prepare()
        if selected_model == "All Models":
            chosen = ALL_MODELS
        else:
            chosen = {selected_model: ALL_MODELS[selected_model]}
        pipeline = Pipeline(loader)
        results = pipeline.run_all(chosen)

    # Success message after completion - centered like spinner
    st.markdown("""
    <div style="text-align: center; margin: 20px auto; padding: 15px; background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 12px; color: white; font-size: 1.2rem; font-weight: 600; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);">
        ✅ Training and Evaluation completed
    </div>
    """, unsafe_allow_html=True)

    # Infographic cards
    show_infographic_cards(loader.df, loader.dataset_name, target_labels)
    st.markdown("---")

    # Tabbed navigation
    tab_data, tab_models, tab_compare = st.tabs([
        "Processed Data Set", "Indvidual Models Results", "Results Comparison"
    ])

    with tab_data:
        show_dataset_info(loader.df, loader.dataset_name, target_labels)

    with tab_models:
        show_model_metrics(results, target_labels)

    with tab_compare:
        show_best_models(results)
        st.markdown("---")
        comparison_df = show_comparison_table(results)
        st.markdown("---")
        show_visualisation(results)
        st.markdown("---")
        show_download(comparison_df, label)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

# Handle dataset loading
if dataset_source == "SK Learn Built-in Dataset":
    if st.session_state.loader is None:
        loader = DataLoader()
        loader.load_dataset("Breast Cancer")
        st.session_state.loader = loader
        st.success("✅ Built-in dataset loaded from SK Learn successfully!")
    
    # Auto-run on first load for built-in dataset
    if not st.session_state.auto_run_completed and st.session_state.loader is not None:
        st.session_state.auto_run_completed = True
        run_pipeline(st.session_state.loader, "breast_cancer")
    
    # Show train button and run pipeline when clicked
    if train_button and st.session_state.loader:
        run_pipeline(st.session_state.loader, "breast_cancer")

else:  # Upload CSV (handled in sidebar)
    # Get uploaded file from sidebar
    uploaded_file = st.session_state.get("csv_upload", None)
    
    if uploaded_file is not None:
        try:
            # Preview the uploaded dataset
            preview_df = pd.read_csv(uploaded_file)
            st.markdown("### 📊 Dataset Preview")
            st.success("✅ Dataset uploaded successfully!")
            st.dataframe(preview_df.head(), width="stretch")

            # Target column selection
            if "diagnosis" in preview_df.columns:
                target_col = "diagnosis"
                st.info(f"🎯 Target column detected: **{target_col}**")
            else:
                target_col = st.selectbox("🎯 Select target column:", preview_df.columns)

            # Store loader in session state
            if st.session_state.loader is None or st.session_state.loader.dataset_name != "Custom Dataset":
                uploaded_file.seek(0)
                loader = DataLoader()
                loader.load_custom(uploaded_file, target_col)
                st.session_state.loader = loader
                st.success("✅ Custom dataset processed successfully!")

            # Show train button and run pipeline when clicked
            if train_button and st.session_state.loader:
                run_pipeline(st.session_state.loader, "custom")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    else:
        st.markdown("### 📁 Upload Custom Dataset")
        st.info("👈 Please upload a CSV file using the sidebar to get started.")

# Footer
st.markdown("---")
st.caption("Train, evaluate, and compare 6 ML models on the Wisconsin Breast Cancer dataset")
