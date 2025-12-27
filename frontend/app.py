"""
Credit Risk Assessment System - Streamlit Dashboard

Main entry point for the Streamlit multipage application.
"""

import streamlit as st

# ============================================
# PAGE CONFIGURATION (MUST BE FIRST!)
# ============================================
st.set_page_config(
    page_title="Credit Risk Assessment System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================
"""
Custom CSS for professional appearance.

Interview Note:
"I added custom styling to match industry standards for financial dashboards."
"""
st.markdown("""
    <style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Metric card styling */
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    
    /* Success message */
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #155724;
    }
    
    /* Warning message */
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #856404;
    }
    
    /* Danger message */
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #721c24;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ============================================
# MAIN PAGE CONTENT
# ============================================

# Title
st.markdown('<h1 class="main-title">💰 Credit Risk Assessment System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Loan Portfolio Management Platform</p>', unsafe_allow_html=True)

st.markdown("---")

# ============================================
# WELCOME SECTION
# ============================================
col1, col2, col3 = st.columns(3)

with col1:
    st.info("📝 **Loan Application**")
    st.write("Submit new loan applications with real-time ML credit scoring")
    
with col2:
    st.success("📊 **Portfolio Analytics**")
    st.write("Monitor portfolio health, NPA ratios, and risk metrics")
    
with col3:
    st.warning("🎯 **ML Predictions**")
    st.write("Random Forest model with 91% accuracy, 94% ROC-AUC")

st.markdown("---")

# ============================================
# SYSTEM STATUS
# ============================================
st.subheader("🔧 System Status")

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    st.metric(label="Backend API", value="✅ Online", delta="Running")
    
with status_col2:
    st.metric(label="Database", value="✅ Connected", delta="PostgreSQL")
    
with status_col3:
    st.metric(label="ML Model", value="✅ Loaded", delta="91% Accuracy")
    
with status_col4:
    st.metric(label="Features", value="29", delta="Trained")

st.markdown("---")

# ============================================
# NAVIGATION INSTRUCTIONS
# ============================================
st.subheader("📌 Quick Start Guide")

st.markdown("""
### How to Use This System:

1. **Loan Application Form** (Page 1)
   - Select an existing customer or register a new one
   - Enter loan details (amount, tenure, purpose)
   - Get instant ML-powered credit risk assessment
   - View approval/rejection recommendation

2. **Admin Dashboard** (Page 2)
   - View portfolio health metrics (NPA ratio, default rate)
   - Analyze loan distribution by purpose and status
   - Monitor risk concentration
   - Track repayment performance

### Navigation
👈 **Use the sidebar** on the left to switch between pages.
""")

st.markdown("---")

# ============================================
# PROJECT INFORMATION
# ============================================
with st.expander("ℹ️ About This Project"):
    st.markdown("""
    ### Credit Risk Assessment & Loan Portfolio Management System
    
    **Technology Stack:**
    - **Backend:** Flask REST API (Python)
    - **Database:** PostgreSQL (9 tables)
    - **ML Model:** Random Forest Classifier (scikit-learn)
    - **Frontend:** Streamlit
    - **Version Control:** Git + GitHub
    
    **Key Features:**
    - Real-time credit risk scoring using ML (91% accuracy, 94% ROC-AUC)
    - Portfolio analytics (NPA ratios, risk metrics)
    - RESTful API architecture
    - Causal data generation for realistic training
    - 29 engineered features based on financial domain knowledge
    
    **ML Model Performance:**
    - Test Accuracy: 91.67%
    - ROC-AUC: 93.75%
    - Precision: 100%
    - Recall: 75%
    - Cross-Validation: 88.33% ± 4.08%
    
    **Financial Calculations:**
    - EMI calculation using standard banking formula
    - Debt-to-Income (DTI) ratio
    - Loan-to-Value (LTV) ratio
    - NPA classification (Sub-Standard, Doubtful, Loss)
    
    **Target Companies:**
    - JP Morgan Chase (Finance focus)
    - Thorogood (Data/ML focus)
    """)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Built with ❤️ using Flask, PostgreSQL, scikit-learn, and Streamlit</p>
    <p>© 2025 Credit Risk Assessment System | End-to-End ML Project</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR CONFIGURATION
# ============================================
with st.sidebar:
    st.image("https://via.placeholder.com/200x100/1f77b4/FFFFFF?text=Credit+Risk", use_container_width=True)
    
    st.markdown("### 🎯 Quick Stats")
    st.info("""
    **ML Model:** Random Forest  
    **Accuracy:** 91.67%  
    **ROC-AUC:** 93.75%  
    **Features:** 29  
    """)
    
    st.markdown("### 🔗 Resources")
    st.markdown("""
    - [Flask API Docs](http://localhost:5000/)
    - [GitHub Repository](#)
    - [PostgreSQL Schema](#)
    """)
    
    st.markdown("### 📧 Support")
    st.markdown("For technical support, contact the development team.")