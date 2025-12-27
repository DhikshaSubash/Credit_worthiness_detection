"""
Loan Application Form Page

Allows users to:
1. Select existing customer or register new one
2. Submit loan application
3. Get real-time ML credit risk assessment
4. View approval/rejection decision
"""

import streamlit as st
import requests
import json
from datetime import datetime

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Loan Application",
    page_icon="📝",
    layout="wide"
)

# ============================================
# API CONFIGURATION
# ============================================
API_BASE_URL = "http://localhost:5000/api"

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_customers():
    """Fetch all customers from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/customers/", params={"limit": 100})
        if response.status_code == 200:
            return response.json()['customers']
        else:
            st.error(f"Failed to fetch customers: {response.text}")
            return []
    except Exception as e:
        st.error(f"API connection error: {str(e)}")
        st.warning("Make sure Flask server is running: `python backend/app.py`")
        return []


def submit_loan_application(customer_id, loan_amount, loan_tenure, interest_rate, loan_purpose):
    """Submit loan application to API"""
    try:
        payload = {
            "customer_id": int(customer_id),
            "loan_amount": float(loan_amount),
            "loan_tenure_months": int(loan_tenure),
            "interest_rate": float(interest_rate),
            "loan_purpose": loan_purpose
        }
        
        response = requests.post(
            f"{API_BASE_URL}/loans/apply",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        return response.status_code, response.json()
        
    except Exception as e:
        return 500, {"error": str(e)}


# ============================================
# MAIN PAGE
# ============================================

st.title("📝 Loan Application Form")
st.markdown("Submit a new loan application with AI-powered credit risk assessment")
st.markdown("---")

# ============================================
# CUSTOMER SELECTION
# ============================================
st.subheader("Step 1: Select Customer")

customers = get_customers()

if customers:
    # Create customer options for dropdown
    customer_options = {
        f"{c['full_name']} (ID: {c['customer_id']}, {c['city']})": c['customer_id'] 
        for c in customers
    }
    
    selected_customer_label = st.selectbox(
        "Choose an existing customer:",
        options=list(customer_options.keys()),
        help="Select the customer who is applying for the loan"
    )
    
    selected_customer_id = customer_options[selected_customer_label]
    
    # Display customer info
    with st.expander("📋 Customer Details"):
        selected_customer = next(c for c in customers if c['customer_id'] == selected_customer_id)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Name:** {selected_customer['full_name']}")
            st.write(f"**Email:** {selected_customer['email']}")
            st.write(f"**Phone:** {selected_customer['phone']}")
        with col2:
            st.write(f"**City:** {selected_customer['city']}")
            st.write(f"**State:** {selected_customer['state']}")
            st.write(f"**Registered:** {selected_customer['created_at']}")
else:
    st.warning("⚠️ No customers found. Make sure the Flask API is running and database is seeded.")
    st.stop()

st.markdown("---")

# ============================================
# LOAN APPLICATION FORM
# ============================================
st.subheader("Step 2: Loan Details")

with st.form("loan_application_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        loan_amount = st.number_input(
            "Loan Amount (₹)",
            min_value=100000,
            max_value=10000000,
            value=1000000,
            step=100000,
            help="Enter the requested loan amount"
        )
        
        loan_tenure = st.selectbox(
            "Loan Tenure (Months)",
            options=[12, 24, 36, 48, 60, 84, 120],
            index=4,  # Default 60 months
            help="Select the repayment period"
        )
        
        interest_rate = st.slider(
            "Interest Rate (%)",
            min_value=7.0,
            max_value=15.0,
            value=9.5,
            step=0.5,
            help="Annual interest rate"
        )
    
    with col2:
        loan_purpose = st.selectbox(
            "Loan Purpose",
            options=[
                "Home Purchase",
                "Vehicle Purchase",
                "Business Expansion",
                "Education",
                "Medical Emergency",
                "Debt Consolidation",
                "Home Renovation",
                "Wedding Expenses"
            ],
            help="Select the purpose of the loan"
        )
        
        # Calculate estimated EMI
        r = interest_rate / (12 * 100)
        n = loan_tenure
        if r > 0:
            emi = (loan_amount * r * (1 + r) ** n) / ((1 + r) ** n - 1)
        else:
            emi = loan_amount / n
        
        st.info(f"**Estimated EMI:** ₹{emi:,.2f}/month")
        st.info(f"**Total Repayment:** ₹{emi * loan_tenure:,.2f}")
        st.info(f"**Total Interest:** ₹{(emi * loan_tenure) - loan_amount:,.2f}")
    
    # Submit button
    submit_button = st.form_submit_button("🚀 Submit Application", use_container_width=True)

# ============================================
# PROCESS SUBMISSION
# ============================================
if submit_button:
    with st.spinner("Processing application with ML model..."):
        status_code, result = submit_loan_application(
            selected_customer_id,
            loan_amount,
            loan_tenure,
            interest_rate,
            loan_purpose
        )
    
    st.markdown("---")
    
    # ============================================
    # DISPLAY RESULTS
    # ============================================
    if status_code == 201:
        st.success("✅ Application Submitted Successfully!")
        
        # Display results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Application ID",
                value=f"#{result['application_id']}",
                help="Unique application identifier"
            )
        
        with col2:
            st.metric(
                label="Credit Score",
                value=f"{result['credit_score']:.0f}",
                delta="Good" if result['credit_score'] > 700 else "Fair" if result['credit_score'] > 600 else "Poor",
                help="ML-generated credit score (300-850)"
            )
        
        with col3:
            st.metric(
                label="Risk Probability",
                value=f"{result['risk_probability'] * 100:.2f}%",
                delta="Low" if result['risk_probability'] < 0.3 else "Medium" if result['risk_probability'] < 0.5 else "High",
                help="Probability of default predicted by ML model"
            )
        
        # ============================================
        # DECISION DISPLAY
        # ============================================
        st.markdown("### 🎯 Decision")
        
        if result['status'] == 'Approved':
            st.markdown(f"""
            <div class="success-box" style="background-color: #d4edda; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #28a745;">
                <h3 style="color: #155724; margin: 0;">✅ APPLICATION APPROVED</h3>
                <p style="color: #155724; margin-top: 0.5rem; font-size: 1.1rem;">
                    <strong>Recommendation:</strong> {result['recommendation']}<br>
                    <strong>Risk Level:</strong> Low Risk<br>
                    <strong>Next Steps:</strong> Proceed with loan disbursement process
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.balloons()
            
        elif result['status'] == 'Rejected':
            st.markdown(f"""
            <div class="danger-box" style="background-color: #f8d7da; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #dc3545;">
                <h3 style="color: #721c24; margin: 0;">❌ APPLICATION REJECTED</h3>
                <p style="color: #721c24; margin-top: 0.5rem; font-size: 1.1rem;">
                    <strong>Recommendation:</strong> {result['recommendation']}<br>
                    <strong>Risk Level:</strong> High Risk<br>
                    <strong>Reason:</strong> Credit risk score indicates high probability of default
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        else:  # Pending
            st.markdown(f"""
            <div class="warning-box" style="background-color: #fff3cd; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #ffc107;">
                <h3 style="color: #856404; margin: 0;">⏳ PENDING REVIEW</h3>
                <p style="color: #856404; margin-top: 0.5rem; font-size: 1.1rem;">
                    <strong>Recommendation:</strong> {result['recommendation']}<br>
                    <strong>Risk Level:</strong> Medium Risk<br>
                    <strong>Next Steps:</strong> Application requires manual review by credit team
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # ============================================
        # ADDITIONAL DETAILS
        # ============================================
        with st.expander("📊 Detailed Analysis"):
            st.json(result)
    
    else:
        st.error(f"❌ Application submission failed")
        st.error(f"Error: {result.get('error', 'Unknown error')}")
        st.code(json.dumps(result, indent=2))

# ============================================
# SIDEBAR INFO
# ============================================
with st.sidebar:
    st.markdown("### 📝 Application Tips")
    st.info("""
    **For Better Approval Chances:**
    - Keep DTI ratio below 40%
    - Choose appropriate tenure
    - Maintain good credit history
    - Provide accurate information
    """)
    
    st.markdown("### 📊 Credit Score Guide")
    st.markdown("""
    - **750-850:** Excellent
    - **700-749:** Good
    - **650-699:** Fair
    - **600-649:** Poor
    - **<600:** Very Poor
    """)