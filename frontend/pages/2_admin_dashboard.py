"""
Admin Dashboard - Portfolio Analytics

Displays:
1. Portfolio health metrics (NPA ratio, default rate)
2. Loan distribution visualizations
3. Risk concentration analysis
4. Repayment performance tracking
"""

import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Admin Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================
# API CONFIGURATION
# ============================================
API_BASE_URL = "http://localhost:5000/api"

# ============================================
# HELPER FUNCTIONS
# ============================================

@st.cache_data(ttl=60)  # Cache for 60 seconds
def fetch_portfolio_summary():
    """Fetch portfolio summary from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/portfolio/summary")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None


@st.cache_data(ttl=60)
def fetch_npa_analysis():
    """Fetch NPA breakdown"""
    try:
        response = requests.get(f"{API_BASE_URL}/portfolio/npa-analysis")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None


@st.cache_data(ttl=60)
def fetch_repayment_stats():
    """Fetch repayment statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/portfolio/repayment-stats")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None


@st.cache_data(ttl=60)
def fetch_loan_distribution():
    """Fetch loan distribution"""
    try:
        response = requests.get(f"{API_BASE_URL}/portfolio/loan-distribution")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None


# ============================================
# MAIN PAGE
# ============================================

st.title("📊 Portfolio Analytics Dashboard")
st.markdown("Real-time monitoring of loan portfolio health and risk metrics")
st.markdown("---")

# Fetch data
portfolio_data = fetch_portfolio_summary()

if not portfolio_data:
    st.error("⚠️ Cannot connect to API. Make sure Flask server is running.")
    st.code("python backend/app.py")
    st.stop()

# ============================================
# KEY METRICS ROW
# ============================================
st.subheader("🎯 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_loans = portfolio_data['loan_statistics']['total_loans']
    st.metric(
        label="Total Loans",
        value=total_loans,
        delta=f"{portfolio_data['loan_statistics']['active_loans']} Active",
        help="Total number of disbursed loans"
    )

with col2:
    disbursed = portfolio_data['financial_metrics']['total_disbursed']
    st.metric(
        label="Total Disbursed",
        value=f"₹{disbursed/10000000:.2f} Cr",
        delta="Cumulative",
        help="Total amount disbursed across all loans"
    )

with col3:
    npa_ratio = portfolio_data['risk_metrics']['npa_ratio']
    npa_status = "🟢" if npa_ratio < 3 else "🟡" if npa_ratio < 5 else "🔴"
    st.metric(
        label=f"{npa_status} NPA Ratio",
        value=f"{npa_ratio}%",
        delta="Target <3%",
        delta_color="inverse",
        help="Non-Performing Asset ratio (Industry standard: <3%)"
    )

with col4:
    default_rate = portfolio_data['risk_metrics']['default_rate']
    st.metric(
        label="Default Rate",
        value=f"{default_rate}%",
        delta=f"{portfolio_data['loan_statistics']['defaulted_loans']} Defaults",
        delta_color="inverse",
        help="Percentage of loans that defaulted"
    )

st.markdown("---")

# ============================================
# PORTFOLIO HEALTH SECTION
# ============================================
st.subheader("💼 Portfolio Health Overview")

col1, col2 = st.columns(2)

with col1:
    # Loan Status Distribution (Pie Chart)
    st.markdown("#### Loan Status Distribution")
    
    loan_stats = portfolio_data['loan_statistics']
    status_data = pd.DataFrame({
        'Status': ['Active', 'Closed', 'Defaulted'],
        'Count': [
            loan_stats['active_loans'],
            loan_stats['closed_loans'],
            loan_stats['defaulted_loans']
        ]
    })
    
    fig_status = px.pie(
        status_data,
        values='Count',
        names='Status',
        title='Loan Portfolio Composition',
        color='Status',
        color_discrete_map={
            'Active': '#2ecc71',
            'Closed': '#3498db',
            'Defaulted': '#e74c3c'
        },
        hole=0.4
    )
    fig_status.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_status, use_container_width=True)

with col2:
    # Financial Metrics (Bar Chart)
    st.markdown("#### Financial Overview")
    
    fin_metrics = portfolio_data['financial_metrics']
    financial_data = pd.DataFrame({
        'Metric': ['Disbursed', 'Outstanding', 'NPA Amount'],
        'Amount (Cr)': [
            fin_metrics['total_disbursed'] / 10000000,
            fin_metrics['total_outstanding'] / 10000000,
            fin_metrics['total_npa_amount'] / 10000000
        ]
    })
    
    fig_financial = px.bar(
        financial_data,
        x='Metric',
        y='Amount (Cr)',
        title='Financial Metrics (₹ Crores)',
        color='Metric',
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#d62728']
    )
    fig_financial.update_layout(showlegend=False)
    st.plotly_chart(fig_financial, use_container_width=True)

st.markdown("---")

# ============================================
# APPLICATION STATISTICS
# ============================================
st.subheader("📋 Application Statistics")

app_stats = portfolio_data['application_statistics']

col1, col2, col3 = st.columns(3)

with col1:
    approval_rate = app_stats['approved'] / app_stats['total_applications'] * 100
    st.metric("Approval Rate", f"{approval_rate:.1f}%", f"{app_stats['approved']}/{app_stats['total_applications']}")

with col2:
    rejection_rate = app_stats['rejected'] / app_stats['total_applications'] * 100
    st.metric("Rejection Rate", f"{rejection_rate:.1f}%", f"{app_stats['rejected']}/{app_stats['total_applications']}")

with col3:
    st.metric("Pending Review", app_stats['pending'], "Applications")

# Application funnel chart
funnel_data = pd.DataFrame({
    'Stage': ['Total Applications', 'Approved', 'Disbursed'],
    'Count': [
        app_stats['total_applications'],
        app_stats['approved'],
        loan_stats['total_loans']
    ]
})

fig_funnel = go.Figure(go.Funnel(
    y=funnel_data['Stage'],
    x=funnel_data['Count'],
    textinfo="value+percent initial",
    marker=dict(color=['#3498db', '#2ecc71', '#1f77b4'])
))
fig_funnel.update_layout(title='Application Conversion Funnel')
st.plotly_chart(fig_funnel, use_container_width=True)

st.markdown("---")

# ============================================
# NPA ANALYSIS
# ============================================
npa_data = fetch_npa_analysis()

if npa_data and npa_data['total_npa_loans'] > 0:
    st.subheader("⚠️ NPA Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total NPA Loans", npa_data['total_npa_loans'])
        st.metric("Total NPA Amount", f"₹{npa_data['total_npa_amount']/100000:.2f} L")
        
        # NPA Classification
        if npa_data['npa_by_classification']:
            st.markdown("#### NPA by Classification")
            for classification, count in npa_data['npa_by_classification'].items():
                st.write(f"**{classification}:** {count} loans")
    
    with col2:
        # NPA Amount by Classification (Pie Chart)
        if npa_data['npa_amount_by_classification']:
            npa_class_df = pd.DataFrame({
                'Classification': list(npa_data['npa_amount_by_classification'].keys()),
                'Amount': list(npa_data['npa_amount_by_classification'].values())
            })
            
            fig_npa = px.pie(
                npa_class_df,
                values='Amount',
                names='Classification',
                title='NPA Amount by Classification',
                color_discrete_sequence=px.colors.sequential.Reds
            )
            st.plotly_chart(fig_npa, use_container_width=True)
    
    st.markdown("---")

# ============================================
# REPAYMENT PERFORMANCE
# ============================================
repayment_data = fetch_repayment_stats()

if repayment_data:
    st.subheader("💳 Repayment Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Repayments", repayment_data['total_repayments'])
    
    with col2:
        st.metric("On-Time Payments", 
                 f"{repayment_data['on_time_percentage']:.1f}%",
                 f"{repayment_data['on_time_payments']} payments")
    
    with col3:
        st.metric("Late Payments", 
                 repayment_data['late_payments'],
                 delta="Track closely",
                 delta_color="inverse")
    
    with col4:
        st.metric("Late Fees Collected", 
                 f"₹{repayment_data['total_late_fees_collected']:,.0f}")
    
    # Repayment status visualization
    repay_df = pd.DataFrame({
        'Status': ['On-Time', 'Late'],
        'Count': [repayment_data['on_time_payments'], repayment_data['late_payments']]
    })
    
    fig_repay = px.bar(
        repay_df,
        x='Status',
        y='Count',
        title='Repayment Status Distribution',
        color='Status',
        color_discrete_map={'On-Time': '#2ecc71', 'Late': '#e74c3c'}
    )
    st.plotly_chart(fig_repay, use_container_width=True)
    
    st.markdown("---")

# ============================================
# LOAN DISTRIBUTION
# ============================================
loan_dist = fetch_loan_distribution()

if loan_dist:
    st.subheader("📈 Loan Distribution Analysis")
    
    # By Purpose
    if loan_dist['by_purpose']:
        st.markdown("#### Distribution by Purpose")
        
        purpose_df = pd.DataFrame([
            {'Purpose': k, 'Count': v['count'], 'Amount': v['total_amount']}
            for k, v in loan_dist['by_purpose'].items()
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_purpose_count = px.bar(
                purpose_df,
                x='Purpose',
                y='Count',
                title='Number of Loans by Purpose',
                color='Count',
                color_continuous_scale='Blues'
            )
            fig_purpose_count.update_xaxes(tickangle=45)
            st.plotly_chart(fig_purpose_count, use_container_width=True)
        
        with col2:
            fig_purpose_amount = px.bar(
                purpose_df,
                x='Purpose',
                y='Amount',
                title='Loan Amount by Purpose (₹)',
                color='Amount',
                color_continuous_scale='Greens'
            )
            fig_purpose_amount.update_xaxes(tickangle=45)
            st.plotly_chart(fig_purpose_amount, use_container_width=True)

# ============================================
# RISK ALERTS
# ============================================
st.markdown("---")
st.subheader("⚠️ Risk Alerts")

alerts = []

if npa_ratio > 5:
    alerts.append(("🔴 CRITICAL", f"NPA ratio ({npa_ratio}%) exceeds regulatory limit (5%)"))
elif npa_ratio > 3:
    alerts.append(("🟡 WARNING", f"NPA ratio ({npa_ratio}%) above target threshold (3%)"))

if default_rate > 10:
    alerts.append(("🔴 CRITICAL", f"Default rate ({default_rate}%) is high"))

if repayment_data and repayment_data['on_time_percentage'] < 80:
    alerts.append(("🟡 WARNING", f"On-time payment rate ({repayment_data['on_time_percentage']:.1f}%) is below target (80%)"))

if not alerts:
    st.success("✅ No critical alerts. Portfolio health is good.")
else:
    for level, message in alerts:
        if "CRITICAL" in level:
            st.error(f"{level}: {message}")
        else:
            st.warning(f"{level}: {message}")

# ============================================
# REFRESH BUTTON
# ============================================
st.markdown("---")
if st.button("🔄 Refresh Dashboard", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### 📊 Dashboard Info")
    st.info("""
    This dashboard provides real-time
    insights into portfolio health,
    risk metrics, and loan performance.
    
    Data refreshes every 60 seconds.
    """)
    
    st.markdown("### 🎯 Key Metrics")
    st.markdown("""
    **NPA Ratio:** < 3% (Good)  
    **Default Rate:** < 5% (Good)  
    **On-time %:** > 85% (Good)  
    """)
    
    st.markdown("### 📈 Export Options")
    st.download_button(
        label="📥 Download Portfolio Report",
        data=str(portfolio_data),
        file_name=f"portfolio_report_{pd.Timestamp.now().strftime('%Y%m%d')}.json",
        mime="application/json"
    )