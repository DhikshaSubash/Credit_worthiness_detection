"""
Financial calculation utilities.

Contains formulas for:
- EMI (Equated Monthly Installment)
- NPA Ratio (Non-Performing Asset Ratio)
- LTV (Loan-to-Value Ratio)
- Default Rate
- Other credit risk metrics

Interview Gold: Know these formulas by heart!
"""


def calculate_emi(principal, annual_rate, tenure_months):
    """
    Calculate EMI using the standard loan formula.
    
    Formula:
    EMI = [P × r × (1 + r)^n] / [(1 + r)^n - 1]
    
    Where:
    - P = Principal loan amount
    - r = Monthly interest rate (annual rate / 12 / 100)
    - n = Loan tenure in months
    
    Args:
        principal (float): Loan amount
        annual_rate (float): Annual interest rate (e.g., 9.5 for 9.5%)
        tenure_months (int): Loan tenure in months
    
    Returns:
        float: Monthly EMI amount
    
    Example:
        >>> calculate_emi(1000000, 9.5, 60)
        20978.97
    
    Interview Note:
    "This is the industry-standard EMI formula used by all banks.
    It ensures equal monthly payments throughout the loan tenure,
    with interest component decreasing and principal component
    increasing over time."
    """
    # Convert annual rate to monthly decimal
    monthly_rate = annual_rate / (12 * 100)
    
    # Handle edge case: 0% interest
    if monthly_rate == 0:
        return principal / tenure_months
    
    # Apply EMI formula
    numerator = principal * monthly_rate * ((1 + monthly_rate) ** tenure_months)
    denominator = ((1 + monthly_rate) ** tenure_months) - 1
    
    emi = numerator / denominator
    
    return round(emi, 2)


def calculate_npa_ratio(npa_amount, total_outstanding):
    """
    Calculate NPA (Non-Performing Asset) Ratio.
    
    Formula:
    NPA Ratio = (Total NPA Amount / Total Outstanding Loans) × 100
    
    Args:
        npa_amount (float): Total amount in NPA category
        total_outstanding (float): Total outstanding loan amount
    
    Returns:
        float: NPA ratio as a percentage
    
    Example:
        >>> calculate_npa_ratio(5000000, 50000000)
        10.0
    
    Interview Note:
    "NPA Ratio is a key banking metric monitored by RBI (Reserve Bank of India).
    Banks must keep NPA below regulatory limits (typically 3-4%).
    High NPA indicates poor credit quality and affects profitability.
    
    RBI Classification:
    - Sub-Standard: 90-180 days overdue
    - Doubtful: 180-365 days overdue
    - Loss: >365 days overdue"
    """
    if total_outstanding == 0:
        return 0.0
    
    npa_ratio = (npa_amount / total_outstanding) * 100
    return round(npa_ratio, 2)


def calculate_default_rate(defaulted_loans, total_loans):
    """
    Calculate Default Rate (Loss Rate).
    
    Formula:
    Default Rate = (Number of Defaulted Loans / Total Loans) × 100
    
    Args:
        defaulted_loans (int): Number of loans in default
        total_loans (int): Total number of loans
    
    Returns:
        float: Default rate as a percentage
    
    Example:
        >>> calculate_default_rate(5, 50)
        10.0
    
    Interview Note:
    "Default Rate measures the percentage of loans that have defaulted.
    It's different from NPA ratio (which measures amount).
    Both metrics together give a complete picture of portfolio health."
    """
    if total_loans == 0:
        return 0.0
    
    default_rate = (defaulted_loans / total_loans) * 100
    return round(default_rate, 2)


def calculate_ltv_ratio(loan_amount, collateral_value):
    """
    Calculate LTV (Loan-to-Value) Ratio.
    
    Formula:
    LTV = (Loan Amount / Collateral Value) × 100
    
    Args:
        loan_amount (float): Loan amount disbursed
        collateral_value (float): Market value of collateral
    
    Returns:
        float: LTV ratio as a percentage
    
    Example:
        >>> calculate_ltv_ratio(800000, 1000000)
        80.0
    
    Interview Note:
    "LTV measures collateral coverage. Lower LTV = Lower risk.
    
    Typical LTV limits:
    - Home loans: 75-90%
    - Car loans: 80-85%
    - Gold loans: 70-75%
    
    If LTV > 100%, loan is under-collateralized (high risk).
    Banks require LTV below certain thresholds to mitigate loss risk."
    """
    if collateral_value == 0:
        return 100.0  # No collateral = 100% risk exposure
    
    ltv = (loan_amount / collateral_value) * 100
    return round(ltv, 2)


def calculate_debt_to_income_ratio(total_debt, monthly_income):
    """
    Calculate Debt-to-Income (DTI) Ratio.
    
    Formula:
    DTI = (Total Monthly Debt Payments / Gross Monthly Income) × 100
    
    Args:
        total_debt (float): Total monthly debt obligations
        monthly_income (float): Gross monthly income
    
    Returns:
        float: DTI ratio as a percentage
    
    Example:
        >>> calculate_debt_to_income_ratio(30000, 100000)
        30.0
    
    Interview Note:
    "DTI measures borrower's ability to repay.
    
    Industry Standards:
    - DTI < 40%: Low risk (borrower can comfortably repay)
    - DTI 40-50%: Medium risk
    - DTI > 50%: High risk (borrower is over-leveraged)
    
    This is a key input for ML credit scoring models."
    """
    if monthly_income == 0:
        return 100.0  # No income = 100% risk
    
    dti = (total_debt / monthly_income) * 100
    return round(dti, 2)


def calculate_risk_weighted_assets(loan_amount, risk_weight):
    """
    Calculate Risk-Weighted Assets (Basel III).
    
    Formula:
    RWA = Loan Amount × Risk Weight
    
    Args:
        loan_amount (float): Exposure amount
        risk_weight (float): Risk weight (0.0 - 1.0)
    
    Returns:
        float: Risk-weighted asset value
    
    Example:
        >>> calculate_risk_weighted_assets(1000000, 0.75)
        750000.0
    
    Interview Note (Advanced):
    "This is a Basel III banking regulation concept.
    Different asset classes have different risk weights:
    
    - Sovereign debt: 0% (government bonds)
    - Residential mortgages: 35-75%
    - Unsecured consumer loans: 75-100%
    - Corporate loans: 100-150%
    
    Banks must maintain capital reserves based on RWA."
    """
    rwa = loan_amount * risk_weight
    return round(rwa, 2)


def calculate_provision_coverage_ratio(provisions, npas):
    """
    Calculate Provision Coverage Ratio (PCR).
    
    Formula:
    PCR = (Total Provisions / Total NPAs) × 100
    
    Args:
        provisions (float): Total provisions set aside
        npas (float): Total NPA amount
    
    Returns:
        float: PCR as a percentage
    
    Example:
        >>> calculate_provision_coverage_ratio(4000000, 5000000)
        80.0
    
    Interview Note:
    "PCR measures how much of NPAs are covered by provisions.
    
    Higher PCR = Better (bank has set aside more reserves).
    RBI mandates minimum PCR of 70%.
    
    If PCR < 70%, bank must increase provisioning (hits profitability)."
    """
    if npas == 0:
        return 100.0
    
    pcr = (provisions / npas) * 100
    return round(pcr, 2)


# ============================================
# AMORTIZATION SCHEDULE
# ============================================

def generate_amortization_schedule(principal, annual_rate, tenure_months):
    """
    Generate complete loan amortization schedule.
    
    Returns month-by-month breakdown of principal and interest.
    
    Args:
        principal (float): Loan amount
        annual_rate (float): Annual interest rate
        tenure_months (int): Loan tenure in months
    
    Returns:
        list: List of dicts with month-wise breakdown
    
    Example:
        >>> schedule = generate_amortization_schedule(1000000, 9.5, 12)
        >>> schedule[0]
        {
            'month': 1,
            'emi': 87183.92,
            'principal': 79267.25,
            'interest': 7916.67,
            'balance': 920732.75
        }
    
    Interview Note:
    "Amortization shows how each EMI is split between principal and interest.
    Early payments are mostly interest. Later payments are mostly principal.
    This is useful for generating repayment schedules for borrowers."
    """
    emi = calculate_emi(principal, annual_rate, tenure_months)
    monthly_rate = annual_rate / (12 * 100)
    
    schedule = []
    balance = principal
    
    for month in range(1, tenure_months + 1):
        # Interest for this month
        interest_payment = balance * monthly_rate
        
        # Principal for this month
        principal_payment = emi - interest_payment
        
        # Remaining balance
        balance = balance - principal_payment
        
        # Handle last month rounding
        if month == tenure_months:
            principal_payment = principal_payment + balance
            balance = 0
        
        schedule.append({
            'month': month,
            'emi': round(emi, 2),
            'principal': round(principal_payment, 2),
            'interest': round(interest_payment, 2),
            'balance': round(max(balance, 0), 2)
        })
    
    return schedule