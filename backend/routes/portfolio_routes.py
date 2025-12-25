"""
Portfolio Management & Analytics API Endpoints.

Handles:
- Portfolio health metrics (NPA ratio, default rate)
- Risk concentration analysis
- Loan-to-Value (LTV) calculations
- Repayment statistics

This is what makes your project stand out for finance roles!
Companies like JPMC care deeply about portfolio risk management.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Blueprint, request, jsonify
from sqlalchemy import func
from decimal import Decimal
from backend.database import get_db_session, close_db_session
from backend.models import Loan, NPATracking, Repayment, Application, Collateral
from backend.utils.calculations import (
    calculate_npa_ratio,
    calculate_default_rate,
    calculate_ltv_ratio
)

# ============================================
# BLUEPRINT DEFINITION
# ============================================
portfolio_bp = Blueprint('portfolio', __name__)


# ============================================
# ENDPOINT 1: PORTFOLIO SUMMARY
# ============================================
@portfolio_bp.route('/summary', methods=['GET'])
def get_portfolio_summary():
    """
    Get high-level portfolio health metrics.
    
    This is the KEY endpoint that demonstrates your understanding of
    financial risk management - critical for JPMC interviews!
    
    Response (200):
    {
        "total_loans": 50,
        "active_loans": 32,
        "closed_loans": 12,
        "defaulted_loans": 6,
        "total_disbursed": 75000000.00,
        "total_outstanding": 45000000.00,
        "npa_ratio": 0.12,
        "default_rate": 0.12,
        "average_loan_size": 1500000.00
    }
    
    Interview Note:
    - NPA Ratio = (NPA Amount / Total Outstanding) × 100
    - Default Rate = (Defaulted Loans / Total Loans) × 100
    - These are standard banking metrics
    """
    session = get_db_session()
    
    try:
        # ============================================
        # LOAN STATISTICS
        # ============================================
        total_loans = session.query(Loan).count()
        active_loans = session.query(Loan).filter_by(loan_status='Active').count()
        closed_loans = session.query(Loan).filter_by(loan_status='Closed').count()
        defaulted_loans = session.query(Loan).filter_by(loan_status='Defaulted').count()
        
        # ============================================
        # FINANCIAL AGGREGATES
        # ============================================
        """
        Using SQLAlchemy's func.sum() for efficient database-level aggregation.
        
        Why aggregate in database?
        - Much faster than fetching all records and summing in Python
        - Database is optimized for aggregations
        - Reduces network overhead
        """
        total_disbursed = session.query(
            func.sum(Loan.disbursed_amount)
        ).scalar() or Decimal('0')
        
        total_outstanding = session.query(
            func.sum(Loan.outstanding_balance)
        ).filter(Loan.loan_status == 'Active').scalar() or Decimal('0')
        
        # Calculate average loan size
        avg_loan_size = session.query(
            func.avg(Loan.loan_amount)
        ).scalar() or Decimal('0')
        
        # ============================================
        # NPA CALCULATIONS
        # ============================================
        """
        NPA (Non-Performing Asset) = Loans overdue > 90 days
        
        NPA Ratio formula:
        NPA Ratio = (Total NPA Amount / Total Outstanding Loans) × 100
        
        Why it matters:
        - Banks must maintain NPA ratio below regulatory limits (typically <3%)
        - High NPA indicates poor credit quality
        - Critical metric for RBI and investors
        """
        npa_amount = session.query(
            func.sum(NPATracking.overdue_amount)
        ).scalar() or Decimal('0')
        
        npa_ratio = calculate_npa_ratio(float(npa_amount), float(total_outstanding))
        default_rate = calculate_default_rate(defaulted_loans, total_loans)
        
        # ============================================
        # APPLICATION STATISTICS
        # ============================================
        total_applications = session.query(Application).count()
        approved_applications = session.query(Application).filter_by(
            application_status='Approved'
        ).count()
        rejected_applications = session.query(Application).filter_by(
            application_status='Rejected'
        ).count()
        pending_applications = session.query(Application).filter_by(
            application_status='Pending'
        ).count()
        
        approval_rate = (approved_applications / total_applications * 100) if total_applications > 0 else 0
        
        # ============================================
        # BUILD RESPONSE
        # ============================================
        return jsonify({
            "loan_statistics": {
                "total_loans": total_loans,
                "active_loans": active_loans,
                "closed_loans": closed_loans,
                "defaulted_loans": defaulted_loans
            },
            "financial_metrics": {
                "total_disbursed": float(total_disbursed),
                "total_outstanding": float(total_outstanding),
                "average_loan_size": float(avg_loan_size),
                "total_npa_amount": float(npa_amount)
            },
            "risk_metrics": {
                "npa_ratio": round(npa_ratio, 2),
                "default_rate": round(default_rate, 2),
                "approval_rate": round(approval_rate, 2)
            },
            "application_statistics": {
                "total_applications": total_applications,
                "approved": approved_applications,
                "rejected": rejected_applications,
                "pending": pending_applications
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to fetch portfolio summary: {str(e)}"
        }), 500
        
    finally:
        close_db_session()


# ============================================
# ENDPOINT 2: NPA BREAKDOWN
# ============================================
@portfolio_bp.route('/npa-analysis', methods=['GET'])
def get_npa_analysis():
    """
    Get detailed NPA classification breakdown.
    
    Response (200):
    {
        "total_npa_loans": 6,
        "npa_by_classification": {
            "Sub-Standard": 2,
            "Doubtful": 3,
            "Loss": 1
        },
        "npa_amount_by_classification": {...}
    }
    
    Interview Note:
    - Sub-Standard: Overdue 90-180 days
    - Doubtful: Overdue 180-365 days
    - Loss: Overdue >365 days (unlikely to recover)
    """
    session = get_db_session()
    
    try:
        # Get NPA classification breakdown
        npa_breakdown = session.query(
            NPATracking.npa_classification,
            func.count(NPATracking.npa_id).label('count'),
            func.sum(NPATracking.overdue_amount).label('total_amount')
        ).group_by(NPATracking.npa_classification).all()
        
        # Format results
        classification_count = {}
        classification_amount = {}
        
        for classification, count, amount in npa_breakdown:
            if classification:
                classification_count[classification] = count
                classification_amount[classification] = float(amount) if amount else 0
        
        total_npa_loans = session.query(NPATracking).count()
        total_npa_amount = session.query(
            func.sum(NPATracking.overdue_amount)
        ).scalar() or Decimal('0')
        
        return jsonify({
            "total_npa_loans": total_npa_loans,
            "total_npa_amount": float(total_npa_amount),
            "npa_by_classification": classification_count,
            "npa_amount_by_classification": classification_amount
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to fetch NPA analysis: {str(e)}"
        }), 500
        
    finally:
        close_db_session()


# ============================================
# ENDPOINT 3: REPAYMENT PERFORMANCE
# ============================================
@portfolio_bp.route('/repayment-stats', methods=['GET'])
def get_repayment_stats():
    """
    Get repayment performance metrics.
    
    Response (200):
    {
        "total_repayments": 786,
        "on_time_payments": 700,
        "late_payments": 86,
        "on_time_percentage": 89.06,
        "total_late_fees": 25000.00
    }
    
    Interview Note:
    - On-time payment % is a leading indicator of portfolio health
    - High late payment rate signals potential future NPAs
    """
    session = get_db_session()
    
    try:
        # Total repayments
        total_repayments = session.query(Repayment).count()
        
        # Late payments (payment_date > emi_due_date)
        late_payments = session.query(Repayment).filter(
            Repayment.payment_date > Repayment.emi_due_date
        ).count()
        
        on_time_payments = total_repayments - late_payments
        on_time_percentage = (on_time_payments / total_repayments * 100) if total_repayments > 0 else 0
        
        # Total late fees collected
        total_late_fees = session.query(
            func.sum(Repayment.late_fee)
        ).scalar() or Decimal('0')
        
        # Average repayment amount
        avg_repayment = session.query(
            func.avg(Repayment.amount_paid)
        ).scalar() or Decimal('0')
        
        return jsonify({
            "total_repayments": total_repayments,
            "on_time_payments": on_time_payments,
            "late_payments": late_payments,
            "on_time_percentage": round(on_time_percentage, 2),
            "total_late_fees_collected": float(total_late_fees),
            "average_repayment_amount": float(avg_repayment)
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to fetch repayment stats: {str(e)}"
        }), 500
        
    finally:
        close_db_session()


# ============================================
# ENDPOINT 4: COLLATERAL ANALYSIS
# ============================================
@portfolio_bp.route('/collateral-analysis', methods=['GET'])
def get_collateral_analysis():
    """
    Get collateral coverage and LTV analysis.
    
    Response (200):
    {
        "total_collateral_value": 100000000.00,
        "total_loan_amount": 75000000.00,
        "average_ltv": 0.75,
        "collateral_by_type": {...}
    }
    
    Interview Note:
    - LTV (Loan-to-Value) = Loan Amount / Collateral Value
    - Lower LTV = Lower risk (more collateral coverage)
    - Banks typically cap LTV at 80-90%
    """
    session = get_db_session()
    
    try:
        # Total collateral value
        total_collateral_value = session.query(
            func.sum(Collateral.collateral_value)
        ).scalar() or Decimal('0')
        
        # Get collateral breakdown by type
        collateral_breakdown = session.query(
            Collateral.collateral_type,
            func.count(Collateral.collateral_id).label('count'),
            func.sum(Collateral.collateral_value).label('total_value')
        ).group_by(Collateral.collateral_type).all()
        
        collateral_by_type = {}
        for coll_type, count, value in collateral_breakdown:
            collateral_by_type[coll_type] = {
                "count": count,
                "total_value": float(value) if value else 0
            }
        
        # Calculate average LTV
        loans_with_collateral = session.query(
            Loan.loan_amount,
            Collateral.collateral_value
        ).join(Collateral, Loan.loan_id == Collateral.loan_id).all()
        
        if loans_with_collateral:
            ltv_ratios = [
                calculate_ltv_ratio(float(loan_amt), float(coll_val))
                for loan_amt, coll_val in loans_with_collateral
            ]
            avg_ltv = sum(ltv_ratios) / len(ltv_ratios)
        else:
            avg_ltv = 0
        
        return jsonify({
            "total_collateral_value": float(total_collateral_value),
            "collateral_by_type": collateral_by_type,
            "average_ltv_ratio": round(avg_ltv, 2),
            "loans_with_collateral": len(loans_with_collateral)
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to fetch collateral analysis: {str(e)}"
        }), 500
        
    finally:
        close_db_session()


# ============================================
# ENDPOINT 5: LOAN DISTRIBUTION
# ============================================
@portfolio_bp.route('/loan-distribution', methods=['GET'])
def get_loan_distribution():
    """
    Get loan distribution by purpose, amount buckets, and status.
    
    Useful for portfolio diversification analysis.
    
    Response (200):
    {
        "by_purpose": {...},
        "by_amount_bucket": {...},
        "by_status": {...}
    }
    """
    session = get_db_session()
    
    try:
        # Distribution by loan purpose
        purpose_dist = session.query(
            Application.loan_purpose,
            func.count(Application.application_id).label('count'),
            func.sum(Application.loan_amount).label('total_amount')
        ).group_by(Application.loan_purpose).all()
        
        by_purpose = {}
        for purpose, count, amount in purpose_dist:
            by_purpose[purpose] = {
                "count": count,
                "total_amount": float(amount) if amount else 0
            }
        
        # Distribution by loan status
        status_dist = session.query(
            Loan.loan_status,
            func.count(Loan.loan_id).label('count')
        ).group_by(Loan.loan_status).all()
        
        by_status = {status: count for status, count in status_dist}
        
        return jsonify({
            "by_purpose": by_purpose,
            "by_status": by_status
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to fetch loan distribution: {str(e)}"
        }), 500
        
    finally:
        close_db_session()