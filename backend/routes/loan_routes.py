"""
Loan Management API Endpoints.

Handles:
- Loan application submission (with ML & SHAP integration)
- Application status retrieval
- Loan approval/rejection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
from decimal import Decimal
from backend.database import get_db_session, close_db_session
from backend.models import Application, Customer, Loan, ApprovedLoan
from ml.predict import predict_credit_risk

# ============================================
# BLUEPRINT DEFINITION
# ============================================
loan_bp = Blueprint('loan', __name__)


# ============================================
# ENDPOINT 1: SUBMIT LOAN APPLICATION
# ============================================
@loan_bp.route('/apply', methods=['POST'])
def submit_application():
    """
    Submit a new loan application.
    Integrates ML Model + SHAP Explanations.
    """
    session = get_db_session()
    
    try:
        data = request.get_json()
        
        # 1. INPUT VALIDATION
        required_fields = ['customer_id', 'loan_amount', 'loan_purpose', 'loan_tenure_months', 'interest_rate']
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
        
        # 2. ML MODEL PREDICTION
        # This returns a dict with 'credit_score', 'risk_probability', AND 'contributors'
        prediction = predict_credit_risk(
            customer_id=data['customer_id'],
            loan_amount=data['loan_amount'],
            loan_tenure_months=data['loan_tenure_months'],
            interest_rate=data['interest_rate'],
            loan_purpose=data['loan_purpose']
        )
        
        # 3. DETERMINE STATUS
        if prediction['risk_level'] == 'Low':
            status = 'Approved'
        elif prediction['risk_level'] == 'High':
            status = 'Rejected'
        else:
            status = 'Pending'

        # 4. SAVE TO DATABASE
        new_application = Application(
            customer_id=data['customer_id'],
            loan_amount=Decimal(str(data['loan_amount'])),
            loan_purpose=data['loan_purpose'],
            loan_tenure_months=data['loan_tenure_months'],
            interest_rate=Decimal(str(data['interest_rate'])),
            application_status=status,
            credit_score=Decimal(str(prediction['credit_score'])),
            risk_probability=Decimal(str(prediction['risk_probability'])),
            remarks=prediction['recommendation']
        )
        
        session.add(new_application)
        session.commit()
        
        # 5. CONSTRUCT RESPONSE (THE FIX IS HERE)
        response = {
            "message": "Loan application submitted successfully",
            "application_id": new_application.application_id,
            "status": status,
            "credit_score": prediction['credit_score'],
            "risk_probability": prediction['risk_probability'],
            "recommendation": prediction['recommendation'],
            
            # --- THIS IS THE CRITICAL LINE ---
            # Pass the SHAP data from the ML model to the Frontend
            "contributors": prediction.get('contributors', []), 
            "factors": prediction.get('factors', {})
            # ---------------------------------
        }
        
        return jsonify(response), 201
        
    except Exception as e:
        session.rollback()
        return jsonify({"error": f"Failed to submit application: {str(e)}"}), 500
        
    finally:
        close_db_session()


# ============================================
# ENDPOINT 2: GET ALL APPLICATIONS
# ============================================
@loan_bp.route('/applications', methods=['GET'])
def get_applications():
    session = get_db_session()
    try:
        query = session.query(Application)
        
        # Filters
        status = request.args.get('status')
        if status: query = query.filter_by(application_status=status)
        
        customer_id = request.args.get('customer_id', type=int)
        if customer_id: query = query.filter_by(customer_id=customer_id)
        
        # Pagination
        limit = request.args.get('limit', default=50, type=int)
        offset = request.args.get('offset', default=0, type=int)
        
        applications = query.order_by(Application.application_date.desc()).limit(limit).offset(offset).all()
        total = query.count()
        
        app_list = [{
            "application_id": app.application_id,
            "customer_id": app.customer_id,
            "customer_name": app.customer.full_name,
            "loan_amount": float(app.loan_amount),
            "loan_purpose": app.loan_purpose,
            "tenure_months": app.loan_tenure_months,
            "interest_rate": float(app.interest_rate),
            "status": app.application_status,
            "credit_score": float(app.credit_score) if app.credit_score else None,
            "risk_probability": float(app.risk_probability) if app.risk_probability else None,
            "application_date": app.application_date.strftime('%Y-%m-%d')
        } for app in applications]
        
        return jsonify({
            "applications": app_list,
            "total": total,
            "limit": limit,
            "offset": offset
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        close_db_session()

# ============================================
# ENDPOINT 3: GET LOANS
# ============================================
@loan_bp.route('/loans', methods=['GET'])
def get_loans():
    session = get_db_session()
    try:
        query = session.query(Loan)
        status = request.args.get('status')
        if status: query = query.filter_by(loan_status=status)
        
        loans = query.limit(50).all()
        
        loan_list = [{
            "loan_id": l.loan_id,
            "customer_name": l.customer.full_name,
            "amount": float(l.loan_amount),
            "status": l.loan_status
        } for l in loans]
        
        return jsonify({"loans": loan_list}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        close_db_session()