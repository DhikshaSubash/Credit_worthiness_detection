"""
Machine Learning Prediction Module.

This module loads the trained model and makes predictions
on new loan applications.

Used by Flask API to predict credit risk in real-time.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np
from backend.database import get_db_session, close_db_session
from backend.models import Customer, Employment
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not installed. Explainability features disabled.")

# ============================================
# LOAD TRAINED MODEL
# ============================================
"""
Load model once when module is imported (not every prediction).

Why load once?
- Loading .pkl file is slow (~100ms)
- Keep model in memory for fast predictions (~1ms)
- Standard practice in production ML systems
"""
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'credit_model.pkl')
MODEL_DATA = None

def load_model():
    """Load trained model from disk with lazy loading."""
    global MODEL_DATA
    
    if MODEL_DATA is None:
        try:
            print(f"Loading model from {MODEL_PATH}...")
            MODEL_DATA = joblib.load(MODEL_PATH)
            print(f"✓ Model loaded successfully")
            print(f"✓ Training date: {MODEL_DATA.get('training_date', 'Unknown')}")
        except FileNotFoundError:
            print("✗ Model file not found. Please train the model first:")
            print("  python ml/train_model.py")
            raise
    
    return MODEL_DATA


def prepare_features_for_prediction(customer_id, loan_amount, loan_tenure_months, 
                                    interest_rate, loan_purpose):
    """
    Prepare features for a new loan application.
    Must EXACTLY match training feature engineering.
    """
    session = get_db_session()
    
    try:
        # FETCH CUSTOMER DATA
        customer = session.query(Customer).filter_by(customer_id=customer_id).first()
        if not customer:
            raise ValueError(f"Customer {customer_id} not found")
        
        employment = session.query(Employment).filter_by(customer_id=customer_id).first()
        if not employment:
            raise ValueError(f"Employment data for customer {customer_id} not found")
        
        # CALCULATE AGE
        from datetime import datetime
        today = datetime.now().date()
        age = (today - customer.date_of_birth).days / 365.25
        
        # ENGINEER FEATURES (Same as training!)
        estimated_emi = loan_amount / loan_tenure_months
        debt_to_income_ratio = (estimated_emi / float(employment.monthly_income)) * 100
        loan_to_income_ratio = loan_amount / float(employment.monthly_income)
        
        high_risk_flag = int(
            (debt_to_income_ratio > 50) or
            (loan_to_income_ratio > 36) or
            (float(employment.monthly_income) < 30000)
        )
        
        # Age Group Encoding
        if age <= 25: age_group_encoded = 0
        elif age <= 35: age_group_encoded = 1
        elif age <= 45: age_group_encoded = 2
        elif age <= 55: age_group_encoded = 3
        else: age_group_encoded = 4
        
        # Experience Encoding
        exp = float(employment.years_of_experience)
        if exp <= 2: experience_encoded = 0
        elif exp <= 5: experience_encoded = 1
        elif exp <= 10: experience_encoded = 2
        else: experience_encoded = 3
        
        # Gender Encoding
        gender_encoded = 1 if customer.gender == 'Male' else 0
        
        # CREATE FEATURE DICTIONARY
        features = {
            'age': age,
            'gender_encoded': gender_encoded,
            'age_group_encoded': age_group_encoded,
            'monthly_income': float(employment.monthly_income),
            'years_of_experience': float(employment.years_of_experience),
            'experience_encoded': experience_encoded,
            'loan_amount': loan_amount,
            'loan_tenure_months': loan_tenure_months,
            'interest_rate': interest_rate,
            'debt_to_income_ratio': debt_to_income_ratio,
            'loan_to_income_ratio': loan_to_income_ratio,
            'high_risk_flag': high_risk_flag,
            'estimated_emi': estimated_emi,
            # One-Hot Encodings
            'employment_Self-Employed': 1 if employment.employment_type == 'Self-Employed' else 0,
            'employment_Salaried': 1 if employment.employment_type == 'Salaried' else 0,
            'purpose_Business Expansion': 1 if loan_purpose == 'Business Expansion' else 0,
            'purpose_Debt Consolidation': 1 if loan_purpose == 'Debt Consolidation' else 0,
            'purpose_Education': 1 if loan_purpose == 'Education' else 0,
            'purpose_Home Purchase': 1 if loan_purpose == 'Home Purchase' else 0,
            'purpose_Home Renovation': 1 if loan_purpose == 'Home Renovation' else 0,
            'purpose_Medical Emergency': 1 if loan_purpose == 'Medical Emergency' else 0,
            'purpose_Vehicle Purchase': 1 if loan_purpose == 'Vehicle Purchase' else 0,
            'purpose_Wedding Expenses': 1 if loan_purpose == 'Wedding Expenses' else 0,
            'state_Karnataka': 1 if customer.state == 'Karnataka' else 0,
            'state_Maharashtra': 1 if customer.state == 'Maharashtra' else 0,
            'state_Other': 1 if customer.state not in ['Karnataka', 'Maharashtra', 'Tamil Nadu', 'Telangana', 'Delhi'] else 0,
            'state_Tamil Nadu': 1 if customer.state == 'Tamil Nadu' else 0,
            'state_Telangana': 1 if customer.state == 'Telangana' else 0,
        }
        
        return pd.DataFrame([features])
        
    finally:
        close_db_session()

def align_features_with_model(feature_df, model_feature_names):
    """Ensure features match model's expected feature order."""
    aligned = pd.DataFrame(0, index=[0], columns=model_feature_names)
    for col in feature_df.columns:
        if col in aligned.columns:
            aligned[col] = feature_df[col].values[0]
    return aligned

def get_shap_explanation(model, input_df):
    """
    Calculate SHAP values to explain WHY the model made a specific prediction.
    Returns top 5 features driving the risk.
    """
    if not HAS_SHAP:
        return []

    try:
        # Create TreeExplainer (optimized for Random Forest)
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        # For binary classification, shap_values is a list of arrays [class_0_shap, class_1_shap]
        shap_values = explainer.shap_values(input_df)
        
        # We care about Class 1 (High Risk)
        if isinstance(shap_values, list):
            risk_contributors = shap_values[1][0]
        else:
            risk_contributors = shap_values[0]

        # Pair feature names with their impact
        feature_names = input_df.columns
        contributions = zip(feature_names, risk_contributors)
        
        # Sort by absolute impact (magnitude of contribution)
        sorted_contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
        
        # Return top 5 most important factors for THIS specific prediction
        return [
            {"feature": k, "impact": float(v)} 
            for k, v in sorted_contributions[:5]
        ]
    except Exception as e:
        print(f"Error calculating SHAP values: {str(e)}")
        return []

def predict_credit_risk(customer_id, loan_amount, loan_tenure_months, 
                        interest_rate, loan_purpose):
    """
    Predict credit risk for a loan application.
    """
    # Load model
    model_data = load_model()
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    # Prepare features
    feature_df = prepare_features_for_prediction(
        customer_id, loan_amount, loan_tenure_months, 
        interest_rate, loan_purpose
    )
    
    # Align features with model
    aligned_features = align_features_with_model(feature_df, feature_names)
    
    # MAKE PREDICTION
    risk_probability = model.predict_proba(aligned_features)[0][1]
    
    # CALCULATE CREDIT SCORE
    credit_score = 850 - (risk_probability * 550)
    credit_score = max(300, min(850, credit_score))
    
    # DETERMINE RISK LEVEL & RECOMMENDATION
    if risk_probability < 0.3:
        risk_level = 'Low'
        recommendation = 'Approve'
    elif risk_probability < 0.5:
        risk_level = 'Medium'
        recommendation = 'Manual Review Required'
    else:
        risk_level = 'High'
        recommendation = 'Reject or Require Collateral'
    
    # NEW: GET SHAP EXPLANATION
    contributors = get_shap_explanation(model, aligned_features)

    # BUILD RESPONSE
    result = {
        'credit_score': round(credit_score, 2),
        'risk_probability': round(risk_probability, 4),
        'risk_level': risk_level,
        'recommendation': recommendation,
        'model_confidence': round(max(risk_probability, 1 - risk_probability), 4),
        'contributors': contributors, # <--- Added this for Frontend Visualization
        'factors': {
            'debt_to_income_ratio': round(float(aligned_features.get('debt_to_income_ratio', [0]).values[0] if 'debt_to_income_ratio' in aligned_features.columns else 0), 2),
            'loan_to_income_ratio': round(float(aligned_features.get('loan_to_income_ratio', [0]).values[0] if 'loan_to_income_ratio' in aligned_features.columns else 0), 2),
            'monthly_income': float(aligned_features.get('monthly_income', [0]).values[0] if 'monthly_income' in aligned_features.columns else 0),
        }
    }
    
    return result


def batch_predict(applications):
    """
    Predict credit risk for multiple applications at once.
    
    Used for batch processing (e.g., overnight scoring of pending applications).
    
    Args:
        applications (list): List of application dicts
    
    Returns:
        list: List of prediction results
    
    Interview Note:
    "Batch prediction is more efficient than individual predictions.
    In production, we'd score pending applications overnight and
    cache results. This reduces API latency during business hours."
    """
    model_data = load_model()
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    results = []
    for app in applications:
        try:
            result = predict_credit_risk(
                app['customer_id'],
                app['loan_amount'],
                app['loan_tenure_months'],
                app['interest_rate'],
                app['loan_purpose']
            )
            results.append(result)
        except Exception as e:
            results.append({
                'error': str(e),
                'customer_id': app.get('customer_id')
            })
    
    return results


# ============================================
# TEST EXECUTION
# ============================================
if __name__ == "__main__":
    """
    Test prediction on a sample customer.
    
    Run this to test predictions:
        python ml/predict.py
    """
    print("="*60)
    print("TESTING CREDIT RISK PREDICTION")
    print("="*60)
    
    # First, get a valid customer ID from database
    try:
        session = get_db_session()
        first_customer = session.query(Customer).first()
        
        if not first_customer:
            print("\n✗ No customers found in database. Please run seed_data.py first.")
            close_db_session()
            exit(1)
        
        customer_id = first_customer.customer_id
        print(f"\n  Using Customer ID: {customer_id} ({first_customer.full_name})")
        close_db_session()
        
        # Test prediction
        result = predict_credit_risk(
            customer_id=customer_id,
            loan_amount=1000000,
            loan_tenure_months=60,
            interest_rate=9.5,
            loan_purpose='Home Purchase'
        )
        
        print("\nPrediction Result:")
        print(f"  Credit Score: {result['credit_score']}")
        print(f"  Risk Probability: {result['risk_probability']}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Recommendation: {result['recommendation']}")
        print(f"  Model Confidence: {result['model_confidence']}")
        print(f"\n  Key Factors:")
        for key, value in result['factors'].items():
            print(f"    {key}: {value}")
        
        print("\n✓ Prediction successful!")
        
    except Exception as e:
        print(f"\n✗ Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()