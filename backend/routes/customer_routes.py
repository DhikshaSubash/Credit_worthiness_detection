"""
Customer Management API Endpoints.

Handles:
- Customer registration
- Customer retrieval
- Customer information updates

These endpoints are called by the Streamlit frontend when users
submit the loan application form.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Blueprint, request, jsonify
from datetime import datetime
from backend.database import get_db_session, close_db_session
from backend.models import Customer, Employment

# ============================================
# BLUEPRINT DEFINITION
# ============================================
"""
Blueprint = A collection of related routes.

Why use blueprints?
- Keeps routes organized by feature
- Makes code easier to maintain
- Allows modular testing

This blueprint handles all /api/customers/* endpoints
"""
customer_bp = Blueprint('customer', __name__)


# ============================================
# ENDPOINT 1: REGISTER CUSTOMER
# ============================================
@customer_bp.route('/register', methods=['POST'])
def register_customer():
    """
    Register a new customer with employment details.
    
    This endpoint is called when a user fills out the loan application form.
    It creates both Customer and Employment records in a single transaction.
    
    Request Body (JSON):
    {
        "full_name": "Arjun Sharma",
        "date_of_birth": "1990-05-15",
        "gender": "Male",
        "email": "arjun.sharma@email.com",
        "phone": "9876543210",
        "address": "123 MG Road",
        "city": "Bangalore",
        "state": "Karnataka",
        "pincode": "560001",
        "pan_number": "ABCDE1234F",
        "aadhar_number": "123456789012",
        "employer_name": "TCS",
        "job_title": "Software Engineer",
        "employment_type": "Salaried",
        "monthly_income": 80000,
        "years_of_experience": 5.5,
        "employer_phone": "9876543211",
        "employment_start_date": "2018-06-01"
    }
    
    Response (Success - 201):
    {
        "message": "Customer registered successfully",
        "customer_id": 51,
        "customer_name": "Arjun Sharma"
    }
    
    Response (Error - 400/500):
    {
        "error": "Error message here"
    }
    
    Interview Notes:
    - Uses POST method (creates new resource)
    - Returns 201 (Created) on success, not 200 (OK)
    - Validates all required fields before database insertion
    - Uses database transactions (commit/rollback)
    """
    session = get_db_session()
    
    try:
        # Get JSON data from request body
        data = request.get_json()
        
        # ============================================
        # INPUT VALIDATION
        # ============================================
        """
        Why validate input?
        - Prevents SQL injection (SQLAlchemy helps, but still good practice)
        - Ensures data quality
        - Provides clear error messages to frontend
        - Avoids database constraint violations
        """
        required_fields = [
            'full_name', 'date_of_birth', 'gender', 'email', 'phone',
            'address', 'city', 'state', 'pincode', 'pan_number', 'aadhar_number',
            'employer_name', 'job_title', 'employment_type', 'monthly_income',
            'years_of_experience', 'employment_start_date'
        ]
        
        # Check if all required fields are present
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Check if email already exists (unique constraint)
        existing_customer = session.query(Customer).filter_by(email=data['email']).first()
        if existing_customer:
            return jsonify({
                "error": "Customer with this email already exists",
                "existing_customer_id": existing_customer.customer_id
            }), 400
        
        # Check if PAN already exists (unique constraint)
        existing_pan = session.query(Customer).filter_by(pan_number=data['pan_number']).first()
        if existing_pan:
            return jsonify({
                "error": "Customer with this PAN already exists"
            }), 400
        
        # Check if Aadhar already exists (unique constraint)
        existing_aadhar = session.query(Customer).filter_by(aadhar_number=data['aadhar_number']).first()
        if existing_aadhar:
            return jsonify({
                "error": "Customer with this Aadhar already exists"
            }), 400
        
        # ============================================
        # CREATE CUSTOMER RECORD
        # ============================================
        """
        Date parsing: Convert string to Python date object.
        
        Why?
        - Database expects Date type, not string
        - SQLAlchemy handles conversion, but explicit is better
        - Allows validation of date format
        """
        try:
            dob = datetime.strptime(data['date_of_birth'], '%Y-%m-%d').date()
            emp_start_date = datetime.strptime(data['employment_start_date'], '%Y-%m-%d').date()
        except ValueError:
            return jsonify({
                "error": "Invalid date format. Use YYYY-MM-DD"
            }), 400
        
        # Create Customer object
        new_customer = Customer(
            full_name=data['full_name'],
            date_of_birth=dob,
            gender=data['gender'],
            email=data['email'],
            phone=data['phone'],
            address=data['address'],
            city=data['city'],
            state=data['state'],
            pincode=data['pincode'],
            pan_number=data['pan_number'],
            aadhar_number=data['aadhar_number']
        )
        
        # Add to session (not yet committed to database)
        session.add(new_customer)
        session.flush()  # Get customer_id without committing
        
        # ============================================
        # CREATE EMPLOYMENT RECORD
        # ============================================
        """
        Why flush() before creating Employment?
        - We need customer_id for the foreign key
        - flush() executes INSERT and returns the ID
        - But transaction isn't committed yet (can still rollback)
        """
        new_employment = Employment(
            customer_id=new_customer.customer_id,
            employer_name=data['employer_name'],
            job_title=data['job_title'],
            employment_type=data['employment_type'],
            monthly_income=data['monthly_income'],
            years_of_experience=data['years_of_experience'],
            employer_phone=data.get('employer_phone', ''),
            employment_start_date=emp_start_date
        )
        
        session.add(new_employment)
        
        # ============================================
        # COMMIT TRANSACTION
        # ============================================
        """
        Commit = Save all changes to database.
        
        Why commit at the end?
        - Ensures atomicity (both Customer and Employment succeed or fail together)
        - If Employment creation fails, Customer won't be saved
        - This is called a "transaction"
        """
        session.commit()
        
        return jsonify({
            "message": "Customer registered successfully",
            "customer_id": new_customer.customer_id,
            "customer_name": new_customer.full_name,
            "email": new_customer.email
        }), 201  # 201 = Created
        
    except Exception as e:
        # Rollback transaction if any error occurs
        session.rollback()
        return jsonify({
            "error": f"Failed to register customer: {str(e)}"
        }), 500
        
    finally:
        # Always close the session (prevents connection leaks)
        close_db_session()


# ============================================
# ENDPOINT 2: GET ALL CUSTOMERS
# ============================================
@customer_bp.route('/', methods=['GET'])
def get_customers():
    """
    Retrieve all customers.
    
    Usage:
        GET /api/customers/
    
    Query Parameters (Optional):
        ?limit=10 - Limit number of results
        ?offset=0 - Skip first N results (for pagination)
    
    Response (200):
    {
        "customers": [
            {
                "customer_id": 1,
                "full_name": "Arjun Sharma",
                "email": "arjun@email.com",
                "phone": "9876543210",
                "city": "Bangalore"
            },
            ...
        ],
        "total": 50
    }
    
    Interview Note:
    - Implements pagination (important for large datasets)
    - Returns only necessary fields (not entire Customer object)
    - Uses GET method (read operation)
    """
    session = get_db_session()
    
    try:
        # Get pagination parameters from query string
        limit = request.args.get('limit', default=100, type=int)
        offset = request.args.get('offset', default=0, type=int)
        
        # Query customers with pagination
        customers = session.query(Customer)\
            .limit(limit)\
            .offset(offset)\
            .all()
        
        # Get total count (for pagination metadata)
        total = session.query(Customer).count()
        
        # Convert to JSON-serializable format
        customer_list = []
        for customer in customers:
            customer_list.append({
                "customer_id": customer.customer_id,
                "full_name": customer.full_name,
                "email": customer.email,
                "phone": customer.phone,
                "city": customer.city,
                "state": customer.state,
                "created_at": customer.created_at.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return jsonify({
            "customers": customer_list,
            "total": total,
            "limit": limit,
            "offset": offset
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to fetch customers: {str(e)}"
        }), 500
        
    finally:
        close_db_session()


# ============================================
# ENDPOINT 3: GET CUSTOMER BY ID
# ============================================
@customer_bp.route('/<int:customer_id>', methods=['GET'])
def get_customer_by_id(customer_id):
    """
    Retrieve a specific customer with full details including employment.
    
    Usage:
        GET /api/customers/51
    
    Response (200):
    {
        "customer_id": 51,
        "full_name": "Arjun Sharma",
        "email": "arjun@email.com",
        "phone": "9876543210",
        "employment": {
            "employer_name": "TCS",
            "job_title": "Software Engineer",
            "monthly_income": 80000
        }
    }
    
    Response (404):
    {
        "error": "Customer not found"
    }
    
    Interview Note:
    - Uses path parameter (customer_id in URL)
    - Joins Customer and Employment tables (uses ORM relationships)
    - Returns 404 if customer doesn't exist
    """
    session = get_db_session()
    
    try:
        # Query customer by ID
        customer = session.query(Customer).filter_by(customer_id=customer_id).first()
        
        if not customer:
            return jsonify({
                "error": "Customer not found"
            }), 404
        
        # Get employment details (using ORM relationship)
        employment = session.query(Employment).filter_by(customer_id=customer_id).first()
        
        # Build response
        customer_data = {
            "customer_id": customer.customer_id,
            "full_name": customer.full_name,
            "date_of_birth": customer.date_of_birth.strftime('%Y-%m-%d'),
            "gender": customer.gender,
            "email": customer.email,
            "phone": customer.phone,
            "address": customer.address,
            "city": customer.city,
            "state": customer.state,
            "pincode": customer.pincode,
            "pan_number": customer.pan_number,
            "aadhar_number": customer.aadhar_number,
            "created_at": customer.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add employment details if exists
        if employment:
            customer_data["employment"] = {
                "employer_name": employment.employer_name,
                "job_title": employment.job_title,
                "employment_type": employment.employment_type,
                "monthly_income": float(employment.monthly_income),
                "years_of_experience": float(employment.years_of_experience)
            }
        
        return jsonify(customer_data), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to fetch customer: {str(e)}"
        }), 500
        
    finally:
        close_db_session()