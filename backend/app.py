"""
Main Flask application for Credit Risk Assessment API.

This is the entry point for the backend server.
It initializes Flask, registers blueprints, and starts the API server.

Why Flask?
- Lightweight and perfect for REST APIs
- Easy to integrate with SQLAlchemy
- Industry-standard for Python web services
- Used by companies like Netflix, LinkedIn, Uber
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify
from flask_cors import CORS
from config import Config

# Import route blueprints
from backend.routes.customer_routes import customer_bp
from backend.routes.loan_routes import loan_bp
from backend.routes.portfolio_routes import portfolio_bp

# ============================================
# FLASK APP INITIALIZATION
# ============================================

def create_app():
    """
    Application Factory Pattern.
    
    Why use this pattern?
    - Makes testing easier (can create multiple app instances)
    - Allows configuration changes without modifying core code
    - Industry best practice for Flask applications
    
    Returns:
        Flask: Configured Flask application instance
    """
    app = Flask(__name__)
    
    # Load configuration from config.py
    app.config.from_object(Config)
    
    # Enable CORS (Cross-Origin Resource Sharing)
    # This allows Streamlit (frontend) to call Flask (backend) APIs
    # Without CORS, browser blocks requests from different ports
    CORS(app, resources={
        r"/api/*": {
            "origins": "*",  # In production, specify exact origins
            "methods": ["GET", "POST", "PUT", "DELETE"],
            "allow_headers": ["Content-Type"]
        }
    })
    
    # ============================================
    # REGISTER BLUEPRINTS (Route Groups)
    # ============================================
    """
    Blueprints organize routes by feature.
    
    Why blueprints?
    - Keeps code modular and organized
    - Each feature (customers, loans, portfolio) has its own file
    - Easy to add/remove features without touching other code
    
    URL structure:
    - /api/customers/* → Customer operations
    - /api/loans/* → Loan operations
    - /api/portfolio/* → Portfolio analytics
    """
    app.register_blueprint(customer_bp, url_prefix='/api/customers')
    app.register_blueprint(loan_bp, url_prefix='/api/loans')
    app.register_blueprint(portfolio_bp, url_prefix='/api/portfolio')
    
    # ============================================
    # ROOT ROUTE (Health Check)
    # ============================================
    @app.route('/')
    def home():
        """
        Root endpoint - acts as a health check.
        
        Usage:
            GET http://localhost:5000/
        
        Returns:
            JSON: API status and available endpoints
        """
        return jsonify({
            "message": "Credit Risk Assessment API",
            "status": "running",
            "version": "1.0.0",
            "endpoints": {
                "customers": "/api/customers",
                "loans": "/api/loans",
                "portfolio": "/api/portfolio"
            }
        }), 200
    
    @app.route('/health')
    def health():
        """
        Dedicated health check endpoint.
        Used by monitoring tools to verify API is responsive.
        
        Usage:
            GET http://localhost:5000/health
        """
        return jsonify({"status": "healthy"}), 200
    
    # ============================================
    # ERROR HANDLERS
    # ============================================
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors (endpoint not found)"""
        return jsonify({
            "error": "Endpoint not found",
            "message": "The requested URL does not exist"
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors (server errors)"""
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }), 500
    
    return app


# ============================================
# APPLICATION ENTRY POINT
# ============================================

if __name__ == '__main__':
    """
    Start the Flask development server.
    
    Run this file to start the API:
        python backend/app.py
    
    The API will be accessible at:
        http://127.0.0.1:5000
    
    Interview Note:
    - debug=True enables auto-reload and detailed error messages
    - In production, set debug=False and use proper WSGI server (Gunicorn, uWSGI)
    - host='127.0.0.1' means only accessible locally
    - Use host='0.0.0.0' to make accessible from other devices on network
    """
    app = create_app()
    
    print("\n" + "="*60)
    print("🚀 CREDIT RISK ASSESSMENT API STARTING")
    print("="*60)
    print(f"📍 API running at: http://{Config.API_HOST}:{Config.API_PORT}")
    print(f"📊 Database: {Config.DB_NAME}")
    print(f"🔍 Debug mode: {Config.DEBUG}")
    print("="*60 + "\n")
    
    app.run(
        host=Config.API_HOST,
        port=Config.API_PORT,
        debug=Config.DEBUG,
        use_reloader=False  # Prevent double logging in some environments
    )