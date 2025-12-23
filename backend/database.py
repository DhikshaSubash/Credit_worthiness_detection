"""
Database connection and session management using SQLAlchemy.

This module handles:
- Creating database engine (connection pool)
- Managing database sessions (transactions)
- Providing a base class for ORM models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from config import Config

# DATABASE ENGINE
"""
The Engine is the starting point for SQLAlchemy.
It maintains a connection pool to the database.

Why we use it:
- Connection pooling improves performance (reuses connections instead of creating new ones)
- Handles database connection lifecycle automatically
"""
engine = create_engine(
    Config.SQLALCHEMY_DATABASE_URI,
    echo=Config.SQLALCHEMY_ECHO,  # Set to True to see SQL queries
    pool_pre_ping=True,  # Checks if connection is alive before using it
    pool_size=10,  # Number of connections to keep in pool
    max_overflow=20  # Max additional connections if pool is full
)

# SESSION FACTORY
"""
Sessions represent a "workspace" for database operations.
Think of it like a transaction - you can add/update/delete objects,
then commit() to save changes or rollback() to undo.

Why scoped_session:
- Thread-safe: Each thread gets its own session
- Auto-cleanup: Removes sessions when done
- Required for web applications (Flask handles multiple requests simultaneously)
"""
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

# ============================================
# BASE CLASS FOR ORM MODELS
# ============================================
"""
All our table models will inherit from this Base class.
It provides:
- Automatic table name generation from class names
- Common methods (query, insert, update, delete)
- Mapping between Python classes and database tables
"""
Base = declarative_base()

# ============================================
# HELPER FUNCTIONS
# ============================================

def init_db():
    """
    Initialize the database by creating all tables.
    
    This function:
    1. Imports all models (so SQLAlchemy knows about them)
    2. Creates tables if they don't exist
    3. Should be called once when setting up the application
    
    Usage:
        from backend.database import init_db
        init_db()
    """
    # Import all models here so they're registered with Base
    from backend.models import (
        Customer, Application, Employment, Loan, 
        Collateral, Guarantor, ApprovedLoan, Repayment, NPATracking
    )
    
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ All tables created successfully!")


def get_db_session():
    """
    Get a new database session.
    
    This is used in Flask routes to interact with the database.
    
    Usage in Flask route:
        from backend.database import get_db_session
        
        @app.route('/customers')
        def get_customers():
            session = get_db_session()
            customers = session.query(Customer).all()
            session.close()
            return customers
    
    Returns:
        Session: A new database session
    """
    return Session()


def close_db_session():
    """
    Close the current database session.
    Should be called after database operations are complete.
    
    Usage:
        session = get_db_session()
        # ... do database operations ...
        close_db_session()
    """
    Session.remove()


# ============================================
# TEST CONNECTION
# ============================================
if __name__ == "__main__":
    """
    Test database connection when this file is run directly.
    
    Run this to verify database setup:
        python backend/database.py
    """
    try:
        # Try to connect to database
        connection = engine.connect()
        print("Database connection successful!")
        print(f"Connected to: {Config.DB_NAME}")
        print(f"Host: {Config.DB_HOST}:{Config.DB_PORT}")
        connection.close()
    except Exception as e:
        print("Database connection failed!")
        print(f"Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check if PostgreSQL is running")
        print("2. Verify credentials in config.py")
        print("3. Ensure credit_risk_db database exists")