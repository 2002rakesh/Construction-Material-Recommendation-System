import os
import pickle
import pandas as pd
import numpy as np
import sqlite3
import re  # For email and phone number validation
from flask import Flask, request, jsonify, send_from_directory, send_file, render_template
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import jwt
import datetime
import secrets
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from dotenv import load_dotenv  # For loading environment variables

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')  # Secure key for JWT

# Global variables for model and material data
recommendation_model = None
material_data = None

# File paths for model and data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'construction_material_recommendation_system.csv')
PICKLE_PATH = os.path.join(BASE_DIR, 'recommendation_model.pkl')

# MailerSend SMTP Configuration using environment variables
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
SMTP_USERNAME = os.getenv('SMTP_USERNAME', 'materialrecommendationsystem@gmail.com')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '2005password')
SENDER_EMAIL = os.getenv('SENDER_EMAIL', 'materialrecommendationsystem@gmail.com')

# Validate SMTP configuration
if not all([SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SENDER_EMAIL]):
    raise ValueError("SMTP configuration is incomplete. Please check your environment variables.")

# Database setup: Initialize tables for users, projects, recommendations, activity, OTPs, and suppliers
def init_db():
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    
    # Create users table with required fields
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL,
        role TEXT NOT NULL,
        approved INTEGER DEFAULT 0,
        email TEXT NOT NULL,
        name TEXT NOT NULL,
        mobile_number TEXT NOT NULL
    )''')
    
    # Create projects table for storing project details
    c.execute('''CREATE TABLE IF NOT EXISTS projects (
        project_id TEXT PRIMARY KEY,
        user TEXT,
        category TEXT,
        environmental_suitability TEXT,
        supplier_availability TEXT,
        fire_resistance TEXT,
        durability REAL,
        cost REAL,
        lead_time REAL,
        sustainability REAL,
        thermal_conductivity REAL,
        compressive_strength REAL,
        created_at TEXT,
        updated_at TEXT,
        FOREIGN KEY(user) REFERENCES users(username)
    )''')
    
    # Create recommendations table to store material recommendations
    c.execute('''CREATE TABLE IF NOT EXISTS recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id TEXT,
        username TEXT,
        created_at TEXT,
        material_name TEXT,
        durability REAL,
        cost REAL,
        suitability TEXT,
        supplier_name TEXT,
        supplier_price REAL,
        supplier_availability TEXT,
        score REAL,
        is_current INTEGER DEFAULT 1,
        FOREIGN KEY(project_id) REFERENCES projects(project_id),
        FOREIGN KEY(username) REFERENCES users(username)
    )''')
    
    # Migration: Add username and is_current columns to recommendations if they don't exist
    try:
        c.execute('ALTER TABLE recommendations ADD COLUMN username TEXT')
        c.execute('ALTER TABLE recommendations ADD CONSTRAINT fk_username FOREIGN KEY (username) REFERENCES users(username)')
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise e
    
    try:
        c.execute('ALTER TABLE recommendations ADD COLUMN is_current INTEGER DEFAULT 1')
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise e
    
    # Create user activity table for logging actions
    c.execute('''CREATE TABLE IF NOT EXISTS user_activity (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        action TEXT,
        timestamp TEXT,
        details TEXT,
        FOREIGN KEY(username) REFERENCES users(username)
    )''')
    
    # Create password reset OTPs table for forgot password functionality
    c.execute('''CREATE TABLE IF NOT EXISTS password_reset_otps (
        username TEXT PRIMARY KEY,
        otp TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        FOREIGN KEY(username) REFERENCES users(username)
    )''')
    
    # Create suppliers table for storing supplier information
    c.execute('''CREATE TABLE IF NOT EXISTS suppliers (
        supplier_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        location TEXT NOT NULL,
        material_type TEXT NOT NULL,
        contact TEXT NOT NULL,
        created_at TEXT,
        updated_at TEXT
    )''')
    
    conn.commit()
    conn.close()

    # Initialize default admin user if not exists
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    default_admin = ('admin', generate_password_hash('admin'), 'admin', 1, 'admin@example.com', 'Admin User', '1234567890')
    c.execute('INSERT OR IGNORE INTO users (username, password, role, approved, email, name, mobile_number) VALUES (?, ?, ?, ?, ?, ?, ?)', default_admin)
    
    # Initialize default suppliers if the table is empty
    c.execute('SELECT COUNT(*) FROM suppliers')
    if c.fetchone()[0] == 0:
        default_suppliers = [
            ('SUP001', 'Supplier A', 'New York', 'Concrete', 'contact@suppliera.com', datetime.datetime.now(datetime.UTC).isoformat(), None),
            ('SUP002', 'Supplier B', 'Los Angeles', 'Steel', 'contact@supplierb.com', datetime.datetime.now(datetime.UTC).isoformat(), None),
            ('SUP003', 'Supplier C', 'Chicago', 'Wood', 'contact@supplierc.com', datetime.datetime.now(datetime.UTC).isoformat(), None)
        ]
        c.executemany('INSERT INTO suppliers (supplier_id, name, location, material_type, contact, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)', default_suppliers)
    
    conn.commit()
    conn.close()

# Load the recommendation model and material data during app startup
def load_resources():
    global recommendation_model, material_data
    try:
        # Load the machine learning model from pickle file
        if not os.path.exists(PICKLE_PATH):
            raise FileNotFoundError(f"Pickle file not found: {PICKLE_PATH}")

        with open(PICKLE_PATH, 'rb') as f:
            recommendation_model = pickle.load(f)
        print("Recommendation model loaded successfully")

        # Validate that the model has a predict method
        if not hasattr(recommendation_model, 'predict'):
            raise ValueError("Loaded pickle file does not contain a valid model with a predict method")

        # Load material data from CSV file
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

        material_data = pd.read_csv(CSV_PATH)
        # Ensure all required columns are present in the CSV
        required_columns = [
            'Material ID', 'Material Name', 'Category', 'Durability Rating',
            'Cost per Unit ($)', 'Environmental Suitability', 'Supplier Availability',
            'Lead Time (days)', 'Sustainability Score', 'Thermal Conductivity (W/m·K)',
            'Compressive Strength (MPa)', 'Fire Resistance Rating'
        ]
        missing_columns = [col for col in required_columns if col not in material_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")

        # Convert numeric columns to appropriate types and handle missing values
        numeric_columns = [
            'Durability Rating', 'Cost per Unit ($)', 'Lead Time (days)',
            'Sustainability Score', 'Thermal Conductivity (W/m·K)', 'Compressive Strength (MPa)'
        ]
        for col in numeric_columns:
            material_data[col] = pd.to_numeric(material_data[col], errors='coerce').fillna(0)

        # Map fire resistance ratings to numerical values for model processing
        fire_rating_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
        material_data['Fire Resistance'] = material_data['Fire Resistance Rating'].map(fire_rating_map).fillna(0)

        print("Material data loaded successfully")
    except Exception as e:
        print(f"Error loading resources: {str(e)}")
        raise

# Initialize database and load resources
try:
    init_db()
    load_resources()
except Exception as e:
    print(f"Failed to initialize application: {str(e)}")
    exit(1)

# In-memory materials database for CRUD operations
materials_db = [
    {'id': i, 'name': row['Material Name'], 'durability': float(row['Durability Rating']),
     'cost': float(row['Cost per Unit ($)']), 'suitability': row['Environmental Suitability'],
     'lead_time': float(row['Lead Time (days)']), 'sustainability': float(row['Sustainability Score']),
     'thermal_conductivity': float(row['Thermal Conductivity (W/m·K)']), 'compressive_strength': float(row['Compressive Strength (MPa)'])}
    for i, row in material_data.iterrows()
]

# Helper function to log user activities in the database
def log_user_activity(username, action, details):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    timestamp = datetime.datetime.now(datetime.UTC).isoformat()
    c.execute('INSERT INTO user_activity (username, action, timestamp, details) VALUES (?, ?, ?, ?)',
              (username, action, timestamp, details))
    conn.commit()
    conn.close()

# Decorator to require JWT token for protected routes
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        try:
            token = token.split(" ")[1]  # Extract token from "Bearer <token>"
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = data['username']
            # Check if user is approved
            conn = sqlite3.connect('material_recommendation.db')
            c = conn.cursor()
            c.execute('SELECT approved FROM users WHERE username = ?', (current_user,))
            result = c.fetchone()
            conn.close()
            if not result or result[0] == 0:
                return jsonify({'error': 'User not approved by admin'}), 403
        except:
            return jsonify({'error': 'Token is invalid'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

# Serve static files (e.g., CSS, JS)
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Serve the CSV file for frontend access
@app.route('/construction_material_recommendation_system.csv')
def serve_csv():
    try:
        return send_file(CSV_PATH, mimetype='text/csv')
    except FileNotFoundError:
        return jsonify({'error': 'CSV file not found'}), 404

# Serve the main HTML page
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# User registration endpoint with validation
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    name = data.get('name')
    mobile_number = data.get('mobile_number')

    # Validate mandatory fields
    if not all([username, password, email, name, mobile_number]):
        return jsonify({'error': 'All fields (username, password, email, name, mobile number) are required'}), 400

    # Validate username: alphanumeric, 3-20 characters
    if not re.match(r'^[a-zA-Z0-9]{3,20}$', username):
        return jsonify({'error': 'Username must be 3-20 characters long and contain only letters and numbers'}), 400

    # Validate password: at least 6 characters
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters long'}), 400

    # Validate email format
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return jsonify({'error': 'Invalid email format'}), 400

    # Validate mobile number: exactly 10 digits
    if not re.match(r'^\d{10}$', mobile_number):
        return jsonify({'error': 'Mobile number must be exactly 10 digits'}), 400

    # Check if username already exists
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT username FROM users WHERE username = ?', (username,))
    if c.fetchone():
        conn.close()
        return jsonify({'error': 'User already exists'}), 400

    # Hash the password and insert the new user
    hashed_password = generate_password_hash(password)
    c.execute('INSERT INTO users (username, password, role, approved, email, name, mobile_number) VALUES (?, ?, ?, ?, ?, ?, ?)',
              (username, hashed_password, 'core', 0, email, name, mobile_number))
    conn.commit()
    conn.close()

    log_user_activity(username, 'register', 'User registered, awaiting admin approval')
    return jsonify({'message': 'User registered successfully, awaiting admin approval'}), 201

# Get user profile details
@app.route('/profile', methods=['GET'])
@token_required
def get_profile(current_user):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT username, role, email, name, mobile_number FROM users WHERE username = ?', (current_user,))
    user = c.fetchone()
    conn.close()

    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({
        'username': user[0],
        'role': user[1],
        'email': user[2],
        'name': user[3],
        'mobile_number': user[4]
    }), 200

# Update user profile details
@app.route('/update-profile', methods=['PUT'])
@token_required
def update_profile(current_user):
    data = request.get_json()
    email = data.get('email')
    name = data.get('name')
    mobile_number = data.get('mobile_number')

    # Validate mandatory fields
    if not all([email, name, mobile_number]):
        return jsonify({'error': 'All fields (email, name, mobile number) are required'}), 400

    # Validate email format
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return jsonify({'error': 'Invalid email format'}), 400

    # Validate mobile number: exactly 10 digits
    if not re.match(r'^\d{10}$', mobile_number):
        return jsonify({'error': 'Mobile number must be exactly 10 digits'}), 400

    # Update user details in the database
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('UPDATE users SET email = ?, name = ?, mobile_number = ? WHERE username = ?',
              (email, name, mobile_number, current_user))
    conn.commit()

    # Fetch updated user data to return
    c.execute('SELECT username, role, email, name, mobile_number FROM users WHERE username = ?', (current_user,))
    user = c.fetchone()
    conn.close()

    if not user:
        return jsonify({'error': 'User not found after update'}), 404

    log_user_activity(current_user, 'update_profile', 'Updated user profile')
    return jsonify({
        'message': 'Profile updated successfully',
        'user': {
            'username': user[0],
            'role': user[1],
            'email': user[2],
            'name': user[3],
            'mobile_number': user[4]
        }
    }), 200

# Promote user to admin (admin only)
@app.route('/promote-to-admin', methods=['POST'])
@token_required
def promote_to_admin(current_user):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE username = ?', (current_user,))
    user_role = c.fetchone()[0]
    if user_role != 'admin':
        conn.close()
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.get_json()
    username_to_promote = data.get('username')
    if not username_to_promote:
        conn.close()
        return jsonify({'error': 'Username required'}), 400

    c.execute('SELECT role FROM users WHERE username = ?', (username_to_promote,))
    result = c.fetchone()
    if not result:
        conn.close()
        return jsonify({'error': 'User not found'}), 404
    if result[0] == 'admin':
        conn.close()
        return jsonify({'error': 'User is already an admin'}), 400

    c.execute('UPDATE users SET role = ? WHERE username = ?', ('admin', username_to_promote))
    conn.commit()
    conn.close()

    log_user_activity(current_user, 'promote_to_admin', f'Promoted user {username_to_promote} to admin')
    return jsonify({'message': f'User {username_to_promote} promoted to admin'}), 200

# Demote user from admin (admin only)
@app.route('/demote-from-admin', methods=['POST'])
@token_required
def demote_from_admin(current_user):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE username = ?', (current_user,))
    user_role = c.fetchone()[0]
    if user_role != 'admin':
        conn.close()
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.get_json()
    username_to_demote = data.get('username')
    if not username_to_demote:
        conn.close()
        return jsonify({'error': 'Username required'}), 400

    if username_to_demote == current_user:
        conn.close()
        return jsonify({'error': 'Cannot demote yourself'}), 403

    c.execute('SELECT role FROM users WHERE username = ?', (username_to_demote,))
    result = c.fetchone()
    if not result:
        conn.close()
        return jsonify({'error': 'User not found'}), 404
    if result[0] != 'admin':
        conn.close()
        return jsonify({'error': 'User is not an admin'}), 400

    c.execute('UPDATE users SET role = ? WHERE username = ?', ('core', username_to_demote))
    conn.commit()
    conn.close()

    log_user_activity(current_user, 'demote_from_admin', f'Demoted user {username_to_demote} from admin')
    return jsonify({'message': f'User {username_to_demote} demoted from admin'}), 200

# Admin user management: view users
@app.route('/admin/users', methods=['GET'])
@token_required
def get_users(current_user):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE username = ?', (current_user,))
    user_role = c.fetchone()[0]
    if user_role != 'admin':
        conn.close()
        return jsonify({'error': 'Unauthorized'}), 403

    c.execute('SELECT username, role, approved, email, name, mobile_number FROM users WHERE username != ?', (current_user,))
    users_list = [{'username': row[0], 'role': row[1], 'approved': bool(row[2]), 'email': row[3], 'name': row[4], 'mobile_number': row[5]} for row in c.fetchall()]
    conn.close()
    return jsonify(users_list), 200

# Admin user management: approve user
@app.route('/admin/approve-user', methods=['POST'])
@token_required
def approve_user(current_user):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE username = ?', (current_user,))
    user_role = c.fetchone()[0]
    if user_role != 'admin':
        conn.close()
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.get_json()
    username = data.get('username')
    if not username:
        conn.close()
        return jsonify({'error': 'Username required'}), 400

    c.execute('SELECT username FROM users WHERE username = ?', (username,))
    if not c.fetchone():
        conn.close()
        return jsonify({'error': 'User not found'}), 404

    c.execute('UPDATE users SET approved = 1 WHERE username = ?', (username,))
    conn.commit()
    conn.close()

    log_user_activity(current_user, 'approve_user', f'Approved user {username}')
    return jsonify({'message': f'User {username} approved'}), 200

# Admin user management: delete user
@app.route('/admin/delete-user', methods=['POST'])
@token_required
def delete_user(current_user):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE username = ?', (current_user,))
    user_role = c.fetchone()[0]
    if user_role != 'admin':
        conn.close()
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.get_json()
    username = data.get('username')
    if not username:
        conn.close()
        return jsonify({'error': 'Username required'}), 400

    if username == current_user:
        conn.close()
        return jsonify({'error': 'Cannot delete yourself'}), 403

    c.execute('SELECT username FROM users WHERE username = ?', (username,))
    if not c.fetchone():
        conn.close()
        return jsonify({'error': 'User not found'}), 404

    c.execute('DELETE FROM users WHERE username = ?', (username,))
    c.execute('DELETE FROM projects WHERE user = ?', (username,))
    c.execute('DELETE FROM recommendations WHERE username = ?', (username,))
    c.execute('DELETE FROM user_activity WHERE username = ?', (username,))
    conn.commit()
    conn.close()

    log_user_activity(current_user, 'delete_user', f'Deleted user {username}')
    return jsonify({'message': f'User {username} deleted'}), 200

# Admin user management: view user activity
@app.route('/admin/user-activity', methods=['GET'])
@token_required
def get_user_activity(current_user):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE username = ?', (current_user,))
    user_role = c.fetchone()[0]
    if user_role != 'admin':
        conn.close()
        return jsonify({'error': 'Unauthorized'}), 403

    username = request.args.get('username')
    query = 'SELECT username, action, timestamp, details FROM user_activity'
    params = ()
    if username:
        query += ' WHERE username = ?'
        params = (username,)

    c.execute(query, params)
    activity = [{'username': row[0], 'action': row[1], 'timestamp': row[2], 'details': row[3]} for row in c.fetchall()]
    conn.close()
    return jsonify(activity), 200

# Supplier management: view suppliers
@app.route('/suppliers', methods=['GET'])
@token_required
def get_suppliers(current_user):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT * FROM suppliers')
    suppliers_list = [
        {
            'supplier_id': row[0],
            'name': row[1],
            'location': row[2],
            'material_type': row[3],
            'contact': row[4],
            'created_at': row[5],
            'updated_at': row[6]
        }
        for row in c.fetchall()
    ]
    conn.close()
    log_user_activity(current_user, 'view_suppliers', 'Viewed supplier database')
    return jsonify(suppliers_list), 200

# Supplier management: add supplier (admin only)
@app.route('/supplier', methods=['POST'])
@token_required
def add_supplier(current_user):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE username = ?', (current_user,))
    user_role = c.fetchone()[0]
    if user_role != 'admin':
        conn.close()
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.get_json()
    required_fields = ['name', 'location', 'material_type', 'contact']
    if not all(field in data for field in required_fields):
        conn.close()
        return jsonify({'error': 'Missing required fields'}), 400

    # Validate email format for contact if it looks like an email
    if '@' in data['contact'] and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', data['contact']):
        conn.close()
        return jsonify({'error': 'Invalid email format for contact'}), 400

    # Generate supplier_id
    c.execute('SELECT COUNT(*) FROM suppliers')
    supplier_count = c.fetchone()[0]
    supplier_id = f'SUP{str(supplier_count + 1).zfill(3)}'

    supplier_data = (
        supplier_id,
        data['name'],
        data['location'],
        data['material_type'],
        data['contact'],
        datetime.datetime.now(datetime.UTC).isoformat(),
        None
    )

    c.execute('''INSERT INTO suppliers (supplier_id, name, location, material_type, contact, created_at, updated_at)
              VALUES (?, ?, ?, ?, ?, ?, ?)''', supplier_data)
    conn.commit()
    conn.close()

    log_user_activity(current_user, 'add_supplier', f'Added supplier with ID {supplier_id}')
    return jsonify({'message': 'Supplier added successfully', 'supplier_id': supplier_id}), 201

# Supplier management: update supplier (admin only)
@app.route('/supplier/<supplier_id>', methods=['PUT'])
@token_required
def update_supplier(current_user, supplier_id):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE username = ?', (current_user,))
    user_role = c.fetchone()[0]
    if user_role != 'admin':
        conn.close()
        return jsonify({'error': 'Unauthorized'}), 403

    c.execute('SELECT * FROM suppliers WHERE supplier_id = ?', (supplier_id,))
    if not c.fetchone():
        conn.close()
        return jsonify({'error': 'Supplier not found'}), 404

    data = request.get_json()
    required_fields = ['name', 'location', 'material_type', 'contact']
    if not all(field in data for field in required_fields):
        conn.close()
        return jsonify({'error': 'Missing required fields'}), 400

    # Validate email format for contact if it looks like an email
    if '@' in data['contact'] and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', data['contact']):
        conn.close()
        return jsonify({'error': 'Invalid email format for contact'}), 400

    updated_data = (
        data['name'],
        data['location'],
        data['material_type'],
        data['contact'],
        datetime.datetime.now(datetime.UTC).isoformat(),
        supplier_id
    )

    c.execute('''UPDATE suppliers SET name = ?, location = ?, material_type = ?, contact = ?, updated_at = ?
              WHERE supplier_id = ?''', updated_data)
    conn.commit()
    conn.close()

    log_user_activity(current_user, 'update_supplier', f'Updated supplier with ID {supplier_id}')
    return jsonify({'message': 'Supplier updated successfully'}), 200

# Supplier management: delete supplier (admin only)
@app.route('/supplier/<supplier_id>', methods=['DELETE'])
@token_required
def delete_supplier(current_user, supplier_id):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE username = ?', (current_user,))
    user_role = c.fetchone()[0]
    if user_role != 'admin':
        conn.close()
        return jsonify({'error': 'Unauthorized'}), 403

    c.execute('SELECT * FROM suppliers WHERE supplier_id = ?', (supplier_id,))
    if not c.fetchone():
        conn.close()
        return jsonify({'error': 'Supplier not found'}), 404

    c.execute('DELETE FROM suppliers WHERE supplier_id = ?', (supplier_id,))
    conn.commit()
    conn.close()

    log_user_activity(current_user, 'delete_supplier', f'Deleted supplier with ID {supplier_id}')
    return jsonify({'message': 'Supplier deleted successfully'}), 200

# User login endpoint
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT password, role, approved FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()

    if not user or not check_password_hash(user[0], password):
        return jsonify({'error': 'Invalid credentials'}), 401

    if not user[2]:  # Check approved status
        return jsonify({'error': 'User not approved by admin'}), 403

    token = jwt.encode({
        'username': username,
        'exp': datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=24)
    }, app.config['SECRET_KEY'])

    log_user_activity(username, 'login', 'User logged in')
    return jsonify({'token': token, 'role': user[1]}), 200

# Forgot password - request OTP and send via email
@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    data = request.get_json()
    username = data.get('username')
    email_template = data.get('emailTemplate')

    if not username:
        return jsonify({'error': 'Username required'}), 400

    if not email_template:
        return jsonify({'error': 'Email template is required'}), 400

    # Check if user exists and get their email
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT username, email FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    if not user:
        conn.close()
        return jsonify({'error': 'User not found'}), 404

    # Generate 6-digit OTP
    otp = str(secrets.randbelow(1000000)).zfill(6)
    expires_at = (datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=10)).isoformat()
    c.execute('INSERT OR REPLACE INTO password_reset_otps (username, otp, expires_at) VALUES (?, ?, ?)',
              (username, otp, expires_at))
    conn.commit()
    conn.close()

    # Send OTP via email using MailerSend
    try:
        # Use the provided email template and insert the OTP
        html_body = email_template.replace('{otp}', otp)

        # Create email message
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = user[1]  # User's email
        msg['Subject'] = 'Password Reset OTP for Construction Material Recommendation System'

        # Attach HTML content
        html_content = MIMEText(html_body, 'html')
        msg.attach(html_content)

        # Connect to MailerSend SMTP server and send email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(SENDER_EMAIL, user[1], msg.as_string())
        server.quit()

        log_user_activity(username, 'forgot_password', 'Requested OTP for password reset - sent via email')
        return jsonify({'message': 'OTP sent to your email'}), 200
    except Exception as e:
        print(f"Failed to send OTP email: {str(e)}")  # Log the exact error for debugging
        log_user_activity(username, 'forgot_password_failed', f'Failed to send OTP email: {str(e)}')
        return jsonify({'error': f'Failed to send OTP email: {str(e)}'}), 500

# Reset password with OTP validation
@app.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json()
    otp = data.get('otp')
    new_password = data.get('new_password')

    if not otp or not new_password:
        return jsonify({'error': 'OTP and new password required'}), 400

    # Validate password: at least 6 characters
    if len(new_password) < 6:
        return jsonify({'error': 'New password must be at least 6 characters long'}), 400

    # Check if OTP is valid and not expired
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT username, expires_at FROM password_reset_otps WHERE otp = ?', (otp,))
    otp_data = c.fetchone()
    if not otp_data:
        conn.close()
        return jsonify({'error': 'Invalid or expired OTP'}), 400

    username, expires_at = otp_data
    if datetime.datetime.fromisoformat(expires_at) < datetime.datetime.now(datetime.UTC):
        c.execute('DELETE FROM password_reset_otps WHERE otp = ?', (otp,))
        conn.commit()
        conn.close()
        return jsonify({'error': 'OTP has expired'}), 400

    # Update password and delete OTP
    hashed_password = generate_password_hash(new_password)
    c.execute('UPDATE users SET password = ? WHERE username = ?', (hashed_password, username))
    c.execute('DELETE FROM password_reset_otps WHERE otp = ?', (otp,))
    conn.commit()
    conn.close()

    log_user_activity(username, 'reset_password', 'User reset their password')
    return jsonify({'message': 'Password reset successfully'}), 200

# Create or update project
@app.route('/project', methods=['POST'])
@token_required
def create_project(current_user):
    data = request.get_json()
    required_fields = [
        'category', 'environmental_suitability', 'supplier_availability',
        'fire_resistance', 'durability', 'cost',
        'sustainability', 'thermal_conductivity', 'compressive_strength'
    ]

    # Check for missing fields
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400

    # Validate numeric fields
    numeric_fields = ['durability', 'cost', 'sustainability', 'thermal_conductivity', 'compressive_strength']
    for field in numeric_fields:
        # Check if the field is empty or not a valid number
        if data[field] is None or data[field] == '':
            return jsonify({'error': f'{field} cannot be empty'}), 400
        try:
            float_val = float(data[field])
            if float_val <= 0:
                return jsonify({'error': f'{field} must be a positive number'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': f'{field} must be a valid number'}), 400

    # Set lead_time to a default value if not provided
    lead_time = float(data.get('lead_time', 0))  # Default to 0 if not provided

    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM projects')
    project_count = c.fetchone()[0]
    project_id = str(project_count + 1)

    project_data = (
        project_id,
        current_user,
        data['category'],
        data['environmental_suitability'],
        data['supplier_availability'],
        data['fire_resistance'],
        float(data['durability']),
        float(data['cost']),
        lead_time,
        float(data['sustainability']),
        float(data['thermal_conductivity']),
        float(data['compressive_strength']),
        datetime.datetime.now(datetime.UTC).isoformat(),
        None
    )

    c.execute('''INSERT INTO projects (project_id, user, category, environmental_suitability, supplier_availability,
              fire_resistance, durability, cost, lead_time, sustainability, thermal_conductivity, compressive_strength,
              created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', project_data)
    conn.commit()
    conn.close()

    log_user_activity(current_user, 'create_project', f'Created project with ID {project_id}')
    return jsonify({'project_id': project_id}), 201

@app.route('/project/<project_id>', methods=['PUT'])
@token_required
def update_project(current_user, project_id):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT user FROM projects WHERE project_id = ?', (project_id,))
    project = c.fetchone()
    if not project:
        conn.close()
        return jsonify({'error': 'Project not found'}), 404

    if project[0] != current_user:
        conn.close()
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.get_json()
    required_fields = [
        'category', 'environmental_suitability', 'supplier_availability',
        'fire_resistance', 'durability', 'cost',
        'sustainability', 'thermal_conductivity', 'compressive_strength'
    ]

    if not all(field in data for field in required_fields):
        conn.close()
        return jsonify({'error': 'Missing required fields'}), 400

    # Set lead_time to a default value if not provided
    lead_time = float(data.get('lead_time', 0))  # Default to 0 if not provided

    updated_data = (
        data['category'],
        data['environmental_suitability'],
        data['supplier_availability'],
        data['fire_resistance'],
        float(data['durability']),
        float(data['cost']),
        lead_time,
        float(data['sustainability']),
        float(data['thermal_conductivity']),
        float(data['compressive_strength']),
        datetime.datetime.now(datetime.UTC).isoformat(),
        project_id
    )

    c.execute('''UPDATE projects SET category = ?, environmental_suitability = ?, supplier_availability = ?,
              fire_resistance = ?, durability = ?, cost = ?, lead_time = ?, sustainability = ?,
              thermal_conductivity = ?, compressive_strength = ?, updated_at = ?
              WHERE project_id = ?''', updated_data)
    conn.commit()
    conn.close()

    log_user_activity(current_user, 'update_project', f'Updated project with ID {project_id}')
    return jsonify({'message': 'Project updated successfully'}), 200

# View all projects for the current user
@app.route('/user/projects', methods=['GET'])
@token_required
def get_user_projects(current_user):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT * FROM projects WHERE user = ?', (current_user,))
    projects_list = []
    for row in c.fetchall():
        projects_list.append({
            'project_id': row[0],
            'user': row[1],
            'category': row[2],
            'environmental_suitability': row[3],
            'supplier_availability': row[4],
            'fire_resistance': row[5],
            'durability': row[6],
            'cost': row[7],
            'lead_time': row[8],
            'sustainability': row[9],
            'thermal_conductivity': row[10],
            'compressive_strength': row[11],
            'created_at': row[12],
            'updated_at': row[13]
        })
    conn.close()
    return jsonify(projects_list), 200

# View all projects (admin only)
@app.route('/projects', methods=['GET'])
@token_required
def get_projects(current_user):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE username = ?', (current_user,))
    user_role = c.fetchone()[0]
    if user_role != 'admin':
        conn.close()
        return jsonify({'error': 'Unauthorized'}), 403

    c.execute('SELECT * FROM projects')
    projects_list = []
    for row in c.fetchall():
        projects_list.append({
            'project_id': row[0],
            'user': row[1],
            'category': row[2],
            'environmental_suitability': row[3],
            'supplier_availability': row[4],
            'fire_resistance': row[5],
            'durability': row[6],
            'cost': row[7],
            'lead_time': row[8],
            'sustainability': row[9],
            'thermal_conductivity': row[10],
            'compressive_strength': row[11],
            'created_at': row[12],
            'updated_at': row[13]
        })
    conn.close()
    return jsonify(projects_list), 200

# Generate recommendations using the pickle model
@app.route('/recommend/<project_id>', methods=['GET'])
@token_required
def recommend(current_user, project_id):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT * FROM projects WHERE project_id = ? AND user = ?', (project_id, current_user))
    project = c.fetchone()
    if not project:
        conn.close()
        return jsonify({'error': 'Project not found or unauthorized'}), 404

    # Fetch available suppliers
    c.execute('SELECT * FROM suppliers')
    suppliers = [{'supplier_id': row[0], 'name': row[1], 'material_type': row[3]} for row in c.fetchall()]

    project_input = {
        'category': project[2],
        'environmental_suitability': project[3],
        'supplier_availability': project[4],
        'fire_resistance': project[5],
        'durability': project[6],
        'cost': project[7],
        'lead_time': project[8],
        'sustainability': project[9],
        'thermal_conductivity': project[10],
        'compressive_strength': project[11]
    }

    try:
        # Prepare project input for the model
        fire_rating_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
        project_input['fire_resistance'] = fire_rating_map.get(project_input['fire_resistance'], 0)

        # Pair project input with each material for scoring
        input_data = []
        for _, material in material_data.iterrows():
            input_row = {
                'category': project_input['category'],
                'environmental_suitability': project_input['environmental_suitability'],
                'supplier_availability': project_input['supplier_availability'],
                'fire_resistance': project_input['fire_resistance'],
                'durability': project_input['durability'],
                'cost': project_input['cost'],
                'lead_time': project_input['lead_time'],
                'sustainability': project_input['sustainability'],
                'thermal_conductivity': project_input['thermal_conductivity'],
                'compressive_strength': project_input['compressive_strength'],
                'material_category': material['Category'],
                'material_env_suitability': material['Environmental Suitability'],
                'material_supplier_availability': material['Supplier Availability'],
                'material_fire_resistance': material['Fire Resistance'],
                'material_durability': material['Durability Rating'],
                'material_cost': material['Cost per Unit ($)'],
                'material_lead_time': material['Lead Time (days)'],
                'material_sustainability': material['Sustainability Score'],
                'material_thermal_conductivity': material['Thermal Conductivity (W/m·K)'],
                'material_compressive_strength': material['Compressive Strength (MPa)']
            }
            input_data.append(input_row)

        # Convert input data to DataFrame for model prediction
        input_df = pd.DataFrame(input_data)

        # Log input DataFrame for debugging
        print("Input DataFrame shape:", input_df.shape)
        print("Input DataFrame columns:", input_df.columns.tolist())

        # Ensure numerical columns are of the correct type
        numerical_columns = [
            'fire_resistance', 'durability', 'cost', 'lead_time', 'sustainability',
            'thermal_conductivity', 'compressive_strength', 'material_fire_resistance',
            'material_durability', 'material_cost', 'material_lead_time',
            'material_sustainability', 'material_thermal_conductivity', 'material_compressive_strength'
        ]
        for col in numerical_columns:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

        # Check for NaN or infinite values and handle them
        if input_df.isna().any().any():
            print("NaN values found in input_df:", input_df.isna().sum())
            input_df = input_df.fillna(0)
        input_df.replace([np.inf, -np.inf], 0, inplace=True)

        # Use the model's pipeline to preprocess and predict scores
        try:
            scores = recommendation_model.predict(input_df)
        except Exception as model_error:
            conn.close()
            print(f"Model prediction error: {str(model_error)}")
            return jsonify({'error': f'Model prediction failed: {str(model_error)}'}), 500

        # Validate scores length matches input
        if len(scores) != len(input_df):
            conn.close()
            return jsonify({'error': 'Mismatch between input data and prediction scores'}), 500

        # Mark previous recommendations as not current
        c.execute('UPDATE recommendations SET is_current = 0 WHERE project_id = ? AND username = ?', (project_id, current_user))

        # Combine scores with material data
        material_data_copy = material_data.copy()
        material_data_copy['score'] = scores
        sorted_materials = material_data_copy.sort_values(by='score', ascending=False).head(5)

        # Format recommendations for response
        recs = []
        for idx, row in sorted_materials.iterrows():
            # Find a matching supplier based on material category
            matching_supplier = next((s for s in suppliers if s['material_type'].lower() == row['Category'].lower()), None)
            supplier_name = matching_supplier['name'] if matching_supplier else 'Unknown Supplier'
            supplier_availability = row['Supplier Availability']

            rec = {
                'material_name': row['Material Name'],
                'durability': float(row['Durability Rating']),
                'cost': float(row['Cost per Unit ($)']),
                'suitability': row['Environmental Suitability'],
                'supplier': {
                    'name': supplier_name,
                    'price': float(row['Cost per Unit ($)']) * 1.1,
                    'availability': supplier_availability
                },
                'score': float(row['score'])
            }
            recs.append(rec)

        # Store recommendations in the database
        created_at = datetime.datetime.now(datetime.UTC).isoformat()
        inserted_rows = 0
        for rec in recs:
            try:
                c.execute('''INSERT INTO recommendations (project_id, username, created_at, material_name, durability, cost, suitability,
                          supplier_name, supplier_price, supplier_availability, score, is_current)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)''',
                          (project_id, current_user, created_at, rec['material_name'], rec['durability'], rec['cost'], rec['suitability'],
                           rec['supplier']['name'], rec['supplier']['price'], rec['supplier']['availability'], rec['score']))
                inserted_rows += 1
                print(f"Inserted recommendation for material {rec['material_name']} with project_id {project_id} and username {current_user}")
            except sqlite3.Error as db_error:
                conn.rollback()
                conn.close()
                print(f"Database insertion error: {str(db_error)}")
                return jsonify({'error': f'Database insertion failed: {str(db_error)}'}), 500

        # Log the number of rows inserted
        print(f"Successfully inserted {inserted_rows} recommendations into the database")

        # Verify insertion by querying the database
        c.execute('SELECT COUNT(*) FROM recommendations WHERE project_id = ? AND username = ?', (project_id, current_user))
        count = c.fetchone()[0]
        print(f"Database verification: Found {count} recommendations for project_id {project_id} and username {current_user}")

        conn.commit()
        conn.close()

        log_user_activity(current_user, 'generate_recommendations', f'Generated recommendations for project ID {project_id}')
        return jsonify(recs), 200

    except Exception as e:
        conn.rollback()
        conn.close()
        print(f"General error in recommend endpoint: {str(e)}")
        return jsonify({'error': f'Failed to generate recommendations: {str(e)}'}), 500

# Get current recommendations
@app.route('/current-recommendations/<project_id>', methods=['GET'])
@token_required
def get_current_recommendations(current_user, project_id):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT user FROM projects WHERE project_id = ?', (project_id,))
    project = c.fetchone()
    if not project:
        conn.close()
        return jsonify({'error': 'Project not found'}), 404

    if project[0] != current_user:
        conn.close()
        return jsonify({'error': 'Unauthorized'}), 403

    c.execute('''SELECT id, created_at, material_name, durability, cost, suitability, supplier_name,
              supplier_price, supplier_availability, score 
              FROM recommendations 
              WHERE project_id = ? AND username = ? AND is_current = 1
              ORDER BY created_at DESC''', 
              (project_id, current_user))
    recs = []
    for row in c.fetchall():
        rec = {
            'recommendation_id': row[0],
            'created_at': row[1],
            'material_name': row[2],
            'durability': row[3],
            'cost': row[4],
            'suitability': row[5],
            'supplier': {
                'name': row[6],
                'price': row[7],
                'availability': row[8]
            },
            'score': row[9]
        }
        recs.append(rec)
    
    conn.close()
    log_user_activity(current_user, 'view_current_recommendations', f'Viewed current recommendations for project ID {project_id}')
    return jsonify(recs), 200

# Get all recommendation history
@app.route('/recommendations/<project_id>', methods=['GET'])
@token_required
def get_recommendations(current_user, project_id):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT user FROM projects WHERE project_id = ?', (project_id,))
    project = c.fetchone()
    if not project:
        conn.close()
        return jsonify({'error': 'Project not found'}), 404

    if project[0] != current_user:
        conn.close()
        return jsonify({'error': 'Unauthorized'}), 403

    # Log the query parameters
    print(f"Querying recommendations for project_id {project_id} and username {current_user}")

    c.execute('''SELECT id, created_at, material_name, durability, cost, suitability, supplier_name,
              supplier_price, supplier_availability, score, is_current
              FROM recommendations 
              WHERE project_id = ? AND username = ?
              ORDER BY created_at DESC''', 
              (project_id, current_user))
    recs = []
    for row in c.fetchall():
        rec = {
            'recommendation_id': row[0],
            'created_at': row[1],
            'material_name': row[2],
            'durability': row[3],
            'cost': row[4],
            'suitability': row[5],
            'supplier': {
                'name': row[6],
                'price': row[7],
                'availability': row[8]
            },
            'score': row[9],
            'is_current': bool(row[10])
        }
        recs.append(rec)
        print(f"Retrieved recommendation: {rec['material_name']} at {rec['created_at']}")
    
    # Log the number of records retrieved
    print(f"Retrieved {len(recs)} recommendations for project_id {project_id} and user {current_user}")
    
    conn.close()

    log_user_activity(current_user, 'view_recommendation_history', f'Viewed recommendation history for project ID {project_id}')
    return jsonify(recs), 200

# Material management (CRUD operations)
@app.route('/materials', methods=['GET'])
@token_required
def get_materials(current_user):
    # Define suitability mapping for filtering
    suitability_map = {
        'All': 4,
        'Coastal': 3,
        'Humid': 2,
        'Dry': 1
    }

    filters = request.args
    filtered_materials = materials_db

    if 'name' in filters:
        filtered_materials = [m for m in filtered_materials if filters['name'].lower() in m['name'].lower()]
    if 'minDurability' in filters and filters['minDurability']:
        filtered_materials = [m for m in filtered_materials if m['durability'] >= float(filters['minDurability'])]
    if 'maxDurability' in filters and filters['maxDurability']:
        filtered_materials = [m for m in filtered_materials if m['durability'] <= float(filters['maxDurability'])]
    if 'minCost' in filters and filters['minCost']:
        filtered_materials = [m for m in filtered_materials if m['cost'] >= float(filters['minCost'])]
    if 'maxCost' in filters and filters['maxCost']:
        filtered_materials = [m for m in filtered_materials if m['cost'] <= float(filters['maxCost'])]
    if 'minSuitability' in filters and filters['minSuitability']:
        min_suitability = float(filters['minSuitability'])
        filtered_materials = [
            m for m in filtered_materials 
            if suitability_map.get(m.get('suitability', ''), 0) >= min_suitability
        ]
    if 'maxSuitability' in filters and filters['maxSuitability']:
        max_suitability = float(filters['maxSuitability'])
        filtered_materials = [
            m for m in filtered_materials 
            if suitability_map.get(m.get('suitability', ''), 0) <= max_suitability
        ]

    log_user_activity(current_user, 'view_materials', 'Viewed materials with filters')
    return jsonify(filtered_materials), 200

@app.route('/material', methods=['POST'])
@token_required
def add_material(current_user):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE username = ?', (current_user,))
    user_role = c.fetchone()[0]
    conn.close()
    if user_role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.get_json()
    required_fields = ['name', 'durability', 'cost', 'suitability']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400

    material_id = len(materials_db) + 1
    material = {
        'id': material_id,
        'name': data['name'],
        'durability': float(data['durability']),
        'cost': float(data['cost']),
        'suitability': data['suitability']
    }
    materials_db.append(material)

    log_user_activity(current_user, 'add_material', f'Added material with ID {material_id}')
    return jsonify({'message': 'Material added successfully', 'id': material_id}), 201

@app.route('/material/<material_id>', methods=['PUT'])
@token_required
def update_material(current_user, material_id):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE username = ?', (current_user,))
    user_role = c.fetchone()[0]
    conn.close()
    if user_role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403

    material = next((m for m in materials_db if m['id'] == int(material_id)), None)
    if not material:
        return jsonify({'error': 'Material not found'}), 404

    data = request.get_json()
    required_fields = ['name', 'durability', 'cost', 'suitability']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400

    material.update({
        'name': data['name'],
        'durability': float(data['durability']),
        'cost': float(data['cost']),
        'suitability': data['suitability']
    })

    log_user_activity(current_user, 'update_material', f'Updated material with ID {material_id}')
    return jsonify({'message': 'Material updated successfully'}), 200

@app.route('/material/<material_id>', methods=['DELETE'])
@token_required
def delete_material(current_user, material_id):
    conn = sqlite3.connect('material_recommendation.db')
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE username = ?', (current_user,))
    user_role = c.fetchone()[0]
    conn.close()
    if user_role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403

    global materials_db
    material = next((m for m in materials_db if m['id'] == int(material_id)), None)
    if not material:
        return jsonify({'error': 'Material not found'}), 404

    materials_db = [m for m in materials_db if m['id'] != int(material_id)]

    log_user_activity(current_user, 'delete_material', f'Deleted material with ID {material_id}')
    return jsonify({'message': 'Material deleted successfully'}), 200

if __name__ == '__main__':
    app.run(debug=True)