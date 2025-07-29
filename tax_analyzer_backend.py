# 1. Create startup.sh file in your project root
#!/bin/bash
# startup.sh
set -e

echo "Starting Tax Analyzer Backend..."

# Activate virtual environment if it exists
if [ -d "antenv" ]; then
    source antenv/bin/activate
    echo "Virtual environment activated"
fi

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the Flask application
echo "Starting Flask application on port 8000..."
python tax_analyzer_backend.py

---

# 2. Update requirements.txt (fix duplicate werkzeug)
Flask==2.3.2
Flask-Cors==4.0.0
psycopg2-binary==2.9.9
PyMuPDF==1.23.22
requests==2.31.0
python-dotenv==1.0.0
werkzeug==2.3.7
gunicorn==21.2.0

---

# 3. Create/update tax_analyzer_backend.py (main fixes)
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import fitz
import requests
import json
from dotenv import load_dotenv
import contextlib

load_dotenv()
app = Flask(__name__)
CORS(app)

# --- Database Connection Details (from Environment Variables) ---
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Gemini API Details (from Environment Variables) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

# --- Database Connection Pooling ---
app.db_pool = None

def init_db_pool():
    """Initialize database connection pool with error handling"""
    global app
    try:
        if DATABASE_URL:
            app.db_pool = pool.SimpleConnectionPool(1, 10, dsn=DATABASE_URL)
            print("Database connection pool created successfully")
        else:
            print("WARNING: DATABASE_URL not set, database features will be disabled")
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        print("Database features will be disabled")
        app.db_pool = None

@contextlib.contextmanager
def get_db_connection():
    """Gets a connection from the pool."""
    if not app.db_pool:
        raise Exception("Database pool is not available.")
    
    conn = None
    try:
        conn = app.db_pool.getconn()
        yield conn
    finally:
        if conn:
            app.db_pool.putconn(conn)

def initialize_database():
    """Creates or alters the users table to include new fields."""
    if not app.db_pool:
        print("Skipping database initialization - pool not available")
        return
        
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        first_name VARCHAR(100) NOT NULL,
                        last_name VARCHAR(100) NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        dob DATE,
                        mobile_number VARCHAR(25)
                    );
                """)
                conn.commit()
                print("Database schema verified successfully.")
    except Exception as e:
        print(f"DATABASE SCHEMA ERROR: {e}")

def extract_text_from_pdf(pdf_bytes):
    """Extracts text from a PDF."""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"PDF EXTRACTION ERROR: {e}")
        return None

def call_gemini_api(text):
    """Calls the Gemini API to summarize the extracted text."""
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not configured")
        return None
        
    prompt = f"""
    You are a meticulous tax notice analyst. Your task is to analyze the following text from an IRS notice and extract specific information into a single, well-structured JSON object. Do not omit any fields. If a field's information cannot be found, return an empty string "" for that value.

    Based on the text provided, find and populate the following JSON structure:
    {{
      "noticeType": "The notice code, like 'CP23' or 'CP503C'",
      "noticeFor": "The full name of the taxpayer, e.g., 'JAMES & KAREN Q. HINDS'",
      "address": "The full address of the taxpayer, with newlines as \\n, e.g., '22 BOULDER STREET\\nHANSON, CT 00000-7253'",
      "ssn": "The Social Security Number, masked, e.g., 'nnn-nn-nnnn'",
      "amountDue": "The final total amount due as a string, e.g., '$500.73'",
      "payBy": "The payment due date as a string, e.g., 'February 20, 2018'",
      "breakdown": [
        {{ "item": "The first line item in the billing summary", "amount": "Its corresponding amount" }},
        {{ "item": "The second line item", "amount": "Its amount" }}
      ],
      "noticeMeaning": "A concise, 2-line professional explanation of what this specific notice type means.",
      "whyText": "A paragraph explaining exactly why the user received this notice, based on the text.",
      "fixSteps": {{
        "agree": "A string explaining the steps to take if the user agrees.",
        "disagree": "A string explaining the steps to take if the user disagrees."
      }},
      "paymentOptions": {{
        "online": "The URL for online payments, e.g., 'www.irs.gov/payments'",
        "mail": "Instructions for paying by mail.",
        "plan": "The URL for setting up a payment plan, e.g., 'www.irs.gov/paymentplan'"
      }},
      "helpInfo": {{
        "contact": "The primary contact phone number for questions.",
        "advocate": "Information about the Taxpayer Advocate Service, including their phone number."
      }}
    }}

    Here is the text to analyze:
    ---
    {text}
    ---
    """
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(GEMINI_API_URL, json=payload, timeout=45)
        response.raise_for_status()
        result = response.json()

        if 'candidates' in result and result['candidates'] and 'content' in result['candidates'][0] and 'parts' in result['candidates'][0]['content'] and result['candidates'][0]['content']['parts']:
            summary_json_string = result['candidates'][0]['content']['parts'][0]['text']
            if summary_json_string.strip().startswith("```json"):
                summary_json_string = summary_json_string.strip()[7:-3]
            return summary_json_string
        else:
            print("GEMINI API ERROR: Unexpected response structure.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"GEMINI API REQUEST FAILED: {e}")
        return None
    except Exception as e:
        print(f"GEMINI API UNKNOWN ERROR: {e}")
        return None

# Health check endpoint for Azure
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Azure App Service"""
    return jsonify({"status": "healthy", "service": "tax-analyzer-backend"}), 200

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({"message": "Tax Analyzer Backend API", "status": "running"}), 200

@app.route('/register', methods=['POST'])
def register_user():
    if not app.db_pool:
        return jsonify({"success": False, "message": "Database service unavailable."}), 503
        
    data = request.get_json()
    required_fields = ['firstName', 'lastName', 'email', 'password', 'dob', 'mobileNumber']
    if not data or not all(k in data for k in required_fields):
        return jsonify({"success": False, "message": "Missing required fields."}), 400

    password_hash = generate_password_hash(data['password'])
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT id FROM users WHERE email = %s;", (data['email'],))
                if cur.fetchone():
                    return jsonify({"success": False, "message": "This email address is already in use."}), 409

                sql = """
                    INSERT INTO users (first_name, last_name, email, password_hash, dob, mobile_number)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id, first_name, email;
                """
                cur.execute(sql, (data['firstName'], data['lastName'], data['email'], password_hash, data['dob'], data['mobileNumber']))
                new_user = cur.fetchone()
                conn.commit()
                return jsonify({"success": True, "user": dict(new_user)}), 201
    except Exception as e:
        print(f"REGISTRATION DB ERROR: {e}")
        return jsonify({"success": False, "message": "An internal error occurred."}), 500

@app.route('/login', methods=['POST'])
def login_user():
    if not app.db_pool:
        return jsonify({"success": False, "message": "Database service unavailable."}), 503
        
    data = request.get_json()
    if not data or not all(k in data for k in ['email', 'password']):
        return jsonify({"success": False, "message": "Missing email or password."}), 400

    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM users WHERE email = %s;", (data['email'],))
                user = cur.fetchone()
                if user and check_password_hash(user['password_hash'], data['password']):
                    user_data = {"id": user['id'], "firstName": user['first_name'], "email": user['email']}
                    return jsonify({"success": True, "user": user_data}), 200
                else:
                    return jsonify({"success": False, "message": "Invalid email or password."}), 401
    except Exception as e:
        print(f"LOGIN DB ERROR: {e}")
        return jsonify({"success": False, "message": "An internal error occurred."}), 500

@app.route('/summarize', methods=['POST'])
def summarize_notice():
    if 'notice_pdf' not in request.files:
        return jsonify({"success": False, "message": "No PDF file provided."}), 400

    file = request.files['notice_pdf']
    pdf_bytes = file.read()
    raw_text = extract_text_from_pdf(pdf_bytes)
    if not raw_text:
        return jsonify({"success": False, "message": "Could not read text from PDF."}), 500

    summary_json = call_gemini_api(raw_text)
    if not summary_json:
        return jsonify({"success": False, "message": "Failed to get summary from AI."}), 500

    try:
        summary_data = json.loads(summary_json)
        return jsonify({"success": True, "summary": summary_data}), 200
    except json.JSONDecodeError:
        print(f"AI returned invalid JSON: {summary_json}")
        return jsonify({"success": False, "message": "AI returned an invalid format."}), 500

# Initialize database connection and schema
print("Initializing Tax Analyzer Backend...")
init_db_pool()

# Initialize database schema if pool is available
if app.db_pool:
    with app.app_context():
        initialize_database()

if __name__ == '__main__':
    # Azure App Service uses PORT environment variable
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting server on port {port}")
    
    # For production, we should use a proper WSGI server
    # But for Azure App Service, this should work
    app.run(debug=False, host='0.0.0.0', port=port)

---

# 4. Create .deployment file in project root
[config]
command = bash startup.sh

---

# 5. Update GitHub Actions workflow
name: Build and deploy Python app to Azure Web App - tax-analyzer-backend

on:
  push:
    branches:
      - main
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python version
        uses: actions/setup-python@v5
        with:
          python-version: '3.9'

      - name: Create and start virtual environment
        run: |
          python -m venv antenv
          source antenv/bin/activate
      
      - name: Install dependencies
        run: |
          source antenv/bin/activate
          pip install --upgrade pip
          pip install -r requirements.txt
        
      - name: Make startup script executable
        run: chmod +x startup.sh

      - name: Zip artifact for deployment
        run: zip release.zip ./* -r

      - name: Upload artifact for deployment jobs
        uses: actions/upload-artifact@v4
        with:
          name: python-app
          path: |
            release.zip
            !antenv/

  deploy:
    runs-on: ubuntu-latest
    needs: build
    permissions:
      id-token: write
      contents: read

    steps:
      - name: Download artifact from build job
        uses: actions/download-artifact@v4
        with:
          name: python-app

      - name: Unzip artifact for deployment
        run: unzip release.zip

      - name: Login to Azure
        uses: azure/login@v2
        with:
          client-id: ${{ secrets.AZUREAPPSERVICE_CLIENTID_7C2650E62F714687BA80D61865985BC6 }}
          tenant-id: ${{ secrets.AZUREAPPSERVICE_TENANTID_3BEC2FAD5A034E82A40C6FA6499D457D }}
          subscription-id: ${{ secrets.AZUREAPPSERVICE_SUBSCRIPTIONID_13C3E239FFA446D18C4F6EC4C43F42D3 }}

      - name: 'Deploy to Azure Web App'
        uses: azure/webapps-deploy@v3
        id: deploy-to-webapp
        with:
          app-name: 'tax-analyzer-backend'
          slot-name: 'Production'
          app-settings-json: |
            [
                { "name": "DATABASE_URL", "value": "${{ secrets.DATABASE_URL }}", "slotSetting": false },
                { "name": "GEMINI_API_KEY", "value": "${{ secrets.GEMINI_API_KEY }}", "slotSetting": false },
                { "name": "PORT", "value": "8000", "slotSetting": false },
                { "name": "PYTHONPATH", "value": "/home/site/wwwroot", "slotSetting": false }
            ]
