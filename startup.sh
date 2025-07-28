#!/bin/bash

# Step 1: Install system-level dependencies
# This command installs the Tesseract OCR engine and its English language pack.
echo "INFO: Installing Tesseract OCR engine..."
apt-get update -y
apt-get install -y tesseract-ocr tesseract-ocr-eng

# Step 2: Start the Gunicorn server
# Gunicorn will serve the Flask application.
# It binds to the host and port provided by Azure App Service.
# 'tax_analyzer_backend:app' tells Gunicorn to look for an object named 'app'
# in a file named 'tax_analyzer_backend.py'.
echo "INFO: Starting Gunicorn server..."
gunicorn --bind=0.0.0.0:$PORT --workers=4 --threads=2 --timeout=120 tax_analyzer_backend:app
