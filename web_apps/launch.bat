@echo off
echo ============================================
echo Federated Learning Web Applications
echo ============================================
echo.

echo Checking dependencies...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo Flask not found. Installing dependencies...
    pip install -r requirements.txt
) else (
    echo Dependencies OK!
)

echo.
echo ============================================
echo Available Web Apps:
echo ============================================
echo.
echo 1. Healthcare Demo (Multi-Hospital Collaboration)
echo    URL: http://localhost:5000
echo.
echo 2. Mobile Keyboard Demo (Keyboard Prediction)
echo    URL: http://localhost:5001
echo.
echo 3. Financial Fraud Demo (Cross-Bank Fraud Detection)
echo    URL: http://localhost:5002
echo.
echo ============================================

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Starting Healthcare Web App...
    echo Open your browser to: http://localhost:5000
    echo.
    echo Press Ctrl+C to stop the server
    echo.
    python healthcare_app.py
) else if "%choice%"=="2" (
    echo.
    echo Starting Mobile Keyboard Web App...
    echo Open your browser to: http://localhost:5001
    echo.
    echo Press Ctrl+C to stop the server
    echo.
    python mobile_keyboard_app.py
) else if "%choice%"=="3" (
    echo.
    echo Starting Financial Fraud Web App...
    echo Open your browser to: http://localhost:5002
    echo.
    echo Press Ctrl+C to stop the server
    echo.
    python financial_fraud_app.py
) else (
    echo Invalid choice!
    pause
)