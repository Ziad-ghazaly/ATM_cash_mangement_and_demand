@echo off
REM FastAPI RAG API Startup Script - Windows
REM FILE: services/RAG_API/start_api.bat

echo ========================================
echo  ATM RAG API - Windows Startup
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\" (
    echo [WARNING] Virtual environment not found.
    echo Creating virtual environment...
    python -m venv .venv
    echo.
)

REM Activate virtual environment
echo [1/4] Activating virtual environment...
call .venv\Scripts\activate.bat
echo.

REM Install dependencies
echo [2/4] Installing dependencies...
pip install -r requirements.txt --quiet
echo.

REM Check environment variables
echo [3/4] Checking environment configuration...
python -c "import os; from dotenv import load_dotenv; load_dotenv('../../.env'); exit(0 if os.getenv('AZURE_OPENAI_ENDPOINT') and os.getenv('AZURE_SEARCH_ENDPOINT') else 1)"
if errorlevel 1 (
    echo [ERROR] Missing environment variables in .env
    echo Please configure AZURE_OPENAI_ENDPOINT and AZURE_SEARCH_ENDPOINT
    pause
    exit /b 1
)
echo [OK] Environment configured
echo.

REM Start server
echo [4/4] Starting FastAPI server...
echo.
echo ========================================
echo  Server: http://localhost:8000
echo  API Docs: http://localhost:8000/docs
echo  Health: http://localhost:8000/health
echo ========================================
echo.
echo Press CTRL+C to stop the server
echo.

python app.py