#!/bin/bash

# FastAPI RAG API Startup Script
# FILE: services/RAG_API/start_api.sh

echo "🚀 Starting ATM RAG API..."
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found."
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt --quiet

# Check environment variables
echo "🔍 Checking environment variables..."
if [ -z "$AZURE_OPENAI_ENDPOINT" ]; then
    echo "❌ AZURE_OPENAI_ENDPOINT not set in .env"
    exit 1
fi

if [ -z "$AZURE_SEARCH_ENDPOINT" ]; then
    echo "❌ AZURE_SEARCH_ENDPOINT not set in .env"
    exit 1
fi

echo "✅ Environment configured"
echo ""

# Start FastAPI server
echo "🌐 Starting server on http://localhost:8000"
echo "📚 API docs available at http://localhost:8000/docs"
echo ""
echo "Press CTRL+C to stop the server"
echo "================================================"
echo ""

python app.py