#!/bin/bash

# ============================================
# ATM Intelligence System - Startup Script
# FILE: start.sh
# ============================================

set -e

echo "🚀 Starting ATM Intelligence System..."
echo "======================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f "services/RAG_API/.env" ]; then
    echo -e "${RED}❌ Error: services/RAG_API/.env not found${NC}"
    echo "Please copy .env.template to .env and configure it"
    exit 1
fi

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Warning: Port $1 is already in use${NC}"
        return 1
    fi
    return 0
}

# Check ports
echo "Checking ports..."
check_port 8000 || echo "  RAG API port 8000 is occupied"
check_port 8501 || echo "  Streamlit port 8501 is occupied"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install RAG API dependencies
echo -e "${GREEN}📦 Installing RAG API dependencies...${NC}"
cd services/RAG_API
pip install -q -r requirements.txt
cd ../..

# Install Streamlit dependencies
echo -e "${GREEN}📦 Installing