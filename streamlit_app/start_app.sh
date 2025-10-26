#!/bin/bash

# ATM Cash Predictor - Quick Start Script
# This script sets up and runs the Streamlit application

echo "=================================================="
echo "🏧 ATM Cash Demand Predictor - Quick Start"
echo "=================================================="
echo ""

# Step 1: Navigate to correct directory
echo "📂 Step 1: Navigating to streamlit_app directory..."
cd ~/cloudfiles/code/users/ZIG018/streamlit_app
if [ $? -ne 0 ]; then
    echo "❌ Error: Could not find streamlit_app directory"
    echo "💡 Please check if the path is correct"
    exit 1
fi
echo "✅ Current directory: $(pwd)"
echo ""

# Step 2: Activate conda environment
echo "🐍 Step 2: Activating conda environment..."
source ~/anaconda/etc/profile.d/conda.sh
conda activate azureml_py38
if [ $? -ne 0 ]; then
    echo "⚠️ Warning: Could not activate azureml_py38"
    echo "💡 Continuing with current environment..."
fi
echo "✅ Environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo ""

# Step 3: Verify setup
echo "🔍 Step 3: Verifying setup..."
if [ -f "verify_setup.py" ]; then
    python verify_setup.py
    if [ $? -ne 0 ]; then
        echo "❌ Verification failed. Please check the output above."
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "⚠️ verify_setup.py not found, skipping verification"
fi
echo ""

# Step 4: Check if streamlit is installed
echo "📦 Step 4: Checking Streamlit installation..."
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found"
    echo "📥 Installing required packages..."
    pip install -q streamlit pandas numpy scikit-learn
    echo "✅ Packages installed"
else
    echo "✅ Streamlit found: $(streamlit --version)"
fi
echo ""

# Step 5: Start Streamlit
echo "🚀 Step 5: Starting Streamlit application..."
echo ""
echo "=================================================="
echo "📱 Application will be available at:"
echo "   - Local: http://localhost:8501"
echo "   - Check the PORTS tab in VS Code"
echo "=================================================="
echo ""
echo "⚠️  Press Ctrl+C to stop the application"
echo ""
sleep 2

streamlit run app.py --server.port 8501 --server.address 0.0.0.0