"""
Complete System Setup Script
FILE: setup_system.py (place in root directory)

Run this ONCE to set up the entire ATM RAG system.
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")

def check_python_version():
    """Ensure Python 3.8+"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

def check_env_file():
    """Verify .env file exists with required variables"""
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found in root directory")
        print("   Create .env with your Azure credentials")
        sys.exit(1)
    
    # Read and check required vars
    with open(env_path) as f:
        content = f.read()
    
    required = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY"
    ]
    
    missing = [var for var in required if var not in content]
    if missing:
        print(f"❌ Missing required variables in .env: {', '.join(missing)}")
        sys.exit(1)
    
    print("✅ .env file configured")

def create_virtual_envs():
    """Create virtual environments for both services"""
    print_header("Creating Virtual Environments")
    
    # RAG API venv
    rag_venv = Path("services/RAG_API/.venv")
    if not rag_venv.exists():
        print("📦 Creating RAG API virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(rag_venv)], check=True)
        print("✅ RAG API venv created")
    else:
        print("✅ RAG API venv already exists")
    
    # Streamlit venv
    streamlit_venv = Path("streamlit_app/.venv")
    if not streamlit_venv.exists():
        print("📦 Creating Streamlit virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(streamlit_venv)], check=True)
        print("✅ Streamlit venv created")
    else:
        print("✅ Streamlit venv already exists")

def install_dependencies():
    """Install dependencies for both services"""
    print_header("Installing Dependencies")
    
    # RAG API dependencies
    print("📥 Installing RAG API dependencies...")
    rag_pip = Path("services/RAG_API/.venv/Scripts/pip") if os.name == 'nt' else Path("services/RAG_API/.venv/bin/pip")
    subprocess.run([
        str(rag_pip), "install", "-r", "services/RAG_API/requirements.txt", "--quiet"
    ], check=True)
    print("✅ RAG API dependencies installed")
    
    # Streamlit dependencies
    print("📥 Installing Streamlit dependencies...")
    streamlit_pip = Path("streamlit_app/.venv/Scripts/pip") if os.name == 'nt' else Path("streamlit_app/.venv/bin/pip")
    subprocess.run([
        str(streamlit_pip), "install", "-r", "streamlit_app/requirements.txt", "--quiet"
    ], check=True)
    print("✅ Streamlit dependencies installed")

def setup_azure_search_index():
    """Create Azure AI Search index"""
    print_header("Setting Up Azure AI Search Index")
    
    rag_python = Path("services/RAG_API/.venv/Scripts/python") if os.name == 'nt' else Path("services/RAG_API/.venv/bin/python")
    
    print("🔧 Creating search index...")
    result = subprocess.run([
        str(rag_python), "services/RAG_API/setup_index.py"
    ])
    
    if result.returncode == 0:
        print("✅ Search index created successfully")
    else:
        print("⚠️  Index creation had issues (may already exist)")

def index_knowledge_base():
    """Index knowledge documents"""
    print_header("Indexing Knowledge Base")
    
    rag_python = Path("services/RAG_API/.venv/Scripts/python") if os.name == 'nt' else Path("services/RAG_API/.venv/bin/python")
    
    print("📚 Indexing ATM operational knowledge...")
    result = subprocess.run([
        str(rag_python), "services/RAG_API/index_knowledge.py"
    ])
    
    if result.returncode == 0:
        print("✅ Knowledge base indexed successfully")
    else:
        print("❌ Indexing failed")
        return False
    return True

def print_next_steps():
    """Print instructions for running the system"""
    print_header("Setup Complete! 🎉")
    
    print("""
Next Steps to Run the System:

1️⃣  START RAG API (Terminal 1):
   cd services/RAG_API
   
   Windows:
   .venv\\Scripts\\activate
   python app.py
   
   Linux/Mac:
   source .venv/bin/activate
   python app.py
   
   API will run on: http://localhost:8000
   API Docs: http://localhost:8000/docs

2️⃣  START STREAMLIT (Terminal 2):
   cd streamlit_app
   
   Windows:
   .venv\\Scripts\\activate
   streamlit run app.py
   
   Linux/Mac:
   source .venv/bin/activate
   streamlit run app.py
   
   Streamlit will run on: http://localhost:8501

3️⃣  USE THE SYSTEM:
   - Upload kuwait_data2.csv in Streamlit
   - Make predictions
   - Ask questions to the AI Assistant!

4️⃣  EXAMPLE QUESTIONS:
   - "Which ATMs are at risk of running out of cash?"
   - "What refill amounts do you recommend?"
   - "Show me top performing ATMs"
   - "Compare performance across cities"

═══════════════════════════════════════════════════
For detailed testing guide, see: TESTING_GUIDE.md
═══════════════════════════════════════════════════
    """)

def main():
    print_header("ATM Cash Management RAG System - Setup")
    
    print("🔍 Checking prerequisites...")
    check_python_version()
    check_env_file()
    
    create_virtual_envs()
    install_dependencies()
    setup_azure_search_index()
    
    # Ask if user wants to index knowledge now
    response = input("\n📚 Index knowledge base now? (y/n): ").lower()
    if response == 'y':
        index_knowledge_base()
    else:
        print("ℹ️  You can index later by running:")
        print("   cd services/RAG_API")
        print("   python index_knowledge.py")
    
    print_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed: {e}")
        sys.exit(1)