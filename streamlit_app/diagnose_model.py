"""
Model Diagnostic Tool
Analyzes the model file and provides detailed information
"""

import os
import sys
import pickle

def diagnose_model(model_path):
    """
    Comprehensive model file diagnosis
    """
    print("\n" + "="*70)
    print(" MODEL FILE DIAGNOSTIC")
    print("="*70)
    
    # Check 1: File exists
    print("\n File Check:")
    if not os.path.exists(model_path):
        print(f"    File not found: {model_path}")
        return
    print(f"    File exists: {model_path}")
    
    # Check 2: File size
    file_size = os.path.getsize(model_path)
    print(f"    File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    # Check 3: File permissions
    print(f"    Readable: {os.access(model_path, os.R_OK)}")
    
    # Check 4: Peek at file header
    print("\n File Header Analysis:")
    try:
        with open(model_path, 'rb') as f:
            header = f.read(100)
            print(f"   First 20 bytes: {header[:20]}")
            
            # Check pickle protocol
            if header[0:2] == b'\x80\x03':
                print("    Pickle Protocol: 3")
            elif header[0:2] == b'\x80\x04':
                print("    Pickle Protocol: 4")
            elif header[0:2] == b'\x80\x05':
                print("    Pickle Protocol: 5")
            else:
                print(f"    Unknown format: {header[0:2]}")
    except Exception as e:
        print(f"    Error reading header: {e}")
    
    # Check 5: Python version info
    print("\n Python Environment:")
    print(f"   Version: {sys.version}")
    print(f"   Version Info: {sys.version_info}")
    
    # Check 6: Installed packages
    print("\n Relevant Packages:")
    packages = {
        'sklearn': 'scikit-learn',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'joblib': 'joblib',
        'azureml': 'azureml-core'
    }
    
    for pkg_import, pkg_name in packages.items():
        try:
            module = __import__(pkg_import)
            version = getattr(module, '__version__', 'unknown')
            print(f"   {pkg_name}: {version}")
        except ImportError:
            print(f"   {pkg_name}: not installed")
    
    # Check 7: Try to load and inspect
    print("\nLoad Attempts:")
    
    # Method 1: Standard pickle
    print("\n   Method 1: Standard pickle.load()")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"      SUCCESS!")
        print(f"      Type: {type(model)}")
        print(f"      Module: {type(model).__module__}")
        
        # Try to get model attributes
        if hasattr(model, '__dict__'):
            print(f"      Attributes: {list(model.__dict__.keys())[:5]}...")
        
        return model
        
    except Exception as e:
        try:
            error_str = str(e)
            print(f"       FAILED: {error_str[:100]}")
            
            # Analyze the error safely
            if "STACK_GLOBAL" in error_str.upper():
                print("\n     STACK_GLOBAL Error Detected:")
                print("      This usually means:")
                print("      - Model was saved with a different Python version")
                print("      - Required module is not in Python path")
                print("      - Azure ML specific serialization")
        except:
            print(f"     FAILED: {type(e).__name__}")
    
    # Method 2: Pickle with encoding
    print("\n   Method 2: pickle.load(encoding='latin1')")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f, encoding='latin1')
        print(f"       SUCCESS!")
        print(f"      Type: {type(model)}")
        return model
    except Exception as e:
        print(f"       FAILED: {str(e)[:100]}")
    
    # Method 3: Joblib
    print("\n   Method 3: joblib.load()")
    try:
        import joblib
        model = joblib.load(model_path)
        print(f"       SUCCESS!")
        print(f"      Type: {type(model)}")
        return model
    except Exception as e:
        print(f"       FAILED: {str(e)[:100]}")
    
    # Check 8: Recommendations
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS:")
    print("="*70)
    print("""
1. If STACK_GLOBAL error:
   - Upgrade scikit-learn: pip install --upgrade scikit-learn
   - Install Azure ML SDK: pip install azureml-core
   - Check Python version compatibility

2. If the model was trained in Azure ML Designer:
   - The model might use Azure ML specific format
   - Try exporting as a standard sklearn model
   - Or use Azure ML SDK to load it

3. Alternative solutions:
   - Re-train and save the model using joblib instead of pickle
   - Use the same Python/sklearn versions as training environment
   - Export model from Azure ML in a different format

4. Quick fix commands to try:
   pip install --upgrade scikit-learn numpy pandas
   pip install azureml-core azureml-train-automl-runtime
    """)
    
    print("="*70)

if __name__ == "__main__":
    # Check if path provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "./trained_model/atm_cash_predictor.pkl"
    
    print(f"\n Analyzing: {model_path}\n")
    
    model = diagnose_model(model_path)
    
    if model:
        print("\n" + "="*70)
        print(" MODEL LOADED SUCCESSFULLY!")
        print("="*70)
        print("\nYou can use this model in your Streamlit app.")
    else:
        print("\n" + "="*70)
        print(" COULD NOT LOAD MODEL")
        print("="*70)
        print("\nPlease follow the recommendations above.")