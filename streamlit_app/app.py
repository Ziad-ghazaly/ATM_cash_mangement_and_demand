"""
Unified ATM Cash Management System
FILE: streamlit_app/app_unified.py

Complete interface combining:
- ML Predictions (Random Forest, XGBoost, etc.)
- RAG Intelligent Assistant
- Interactive Analytics Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import math
from zoneinfo import ZoneInfo

from feature_engineering import ATMFeatureEngineer
from model_loader import ModelLoader

# -------------------- Configuration --------------------
import os, requests, streamlit as st

RAG_API_BASE = os.getenv("RAG_API_BASE_URL", "http://localhost:8000")

def ask_backend(question: str, top_k: int = 5):
    r = requests.post(f"{RAG_API_BASE}/ask", json={"question": question, "top_k": top_k}, timeout=60)
    r.raise_for_status()
    return r.json()

st.subheader("Ask your data (RAG)")
q = st.text_input("Type your question:")
if q:
    with st.spinner("Retrieving and generating..."):
        out = ask_backend(q)
    st.markdown("### Answer")
    st.write(out["answer"])
    with st.expander("Retrieved passages"):
        for i, p in enumerate(out["used_context"], 1):
            st.markdown(f"**{i}.** {p[:1000]}{'...' if len(p)>1000 else ''}")


# -------------------- Helpers --------------------
def fmt_kd(n: float) -> str:
    """Format as Kuwaiti Dinars"""
    try:
        if n is None or (isinstance(n, float) and math.isnan(n)):
            return "KD -"
        return f"KD {float(n):,.0f}"
    except:
        return "KD -"

def add_kd_text(fig):
    """Add KD labels to bar charts"""
    fig.update_traces(texttemplate="KD %{text:,.0f}", textposition="outside")
    return fig

def _ensure_day_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayName'] = df['Date'].dt.day_name()
        df['DayOfMonth'] = df['Date'].dt.day
    return df

def call_rag_api(question: str, city: str = None, atm_id: str = None):
    """Call RAG API endpoint"""
    try:
        payload = {"question": question, "use_tools": True}
        if city:
            payload["city"] = city
        if atm_id:
            payload["atm_id"] = atm_id
        
        response = requests.post(f"{RAG_API_URL}/ask", json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to RAG API. Make sure it's running."}
    except Exception as e:
        return {"error": str(e)}

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="ATM Intelligence Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #0F6CBD;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #2f6fe4 0%, #5b3aa0 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.75rem 0;
    }
    .prediction-value {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0.5rem 0 0.75rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #E8F4F8;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #F0F0F0;
        margin-right: 20%;
    }
    .tool-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        background-color: #0F6CBD;
        color: white;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏦 ATM Intelligence Hub</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Cash Forecasting & Intelligent Operations Assistant</p>', unsafe_allow_html=True)

# -------------------- Session State --------------------
if 'model_loader' not in st.session_state:
    st.session_state.model_loader = ModelLoader()
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'engineer' not in st.session_state:
    st.session_state.engineer = ATMFeatureEngineer()
if 'current_model_name' not in st.session_state:
    st.session_state.current_model_name = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_available' not in st.session_state:
    # Test RAG API availability
    try:
        r = requests.get(f"{RAG_API_URL}/health", timeout=5)
        st.session_state.rag_available = r.status_code == 200
    except:
        st.session_state.rag_available = False

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")

    # Model Loading
    st.subheader("Model Management")
    default_path = './trained_model/atm_rf_model.pkl'
    model_path = st.text_input("Model Path:", value=default_path)

    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        st.success(f"✓ Model found ({file_size:.1f} MB)")
    else:
        st.warning("⚠ Model not found")

    if st.button("🔄 Load Model", type="primary", use_container_width=True):
        if not os.path.exists(model_path):
            st.error(f"File not found: {model_path}")
        else:
            try:
                with st.spinner("Loading models..."):
                    model_obj, error = st.session_state.model_loader.load_model(model_path)
                    if model_obj:
                        st.session_state.model = model_obj
                        st.session_state.current_model_name = model_obj.current_model_name
                        st.success("✓ Models loaded successfully")
                        st.rerun()
                    else:
                        st.error(f"Error: {error}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.markdown("---")

    # Model Status
    if st.session_state.model is not None:
        st.subheader("📊 Model Status")
        model_info = st.session_state.model.get_model_info()
        st.success("🟢 Ready")
        st.write(f"**Active:** {model_info['current_model']}")
        
        if model_info.get('metadata'):
            with st.expander("Details"):
                st.write(model_info['metadata'])
    else:
        st.subheader("📊 Model Status")
        st.error("🔴 Not Loaded")

    st.markdown("---")

    # RAG API Status
    st.subheader("🤖 RAG Assistant")
    if st.session_state.rag_available:
        st.success("🟢 Connected")
        st.caption(f"Endpoint: {RAG_API_URL}")
    else:
        st.error("🔴 Offline")
        st.caption("Start RAG API to enable intelligent assistant")
        if st.button("🔄 Retry Connection"):
            try:
                r = requests.get(f"{RAG_API_URL}/health", timeout=5)
                st.session_state.rag_available = r.status_code == 200
                st.rerun()
            except:
                st.session_state.rag_available = False

    st.markdown("---")
    st.info("💡 **Tip:** Upload data and ask questions in natural language!")

# -------------------- MAIN TABS --------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Predictions", "🤖 AI Assistant", "📊 Analytics", "📁 Data Management"])

# ============ TAB 1: PREDICTIONS ============
with tab1:
    st.header("Next-Day Cash Demand Forecasting")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please load model from sidebar to make predictions")
    elif st.session_state.data is None:
        st.warning("⚠️ Please upload historical data in 'Data Management' tab")
    else:
        data = st.session_state.data
        
        # Prediction Interface
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            input_method = st.radio("ATM Selection:", ["Select from List", "Enter Manually"], horizontal=True)
            if input_method == "Select from List" and "ATM_ID" in data.columns:
                atm_ids = sorted(data["ATM_ID"].unique())
                selected_atm = st.selectbox("ATM ID:", options=atm_ids)
            else:
                selected_atm = st.text_input("ATM ID:", value="ATM_001")
        
        with col2:
            pred_type = st.selectbox("Forecast Type:", ["Next Day", "Specific Date", "Multi-Day Range"])
        
        with col3:
            st.metric("Model", st.session_state.current_model_name)
        
        # Date selection
        today_kw = datetime.now(ZoneInfo("Asia/Kuwait")).date()
        next_day_kw = today_kw + timedelta(days=1)
        
        target_dates = []
        if pred_type == "Next Day":
            target_dates = [next_day_kw]
            st.info(f"📅 Predicting for: {next_day_kw}")
        elif pred_type == "Specific Date":
            target_date = st.date_input("Target Date:", min_value=next_day_kw, value=next_day_kw)
            target_dates = [target_date]
        else:  # Multi-Day Range
            col_start, col_end = st.columns(2)
            with col_start:
                start_date = st.date_input("Start Date:", min_value=next_day_kw, value=next_day_kw)
            with col_end:
                end_date = st.date_input("End Date:", min_value=start_date, value=start_date + timedelta(days=6))
            target_dates = pd.date_range(start_date, end_date).tolist()
        
        if st.button("🎯 Generate Forecast", type="primary", use_container_width=True):
            if selected_atm:
                try:
                    with st.spinner("Generating predictions..."):
                        # Filter data for selected ATM
                        atm_data = data[data["ATM_ID"] == selected_atm].copy() if "ATM_ID" in data.columns else data.copy()
                        
                        if atm_data.empty:
                            st.error(f"No data found for ATM: {selected_atm}")
                        else:
                            # Engineer features
                            processed, _, _ = st.session_state.engineer.engineer_all_features(
                                atm_data.copy(),
                                atm_id=selected_atm
                            )
                            
                            if processed is not None and not processed.empty:
                                processed = processed.sort_values("Date")
                                last_idx = processed["Date"].idxmax()
                                
                                predictions = []
                                for pred_date in target_dates:
                                    base = processed.loc[[last_idx]].copy()
                                    pred_ts = pd.Timestamp(pred_date)
                                    
                                    # Update temporal features
                                    base["Date"] = pred_ts
                                    base["Year"] = pred_ts.year
                                    base["Month"] = pred_ts.month
                                    base["Day"] = pred_ts.day
                                    base["Quarter"] = pred_ts.quarter
                                    base["Day_of_Week"] = pred_ts.day_name()
                                    base["Month_Sin"] = np.sin(2 * np.pi * pred_ts.month / 12)
                                    base["Month_Cos"] = np.cos(2 * np.pi * pred_ts.month / 12)
                                    
                                    # Predict
                                    X = st.session_state.engineer.prepare_for_model(base)
                                    raw_pred = float(st.session_state.model.predict(X)[0])
                                    pred_kd = max(int(round(raw_pred / 10) * 10), 0)  # Round to nearest 10 KD
                                    
                                    predictions.append({
                                        "Date": pred_ts.strftime("%Y-%m-%d"),
                                        "Day": pred_ts.strftime("%A"),
                                        "Predicted_Demand": pred_kd
                                    })
                                
                                results_df = pd.DataFrame(predictions)
                                
                                # Display Results
                                st.success("✅ Predictions Generated Successfully")
                                st.markdown("---")
                                
                                if len(results_df) == 1:
                                    pred = results_df.iloc[0]
                                    st.markdown(f"""
                                    <div class="prediction-box">
                                        <h2>{pred['Day']}, {pred['Date']}</h2>
                                        <div class="prediction-value">{fmt_kd(pred['Predicted_Demand'])}</div>
                                        <p>Predicted Cash Demand</p>
                                        <p style="opacity: 0.8;">ATM: {selected_atm} | Model: {st.session_state.current_model_name}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    # Multi-day visualization
                                    fig = px.line(results_df, x='Date', y='Predicted_Demand', markers=True,
                                                  title=f"Predicted Cash Demand - {selected_atm}")
                                    fig.update_layout(yaxis_title="Demand (KD)", height=400)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Demand", fmt_kd(results_df["Predicted_Demand"].sum()))
                                with col2:
                                    st.metric("Average Daily", fmt_kd(results_df["Predicted_Demand"].mean()))
                                with col3:
                                    st.metric("Peak Day", fmt_kd(results_df["Predicted_Demand"].max()))
                                with col4:
                                    st.metric("Days Forecasted", len(results_df))
                                
                                # Table view
                                with st.expander("📋 Detailed Predictions"):
                                    st.dataframe(results_df, use_container_width=True)
                                
                                # Download
                                results_df["ATM_ID"] = selected_atm
                                results_df["Model"] = st.session_state.current_model_name
                                results_df["Generated_At"] = datetime.now(ZoneInfo("Asia/Kuwait")).strftime("%Y-%m-%d %H:%M:%S")
                                
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Predictions (CSV)",
                                    data=csv,
                                    file_name=f"predictions_{selected_atm}_{datetime.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            else:
                                st.error("Failed to process features")
                except Exception as e:
                    st.error(f"Prediction Error: {str(e)}")
                    with st.expander("Error Details"):
                        st.exception(e)

# ============ TAB 2: AI ASSISTANT ============
with tab2:
    st.header("🤖 Intelligent Operations Assistant")
    
    if not st.session_state.rag_available:
        st.error("⚠️ RAG API is not available. Please start the service.")
        st.code(f"cd services/RAG_API && uvicorn app.main:app --reload --port 8000")
    else:
        st.success("✅ AI Assistant Ready")
        
        # Example questions
        with st.expander("💡 Example Questions You Can Ask"):
            st.markdown("""
            **Risk & Alerts:**
            - Which ATMs are at risk of running out of cash tomorrow?
            - Show me critical cashout alerts
            
            **Refill Planning:**
            - What refill amounts do you recommend for this weekend?
            - Which ATMs need urgent replenishment?
            
            **Location Analysis:**
            - Where should we place a new ATM in Kuwait City?
            - Which locations have the highest demand?
            
            **Performance:**
            - Show me the top 10 performing ATMs
            - Compare ATM performance across cities
            
            **Patterns:**
            - What are the peak withdrawal days for ATM_001?
            - Analyze demand trends for mall locations
            
            **Operations:**
            - Give me overall system efficiency metrics
            - Prepare a weekend cash preparation report
            """)
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            filter_city = st.selectbox("Filter by City (optional):", 
                                       ["None"] + (sorted(st.session_state.data['City'].unique().tolist()) if st.session_state.data is not None and 'City' in st.session_state.data.columns else []))
        with col2:
            filter_atm = st.selectbox("Filter by ATM (optional):",
                                      ["None"] + (sorted(st.session_state.data['ATM_ID'].unique().tolist()) if st.session_state.data is not None and 'ATM_ID' in st.session_state.data.columns else []))
        
        # Chat Interface
        st.markdown("---")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {msg["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {msg["content"]}</div>', 
                           unsafe_allow_html=True)
                if 'tools' in msg and msg['tools']:
                    tools_html = "".join([f'<span class="tool-badge">{tool}</span>' for tool in msg['tools']])
                    st.markdown(f'<div style="margin-left: 1rem;">{tools_html}</div>', unsafe_allow_html=True)
        
        # Input
        with st.form("chat_form", clear_on_submit=True):
            user_question = st.text_area("Ask a question about ATM operations:", 
                                         placeholder="e.g., Which ATMs risk cashout tomorrow?",
                                         height=100)
            submitted = st.form_submit_button("🚀 Ask Assistant", type="primary", use_container_width=True)
        
        if submitted and user_question:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Call RAG API
            with st.spinner("🤔 Analyzing your question..."):
                city_filter = filter_city if filter_city != "None" else None
                atm_filter = filter_atm if filter_atm != "None" else None
                
                response = call_rag_api(user_question, city=city_filter, atm_id=atm_filter)
                
                if "error" in response:
                    st.error(f"Error: {response['error']}")
                else:
                    # Add assistant response
                    assistant_msg = {
                        "role": "assistant",
                        "content": response.get("answer", "No answer generated"),
                        "tools": response.get("tools_executed", []),
                        "confidence": response.get("confidence", "unknown")
                    }
                    st.session_state.chat_history.append(assistant_msg)
                    
                    st.rerun()
        
        # Clear chat
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# ============ TAB 3: ANALYTICS ============
with tab3:
    st.header("📊 Interactive Analytics Dashboard")
    
    if st.session_state.data is None:
        st.warning("⚠️ Upload data in 'Data Management' tab to view analytics")
    else:
        data = st.session_state.data
        
        # Quick Stats
        st.subheader("Quick Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total ATMs", data['ATM_ID'].nunique() if 'ATM_ID' in data.columns else 0)
        with col2:
            st.metric("Total Records", f"{len(data):,}")
        with col3:
            st.metric("Cities", data['City'].nunique() if 'City' in data.columns else 0)
        with col4:
            if 'Withdrawals' in data.columns:
                st.metric("Avg Withdrawal", fmt_kd(data['Withdrawals'].mean()))
        
        st.markdown("---")
        
        # Top ATMs
        with st.expander("🏆 Top Performing ATMs", expanded=True):
            metric = st.selectbox("Metric:", ["Withdrawals", "Deposits"], key="top_atm_metric")
            if metric in data.columns:
                top_atms = data.groupby('ATM_ID')[metric].sum().sort_values(ascending=False).head(10)
                fig = px.bar(top_atms, x=top_atms.index, y=top_atms.values, text=top_atms.values,
                            title=f"Top 10 ATMs by {metric}")
                add_kd_text(fig)
                fig.update_layout(xaxis_title="ATM ID", yaxis_title=f"{metric} (KD)", height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # City Distribution
        with st.expander("🌆 Distribution by City", expanded=True):
            if 'City' in data.columns and 'Withdrawals' in data.columns:
                city_data = data.groupby('City')['Withdrawals'].sum().reset_index()
                fig = px.pie(city_data, values='Withdrawals', names='City', hole=0.4,
                            title="Withdrawal Distribution by City")
                st.plotly_chart(fig, use_container_width=True)
        
        # Temporal Trends
        with st.expander("📈 Temporal Trends", expanded=False):
            data_temp = _ensure_day_columns(data)
            if 'Date' in data_temp.columns and 'Withdrawals' in data_temp.columns:
                monthly = data_temp.groupby(data_temp['Date'].dt.to_period('M'))['Withdrawals'].mean()
                monthly.index = monthly.index.to_timestamp()
                fig = px.line(monthly, title="Average Monthly Withdrawals Trend")
                fig.update_layout(xaxis_title="Month", yaxis_title="Avg Withdrawals (KD)", height=400)
                st.plotly_chart(fig, use_container_width=True)

# ============ TAB 4: DATA MANAGEMENT ============
with tab4:
    st.header("📁 Data Management")
    
    uploaded_file = st.file_uploader(
        "Upload ATM Historical Data (CSV)",
        type=['csv'],
        help="Required: Date, ATM_ID, City, Location, Withdrawals, Deposits, DayOfWeek, IsHoliday, Weather"
    )
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            
            # Data preprocessing
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            
            for col in ['Withdrawals', 'Deposits']:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).round(0).astype(int)
            
            st.session_state.data = data
            st.success(f"✅ Successfully loaded {len(data):,} records")
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(data.head(20), use_container_width=True)
            
            # Data summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{len(data):,}")
            with col2:
                st.metric("Columns", len(data.columns))
            with col3:
                if 'Date' in data.columns:
                    date_range = (data['Date'].max() - data['Date'].min()).days
                    st.metric("Date Range (days)", date_range)
            
            # Column info
            with st.expander("📋 Column Information"):
                col_info = pd.DataFrame({
                    'Column': data.columns,
                    'Type': data.dtypes.values,
                    'Non-Null': data.count().values,
                    'Null': data.isnull().sum().values
                })
                st.dataframe(col_info, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.exception(e)
    
    elif st.session_state.data is not None:
        st.info(f"✓ Data already loaded: {len(st.session_state.data):,} records")
        if st.button("🔄 Clear Data"):
            st.session_state.data = None
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>ATM Intelligence Hub v2.0</strong></p>
        <p>Powered by Azure ML Studio, Azure OpenAI & Azure AI Search</p>
        <p style='font-size: 0.9rem;'>© 2024-2025 - Advanced Banking Analytics System</p>
    </div>
""", unsafe_allow_html=True)