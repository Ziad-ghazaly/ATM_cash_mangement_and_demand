"""
ATM Cash Demand Predictor - Complete Multi-Model Streamlit App
FILE LOCATION: ZIG018/streamlit_app/app.py

CRITICAL FIXES:
1. Fixed ML prediction helper functions with proper error handling
2. Fixed date handling in prediction logic
3. Fixed JSON serialization issues
4. Added proper error handling throughout
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import math
from datetime import datetime, date
from feature_engineering import ATMFeatureEngineer
from model_loader import ModelLoader
import json
from zoneinfo import ZoneInfo
import re
import traceback

# -------------------- Helpers --------------------
def fmt_kd(n: float) -> str:
    """Format as Kuwaiti Dinars (no cents)."""
    try:
        if n is None or (isinstance(n, float) and math.isnan(n)):
            return "KD -"
        return f"KD {float(n):,.0f}"
    except:
        return "KD -"

def add_kd_text(fig):
    """Add KD labels to bar charts."""
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
    elif 'DayOfWeek' in df.columns and 'DayName' not in df.columns:
        day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
                   3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        df['DayName'] = df['DayOfWeek'].map(day_map).fillna('Monday')
    return df

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="ATM Cash Demand Predictor",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.1rem;
        color: #0F6CBD;
        text-align: center;
        margin-bottom: 0.25rem;
        font-weight: 700;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.05rem;
        margin-bottom: 1.25rem;
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
    .muted { opacity: 0.9; }
    .badge {
        display:inline-block;padding:4px 8px;border-radius:6px;font-size:12px;font-weight:600;
        background:#eef6ff;color:#0f6cbd;border:1px solid #cfe3ff;margin-left:6px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">ATM Cash Demand Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Multi-Model Cash Forecasting System</p>', unsafe_allow_html=True)

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

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("Configuration")
    st.markdown("---")

    st.subheader("Step 1: Load Models")
    default_path = './trained_model/atm_rf_model.pkl'
    model_path = st.text_input("Model File Path:", value=default_path)

    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        st.success(f"Found model file ({file_size:.1f} MB)")
    else:
        st.warning("Model file not found")

    if st.button("Load Models", type="primary", use_container_width=True):
        if not os.path.exists(model_path):
            st.error(f"File not found: {model_path}")
        else:
            try:
                model_obj, error = st.session_state.model_loader.load_model(model_path)
                if model_obj:
                    st.session_state.model = model_obj
                    st.session_state.current_model_name = model_obj.current_model_name
                    st.success("Models loaded")
                    st.rerun()
                else:
                    st.error(f"Error: {error}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.markdown("---")

    # Model Status & Selection
    if st.session_state.model is not None:
        st.subheader("Model Status")
        st.success("Ready")

        model_info = st.session_state.model.get_model_info()

        st.write(f"**Type:** {'Multi-Model' if model_info['is_multi_model'] else 'Single'}")
        st.write(f"**Current:** {model_info['current_model']}")

        if model_info['is_multi_model']:
            st.write(f"**Available:** {len(model_info['available_models'])}")
            st.markdown("**Models Loaded:**")
            for model_name in model_info['available_models']:
                tag = "(active)" if model_name == model_info['current_model'] else ""
                st.markdown(f"- {model_name} {tag}")

            st.markdown("---")
            st.subheader("Select Model")
            selected_model = st.selectbox(
                "Choose prediction model:",
                options=model_info['available_models'],
                index=model_info['available_models'].index(model_info['current_model'])
                if model_info['current_model'] in model_info['available_models'] else 0,
                help="Switch between Ridge, Lasso, RandomForest, XGBoost, CatBoost"
            )

            if selected_model != model_info['current_model']:
                if st.session_state.model.switch_model(selected_model):
                    st.session_state.current_model_name = selected_model
                    st.success(f"Switched to {selected_model}")
                    st.rerun()

        if model_info.get('metadata'):
            st.markdown("---")
            st.subheader("Metadata")
            metadata = model_info['metadata']
            st.write(f"**Saved:** {metadata.get('saved_at', 'N/A')}")
            st.write(f"**Models:** {metadata.get('total_models', 'N/A')}")
            st.write(f"**Pipelines:** {metadata.get('total_pipelines', 'N/A')}")
    else:
        st.subheader("Model Status")
        st.error("No Model Loaded")
        st.info("Load models to begin")

    st.markdown("---")
    st.info("Azure ML Studio")

# -------------------- MAIN CONTENT --------------------
st.markdown("---")

# Step 2: Upload Data
st.header("Step 2: Upload Historical Data")

uploaded_file = st.file_uploader(
    "Upload CSV with ATM transaction history",
    type=['csv'],
    help="Required columns: Date, ATM_ID, City, Location, Withdrawals, Deposits, DayOfWeek, IsHoliday, Weather"
)

def _coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare uploaded data."""
    df = df.copy()

    # Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Numeric columns -> whole Kuwaiti Dinars
    for col in ['Withdrawals', 'Deposits']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df[col] = df[col].round(0).astype(int)

    # Day of week display name
    if 'DayOfWeek' in df.columns:
        if pd.api.types.is_numeric_dtype(df['DayOfWeek']):
            day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
                       3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
            df['DayName'] = df['DayOfWeek'].map(day_map).fillna('Monday')
        else:
            df['DayName'] = df['DayOfWeek'].astype(str)
    elif 'Date' in df.columns:
        df['DayName'] = df['Date'].dt.day_name()

    # Holiday flag
    if 'IsHoliday' in df.columns:
        df['IsHoliday'] = pd.to_numeric(df['IsHoliday'], errors='coerce').fillna(0).astype(int)

    # Defaults if missing
    if 'Location' not in df.columns:
        df['Location'] = 'bank_branch'
    if 'ATM_ID' not in df.columns:
        df['ATM_ID'] = 'ATM_001'
    if 'City' not in df.columns:
        df['City'] = 'Kuwait City'
    if 'Weather' not in df.columns:
        df['Weather'] = 'clear'

    return _ensure_day_columns(df)

# Process uploaded file
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        data = _coerce_columns(data)
        st.session_state.data = data
        st.success(f"Loaded {len(data):,} records")

        with st.expander("Data Preview", expanded=False):
            st.dataframe(data.head(20), use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Records", f"{len(data):,}")
            with col2:
                st.metric("ATMs", data['ATM_ID'].nunique())
            with col3:
                st.metric("Columns", len(data.columns))
            with col4:
                if 'Date' in data.columns and data['Date'].notna().any():
                    dates = data['Date'].dropna()
                    days = (dates.max() - dates.min()).days + 1
                    st.metric("Days Covered", days)

        # ============ STEP 3: EXPLORATORY DATA ANALYSIS ============
        st.markdown("---")
        st.header("Step 3: Exploratory Data Analysis")

        # Relations (bar & pie)
        with st.expander("Relations: Top 5 and Distributions (Bar & Pie)", expanded=False):
            c1, c2, c3, c4 = st.columns([1.1, 1.1, 1, 1.3])
            with c1:
                city_filter = st.selectbox("Filter by City (optional)",
                                           ["All"] + sorted(data['City'].dropna().unique().tolist()))
            with c2:
                loc_filter = st.selectbox("Filter by Location (optional)",
                                          ["All"] + sorted(data['Location'].dropna().unique().tolist()))
            with c3:
                metric = st.selectbox("Metric", ["Withdrawals", "Deposits"])
            with c4:
                years_available = sorted(_ensure_day_columns(data)['Year'].dropna().unique().tolist())
                years_selected = st.multiselect("Year", years_available, default=years_available)

            # Apply filters
            df_rel = _ensure_day_columns(data).copy()
            if city_filter != "All":
                df_rel = df_rel[df_rel['City'] == city_filter]
            if loc_filter != "All":
                df_rel = df_rel[df_rel['Location'] == loc_filter]
            if years_selected:
                df_rel = df_rel[df_rel['Year'].isin(years_selected)]

            # TOP 5 ATMs
            st.subheader(f"Top 5 ATMs by {metric}")
            if {'ATM_ID', metric}.issubset(df_rel.columns) and not df_rel.empty:
                top5 = (df_rel.groupby('ATM_ID', as_index=False)[metric]
                        .sum()
                        .sort_values(metric, ascending=False)
                        .head(5))
                fig_top = px.bar(
                    top5, x='ATM_ID', y=metric, text=metric,
                    title=f"Top 5 ATMs by Total {metric}"
                )
                add_kd_text(fig_top)
                fig_top.update_layout(
                    height=360, showlegend=False,
                    xaxis_title="ATM ID", yaxis_title=f"{metric} (KD)"
                )
                st.plotly_chart(fig_top, use_container_width=True)
            else:
                st.info("No ATM data available with current filters.")

            # Distribution by City / Location
            st.subheader("Distribution by City / Location")
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                if {'City', metric}.issubset(df_rel.columns) and len(df_rel) > 0:
                    by_city = (df_rel.groupby('City')[metric]
                               .sum().reset_index()
                               .sort_values(metric, ascending=False))
                    fig_city = px.pie(by_city, values=metric, names='City',
                                      title=f"{metric} Distribution by City", hole=0.3)
                    fig_city.update_traces(texttemplate='%{label}<br>%{percent}')
                    st.plotly_chart(fig_city, use_container_width=True)
            with dcol2:
                if {'Location', metric}.issubset(df_rel.columns) and len(df_rel) > 0:
                    by_loc = (df_rel.groupby('Location')[metric]
                              .sum().reset_index()
                              .sort_values(metric, ascending=False))
                    fig_loc = px.bar(by_loc, x='Location', y=metric, text=metric,
                                     title=f"Total {metric} by Location")
                    add_kd_text(fig_loc)
                    fig_loc.update_layout(height=360, showlegend=False,
                                          xaxis_title="Location", yaxis_title=f"{metric} (KD)")
                    st.plotly_chart(fig_loc, use_container_width=True)

            # Top 5 transactions count
            st.subheader("Top 5 Transaction Counts")
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                counts_city = df_rel['City'].value_counts().reset_index()
                counts_city.columns = ['City', 'Count']
                counts_city = counts_city.head(5)
                fig_c = px.bar(counts_city, x='City', y='Count', text='Count',
                               title="Top 5 Transaction Counts by City")
                fig_c.update_traces(textposition='outside')
                fig_c.update_layout(height=360, showlegend=False)
                st.plotly_chart(fig_c, use_container_width=True)
            with fcol2:
                counts_loc = df_rel['Location'].value_counts().reset_index()
                counts_loc.columns = ['Location', 'Count']
                counts_loc = counts_loc.head(5)
                fig_l = px.bar(counts_loc, x='Location', y='Count', text='Count',
                               title="Top 5 Transaction Counts by Location")
                fig_l.update_traces(textposition='outside')
                fig_l.update_layout(height=360, showlegend=False)
                st.plotly_chart(fig_l, use_container_width=True)

        # Temporal patterns
        with st.expander("Temporal Patterns: Mean Withdrawals Over Time (Line Charts)", expanded=False):
            t1, t2, t3 = st.columns([1.2, 1.2, 1.2])
            with t1:
                city_t = st.selectbox("Filter by City", ["All"] + sorted(data['City'].dropna().unique().tolist()), key="temporal_city")
            with t2:
                loc_t = st.selectbox("Filter by Location", ["All"] + sorted(data['Location'].dropna().unique().tolist()), key="temporal_loc")
            with t3:
                timeframe = st.selectbox(
                    "Aggregation",
                    ["Yearly Mean", "Monthly Mean", "Day of Week Mean", "Day of Month Mean"],
                    help="All charts use Withdrawals mean."
                )

            df_t = _ensure_day_columns(data)
            if city_t != "All":
                df_t = df_t[df_t['City'] == city_t]
            if loc_t != "All":
                df_t = df_t[df_t['Location'] == loc_t]

            if 'Date' not in df_t.columns or df_t['Date'].isna().all():
                st.warning("Date column missing or invalid for temporal analysis.")
            else:
                if timeframe == "Yearly Mean":
                    grp = df_t.groupby('Year')['Withdrawals'].mean().reset_index()
                    fig = px.line(grp, x='Year', y='Withdrawals', markers=True, title="Yearly Mean Withdrawals")
                elif timeframe == "Monthly Mean":
                    grp = df_t.groupby(['Year', 'Month'])['Withdrawals'].mean().reset_index()
                    grp['YearMonth'] = pd.to_datetime(grp['Year'].astype(str) + '-' + grp['Month'].astype(str) + '-01')
                    fig = px.line(grp, x='YearMonth', y='Withdrawals', markers=True, title="Monthly Mean Withdrawals")
                elif timeframe == "Day of Week Mean":
                    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    grp = df_t.groupby('DayName')['Withdrawals'].mean().reindex(order).reset_index()
                    fig = px.line(grp, x='DayName', y='Withdrawals', markers=True, title="Day of Week Mean Withdrawals")
                else:
                    grp = df_t.groupby('DayOfMonth')['Withdrawals'].mean().reset_index()
                    fig = px.line(grp, x='DayOfMonth', y='Withdrawals', markers=True, title="Day of Month Mean Withdrawals")

                fig.update_layout(height=380, xaxis_title="", yaxis_title="Withdrawals (KD)")
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
        traceback.print_exc()

# ============ STEP 4: PREDICTION ============
if st.session_state.data is not None and st.session_state.model is not None:
    st.markdown("---")
    st.header("Step 4: Make Predictions")

    def round_to_nearest_auto(value: float) -> int:
        """Automatically round ATM prediction to realistic note denominations."""
        if value < 100:
            denom = 5
        elif value < 500:
            denom = 10
        else:
            denom = 20
        return int(round(value / denom) * denom)

    data = st.session_state.data

    # Use actual current date in Kuwait
    today_kw = datetime.now(ZoneInfo("Asia/Kuwait")).date()
    next_day_kw = today_kw + timedelta(days=1)

    col1, col2 = st.columns([2, 1])
    with col1:
        input_method = st.radio("ATM Selection:", ["Select from List", "Enter Manually"], horizontal=True)
        if input_method == "Select from List" and "ATM_ID" in data.columns:
            atm_ids = sorted(data["ATM_ID"].unique())
            selected_atm = st.selectbox("ATM ID:", options=atm_ids)
        else:
            selected_atm = st.text_input("ATM ID:", value="ATM_001")

    with col2:
        pred_type = st.selectbox("Timeframe:", ["Next Day", "Specific Date"])

    target_date = None
    if pred_type == "Specific Date":
        target_date = st.date_input("Target Date:", min_value=next_day_kw, value=next_day_kw)

    if pred_type == "Next Day":
        st.info(f"Today (Asia/Kuwait): {today_kw}  •  Predicting for: {next_day_kw}")
    else:
        st.info(f"Today (Asia/Kuwait): {today_kw}  •  Predicting for: {target_date}")

    if st.button("Generate Prediction", type="primary", use_container_width=True):
        if selected_atm:
            try:
                atm_data = data[data["ATM_ID"] == selected_atm].copy() if "ATM_ID" in data.columns else data.copy()

                if atm_data.empty:
                    st.error(f"No data found for ATM: {selected_atm}")
                else:
                    processed, issues, suggestions = st.session_state.engineer.engineer_all_features(
                        atm_data.copy(),
                        atm_id=selected_atm
                    )

                    if processed is not None and not processed.empty:
                        processed = processed.sort_values("Date")
                        last_idx = processed["Date"].idxmax()

                        if pred_type == "Next Day":
                            pred_dates = [pd.Timestamp(next_day_kw)]
                        else:
                            pred_dates = [pd.Timestamp(target_date)]

                        predictions = []
                        for pred_date in pred_dates:
                            base = processed.loc[[last_idx]].copy()
                            base["Date"] = pred_date
                            base["Year"] = pred_date.year
                            base["Month"] = pred_date.month
                            base["Day"] = pred_date.day
                            base["Quarter"] = pred_date.quarter
                            base["Day_of_Week"] = pred_date.day_name()
                            base["WeekOfYear"] = int(pd.Timestamp(pred_date).isocalendar().week)
                            base["Month_Sin"] = np.sin(2 * np.pi * pred_date.month / 12)
                            base["Month_Cos"] = np.cos(2 * np.pi * pred_date.month / 12)
                            base["Week_Sin"] = np.sin(2 * np.pi * base["WeekOfYear"] / 52)
                            base["Week_Cos"] = np.cos(2 * np.pi * base["WeekOfYear"] / 52)

                            X = st.session_state.engineer.prepare_for_model(base)
                            raw_pred = float(st.session_state.model.predict(X)[0])
                            non_neg = max(raw_pred, 0.0)
                            pred_kd = round_to_nearest_auto(non_neg)

                            predictions.append({
                                "Date": pred_date.strftime("%Y-%m-%d"),
                                "Day": pred_date.strftime("%A"),
                                "Predicted_Demand": int(pred_kd)
                            })

                        results_df = pd.DataFrame(predictions)

                        st.markdown("---")
                        st.subheader(f"Prediction Results - {selected_atm}")

                        if len(results_df) == 1:
                            pred = results_df.iloc[0]
                            st.markdown(f"""
                            <div class="prediction-box">
                                <h2>{pred['Day']}, {pred['Date']}</h2>
                                <div class="prediction-value">{fmt_kd(pred['Predicted_Demand'])}</div>
                                <p class="muted">Predicted Cash Demand (auto-rounded to nearest 5/10/20 KD)</p>
                                <p class="muted">Model: {st.session_state.current_model_name}</p>
                                <p class="muted">ATM: {selected_atm}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Predicted Amount", fmt_kd(results_df["Predicted_Demand"].iloc[0]))
                        with c2:
                            st.metric("Model Used", st.session_state.current_model_name)
                        with c3:
                            st.metric("Prediction Date", results_df["Date"].iloc[0])

                        st.markdown("---")
                        results_df["ATM_ID"] = selected_atm
                        results_df["Model"] = st.session_state.current_model_name
                        results_df["Generated_At"] = datetime.now(ZoneInfo("Asia/Kuwait")).strftime("%Y-%m-%d %H:%M:%S")

                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name=f"predictions_{selected_atm}_{datetime.now(ZoneInfo('Asia/Kuwait')).strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.error("Failed to process features")
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")
                traceback.print_exc()

elif st.session_state.model is None:
    st.info("Load models from sidebar to continue")
elif st.session_state.data is None:
    st.info("Upload data to continue")

# ============ STEP 5: RAG INTELLIGENT ASSISTANT ============
st.markdown("---")
st.header("Step 5: AI Assistant - Ask Questions About Your ATMs")

# FIXED ML prediction helper function
def _extract_atms_from_text(text: str) -> list:
    """Extract ATM IDs from text query."""
    matches = re.findall(r'ATM[_\-\s]?(\d{1,3})', text, flags=re.IGNORECASE)
    return [f"ATM_{m.zfill(3)}" for m in matches] if matches else []

def _predict_for_atms_safe(atm_ids: list, days_ahead: int = 1) -> list:
    """
    FIXED: Use the already-loaded model and data to get predictions with proper error handling.
    """
    out = []
    if st.session_state.data is None or st.session_state.model is None:
        return out
    
    df = st.session_state.data
    try:
        today_kw = datetime.now(ZoneInfo("Asia/Kuwait")).date()
        pred_date_obj = today_kw + timedelta(days=days_ahead)
        pred_date = pd.Timestamp(pred_date_obj)
    except Exception as e:
        print(f"[Date error] {e}")
        return out

    for atm_id in atm_ids:
        try:
            atm_data = df[df["ATM_ID"] == atm_id].copy()
            if atm_data.empty:
                continue
                
            processed, _, _ = st.session_state.engineer.engineer_all_features(atm_data.copy(), atm_id=atm_id)
            if processed is None or processed.empty:
                continue
                
            processed = processed.sort_values("Date")
            last_idx = processed["Date"].idxmax()
            base = processed.loc[[last_idx]].copy()

            # Set prediction date features
            base["Date"] = pred_date
            base["Year"] = pred_date.year
            base["Month"] = pred_date.month
            base["Day"] = pred_date.day
            base["Quarter"] = pred_date.quarter
            base["Day_of_Week"] = pred_date.day_name()
            base["WeekOfYear"] = int(pred_date.isocalendar().week)
            base["Month_Sin"] = np.sin(2 * np.pi * pred_date.month / 12)
            base["Month_Cos"] = np.cos(2 * np.pi * pred_date.month / 12)
            base["Week_Sin"] = np.sin(2 * np.pi * base["WeekOfYear"] / 52)
            base["Week_Cos"] = np.cos(2 * np.pi * base["WeekOfYear"] / 52)

            X = st.session_state.engineer.prepare_for_model(base)
            raw_pred = float(st.session_state.model.predict(X)[0])
            
            # Auto-round
            if raw_pred < 100: denom = 5
            elif raw_pred < 500: denom = 10
            else: denom = 20
            pred_kd = int(round(max(raw_pred, 0.0) / denom) * denom)

            out.append({
                "ATM_ID": atm_id,
                "City": str(atm_data["City"].iloc[0]) if "City" in atm_data.columns else "Unknown",
                "Location": str(atm_data["Location"].iloc[0]) if "Location" in atm_data.columns else "Unknown",
                "Prediction_Date": pred_date_obj.strftime("%Y-%m-%d"),
                "Predicted_Demand_KD": pred_kd,
                "Model": st.session_state.current_model_name,
            })
        except Exception as e:
            print(f"[Predict error for {atm_id}] {e}")
            continue
    return out

def _build_ml_summary_text(question: str) -> str:
    """
    FIXED: Return a compact summary paragraph with proper error handling.
    """
    try:
        atms = _extract_atms_from_text(question)
        df = st.session_state.data

        # If no specific ATMs mentioned, get top 3 by recent withdrawals
        if not atms and df is not None and "ATM_ID" in df.columns and "Withdrawals" in df.columns and "Date" in df.columns:
            try:
                recent = df.sort_values("Date").groupby("ATM_ID").tail(7)
                atms = (recent.groupby("ATM_ID")["Withdrawals"].sum()
                        .sort_values(ascending=False).head(3).index.tolist())
            except Exception as e:
                print(f"[ML summary - ATM selection error] {e}")
                return ""

        preds = _predict_for_atms_safe(atms[:3]) if atms else []
        if not preds:
            return ""

        # Build readable summary
        parts = [f"{p['ATM_ID']} ({p['City']}) → {p['Predicted_Demand_KD']:,} KD on {p['Prediction_Date']}" for p in preds]
        joined = "; ".join(parts)
        return (
            "ML_PREDICTION_SUMMARY: "
            f"Using model '{st.session_state.current_model_name}', next-day demand is: {joined}. "
            "Use these predictions to refine recommendations and refill priorities where relevant."
        )
    except Exception as e:
        print(f"[ML summary error] {e}")
        traceback.print_exc()
        return ""

# RAG client
try:
    from rag_client import RAGClient
    rag_client = RAGClient()
    rag_available = True
except Exception as e:
    st.error(f"RAG client not available: {e}")
    rag_available = False

if rag_available and st.session_state.data is not None:
    # Check RAG API health
    health = rag_client.health_check()

    if health["status"] == "ok":
        h = health["data"]

        # Status indicators
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric(
                "Azure OpenAI",
                ("✅ Ready" if h.get("aoai_configured") else "❌ Not Configured"),
            )
        with col_b:
            st.metric(
                "Azure AI Search",
                ("✅ Ready" if h.get("search_configured") else "❌ Not Configured"),
            )
        with col_c:
            st.metric(
                "Data Status",
                ("✅ Loaded" if h.get("data_loaded") else "⚠️ Not Uploaded"),
            )

        # Upload to RAG API
        if not h.get("data_loaded"):
            if st.button("📤 Upload Data to RAG System", type="primary", use_container_width=True):
                with st.spinner("Uploading data to RAG API..."):
                    try:
                        def make_df_json_safe(df: pd.DataFrame) -> pd.DataFrame:
                            """Convert DataFrame to JSON-safe format."""
                            df = df.copy()
                            for col in df.columns:
                                if pd.api.types.is_datetime64_any_dtype(df[col]):
                                    df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
                                else:
                                    df[col] = df[col].apply(
                                        lambda x: x.isoformat() if isinstance(x, (datetime, date)) else x
                                    )
                            def to_native(x):
                                if isinstance(x, (np.integer,)):   return int(x)
                                if isinstance(x, (np.floating,)):  return float(x)
                                if pd.isna(x):                     return None
                                return x
                            for col in df.columns:
                                df[col] = df[col].map(to_native)
                            return df

                        with st.spinner("Uploading & indexing data to Azure AI Search... (this can take several minutes for large files)"):
                            safe_df = make_df_json_safe(st.session_state.data)
                            result = rag_client.upload_data(safe_df, timeout_sec=900)
                        if result.get("status") == "success":
                            rows = result.get("data", {}).get("rows") or result.get("rows", 0)
                            idxd = result.get("data", {}).get("indexed_to_search") or result.get("indexed_to_search", 0)
                            st.success(f"✅ Uploaded {rows} rows. Indexed to Search: {idxd}")
                            st.rerun()
                        else:
                            st.error(f"Upload failed: {result.get('message', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Upload failed: {e}")
                        traceback.print_exc()

        st.markdown("---")

        # Chat UI
        st.subheader("💬 Chat with Your ATM Data")

        # Toggle for ML model usage
        col_ml, _ = st.columns([2, 3])
        with col_ml:
            use_ml_for_assistant = st.checkbox(
                "Enable ML model in assistant answers",
                value=True,
                help="If enabled, the app will generate a brief prediction summary and pass it to the assistant to strengthen recommendations."
            )
            st.markdown(
                f"<span class='badge'>ML Assistant: {'ON' if use_ml_for_assistant else 'OFF'}</span>",
                unsafe_allow_html=True
            )

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        with st.expander("💡 Example Questions", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **Risk & Monitoring**
                - Which ATMs are at risk of running out of cash tomorrow?
                - Show me critical ATMs that need immediate attention
                - What's the overall risk distribution across all ATMs?
                """)
                st.markdown("""
                **Refill Planning**
                - What refill amounts do you recommend for this weekend?
                - Which ATMs need urgent refills in the next 24 hours?
                - Calculate optimal refill amounts for high-priority ATMs
                """)
            with col2:
                st.markdown("""
                **Performance Analytics**
                - Which are the top 10 performing ATMs by withdrawals?
                - Compare ATM performance across different cities
                - What's the demand pattern for ATM_001?
                """)
                st.markdown("""
                **Strategic Planning**
                - Where should we place new ATMs for maximum impact?
                - What are the best locations for ATM expansion?
                - Show me weekend preparation recommendations
                """)

        user_question = st.text_input(
            "Ask a question about your ATM operations:",
            value=st.session_state.get('quick_query', ''),
            placeholder="e.g., Which ATMs need refills today?",
            key="rag_query_input"
        )
        if 'quick_query' in st.session_state:
            del st.session_state.quick_query

        ask_col, clear_col = st.columns([4, 1])
        with ask_col:
            ask_button = st.button("🔍 Ask Assistant", type="primary", use_container_width=True)
        with clear_col:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        if ask_button and user_question.strip():
            with st.spinner("🤔 Thinking..."):
                try:
                    # Build optional ML prediction summary
                    ml_summary_text = ""
                    if use_ml_for_assistant and st.session_state.model is not None and st.session_state.data is not None:
                        ml_summary_text = _build_ml_summary_text(user_question)

                    # Compose final query
                    final_question = user_question if not ml_summary_text else f"{user_question}\n\n[{ml_summary_text}]"

                    st.session_state.chat_history.append({"role": "user", "content": user_question})

                    # Call backend
                    result = rag_client.query(final_question, use_tools=True)

                    if result.get("status") == "success":
                        data_resp = result.get("data", {})
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": data_resp.get("answer", "No answer generated"),
                            "sources": data_resp.get("sources", []),
                            "tool_results": data_resp.get("tool_results")
                        })
                    else:
                        st.error(f"Error: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Query error: {str(e)}")
                    traceback.print_exc()

        if st.session_state.chat_history:
            st.markdown("### 💬 Conversation")
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(
                        f"<div style='background:#e3f2fd;padding:1rem;border-radius:8px;margin:.5rem 0;'>"
                        f"<strong>👤 You:</strong><br>{msg['content']}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='background:#f5f5f5;padding:1rem;border-radius:8px;margin:.5rem 0;'>"
                        f"<strong>🤖 AI Assistant:</strong><br>{msg['content']}</div>",
                        unsafe_allow_html=True
                    )

                    # Tool results
                    if msg.get("tool_results"):
                        with st.expander("📊 View Detailed Analytics", expanded=False):
                            tool_name = msg["tool_results"].get("tool", "analysis")
                            st.markdown(f"**Analysis Type:** `{tool_name}`")
                            results = msg["tool_results"].get("results", {})
                            
                            if isinstance(results, list) and len(results) > 0:
                                df_results = pd.DataFrame(results)
                                st.dataframe(df_results, use_container_width=True)
                                
                                chat_idx = st.session_state.chat_history.index(msg)
                                st.download_button(
                                    "📥 Download Results",
                                    df_results.to_csv(index=False),
                                    file_name=f"{tool_name}_results.csv",
                                    mime="text/csv",
                                    key=f"download_{chat_idx}_{tool_name}"
                                )
                            elif isinstance(results, dict):
                                st.json(results)

                    # Knowledge sources
                    if msg.get("sources"):
                        with st.expander("📚 Knowledge Sources", expanded=False):
                            for i, source in enumerate(msg["sources"][:3], 1):
                                score = source.get("score")
                                score_txt = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"
                                st.markdown(
                                    f"**{i}. {source.get('title','Untitled')}**  \n"
                                    f"Category: `{source.get('category','N/A')}`  \n"
                                    f"Relevance Score: {score_txt}"
                                )

    else:
        st.error(f"⚠️ RAG API Connection Error: {health.get('message')}")
        st.info(
            "**To start the RAG API:**\n"
            "1. Open a new terminal\n"
            "2. Navigate to `services/RAG_API/`\n"
            "3. Run: `python -m uvicorn app:app --host 0.0.0.0 --port 8000`\n"
            "4. Refresh this page"
        )

elif rag_available and st.session_state.data is None:
    st.info("📊 Upload historical data first to use the AI Assistant")

else:
    st.warning("RAG client not available. Install dependencies: `pip install httpx`")

# ============ ML PREDICTION API FOR RAG ============
# This allows the RAG service to call your trained ML model

if 'ml_api_mode' in st.query_params:
    # API mode - return JSON predictions without rendering UI
    st.set_page_config(page_title="ML API", layout="wide")
    
    try:
        # Get parameters from query string
        atm_id = st.query_params.get('atm_id', 'ATM_001')
        days_ahead = int(st.query_params.get('days_ahead', 1))
        
        if st.session_state.data is not None and st.session_state.model is not None:
            # Filter data for ATM
            atm_data = st.session_state.data[st.session_state.data["ATM_ID"] == atm_id].copy()
            
            if not atm_data.empty:
                # Engineer features
                processed, _, _ = st.session_state.engineer.engineer_all_features(
                    atm_data.copy(),
                    atm_id=atm_id
                )
                
                if processed is not None and not processed.empty:
                    processed = processed.sort_values("Date")
                    last_idx = processed["Date"].idxmax()
                    
                    # Predict for target date
                    today_kw = datetime.now(ZoneInfo("Asia/Kuwait")).date()
                    pred_date_obj = today_kw + timedelta(days=days_ahead)
                    pred_date = pd.Timestamp(pred_date_obj)
                    
                    base = processed.loc[[last_idx]].copy()
                    base["Date"] = pred_date
                    base["Year"] = pred_date.year
                    base["Month"] = pred_date.month
                    base["Day"] = pred_date.day
                    base["Quarter"] = pred_date.quarter
                    base["Day_of_Week"] = pred_date.day_name()
                    base["WeekOfYear"] = int(pred_date.isocalendar().week)
                    base["Month_Sin"] = np.sin(2 * np.pi * pred_date.month / 12)
                    base["Month_Cos"] = np.cos(2 * np.pi * pred_date.month / 12)
                    base["Week_Sin"] = np.sin(2 * np.pi * base["WeekOfYear"] / 52)
                    base["Week_Cos"] = np.cos(2 * np.pi * base["WeekOfYear"] / 52)
                    
                    # Get prediction
                    X = st.session_state.engineer.prepare_for_model(base)
                    raw_pred = float(st.session_state.model.predict(X)[0])
                    pred_kd = max(raw_pred, 0.0)
                    
                    # Auto-round
                    if pred_kd < 100:
                        denom = 5
                    elif pred_kd < 500:
                        denom = 10
                    else:
                        denom = 20
                    pred_kd = int(round(pred_kd / denom) * denom)
                    
                    # Get ATM details
                    city = str(atm_data["City"].iloc[0]) if "City" in atm_data.columns else "Unknown"
                    location = str(atm_data["Location"].iloc[0]) if "Location" in atm_data.columns else "Unknown"
                    
                    # Return JSON
                    result = {
                        "success": True,
                        "atm_id": atm_id,
                        "prediction": {
                            "ATM_ID": atm_id,
                            "City": city,
                            "Location": location,
                            "Predicted_Demand_KD": float(pred_kd),
                            "Prediction_Date": pred_date_obj.strftime("%Y-%m-%d"),
                            "Days_Ahead": days_ahead,
                            "Model_Used": st.session_state.current_model_name,
                            "Confidence": "High (ML Model)",
                            "Raw_Prediction": float(raw_pred),
                            "Historical_Avg": float(atm_data["Withdrawals"].tail(7).mean())
                        }
                    }
                    st.json(result)
                    st.stop()
                else:
                    st.json({"success": False, "error": "Feature engineering failed"})
                    st.stop()
            else:
                st.json({"success": False, "error": f"No data found for {atm_id}"})
                st.stop()
        else:
            st.json({"success": False, "error": "Model or data not loaded"})
            st.stop()
            
    except Exception as e:
        st.json({"success": False, "error": str(e)})
        st.stop()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1.2rem;'>
        <p style='font-size: 1.05rem;'><strong>ATM Cash Demand Prediction System</strong></p>
        <p>Powered by Azure ML Studio | Multi-Model AI Forecasting</p>
        <p style='font-size: 0.9rem;'>© 2024 - All Rights Reserved</p>
    </div>
""", unsafe_allow_html=True)