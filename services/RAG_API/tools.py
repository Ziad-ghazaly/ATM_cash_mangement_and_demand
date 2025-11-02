"""
Enhanced ATM Analytics Tools - FINAL CORRECTED VERSION
FILE: services/RAG_API/tools.py
"""

from __future__ import annotations
import os
import requests
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============= Globals =============
atm_dataframe: Optional[pd.DataFrame] = None

REQUIRED_COLS = ["ATM_ID", "City", "Location", "Date", "Withdrawals", "Deposits"]
NUMERIC_COLS = ["Withdrawals", "Deposits"]

# ============= JSON Serialization Helpers =============
def safe_json_value(val: Any) -> Any:
    """Convert pandas/numpy types to JSON-safe Python types."""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return float(val)
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, np.bool_):
        return bool(val)
    return val

def sanitize_dict_list(data: List[Dict]) -> List[Dict]:
    """Sanitize list of dicts for JSON serialization."""
    return [
        {k: safe_json_value(v) for k, v in record.items()}
        for record in data
    ]

# ============= Core helpers =============
def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric columns are properly typed."""
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def _validate_columns(df: pd.DataFrame):
    """Validate required columns exist."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"ATM dataset missing required columns: {missing}")

def load_structured() -> pd.DataFrame:
    """
    Normalize and return a feature-complete DataFrame.
    Raises ValueError if data is not loaded.
    """
    if atm_dataframe is None:
        raise ValueError("ATM data not loaded. Upload data via /data/upload endpoint first.")

    df = atm_dataframe.copy()

    # Normalize columns
    if "Location_Type" in df.columns and "Location" not in df.columns:
        df["Location"] = df["Location_Type"]
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    _validate_columns(df)
    df = _ensure_numeric(df)

    # Derived features
    df = df.sort_values("Date")

    if "Withdrawals_RollMean7" not in df.columns:
        df["Withdrawals_RollMean7"] = df.groupby("ATM_ID")["Withdrawals"] \
            .transform(lambda x: x.rolling(7, min_periods=1).mean())

    # Placeholder balance logic
    if "Current_Balance" not in df.columns:
        df["Current_Balance"] = (df["Withdrawals_RollMean7"] * 10).fillna(0)

    if "Predicted_Next_Day" not in df.columns:
        df["Predicted_Next_Day"] = df["Withdrawals_RollMean7"]

    for c in ["Current_Balance", "Predicted_Next_Day", "Withdrawals_RollMean7",
              "Withdrawals", "Deposits"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    return df

# ============= Simple/intent utilities =============
def list_cities() -> List[str]:
    """Get list of unique cities."""
    df = load_structured()
    return sorted([c for c in df["City"].dropna().astype(str).unique() if c])

def count_atms(city: Optional[str] = None) -> Dict:
    """
    Deterministic counts of unique ATMs (overall or for a specific city).
    """
    df = load_structured()
    if city:
        sdf = df[df["City"].astype(str).str.casefold() == city.casefold()]
        return {
            "scope": "city",
            "city": city,
            "unique_atms": int(sdf["ATM_ID"].nunique()),
            "total_withdrawals_kd": float(sdf["Withdrawals"].sum()),
            "avg_withdrawal_kd": float(sdf["Withdrawals"].mean() if len(sdf) else 0.0),
        }

    overall = int(df["ATM_ID"].nunique())
    by_city = (
        df.groupby("City")
          .agg(
              unique_atms=("ATM_ID", "nunique"),
              total_withdrawals_kd=("Withdrawals", "sum"),
              avg_withdrawal_kd=("Withdrawals", "mean"),
          )
          .reset_index()
          .sort_values("unique_atms", ascending=False)
    )
    
    return {
        "scope": "all",
        "unique_atms": overall,
        "by_city": sanitize_dict_list(by_city.to_dict(orient="records")),
    }

def risk_distribution_snapshot() -> Dict:
    """Get risk distribution across all ATMs."""
    df = load_structured()
    latest = df.groupby("ATM_ID").tail(1).copy()
    
    # Safe division
    denom = latest["Current_Balance"].replace(0, np.nan)
    latest["risk_ratio"] = (latest["Predicted_Next_Day"] / denom).fillna(np.inf)
    latest["risk_ratio"] = latest["risk_ratio"].replace([np.inf, -np.inf], 999)

    def cat(r):
        if r >= 999 or r > 1.0: return "CRITICAL"
        if r > 0.9: return "HIGH"
        if r > 0.7: return "MEDIUM"
        return "LOW"

    latest["risk_category"] = latest["risk_ratio"].apply(cat)
    counts = latest["risk_category"].value_counts().to_dict()
    for k in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        counts.setdefault(k, 0)

    return {
        "risk_counts": {k: int(v) for k, v in counts.items()}, 
        "total_atms": int(latest["ATM_ID"].nunique())
    }

def critical_atms(limit: int = 25) -> List[Dict]:
    """Get critically low ATMs."""
    df = load_structured()
    latest = df.groupby("ATM_ID").tail(1).copy()
    
    denom = latest["Current_Balance"].replace(0, np.nan)
    latest["risk_ratio"] = (latest["Predicted_Next_Day"] / denom).fillna(np.inf)
    latest["risk_ratio"] = latest["risk_ratio"].replace([np.inf, -np.inf], 999)
    
    latest["days_until_cashout"] = (
        latest["Current_Balance"] / latest["Predicted_Next_Day"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    crit = latest[latest["risk_ratio"] > 1.0].sort_values("risk_ratio", ascending=False)
    
    cols = ["ATM_ID", "City", "Location", "Current_Balance", "Predicted_Next_Day", "days_until_cashout"]
    return sanitize_dict_list(crit[cols].head(limit).to_dict(orient="records"))

def refill_plan_next_24h(top_n: int = 30) -> List[Dict]:
    """Calculate refill needs for next 24 hours."""
    df = load_structured()
    latest = df.groupby("ATM_ID").tail(1).copy()
    latest["target_balance"] = latest["Withdrawals_RollMean7"] * 2
    latest["refill_kd"] = (latest["target_balance"] - latest["Current_Balance"]).clip(lower=0)

    denom = latest["Current_Balance"].replace(0, np.nan)
    latest["urgency"] = (latest["Predicted_Next_Day"] / denom).fillna(999)
    latest["urgency"] = latest["urgency"].replace([np.inf, -np.inf], 999)

    plan = latest[latest["refill_kd"] > 0].sort_values("urgency", ascending=False)
    
    cols = ["ATM_ID", "City", "Location", "Current_Balance", "Predicted_Next_Day",
            "Withdrawals_RollMean7", "refill_kd"]
    return sanitize_dict_list(plan[cols].head(top_n).to_dict(orient="records"))

# ============= Analytics & KPIs =============
def compute_cashout_risk(threshold: float = 0.2, days_ahead: int = 1) -> List[Dict]:
    """Identify ATMs at risk of running out of cash."""
    df = load_structured().copy()
    latest = df.groupby("ATM_ID").tail(1)
    
    denom = latest["Current_Balance"].replace(0, np.nan)
    latest["risk_ratio"] = (latest["Predicted_Next_Day"] / denom).fillna(999)
    latest["risk_ratio"] = latest["risk_ratio"].replace([np.inf, -np.inf], 999)
    
    latest["risk_percentage"] = (latest["risk_ratio"] * 100).clip(upper=99999).round(1)
    latest["days_until_cashout"] = (
        latest["Current_Balance"] / latest["Predicted_Next_Day"].replace(0, np.nan)
    ).round(1).replace([np.inf, -np.inf], np.nan)

    def cat(r):
        if r >= 999 or r > 1.0: return "CRITICAL"
        if r > 0.9: return "HIGH"
        if r > 0.7: return "MEDIUM"
        return "LOW"

    latest["risk_category"] = latest["risk_ratio"].apply(cat)
    high = latest[latest["risk_ratio"] > (1 - threshold)].sort_values("risk_ratio", ascending=False)

    cols = ["ATM_ID", "City", "Location", "Current_Balance", "Predicted_Next_Day",
            "risk_percentage", "days_until_cashout", "risk_category", "Withdrawals_RollMean7"]
    return sanitize_dict_list(high[cols].head(25).to_dict(orient="records"))

def refill_suggestion(buffer_days: int = 2, max_capacity: int = 50000) -> List[Dict]:
    """Calculate optimal refill amounts for ATMs."""
    df = load_structured().copy()
    latest = df.groupby("ATM_ID").tail(1)

    latest["target_balance"] = buffer_days * latest["Withdrawals_RollMean7"]
    latest["refill_kd"] = (latest["target_balance"] - latest["Current_Balance"]).clip(lower=0)

    # Cap at max capacity
    latest["refill_kd"] = latest["refill_kd"].clip(upper=(max_capacity - latest["Current_Balance"]).clip(lower=0))
    latest["refill_kd"] = (latest["refill_kd"] / 1000).round() * 1000

    denom = latest["Current_Balance"].replace(0, np.nan)
    latest["urgency_score"] = (latest["Predicted_Next_Day"] / denom).fillna(999) * 100
    latest["urgency_score"] = latest["urgency_score"].replace([np.inf, -np.inf], 999).clip(upper=999)

    def lvl(s):
        if s >= 999 or s > 90: return "URGENT"
        if s > 70: return "HIGH"
        if s > 50: return "MEDIUM"
        return "LOW"

    latest["priority"] = latest["urgency_score"].apply(lvl)
    need = latest[latest["refill_kd"] > 0].sort_values("urgency_score", ascending=False)

    cols = ["ATM_ID", "City", "Location", "Current_Balance", "refill_kd",
            "target_balance", "urgency_score", "priority", "Withdrawals_RollMean7"]
    return sanitize_dict_list(need[cols].head(50).to_dict(orient="records"))

def location_optimization_analysis(min_transactions: int = 100) -> Dict:
    """Analyze optimal locations for new ATM placement."""
    df = load_structured().copy()
    grp = df.groupby(["City", "Location"]).agg(
        Total_Withdrawals=("Withdrawals", "sum"),
        Avg_Daily_Demand=("Predicted_Next_Day", "mean"),
        ATM_Count=("ATM_ID", "nunique"),
        Avg_Balance=("Current_Balance", "mean"),
    ).reset_index()

    grp["demand_per_atm"] = grp["Avg_Daily_Demand"]
    
    denom = grp["Avg_Balance"].replace(0, np.nan)
    grp["utilization_rate"] = (grp["Avg_Daily_Demand"] / denom * 100).fillna(0)
    grp["utilization_rate"] = grp["utilization_rate"].replace([np.inf, -np.inf], 0).round(1)
    
    denom2 = grp["ATM_Count"].replace(0, np.nan)
    grp["expansion_score"] = grp["demand_per_atm"] * 0.6 + (100 / denom2).fillna(100) * 0.4
    grp["expansion_score"] = grp["expansion_score"].replace([np.inf, -np.inf], 0)

    expansion = grp.sort_values("expansion_score", ascending=False).head(10)
    best = grp.sort_values("demand_per_atm", ascending=False).head(10)

    return {
        "expansion_opportunities": sanitize_dict_list(expansion.to_dict(orient="records")),
        "best_performing_locations": sanitize_dict_list(best.to_dict(orient="records")),
        "summary": {
            "total_locations": int(len(grp)),
            "avg_demand_per_location": float(grp["Avg_Daily_Demand"].mean() if len(grp) else 0.0),
            "highest_demand_city": str(grp.loc[grp["Total_Withdrawals"].idxmax(), "City"]) if len(grp) else None,
        },
    }

def demand_pattern_analysis(atm_id: Optional[str] = None) -> Dict:
    """Analyze demand patterns and trends."""
    df = load_structured().copy()
    if atm_id:
        df = df[df["ATM_ID"] == atm_id]
    if df.empty:
        return {"error": f"No data found for ATM_ID: {atm_id}"}

    df = df.sort_values("Date")
    df["DayName"] = df["Date"].dt.day_name()

    dow_avg = df.groupby("DayName")["Withdrawals"].mean().round(2).to_dict()
    dow_avg = {k: safe_json_value(v) for k, v in dow_avg.items()}

    peak_days = {"Thursday", "Friday", "Saturday"}
    peak_avg = float(df[df["DayName"].isin(peak_days)]["Withdrawals"].mean() or 0)
    offpeak_avg = float(df[~df["DayName"].isin(peak_days)]["Withdrawals"].mean() or 0)
    
    mean_val = float(df["Withdrawals"].mean() or 1)
    vol = float((df["Withdrawals"].std() or 0) / mean_val * 100 if mean_val else 0)

    if len(df) > 7:
        tmp = df.assign(day_index=range(len(df)))
        corr = tmp["day_index"].corr(tmp["Withdrawals"])
        trend = "Increasing" if corr > 0.1 else ("Decreasing" if corr < -0.1 else "Stable")
    else:
        trend = "Insufficient data"

    return {
        "atm_id": atm_id or "ALL",
        "unique_atms_analyzed": int(df["ATM_ID"].nunique()),
        "day_of_week_avg": dow_avg,
        "peak_day_avg": round(peak_avg, 2),
        "offpeak_day_avg": round(offpeak_avg, 2),
        "peak_multiplier": round(peak_avg / offpeak_avg, 2) if offpeak_avg else 0.0,
        "volatility_pct": round(vol, 2),
        "trend": trend,
        "total_transactions": int(len(df)),
        "avg_withdrawal": round(mean_val, 2),
    }

def operational_efficiency_metrics() -> Dict:
    """Calculate operational efficiency KPIs."""
    df = load_structured().copy()
    latest = df.groupby("ATM_ID").tail(1)

    total_atms = int(latest["ATM_ID"].nunique())
    total_balance = float(latest["Current_Balance"].sum())
    total_pred = float(latest["Predicted_Next_Day"].sum())
    
    denom = latest["Current_Balance"].replace(0, np.nan)
    util = (latest["Predicted_Next_Day"] / denom).replace([np.inf, -np.inf], np.nan).mean() * 100
    util = float(util if not np.isnan(util) else 0.0)

    latest["risk_ratio"] = (latest["Predicted_Next_Day"] / denom).fillna(999)
    latest["risk_ratio"] = latest["risk_ratio"].replace([np.inf, -np.inf], 999)

    counts = {
        "critical": int((latest["risk_ratio"] > 1.0).sum()),
        "high": int(((latest["risk_ratio"] > 0.9) & (latest["risk_ratio"] <= 1.0)).sum()),
        "medium": int(((latest["risk_ratio"] > 0.7) & (latest["risk_ratio"] <= 0.9)).sum()),
        "low": int((latest["risk_ratio"] <= 0.7).sum()),
    }

    by_city = (
        latest.groupby("City")
        .agg(ATM_Count=("ATM_ID","nunique"),
             Total_Balance=("Current_Balance","sum"),
             Total_Demand=("Predicted_Next_Day","sum"))
        .reset_index()
    )

    return {
        "overview": {
            "total_atms": total_atms,
            "total_cash_available_kd": round(total_balance, 2),
            "predicted_demand_tomorrow_kd": round(total_pred, 2),
            "system_utilization_pct": round(util, 2),
            "cash_buffer_days": round(total_balance / total_pred, 2) if total_pred > 0 else 0.0,
        },
        "risk_distribution": counts,
        "by_city": sanitize_dict_list(by_city.to_dict(orient="records")),
    }

def weekend_preparation_report() -> Dict:
    """Estimate weekend demand and identify shortages."""
    df = load_structured().copy()
    latest = df.groupby("ATM_ID").tail(1)
    latest["weekend_demand"] = latest["Withdrawals_RollMean7"] * 2.5
    latest["shortage"] = (latest["weekend_demand"] - latest["Current_Balance"]).clip(lower=0)

    needs = latest[latest["shortage"] > 0].sort_values("shortage", ascending=False)

    cols = ["ATM_ID", "City", "Location", "Current_Balance", "weekend_demand", "shortage"]
    return {
        "total_atms_need_refill": int(len(needs)),
        "total_cash_required_kd": round(float(needs["shortage"].sum()), 2),
        "priority_refills": sanitize_dict_list(needs[cols].head(20).to_dict(orient="records")),
        "recommendation": "Complete refills by Thursday evening for optimal weekend coverage",
    }

def atm_performance_ranking(metric: str = "withdrawals", top_n: int = 20) -> List[Dict]:
    """Rank ATMs by performance metric."""
    df = load_structured().copy()

    if metric == "withdrawals":
        ranked = df.groupby("ATM_ID").agg(
            City=("City","first"),
            Location=("Location","first"),
            Withdrawals=("Withdrawals","sum"),
            Predicted_Next_Day=("Predicted_Next_Day","mean"),
        ).reset_index().sort_values("Withdrawals", ascending=False)
        cols = ["ATM_ID", "City", "Location", "Withdrawals", "Predicted_Next_Day"]

    elif metric == "utilization":
        latest = df.groupby("ATM_ID").tail(1)
        denom = latest["Current_Balance"].replace(0, np.nan)
        latest["utilization"] = (latest["Predicted_Next_Day"] / denom * 100).replace([np.inf, -np.inf], 0)
        ranked = latest.sort_values("utilization", ascending=False)
        cols = ["ATM_ID", "City", "Location", "utilization", "Current_Balance", "Predicted_Next_Day"]

    elif metric == "efficiency":
        agg = df.groupby("ATM_ID").agg(
            City=("City","first"),
            Location=("Location","first"),
            Withdrawals=("Withdrawals","sum"),
            Current_Balance=("Current_Balance","mean"),
        ).reset_index()
        denom = agg["Current_Balance"].replace(0, np.nan)
        agg["efficiency"] = (agg["Withdrawals"] / denom).replace([np.inf, -np.inf], 0)
        ranked = agg.sort_values("efficiency", ascending=False)
        cols = ["ATM_ID", "City", "Location", "efficiency", "Withdrawals", "Current_Balance"]

    else:
        return [{"error": f"Unknown metric: {metric}"}]

    return sanitize_dict_list(ranked[cols].head(top_n).to_dict(orient="records"))

def city_comparison_report() -> Dict:
    """Compare performance across cities."""
    df = load_structured().copy()

    city_metrics = df.groupby("City").agg(
        ATM_Count=("ATM_ID","nunique"),
        Transaction_Count=("ATM_ID","count"),
        Total_Withdrawals=("Withdrawals","sum"),
        Avg_Withdrawals=("Withdrawals","mean"),
        Std_Withdrawals=("Withdrawals","std"),
        Total_Balance=("Current_Balance","sum"),
        Total_Demand=("Predicted_Next_Day","sum"),
    ).reset_index()

    denom1 = city_metrics["ATM_Count"].replace(0, np.nan)
    city_metrics["demand_per_atm"] = (city_metrics["Total_Demand"] / denom1).fillna(0)
    
    denom2 = city_metrics["Total_Demand"].replace(0, np.nan)
    city_metrics["buffer_days"] = (city_metrics["Total_Balance"] / denom2).fillna(0)
    
    denom3 = city_metrics["Avg_Withdrawals"].replace(0, np.nan)
    city_metrics["volatility"] = (city_metrics["Std_Withdrawals"] / denom3 * 100).replace([np.inf, -np.inf], 0).round(2)

    city_metrics = city_metrics.sort_values("Total_Demand", ascending=False)

    return {
        "city_comparison": sanitize_dict_list(city_metrics.to_dict(orient="records")),
        "highest_demand_city": str(city_metrics.iloc[0]["City"]) if len(city_metrics) else None,
        "most_atms_city": str(city_metrics.loc[city_metrics["ATM_Count"].idxmax(), "City"]) if len(city_metrics) else None,
    }

# ============= Prediction (ML API + fallback) =============
def _get_recommendation(predicted_kd: float) -> str:
    """Generate practical recommendation."""
    if predicted_kd > 3000:
        return "HIGH DEMAND: Refill to full capacity by Thursday EOD and schedule morning audit."
    if predicted_kd > 1500:
        return "MODERATE: Ensure buffer > 2× daily mean; refill if balance < 50%."
    return "LOW: Standard monitoring; align refill with routine route."

def _fallback_prediction(df: pd.DataFrame, atm_id: Optional[str], days_ahead: int) -> Dict:
    """Statistical fallback: 7-day mean with weekend uplift."""
    if atm_id:
        df = df[df["ATM_ID"] == atm_id]

    df = df.sort_values("Date")
    df_recent = df.groupby("ATM_ID").tail(7) if not atm_id else df.tail(7)

    predictions: List[Dict] = []
    for atm in df_recent["ATM_ID"].unique():
        atm_data = df_recent[df_recent["ATM_ID"] == atm]
        predicted = float(atm_data["Withdrawals"].mean() or 0)

        try:
            last_date = atm_data["Date"].iloc[-1]
            if pd.notna(last_date) and last_date.dayofweek in [4, 5]:
                predicted *= 1.3
        except Exception:
            pass

        city = str(atm_data["City"].iloc[0]) if len(atm_data) > 0 else "Unknown"
        loc = str(atm_data["Location"].iloc[0]) if len(atm_data) > 0 else "Unknown"

        predictions.append({
            "ATM_ID": atm,
            "City": city,
            "Location": loc,
            "Predicted_Demand_KD": round(predicted, 2),
            "Days_Ahead": days_ahead,
            "Confidence": f"Medium (Statistical - {len(atm_data)}-day avg)",
            "Model_Used": "Statistical Fallback",
            "Recommendation": _get_recommendation(predicted),
        })

    predictions.sort(key=lambda x: x["Predicted_Demand_KD"], reverse=True)

    if atm_id and len(predictions) == 1:
        p = predictions[0]
        return {
            "atm_id": atm_id,
            "prediction": p,
            "details": {
                "predicted_demand_kd": p["Predicted_Demand_KD"],
                "confidence_level": p["Confidence"],
                "recommendation": p["Recommendation"],
                "historical_average": p["Predicted_Demand_KD"],
            },
            "source": "statistical_fallback",
        }

    return {
        "predictions": predictions[:20],
        "total_atms_analyzed": len(predictions),
        "total_predicted_demand_kd": round(sum(p["Predicted_Demand_KD"] for p in predictions), 2),
        "source": "statistical_fallback",
    }

def predict_atm_demand(atm_id: Optional[str] = None, days_ahead: int = 1) -> Dict:
    """
    Predict future demand using ML model or statistical fallback.
    """
    df = load_structured().copy()
    url = os.getenv("STREAMLIT_ML_URL", "http://localhost:8501")

    if atm_id:
        if atm_id not in set(df["ATM_ID"].unique()):
            sample = sorted(df["ATM_ID"].unique().tolist())[:10]
            return {"error": f"ATM_ID '{atm_id}' not found", "available_atms_sample": sample}
        try:
            r = requests.get(f"{url}/?ml_api_mode=true",
                             params={"atm_id": atm_id, "days_ahead": days_ahead},
                             timeout=8)
            if r.status_code == 200 and r.json().get("success"):
                pred = r.json()["prediction"]
                pred["Recommendation"] = _get_recommendation(pred["Predicted_Demand_KD"])
                return {
                    "atm_id": atm_id,
                    "prediction": pred,
                    "details": {
                        "predicted_demand_kd": pred["Predicted_Demand_KD"],
                        "confidence_level": f'High (ML Model: {pred["Model_Used"]})',
                        "recommendation": pred["Recommendation"],
                        "historical_average": pred.get("Historical_Avg"),
                        "prediction_date": pred["Prediction_Date"],
                        "model_used": pred["Model_Used"],
                    },
                    "source": "ml_model",
                }
        except Exception:
            pass
        return _fallback_prediction(df, atm_id, days_ahead)

    # Bulk: top 20 ATMs
    recent = df.groupby("ATM_ID").tail(7)
    top = (recent.groupby("ATM_ID")["Withdrawals"].sum()
           .sort_values(ascending=False).head(20).index.tolist())

    preds: List[Dict] = []
    for atm in top:
        try:
            r = requests.get(f"{url}/?ml_api_mode=true",
                             params={"atm_id": atm, "days_ahead": days_ahead},
                             timeout=5)
            if r.status_code == 200 and r.json().get("success"):
                p = r.json()["prediction"]
                p["Recommendation"] = _get_recommendation(p["Predicted_Demand_KD"])
                preds.append(p)
        except Exception:
            continue

    if not preds:
        return _fallback_prediction(df, None, days_ahead)

    preds.sort(key=lambda x: x["Predicted_Demand_KD"], reverse=True)
    return {
        "predictions": preds,
        "total_atms_analyzed": len(preds),
        "total_predicted_demand_kd": round(sum(p["Predicted_Demand_KD"] for p in preds), 2),
        "source": "ml_model",
    }

# ============= Summaries =============
def summarize_predictions(pred: Dict) -> str:
    """Generate concise text summary of predictions."""
    if not pred:
        return "No prediction results available."

    if "prediction" in pred:
        p = pred["prediction"]
        return (f"Forecast for {p.get('ATM_ID','N/A')} ({p.get('City','N/A')}): "
                f"{int(round(float(p.get('Predicted_Demand_KD',0))))} KD; "
                f"Rec: {p.get('Recommendation','N/A')}.")

    preds = pred.get("predictions", [])
    if not preds:
        return "No prediction results available."

    preds = sorted(preds, key=lambda x: float(x.get("Predicted_Demand_KD", 0)), reverse=True)
    top = "; ".join(f"{p.get('ATM_ID')}({int(float(p.get('Predicted_Demand_KD',0)))} KD)" for p in preds[:5])
    total = int(round(float(pred.get("total_predicted_demand_kd",
                                     sum(float(p.get('Predicted_Demand_KD',0)) for p in preds)))))
    return f"System forecast across {len(preds)} ATMs ≈ {total:,} KD. Top: {top}."

def generate_operational_insights() -> Dict:
    """
    Compact, cross-tool snapshot the LLM/UI can cite quickly.
    """
    try:
        risks = compute_cashout_risk()
    except Exception:
        risks = []
    try:
        refills = refill_suggestion()
    except Exception:
        refills = []

    dist = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in risks:
        rc = r.get("risk_category", "LOW")
        dist[rc] = dist.get(rc, 0) + 1

    total_refill = int(sum(float(x.get("refill_kd", 0)) for x in refills))
    urgent = [x.get("ATM_ID") for x in refills if x.get("priority") in ("URGENT", "HIGH")][:10]

    summary = (
        f"Risk → C:{dist['CRITICAL']}, H:{dist['HIGH']}, M:{dist['MEDIUM']}, L:{dist['LOW']}. "
        f"Refill volume ≈ {total_refill:,} KD. Urgent queue: {', '.join(urgent) if urgent else 'None'}."
    )

    return {
        "risk_counts": dist,
        "total_refill_kd": total_refill,
        "urgent_refill_sample": urgent,
        "summary": summary,
        "risks": risks[:15],
        "refills": refills[:25],
    }