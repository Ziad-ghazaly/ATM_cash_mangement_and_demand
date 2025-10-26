"""
Enhanced ATM Analytics Tools
FILE: services/RAG_API/app/tools.py

Comprehensive business logic for ATM operations including:
- Cash-out risk detection
- Refill suggestions
- Location optimization
- Demand forecasting analysis
- Operational efficiency metrics
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional


def load_structured() -> pd.DataFrame:
    """Load ATM data from Azure Blob Storage (Parquet or CSV)"""
    src = os.getenv("ATM_PARQUET_SAS")
    if not src:
        raise ValueError("ATM_PARQUET_SAS environment variable not set")
    
    try:
        if src.endswith(".parquet"):
            df = pd.read_parquet(src)
        else:
            df = pd.read_csv(src)
        
        # Ensure Date column is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        return df
    except Exception as e:
        raise Exception(f"Failed to load ATM data: {str(e)}")


def compute_cashout_risk(threshold: float = 0.2, days_ahead: int = 1) -> List[Dict]:
    """
    Identify ATMs at risk of running out of cash
    
    Args:
        threshold: Risk threshold (0.2 = 20% safety margin)
        days_ahead: Days to predict ahead (1 = next day)
    
    Returns:
        List of ATMs with risk metrics
    """
    df = load_structured().copy()
    
    # Calculate risk ratio
    df['risk_ratio'] = df['Predicted_Next_Day'] / df['Current_Balance'].clip(lower=1)
    df['risk_percentage'] = (df['risk_ratio'] * 100).round(1)
    
    # Calculate days until cashout
    df['days_until_cashout'] = (df['Current_Balance'] / df['Predicted_Next_Day'].clip(lower=1)).round(1)
    
    # Risk categorization
    def categorize_risk(ratio):
        if ratio > 1.0:
            return "CRITICAL"
        elif ratio > 0.9:
            return "HIGH"
        elif ratio > 0.7:
            return "MEDIUM"
        else:
            return "LOW"
    
    df['risk_category'] = df['risk_ratio'].apply(categorize_risk)
    
    # Filter high-risk ATMs
    high_risk = df[df['risk_ratio'] > (1 - threshold)].copy()
    high_risk = high_risk.sort_values('risk_ratio', ascending=False)
    
    cols = [
        'ATM_ID', 'City', 'Location_Type', 'Current_Balance', 
        'Predicted_Next_Day', 'risk_percentage', 'days_until_cashout', 
        'risk_category', 'Withdrawals_RollMean7'
    ]
    
    return high_risk[cols].head(25).to_dict(orient='records')


def refill_suggestion(buffer_days: int = 2, max_capacity: int = 50000) -> List[Dict]:
    """
    Calculate optimal refill amounts for ATMs
    
    Args:
        buffer_days: Days of cash buffer to maintain
        max_capacity: Maximum ATM capacity in KD
    
    Returns:
        List of refill recommendations
    """
    df = load_structured().copy()
    
    # Calculate target balance (buffer_days * average daily demand)
    df['target_balance'] = buffer_days * df['Withdrawals_RollMean7']
    
    # Refill needed
    df['refill_kd'] = (df['target_balance'] - df['Current_Balance']).clip(lower=0)
    
    # Ensure we don't exceed capacity
    df['refill_kd'] = df.apply(
        lambda x: min(x['refill_kd'], max_capacity - x['Current_Balance']),
        axis=1
    )
    
    # Round to nearest 1000 KD for practical logistics
    df['refill_kd'] = (df['refill_kd'] / 1000).round() * 1000
    
    # Calculate urgency score
    df['urgency_score'] = (df['Predicted_Next_Day'] / df['Current_Balance'].clip(lower=1)) * 100
    
    # Priority categorization
    def priority_level(row):
        if row['urgency_score'] > 90:
            return "URGENT"
        elif row['urgency_score'] > 70:
            return "HIGH"
        elif row['urgency_score'] > 50:
            return "MEDIUM"
        else:
            return "LOW"
    
    df['priority'] = df.apply(priority_level, axis=1)
    
    # Filter only ATMs needing refill
    needs_refill = df[df['refill_kd'] > 0].copy()
    needs_refill = needs_refill.sort_values('urgency_score', ascending=False)
    
    cols = [
        'ATM_ID', 'City', 'Location_Type', 'Current_Balance', 
        'refill_kd', 'target_balance', 'urgency_score', 'priority',
        'Withdrawals_RollMean7'
    ]
    
    return needs_refill[cols].head(50).to_dict(orient='records')


def location_optimization_analysis(min_transactions: int = 100) -> Dict:
    """
    Analyze optimal locations for new ATM placement
    
    Args:
        min_transactions: Minimum transaction threshold for analysis
    
    Returns:
        Location performance metrics and recommendations
    """
    df = load_structured().copy()
    
    # Aggregate by location and city
    location_stats = df.groupby(['City', 'Location_Type']).agg({
        'Withdrawals': 'sum',
        'Predicted_Next_Day': 'mean',
        'ATM_ID': 'count',
        'Current_Balance': 'mean'
    }).reset_index()
    
    location_stats.columns = ['City', 'Location_Type', 'Total_Withdrawals', 
                               'Avg_Daily_Demand', 'ATM_Count', 'Avg_Balance']
    
    # Calculate performance metrics
    location_stats['demand_per_atm'] = location_stats['Avg_Daily_Demand']
    location_stats['utilization_rate'] = (
        location_stats['Avg_Daily_Demand'] / location_stats['Avg_Balance'].clip(lower=1) * 100
    ).round(1)
    
    # Identify underserved areas (high demand, few ATMs)
    location_stats['expansion_score'] = (
        location_stats['demand_per_atm'] * 0.6 + 
        (100 / location_stats['ATM_Count'].clip(lower=1)) * 0.4
    )
    
    # Top expansion opportunities
    expansion = location_stats.sort_values('expansion_score', ascending=False).head(10)
    
    # Best performing locations
    best_performing = location_stats.sort_values('demand_per_atm', ascending=False).head(10)
    
    return {
        'expansion_opportunities': expansion.to_dict(orient='records'),
        'best_performing_locations': best_performing.to_dict(orient='records'),
        'summary': {
            'total_locations': len(location_stats),
            'avg_demand_per_location': location_stats['Avg_Daily_Demand'].mean(),
            'highest_demand_city': location_stats.loc[location_stats['Total_Withdrawals'].idxmax(), 'City']
        }
    }


def demand_pattern_analysis(atm_id: Optional[str] = None) -> Dict:
    """
    Analyze demand patterns and trends
    
    Args:
        atm_id: Specific ATM to analyze (None for all ATMs)
    
    Returns:
        Demand pattern insights
    """
    df = load_structured().copy()
    
    if atm_id:
        df = df[df['ATM_ID'] == atm_id]
    
    if df.empty:
        return {'error': f'No data found for ATM_ID: {atm_id}'}
    
    # Day of week analysis
    if 'Date' in df.columns:
        df['DayOfWeek'] = df['Date'].dt.day_name()
        dow_avg = df.groupby('DayOfWeek')['Withdrawals'].mean().to_dict()
    else:
        dow_avg = {}
    
    # Peak vs off-peak
    peak_days = ['Thursday', 'Friday', 'Saturday']  # Typical peak days in Kuwait
    df['is_peak'] = df['DayOfWeek'].isin(peak_days) if 'DayOfWeek' in df.columns else False
    
    peak_avg = df[df['is_peak']]['Withdrawals'].mean() if 'is_peak' in df.columns else 0
    offpeak_avg = df[~df['is_peak']]['Withdrawals'].mean() if 'is_peak' in df.columns else 0
    
    # Volatility (coefficient of variation)
    volatility = (df['Withdrawals'].std() / df['Withdrawals'].mean() * 100) if len(df) > 1 else 0
    
    # Growth trend (simple linear)
    if len(df) > 7 and 'Date' in df.columns:
        df = df.sort_values('Date')
        df['day_index'] = range(len(df))
        correlation = df['day_index'].corr(df['Withdrawals'])
        trend = "Increasing" if correlation > 0.1 else ("Decreasing" if correlation < -0.1 else "Stable")
    else:
        trend = "Insufficient data"
    
    return {
        'atm_id': atm_id or 'ALL',
        'day_of_week_avg': dow_avg,
        'peak_day_avg': round(peak_avg, 2),
        'offpeak_day_avg': round(offpeak_avg, 2),
        'peak_multiplier': round(peak_avg / offpeak_avg, 2) if offpeak_avg > 0 else 0,
        'volatility_pct': round(volatility, 2),
        'trend': trend,
        'total_transactions': len(df),
        'avg_withdrawal': round(df['Withdrawals'].mean(), 2)
    }


def operational_efficiency_metrics() -> Dict:
    """
    Calculate operational efficiency KPIs across all ATMs
    
    Returns:
        System-wide operational metrics
    """
    df = load_structured().copy()
    
    # Overall metrics
    total_atms = df['ATM_ID'].nunique()
    total_balance = df['Current_Balance'].sum()
    total_predicted_demand = df['Predicted_Next_Day'].sum()
    
    # Utilization rate
    avg_utilization = (df['Predicted_Next_Day'] / df['Current_Balance'].clip(lower=1)).mean() * 100
    
    # ATMs by risk category
    df['risk_ratio'] = df['Predicted_Next_Day'] / df['Current_Balance'].clip(lower=1)
    
    critical = len(df[df['risk_ratio'] > 1.0])
    high = len(df[(df['risk_ratio'] > 0.9) & (df['risk_ratio'] <= 1.0)])
    medium = len(df[(df['risk_ratio'] > 0.7) & (df['risk_ratio'] <= 0.9)])
    low = len(df[df['risk_ratio'] <= 0.7])
    
    # City-level breakdown
    city_stats = df.groupby('City').agg({
        'ATM_ID': 'count',
        'Current_Balance': 'sum',
        'Predicted_Next_Day': 'sum'
    }).reset_index()
    city_stats.columns = ['City', 'ATM_Count', 'Total_Balance', 'Total_Demand']
    
    # Location type breakdown
    location_stats = df.groupby('Location_Type').agg({
        'ATM_ID': 'count',
        'Withdrawals': 'mean'
    }).reset_index()
    location_stats.columns = ['Location_Type', 'ATM_Count', 'Avg_Withdrawals']
    
    return {
        'overview': {
            'total_atms': total_atms,
            'total_cash_available_kd': round(total_balance, 2),
            'predicted_demand_tomorrow_kd': round(total_predicted_demand, 2),
            'system_utilization_pct': round(avg_utilization, 2),
            'cash_buffer_days': round(total_balance / total_predicted_demand, 2) if total_predicted_demand > 0 else 0
        },
        'risk_distribution': {
            'critical': critical,
            'high': high,
            'medium': medium,
            'low': low
        },
        'by_city': city_stats.to_dict(orient='records'),
        'by_location_type': location_stats.to_dict(orient='records')
    }


def weekend_preparation_report() -> Dict:
    """
    Generate weekend cash preparation recommendations
    
    Returns:
        Weekend preparation checklist
    """
    df = load_structured().copy()
    
    # Calculate 2-day buffer for weekend
    df['weekend_demand'] = df['Withdrawals_RollMean7'] * 2.5  # 2.5x for weekend surge
    df['shortage'] = (df['weekend_demand'] - df['Current_Balance']).clip(lower=0)
    
    # ATMs needing pre-weekend refill
    needs_refill = df[df['shortage'] > 0].copy()
    needs_refill = needs_refill.sort_values('shortage', ascending=False)
    
    # Calculate total cash needed
    total_cash_needed = needs_refill['shortage'].sum()
    
    # Priority ATMs (top 20)
    priority_list = needs_refill[[
        'ATM_ID', 'City', 'Location_Type', 'Current_Balance', 
        'weekend_demand', 'shortage'
    ]].head(20).to_dict(orient='records')
    
    return {
        'total_atms_need_refill': len(needs_refill),
        'total_cash_required_kd': round(total_cash_needed, 2),
        'priority_refills': priority_list,
        'recommendation': 'Complete refills by Thursday evening for optimal weekend coverage'
    }


def atm_performance_ranking(metric: str = 'withdrawals', top_n: int = 20) -> List[Dict]:
    """
    Rank ATMs by performance metric
    
    Args:
        metric: 'withdrawals', 'utilization', 'efficiency'
        top_n: Number of top ATMs to return
    
    Returns:
        Ranked list of ATMs
    """
    df = load_structured().copy()
    
    if metric == 'withdrawals':
        ranked = df.sort_values('Withdrawals', ascending=False)
        cols = ['ATM_ID', 'City', 'Location_Type', 'Withdrawals', 'Predicted_Next_Day']
    
    elif metric == 'utilization':
        df['utilization'] = (df['Predicted_Next_Day'] / df['Current_Balance'].clip(lower=1)) * 100
        ranked = df.sort_values('utilization', ascending=False)
        cols = ['ATM_ID', 'City', 'Location_Type', 'utilization', 'Current_Balance', 'Predicted_Next_Day']
    
    elif metric == 'efficiency':
        df['efficiency'] = df['Withdrawals'] / df['Current_Balance'].clip(lower=1)
        ranked = df.sort_values('efficiency', ascending=False)
        cols = ['ATM_ID', 'City', 'Location_Type', 'efficiency', 'Withdrawals', 'Current_Balance']
    
    else:
        return [{'error': f'Unknown metric: {metric}'}]
    
    return ranked[cols].head(top_n).to_dict(orient='records')


def city_comparison_report() -> Dict:
    """
    Compare performance across cities
    
    Returns:
        City-level comparative analysis
    """
    df = load_structured().copy()
    
    city_metrics = df.groupby('City').agg({
        'ATM_ID': 'count',
        'Withdrawals': ['sum', 'mean', 'std'],
        'Current_Balance': 'sum',
        'Predicted_Next_Day': 'sum'
    }).reset_index()
    
    city_metrics.columns = ['City', 'ATM_Count', 'Total_Withdrawals', 'Avg_Withdrawals', 
                             'Std_Withdrawals', 'Total_Balance', 'Total_Demand']
    
    # Calculate derived metrics
    city_metrics['demand_per_atm'] = city_metrics['Total_Demand'] / city_metrics['ATM_Count']
    city_metrics['buffer_days'] = city_metrics['Total_Balance'] / city_metrics['Total_Demand'].clip(lower=1)
    city_metrics['volatility'] = (city_metrics['Std_Withdrawals'] / city_metrics['Avg_Withdrawals'] * 100).round(2)
    
    # Sort by total demand
    city_metrics = city_metrics.sort_values('Total_Demand', ascending=False)
    
    return {
        'city_comparison': city_metrics.to_dict(orient='records'),
        'highest_demand_city': city_metrics.iloc[0]['City'] if len(city_metrics) > 0 else None,
        'most_atms_city': city_metrics.loc[city_metrics['ATM_Count'].idxmax(), 'City'] if len(city_metrics) > 0 else None
    }