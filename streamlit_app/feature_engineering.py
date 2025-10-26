"""
Feature Engineering - RF (Slim) VERSION
FILE: ZIG018/streamlit_app/feature_engineering.py

This version produces EXACTLY the features your RandomForest pipeline
was trained on (13 numeric + 5 categorical). It also tolerates common
column names from uploaded CSVs (e.g., Withdrawals -> Total_Withdrawals,
Deposits -> Total_Deposits, IsHoliday -> Holiday_Flag, Location -> Location_Type, Weather -> Weather_Condition).
"""

import pandas as pd
import numpy as np


class ATMFeatureEngineer:
    def __init__(self):
        pass

    def _coerce_base_columns(self, df: pd.DataFrame, atm_id: str | None) -> pd.DataFrame:
        """Map common raw columns to the canonical names your model expects."""
        df = df.copy()

        # Date
        if "Date" not in df.columns:
            df["Date"] = pd.Timestamp("1970-01-01")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Map raw columns -> canonical names used in training
        # Withdrawals / Deposits (totals)
        if "Total_Withdrawals" not in df.columns:
            if "Withdrawals" in df.columns:
                df["Total_Withdrawals"] = pd.to_numeric(df["Withdrawals"], errors="coerce")
            else:
                df["Total_Withdrawals"] = 0.0

        if "Total_Deposits" not in df.columns:
            if "Deposits" in df.columns:
                df["Total_Deposits"] = pd.to_numeric(df["Deposits"], errors="coerce")
            else:
                df["Total_Deposits"] = 0.0

        # Holiday flag
        if "Holiday_Flag" not in df.columns:
            if "IsHoliday" in df.columns:
                df["Holiday_Flag"] = pd.to_numeric(df["IsHoliday"], errors="coerce").fillna(0).astype(int)
            else:
                df["Holiday_Flag"] = 0

        # Location / Weather categorical
        if "Location_Type" not in df.columns:
            df["Location_Type"] = df.get("Location", "bank_branch").astype(str)
        if "Weather_Condition" not in df.columns:
            df["Weather_Condition"] = df.get("Weather", "clear").astype(str)

        # ATM_ID
        if "ATM_ID" not in df.columns:
            df["ATM_ID"] = atm_id if atm_id else "ATM_001"
        df["ATM_ID"] = df["ATM_ID"].astype(str)

        # Day of week label
        if "Day_of_Week" not in df.columns:
            if "DayName" in df.columns:
                df["Day_of_Week"] = df["DayName"].astype(str)
            else:
                df["Day_of_Week"] = df["Date"].dt.day_name().astype(str)

        # Fill numerics with 0 when missing
        for c in ["Total_Withdrawals", "Total_Deposits"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        return df

    def engineer_all_features(self, df: pd.DataFrame, atm_id: str | None = None):
        """
        Build features used by the RF model. Safe for both training-style
        datasets and the single-ATM slice used in the app.
        Returns: processed_df, issues, suggestions (issues/suggestions kept for API compatibility)
        """
        issues, suggestions = [], []
        df = self._coerce_base_columns(df, atm_id)

        # Sort for lag/rolling
        order_cols = ["ATM_ID", "Date"] if "ATM_ID" in df.columns else ["Date"]
        df = df.sort_values(order_cols).reset_index(drop=True)

        # Time parts
        df["Year"] = df["Date"].dt.year.fillna(1970).astype(int)
        df["Month"] = df["Date"].dt.month.fillna(1).astype(int)
        df["Day"] = df["Date"].dt.day.fillna(1).astype(int)
        df["Quarter"] = df["Date"].dt.quarter.fillna(1).astype(int)

        # Lags (by ATM)
        grp = df.groupby("ATM_ID") if "ATM_ID" in df.columns else [(None, df)]
        df["Lag_1"] = np.nan
        df["Lag_7"] = np.nan
        df["RollingMean_3"] = np.nan
        df["RollingMean_7"] = np.nan

        if isinstance(grp, pd.core.groupby.generic.DataFrameGroupBy):
            for _, g in grp:
                idx = g.index
                df.loc[idx, "Lag_1"] = g["Total_Withdrawals"].shift(1)
                df.loc[idx, "Lag_7"] = g["Total_Withdrawals"].shift(7)
                df.loc[idx, "RollingMean_3"] = g["Total_Withdrawals"].rolling(3, min_periods=1).mean()
                df.loc[idx, "RollingMean_7"] = g["Total_Withdrawals"].rolling(7, min_periods=1).mean()
        else:
            g = df
            df["Lag_1"] = g["Total_Withdrawals"].shift(1)
            df["Lag_7"] = g["Total_Withdrawals"].shift(7)
            df["RollingMean_3"] = g["Total_Withdrawals"].rolling(3, min_periods=1).mean()
            df["RollingMean_7"] = g["Total_Withdrawals"].rolling(7, min_periods=1).mean()

        # Cyclical encodings
        df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
        df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)

        # Fill any NaNs from early rows
        # (during prediction we take the last row, so these won't matter much,
        #  but this keeps the frame complete if you preview it)
        for c in ["Lag_1", "Lag_7", "RollingMean_3", "RollingMean_7"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        return df, issues, suggestions

    def prepare_for_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Project a dataframe to EXACTLY the columns your RF pipeline expects,
        with the correct dtypes and order. Returns a single-row frame (last row).
        """
        df = df.copy()

        # Get ATM_ID value properly from DataFrame
        if "ATM_ID" in df.columns and len(df) > 0:
            atm_id_val = df["ATM_ID"].iloc[-1]
        else:
            atm_id_val = "ATM_001"
        
        # Ensure base and engineered fields exist (idempotent)
        df, _, _ = self.engineer_all_features(df, atm_id=atm_id_val)

        # REQUIRED COLUMN LISTS (must match training)
        numeric_features = [
            "Total_Withdrawals", "Total_Deposits", "Holiday_Flag",
            "Year", "Month", "Day", "Lag_1", "Lag_7",
            "RollingMean_3", "RollingMean_7", "Quarter", "Month_Sin", "Month_Cos",
        ]
        categorical_features = [
            "ATM_ID", "Date", "Day_of_Week", "Location_Type", "Weather_Condition",
        ]

        # Coerce types/formats
        # Date must be STRING for OneHotEncoder in your pipeline
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("1970-01-01")
        else:
            df["Date"] = "1970-01-01"

        for c in ["Year", "Month", "Day", "Quarter"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

        for c in ["Lag_1", "Lag_7", "RollingMean_3", "RollingMean_7",
                  "Total_Withdrawals", "Total_Deposits", "Holiday_Flag",
                  "Month_Sin", "Month_Cos"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        for c in ["ATM_ID", "Day_of_Week", "Location_Type", "Weather_Condition", "Date"]:
            df[c] = df.get(c, "").astype(str)

        # Build final single-row input in the exact order
        required = numeric_features + categorical_features
        df = df[required]

        # Keep only the last row (the app uses the most recent engineered row)
        return df.tail(1)