# save_predictions.py (example)
import pandas as pd
def export_predictions(df: pd.DataFrame, path: str):
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
