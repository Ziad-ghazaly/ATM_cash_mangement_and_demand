import joblib
from typing import Tuple, Optional

class ModelLoader:
    """Simplified loader for a single RandomForest pipeline."""

    def __init__(self):
        self.model = None
        self.current_model_name = "RandomForest"

    def load_model(self, path: str) -> Tuple[Optional[object], Optional[str]]:
        """Load the joblib RandomForest pipeline."""
        try:
            self.model = joblib.load(path)
            return self, None
        except Exception as e:
            return None, str(e)

    def predict(self, X):
        """Proxy predict call."""
        return self.model.predict(X)

    def get_model_info(self):
        """Return basic info for the sidebar."""
        return {
            "is_multi_model": False,
            "current_model": self.current_model_name,
            "available_models": [self.current_model_name],
            "metadata": {"saved_at": "streamlit_app/trained_model/atm_rf_model.pkl"},
        }

    def switch_model(self, *_args, **_kwargs):
        return True
