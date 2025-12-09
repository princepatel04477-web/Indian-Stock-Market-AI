"""
Ensemble Model - Combine predictions from multiple models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

from .lgbm_model import LGBMModel
from .xgb_model import XGBModel

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Ensemble of multiple models with weighted voting"""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.models: Dict[str, object] = {}
        self.weights = weights or {'lgbm': 0.4, 'xgb': 0.3, 'tft': 0.3}
        self.is_trained = False
    
    def add_model(self, name: str, model: object, weight: float = None):
        """Add a model to the ensemble"""
        self.models[name] = model
        if weight is not None:
            self.weights[name] = weight
        
        # Normalize weights
        total = sum(self.weights.get(n, 0) for n in self.models.keys())
        if total > 0:
            for n in self.models.keys():
                self.weights[n] = self.weights.get(n, 0) / total
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Train all models in ensemble"""
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            model.train(X_train, y_train, X_val, y_val)
        
        self.is_trained = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using weighted voting"""
        if not self.is_trained:
            raise ValueError("Ensemble not trained")
        
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get weighted probability predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble not trained")
        
        combined_proba = None
        
        for name, model in self.models.items():
            weight = self.weights.get(name, 1.0 / len(self.models))
            
            try:
                proba = model.predict_proba(X)
                if combined_proba is None:
                    combined_proba = proba * weight
                else:
                    combined_proba += proba * weight
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")
        
        return combined_proba
    
    def get_signal(self, X: pd.DataFrame) -> Dict:
        """Get ensemble trading signal"""
        proba = self.predict_proba(X)
        
        if len(proba.shape) == 1:
            proba = proba.reshape(1, -1)
        
        latest_proba = proba[-1]
        pred_class = np.argmax(latest_proba)
        confidence = latest_proba[pred_class]
        
        # Get individual model signals
        individual_signals = {}
        for name, model in self.models.items():
            try:
                sig = model.get_signal(X)
                individual_signals[name] = sig
            except:
                pass
        
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        
        return {
            'signal': signal_map.get(pred_class, 'HOLD'),
            'confidence': float(confidence),
            'probabilities': {
                'sell': float(latest_proba[0]),
                'hold': float(latest_proba[1]),
                'buy': float(latest_proba[2])
            },
            'individual_models': individual_signals,
            'weights': self.weights
        }
    
    def save(self, path: str):
        """Save all models"""
        from pathlib import Path
        import joblib
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model.save(str(path / name))
        
        joblib.dump(self.weights, str(path / 'weights.pkl'))
        logger.info(f"Saved ensemble to {path}")
    
    def load(self, path: str) -> 'EnsembleModel':
        """Load ensemble models"""
        from pathlib import Path
        import joblib
        
        path = Path(path)
        self.weights = joblib.load(str(path / 'weights.pkl'))
        
        # Load each model
        for model_dir in path.iterdir():
            if model_dir.is_dir():
                name = model_dir.name
                if name == 'lgbm':
                    self.models[name] = LGBMModel().load(str(model_dir / name))
                elif name == 'xgb':
                    self.models[name] = XGBModel().load(str(model_dir / name))
        
        self.is_trained = True
        return self
