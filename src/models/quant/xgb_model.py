"""
XGBoost Model - Gradient boosting for trading signals
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
import joblib
import logging

logger = logging.getLogger(__name__)


class XGBModel:
    """XGBoost model for classification and regression"""
    
    def __init__(self, params: Dict = None, task: str = 'classification'):
        self.task = task
        self.model = None
        self.feature_names = None
        
        default_params = {
            'objective': 'multi:softprob' if task == 'classification' else 'reg:squarederror',
            'num_class': 3 if task == 'classification' else None,
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss' if task == 'classification' else 'rmse',
            'use_label_encoder': False,
            'verbosity': 0
        }
        
        self.params = {**default_params, **(params or {})}
        if self.params.get('num_class') is None:
            del self.params['num_class']
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              feature_names: List[str] = None) -> 'XGBModel':
        """Train the model"""
        self.feature_names = feature_names or list(X_train.columns)
        
        # Create model
        if self.task == 'classification':
            self.model = xgb.XGBClassifier(**self.params)
        else:
            self.model = xgb.XGBRegressor(**self.params)
        
        # Fit
        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        logger.info(f"Trained XGBoost with {self.model.n_estimators} trees")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels"""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if self.model is None:
            return pd.DataFrame()
        
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save(self, path: str):
        """Save model"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save_model(str(path.with_suffix('.json')))
        joblib.dump({
            'params': self.params,
            'feature_names': self.feature_names,
            'task': self.task
        }, str(path.with_suffix('.meta')))
        
        logger.info(f"Saved XGBoost model to {path}")
    
    def load(self, path: str) -> 'XGBModel':
        """Load model"""
        path = Path(path)
        
        if self.task == 'classification':
            self.model = xgb.XGBClassifier()
        else:
            self.model = xgb.XGBRegressor()
        
        self.model.load_model(str(path.with_suffix('.json')))
        
        meta = joblib.load(str(path.with_suffix('.meta')))
        self.params = meta['params']
        self.feature_names = meta['feature_names']
        self.task = meta['task']
        
        return self
    
    def get_signal(self, X: pd.DataFrame) -> Dict:
        """Get trading signal with confidence"""
        proba = self.predict_proba(X)
        
        if len(proba.shape) == 1:
            proba = proba.reshape(1, -1)
        
        latest_proba = proba[-1]
        pred_class = np.argmax(latest_proba)
        confidence = latest_proba[pred_class]
        
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        
        return {
            'signal': signal_map.get(pred_class, 'HOLD'),
            'confidence': float(confidence),
            'probabilities': {
                'sell': float(latest_proba[0]),
                'hold': float(latest_proba[1]),
                'buy': float(latest_proba[2])
            }
        }
