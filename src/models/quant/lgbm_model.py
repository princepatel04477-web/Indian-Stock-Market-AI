"""
LightGBM Model - Fast gradient boosting for trading signals
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import joblib
import logging

logger = logging.getLogger(__name__)


class LGBMModel:
    """LightGBM model for classification and regression"""
    
    def __init__(self, params: Dict = None, task: str = 'classification'):
        self.task = task
        self.model = None
        self.feature_names = None
        
        # Default parameters
        default_params = {
            'objective': 'multiclass' if task == 'classification' else 'regression',
            'num_class': 3 if task == 'classification' else None,
            'metric': 'multi_logloss' if task == 'classification' else 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_estimators': 1000,
            'early_stopping_rounds': 50
        }
        
        self.params = {**default_params, **(params or {})}
        if self.params.get('num_class') is None:
            del self.params['num_class']
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              feature_names: List[str] = None) -> 'LGBMModel':
        """Train the model"""
        self.feature_names = feature_names or list(X_train.columns)
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
        
        # Train
        callbacks = [lgb.early_stopping(self.params.get('early_stopping_rounds', 50))]
        
        self.model = lgb.train(
            params={k: v for k, v in self.params.items() 
                   if k not in ['n_estimators', 'early_stopping_rounds']},
            train_set=train_data,
            num_boost_round=self.params.get('n_estimators', 1000),
            valid_sets=valid_sets,
            callbacks=callbacks
        )
        
        logger.info(f"Trained LightGBM with {self.model.num_trees()} trees")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """Get feature importance"""
        if self.model is None:
            return pd.DataFrame()
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save(self, path: str):
        """Save model to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save both booster and metadata
        self.model.save_model(str(path.with_suffix('.txt')))
        joblib.dump({
            'params': self.params,
            'feature_names': self.feature_names,
            'task': self.task
        }, str(path.with_suffix('.meta')))
        
        logger.info(f"Saved model to {path}")
    
    def load(self, path: str) -> 'LGBMModel':
        """Load model from disk"""
        path = Path(path)
        
        self.model = lgb.Booster(model_file=str(path.with_suffix('.txt')))
        
        meta = joblib.load(str(path.with_suffix('.meta')))
        self.params = meta['params']
        self.feature_names = meta['feature_names']
        self.task = meta['task']
        
        logger.info(f"Loaded model from {path}")
        return self
    
    def get_signal(self, X: pd.DataFrame) -> Dict:
        """Get trading signal with confidence"""
        proba = self.predict_proba(X)
        
        if len(proba.shape) == 1:
            proba = proba.reshape(1, -1)
        
        # Get latest prediction
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
