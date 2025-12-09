"""
Model Trainer - Training pipeline with walk-forward validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import logging

from .lgbm_model import LGBMModel
from .xgb_model import XGBModel
from .ensemble import EnsembleModel

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Training pipeline with proper time-series validation"""
    
    def __init__(self, model_type: str = 'lgbm'):
        self.model_type = model_type
        self.model = None
        self.metrics_history = []
    
    def create_model(self, params: Dict = None) -> object:
        """Create model instance"""
        if self.model_type == 'lgbm':
            return LGBMModel(params)
        elif self.model_type == 'xgb':
            return XGBModel(params)
        elif self.model_type == 'ensemble':
            return EnsembleModel()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def time_series_split(self, df: pd.DataFrame,
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data maintaining temporal order"""
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]
        
        logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test
    
    def walk_forward_split(self, df: pd.DataFrame,
                           train_window: int = 252,
                           test_window: int = 63,
                           step: int = 21) -> List[Tuple]:
        """Generate walk-forward validation splits"""
        splits = []
        n = len(df)
        
        start = 0
        while start + train_window + test_window <= n:
            train_end = start + train_window
            test_end = train_end + test_window
            
            splits.append((
                df.iloc[start:train_end],
                df.iloc[train_end:test_end]
            ))
            
            start += step
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits
    
    def train(self, df: pd.DataFrame,
              feature_cols: List[str],
              label_col: str,
              params: Dict = None) -> object:
        """Train model with validation"""
        # Split data
        train, val, test = self.time_series_split(df)
        
        X_train, y_train = train[feature_cols], train[label_col]
        X_val, y_val = val[feature_cols], val[label_col]
        X_test, y_test = test[feature_cols], test[label_col]
        
        # Drop NaNs
        mask = ~(X_train.isna().any(axis=1) | y_train.isna())
        X_train, y_train = X_train[mask], y_train[mask]
        
        mask = ~(X_val.isna().any(axis=1) | y_val.isna())
        X_val, y_val = X_val[mask], y_val[mask]
        
        # Create and train
        self.model = self.create_model(params)
        self.model.train(X_train, y_train, X_val, y_val, feature_cols)
        
        # Evaluate on test
        metrics = self.evaluate(X_test, y_test)
        self.metrics_history.append(metrics)
        
        logger.info(f"Test metrics: {metrics}")
        return self.model
    
    def train_walk_forward(self, df: pd.DataFrame,
                           feature_cols: List[str],
                           label_col: str,
                           params: Dict = None) -> List[Dict]:
        """Train with walk-forward validation"""
        splits = self.walk_forward_split(df)
        results = []
        
        for i, (train_df, test_df) in enumerate(splits):
            X_train, y_train = train_df[feature_cols], train_df[label_col]
            X_test, y_test = test_df[feature_cols], test_df[label_col]
            
            # Clean NaNs
            mask = ~(X_train.isna().any(axis=1) | y_train.isna())
            X_train, y_train = X_train[mask], y_train[mask]
            
            mask = ~(X_test.isna().any(axis=1) | y_test.isna())
            X_test, y_test = X_test[mask], y_test[mask]
            
            if len(X_train) == 0 or len(X_test) == 0:
                continue
            
            # Train
            model = self.create_model(params)
            model.train(X_train, y_train, feature_names=feature_cols)
            
            # Evaluate
            metrics = self.evaluate(X_test, y_test, model)
            metrics['fold'] = i
            results.append(metrics)
            
            logger.info(f"Fold {i}: accuracy={metrics['accuracy']:.3f}")
        
        # Store the last model
        self.model = model
        self.metrics_history = results
        
        return results
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, 
                 model: object = None) -> Dict:
        """Evaluate model performance"""
        model = model or self.model
        
        if model is None:
            raise ValueError("No model to evaluate")
        
        y_pred = model.predict(X)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0),
            'samples': len(y)
        }
    
    def get_summary(self) -> Dict:
        """Get training summary"""
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        
        return {
            'mean_accuracy': df['accuracy'].mean(),
            'std_accuracy': df['accuracy'].std(),
            'mean_f1': df['f1'].mean(),
            'num_folds': len(df),
            'total_samples': df['samples'].sum()
        }
