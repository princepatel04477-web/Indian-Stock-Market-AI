"""
Data Normalizer - Normalize and scale features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataNormalizer:
    """Normalize features for ML models"""
    
    def __init__(self, method: str = 'robust'):
        """
        Args:
            method: 'standard', 'minmax', or 'robust'
        """
        self.method = method
        self.scalers: Dict[str, object] = {}
        
        scaler_map = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler
        }
        self.scaler_class = scaler_map.get(method, RobustScaler)
    
    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'DataNormalizer':
        """Fit scalers on training data"""
        for col in columns:
            if col in df.columns:
                scaler = self.scaler_class()
                values = df[[col]].dropna().values
                if len(values) > 0:
                    scaler.fit(values)
                    self.scalers[col] = scaler
        
        logger.info(f"Fitted {len(self.scalers)} scalers")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scalers"""
        df = df.copy()
        
        for col, scaler in self.scalers.items():
            if col in df.columns:
                mask = df[col].notna()
                df.loc[mask, col] = scaler.transform(df.loc[mask, [col]])
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(df, columns)
        return self.transform(df)
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform to original scale"""
        df = df.copy()
        
        for col, scaler in self.scalers.items():
            if col in df.columns:
                mask = df[col].notna()
                df.loc[mask, col] = scaler.inverse_transform(df.loc[mask, [col]])
        
        return df
    
    def save(self, path: str):
        """Save scalers to disk"""
        joblib.dump(self.scalers, path)
        logger.info(f"Saved scalers to {path}")
    
    def load(self, path: str):
        """Load scalers from disk"""
        self.scalers = joblib.load(path)
        logger.info(f"Loaded {len(self.scalers)} scalers from {path}")
        return self
    
    @staticmethod
    def winsorize(df: pd.DataFrame, columns: List[str], 
                  lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
        """Clip outliers using percentiles"""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                lower_val = df[col].quantile(lower)
                upper_val = df[col].quantile(upper)
                df[col] = df[col].clip(lower_val, upper_val)
        return df
    
    @staticmethod
    def log_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply log transformation to positive columns"""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))
        return df
