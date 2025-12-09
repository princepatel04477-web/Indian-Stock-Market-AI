"""
Label Creator - Create target labels for ML models
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LabelCreator:
    """Create labels for various trading strategies"""
    
    def __init__(self, 
                 atr_multiplier: float = 0.5,
                 slippage_pct: float = 0.0005,
                 commission_pct: float = 0.0001):
        self.atr_multiplier = atr_multiplier
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct
        self.total_cost = self.slippage_pct + self.commission_pct
    
    def create_labels(self, df: pd.DataFrame, 
                      strategy: str = 'intraday',
                      forward_window: int = None) -> pd.DataFrame:
        """Create labels based on strategy type"""
        df = df.copy()
        
        # Set default forward windows
        windows = {
            'intraday': 15,    # 15 bars (e.g., 15 minutes)
            'swing': 1440,     # 1 day in minutes
            'positional': 7200 # 5 days
        }
        forward_window = forward_window or windows.get(strategy, 60)
        
        df = self.add_forward_returns(df, forward_window)
        df = self.add_binary_labels(df, forward_window)
        df = self.add_multiclass_labels(df, forward_window)
        
        return df
    
    def add_forward_returns(self, df: pd.DataFrame, 
                            window: int) -> pd.DataFrame:
        """Add forward return columns"""
        df[f'fut_ret_{window}'] = df['close'].shift(-window) / df['close'] - 1
        df[f'fut_log_ret_{window}'] = np.log(df['close'].shift(-window) / df['close'])
        
        # Net return after costs
        df[f'fut_ret_net_{window}'] = df[f'fut_ret_{window}'] - self.total_cost
        
        return df
    
    def add_binary_labels(self, df: pd.DataFrame, 
                          window: int) -> pd.DataFrame:
        """Add binary up/down labels"""
        ret_col = f'fut_ret_net_{window}'
        
        if ret_col not in df.columns:
            return df
        
        # Simple direction labels
        df[f'label_binary_{window}'] = (df[ret_col] > 0).astype(int)
        
        return df
    
    def add_multiclass_labels(self, df: pd.DataFrame, 
                               window: int) -> pd.DataFrame:
        """Add multiclass BUY/HOLD/SELL labels"""
        ret_col = f'fut_ret_net_{window}'
        
        if ret_col not in df.columns:
            return df
        
        # ATR-based thresholds
        if 'atr_percent' in df.columns:
            threshold = df['atr_percent'] / 100 * self.atr_multiplier
        else:
            # Fixed threshold as fallback
            threshold = 0.005  # 0.5%
        
        # Create labels: 0=SELL, 1=HOLD, 2=BUY
        df[f'label_{window}'] = 1  # Default HOLD
        df.loc[df[ret_col] > threshold, f'label_{window}'] = 2  # BUY
        df.loc[df[ret_col] < -threshold, f'label_{window}'] = 0  # SELL
        
        return df
    
    def add_regression_labels(self, df: pd.DataFrame,
                               windows: list = [15, 60, 240]) -> pd.DataFrame:
        """Add regression targets (future returns)"""
        for w in windows:
            df[f'target_ret_{w}'] = df['close'].shift(-w) / df['close'] - 1
        
        return df
    
    def get_label_distribution(self, df: pd.DataFrame, 
                                label_col: str) -> Dict:
        """Get distribution of labels"""
        if label_col not in df.columns:
            return {}
        
        counts = df[label_col].value_counts(normalize=True)
        return {
            'distribution': counts.to_dict(),
            'total_samples': len(df),
            'class_balance': counts.min() / counts.max()
        }
    
    @staticmethod
    def label_to_signal(label: int) -> str:
        """Convert numeric label to signal string"""
        mapping = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        return mapping.get(label, 'HOLD')
