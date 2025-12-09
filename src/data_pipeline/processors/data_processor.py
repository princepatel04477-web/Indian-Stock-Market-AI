"""
Data Processor - Clean and resample market data
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process raw market data into clean, analysis-ready format"""
    
    def __init__(self):
        self.market_open = time(9, 15)
        self.market_close = time(15, 30)
        
    def clean_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean OHLCV data"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Ensure timestamp column
        if 'timestamp' not in df.columns:
            df = df.reset_index()
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'timestamp'})
        
        # Convert to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'])
        
        # Sort by time
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Handle missing values
        df['volume'] = df['volume'].fillna(0)
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].ffill()
        
        # Remove invalid rows
        df = df[df['close'] > 0]
        df = df[(df['high'] >= df['low']) & (df['high'] >= df['close'])]
        
        logger.info(f"Cleaned data: {len(df)} rows")
        return df
    
    def resample_ohlcv(self, df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        """Resample OHLCV to different timeframe"""
        if df.empty:
            return df
        
        df = df.set_index('timestamp')
        
        resampled = df.resample(target_interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled.reset_index()
    
    def filter_market_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to market hours only"""
        if df.empty:
            return df
        
        df = df.copy()
        df['time'] = df['timestamp'].dt.time
        
        mask = (df['time'] >= self.market_open) & (df['time'] <= self.market_close)
        filtered = df[mask].drop(columns=['time'])
        
        return filtered.reset_index(drop=True)
    
    def add_session_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trading session information"""
        if df.empty:
            return df
        
        df = df.copy()
        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_monday'] = df['day_of_week'] == 0
        df['is_friday'] = df['day_of_week'] == 4
        
        # Session markers
        df['is_open'] = df['time'] == self.market_open
        df['is_close'] = df['time'] == self.market_close
        
        return df
    
    def compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return columns"""
        if df.empty:
            return df
        
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['intraday_return'] = (df['close'] - df['open']) / df['open']
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        return df
    
    def merge_dataframes(self, dfs: List[pd.DataFrame], 
                         on: str = 'timestamp') -> pd.DataFrame:
        """Merge multiple dataframes on timestamp"""
        if not dfs:
            return pd.DataFrame()
        
        result = dfs[0]
        for df in dfs[1:]:
            result = pd.merge(result, df, on=on, how='outer')
        
        return result.sort_values(on).reset_index(drop=True)
