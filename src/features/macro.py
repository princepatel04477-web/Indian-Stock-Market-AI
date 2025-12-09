"""
Macro Features - Economic and sentiment features
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class MacroFeatures:
    """Calculate macro-economic and calendar features"""
    
    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar-based features"""
        df = df.copy()
        
        if 'timestamp' not in df.columns:
            return df
        
        ts = pd.to_datetime(df['timestamp'])
        
        # Day of week
        df['day_of_week'] = ts.dt.dayofweek
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # Month
        df['month'] = ts.dt.month
        df['is_month_start'] = ts.dt.is_month_start.astype(int)
        df['is_month_end'] = ts.dt.is_month_end.astype(int)
        
        # Quarter
        df['quarter'] = ts.dt.quarter
        df['is_quarter_end'] = ts.dt.is_quarter_end.astype(int)
        
        # Expiry flag (Thursday)
        df['is_expiry_day'] = (df['day_of_week'] == 3).astype(int)
        
        # Days to next expiry
        df['days_to_expiry'] = (3 - df['day_of_week']) % 7
        
        # Time of day (for intraday)
        if ts.dt.hour.nunique() > 1:
            df['hour'] = ts.dt.hour
            df['minute'] = ts.dt.minute
            df['is_opening_hour'] = (df['hour'] == 9).astype(int)
            df['is_closing_hour'] = (df['hour'] == 15).astype(int)
        
        return df
    
    def add_vix_features(self, df: pd.DataFrame, 
                         vix_df: pd.DataFrame = None) -> pd.DataFrame:
        """Add India VIX-based features"""
        if vix_df is None or vix_df.empty:
            return df
        
        df = df.copy()
        
        # Merge VIX data
        vix_df = vix_df.rename(columns={'close': 'vix'})
        vix_df['date'] = pd.to_datetime(vix_df['timestamp']).dt.date
        
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        df = df.merge(vix_df[['date', 'vix']], on='date', how='left')
        
        # VIX features
        df['vix_high'] = (df['vix'] > 20).astype(int)
        df['vix_extreme'] = (df['vix'] > 25).astype(int)
        
        return df.drop(columns=['date'])
    
    def add_fii_dii_features(self, df: pd.DataFrame,
                             fii_dii_df: pd.DataFrame = None) -> pd.DataFrame:
        """Add FII/DII flow features"""
        if fii_dii_df is None or fii_dii_df.empty:
            return df
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        # Merge FII/DII data
        df = df.merge(fii_dii_df, on='date', how='left')
        
        # Cumulative flows
        if 'fii_net' in df.columns:
            df['fii_net_5d'] = df['fii_net'].rolling(5).sum()
            df['dii_net_5d'] = df['dii_net'].rolling(5).sum()
        
        return df.drop(columns=['date'], errors='ignore')
    
    def get_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify market regime (trending/ranging/volatile)"""
        df = df.copy()
        
        if 'atr_percent' not in df.columns or 'adx' not in df.columns:
            return df
        
        vol_threshold = df['atr_percent'].quantile(0.75)
        trend_threshold = 25
        
        conditions = [
            (df['adx'] > trend_threshold) & (df['atr_percent'] > vol_threshold),
            (df['adx'] > trend_threshold) & (df['atr_percent'] <= vol_threshold),
            (df['adx'] <= trend_threshold) & (df['atr_percent'] > vol_threshold),
        ]
        choices = ['trending_volatile', 'trending_calm', 'ranging_volatile']
        
        df['market_regime'] = np.select(conditions, choices, default='ranging_calm')
        
        return df
