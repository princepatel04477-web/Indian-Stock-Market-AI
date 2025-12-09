"""
Technical Indicators - Compute all technical features for trading signals
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TechnicalFeatures:
    """Calculate technical indicators for price data"""
    
    def __init__(self):
        self.ema_periods = [8, 21, 50, 200]
        self.sma_periods = [10, 20, 50, 100, 200]
        self.rsi_period = 14
        self.atr_period = 14
        
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical features"""
        df = df.copy()
        
        # Returns
        df = self.add_returns(df)
        
        # Moving averages
        df = self.add_ema(df)
        df = self.add_sma(df)
        
        # Momentum
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_stochastic(df)
        
        # Volatility
        df = self.add_atr(df)
        df = self.add_bollinger_bands(df)
        
        # Volume
        df = self.add_vwap(df)
        df = self.add_volume_features(df)
        
        # Trend
        df = self.add_adx(df)
        
        # Rolling stats
        df = self.add_rolling_stats(df)
        
        logger.info(f"Computed {len(df.columns)} features")
        return df
    
    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return features"""
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['intraday_return'] = (df['close'] - df['open']) / df['open']
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        return df
    
    def add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add exponential moving averages"""
        for period in self.ema_periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'close_vs_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
        return df
    
    def add_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple moving averages"""
        for period in self.sma_periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        return df
    
    def add_rsi(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """Add Relative Strength Index"""
        period = period or self.rsi_period
        delta = df['close'].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI zones
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        return df
    
    def add_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, 
                 signal: int = 9) -> pd.DataFrame:
        """Add MACD indicator"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # MACD crossover signals
        df['macd_bullish'] = ((df['macd'] > df['macd_signal']) & 
                              (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_bearish'] = ((df['macd'] < df['macd_signal']) & 
                              (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        
        return df
    
    def add_stochastic(self, df: pd.DataFrame, k_period: int = 14, 
                       d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        return df
    
    def add_atr(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """Add Average True Range"""
        period = period or self.atr_period
        
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=period).mean()
        df['atr_percent'] = df['atr'] / df['close'] * 100
        
        return df
    
    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, 
                            std: float = 2.0) -> pd.DataFrame:
        """Add Bollinger Bands"""
        sma = df['close'].rolling(window=period).mean()
        std_dev = df['close'].rolling(window=period).std()
        
        df['bb_upper'] = sma + (std * std_dev)
        df['bb_lower'] = sma - (std * std_dev)
        df['bb_middle'] = sma
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Volume Weighted Average Price"""
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            df['vwap'] = df['close']
            return df
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        df['close_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        if 'volume' not in df.columns:
            return df
        
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # On-Balance Volume
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        
        # Volume-price trend
        df['vpt'] = (df['volume'] * df['returns']).cumsum()
        
        return df
    
    def add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average Directional Index"""
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff().abs()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        atr = df.get('atr', self.add_atr(df.copy())['atr'])
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(window=period).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        return df
    
    def add_rolling_stats(self, df: pd.DataFrame, 
                          windows: List[int] = [5, 15, 60]) -> pd.DataFrame:
        """Add rolling statistics at multiple windows"""
        for w in windows:
            df[f'ret_mean_{w}'] = df['returns'].rolling(w).mean()
            df[f'ret_std_{w}'] = df['returns'].rolling(w).std()
            df[f'ret_skew_{w}'] = df['returns'].rolling(w).skew()
            df[f'close_zscore_{w}'] = (df['close'] - df['close'].rolling(w).mean()) / df['close'].rolling(w).std()
            df[f'high_max_{w}'] = df['high'].rolling(w).max()
            df[f'low_min_{w}'] = df['low'].rolling(w).min()
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, 
                         lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Add lagged features"""
        for lag in lags:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag) if 'volume' in df.columns else 0
        
        return df
