"""
Base Collector - Abstract base class for all data collectors
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import logging
from pathlib import Path
import time
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.settings import data_source_config, market_config

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """Abstract base class for data collectors"""
    
    def __init__(self, name: str):
        self.name = name
        self.config = data_source_config
        self.market_config = market_config
        self.session = None
        self._last_request_time = 0
        
    def _rate_limit(self):
        """Implement rate limiting between requests"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.request_delay_seconds:
            time.sleep(self.config.request_delay_seconds - elapsed)
        self._last_request_time = time.time()
    
    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol
        
        Args:
            symbol: Stock/Index symbol
            start_date: Start date for data
            end_date: End date for data
            interval: Time interval (1m, 5m, 15m, 1h, 1d, etc.)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        pass
    
    @abstractmethod
    def fetch_live_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch live quote for a symbol
        
        Args:
            symbol: Stock/Index symbol
            
        Returns:
            Dictionary with current price data
        """
        pass
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean fetched data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Validated DataFrame
        """
        if df.empty:
            logger.warning(f"{self.name}: Received empty DataFrame")
            return df
            
        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"{self.name}: Missing columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp to datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Localize to IST
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('Asia/Kolkata')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Kolkata')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Handle missing values
        df['volume'] = df['volume'].fillna(0)
        
        # Forward fill price data (for gaps)
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].ffill()
        
        # Ensure positive values
        df = df[df['close'] > 0]
        
        # OHLC sanity check
        df = df[
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        ]
        
        logger.info(f"{self.name}: Validated {len(df)} rows")
        return df
    
    def _log_fetch(self, symbol: str, start: datetime, end: datetime, rows: int):
        """Log fetch operation"""
        logger.info(
            f"{self.name}: Fetched {rows} rows for {symbol} "
            f"from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
        )
    
    def health_check(self) -> bool:
        """Check if the data source is accessible"""
        try:
            # Try to fetch a simple quote
            self.fetch_live_quote("NIFTY 50")
            return True
        except Exception as e:
            logger.error(f"{self.name} health check failed: {e}")
            return False
